if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import tqdm
from torch.utils.data import DataLoader
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, DistributedDataParallelKwargs
import pickle

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.policy.unified_video_action_policy import (
    UnifiedVideoActionPolicy,
)
from unified_video_action.dataset.base_dataset import BaseImageDataset
from unified_video_action.dataset.umi_multi_dataset import UmiMultiDataset
from unified_video_action.common.checkpoint_util import TopKCheckpointManager
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.model.autoregressive.ema_model import EMAModel
from unified_video_action.model.common.lr_scheduler import get_scheduler
from unified_video_action.utils.load_env import load_env_runner, env_rollout
from unified_video_action.eval.eval import test_video_fvd, test_action_l2
from unified_video_action.utils.data_utils import resize_image

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainUnifiedVideoActionWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure policy model
        language_emb_model = cfg.task.dataset.language_emb_model
        if (
            "deepspeed_config" in cfg.training
            and cfg.training.deepspeed_config is not None
        ):
            language_emb_model = (
                None  # HACK: When training umi dataset on multiple nodes
            )
        self.model: UnifiedVideoActionPolicy = hydra.utils.instantiate(
            cfg.model.policy,
            task_name=cfg.task.name,
            task_modes=cfg.task.task_modes,
            normalizer_type=cfg.task.dataset.normalizer_type,
            language_emb_model=language_emb_model,
        )

        self.ema_model: UnifiedVideoActionPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.model.policy.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if (
            "deepspeed_config" in cfg.training
            and cfg.training.deepspeed_config is not None
        ):
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=cfg.training.deepspeed_config
            )
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            accelerator = Accelerator(
                log_with="wandb",
                mixed_precision=self.cfg.training.mixed_precision,
                deepspeed_plugin=deepspeed_plugin,
                kwargs_handlers=[ddp_kwargs],
            )
        else:
            accelerator = Accelerator(
                log_with="wandb", mixed_precision=self.cfg.training.mixed_precision
            )

        if accelerator.is_main_process:
            cfg.logging.name = self.output_dir.split("/")[-1]
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop("project")
            wandb_cfg["resume"] = "allow"

            accelerator.init_trackers(
                project_name=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg},
            )

        if cfg.task.task_type == "multiple_datasets":
            dataset: UmiMultiDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            train_dataloader = dataset.get_dataloader()
            val_dataset = dataset.split_unused_episodes()
            val_dataloader = val_dataset.get_dataloader()
            dataset.set_datasets_attribute("random_img_sampling", True)
            print(
                "train dataset:",
                len(dataset),
                "train dataloader:",
                len(train_dataloader),
            )
            print(
                "val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader)
            )
        else:
            # configure dataset
            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            train_dataloader = DataLoader(dataset, **cfg.dataloader)

            # configure validation dataset
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            print(
                "train dataset:",
                len(dataset),
                "train dataloader:",
                len(train_dataloader),
            )
            print(
                "val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader)
            )

            # compute normalizer on the main process and save to disk
            normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
            if accelerator.is_main_process:
                normalizer = dataset.get_normalizer()
                pickle.dump(normalizer, open(normalizer_path, "wb"))

        if (
            "deepspeed_config" not in cfg.training
            or cfg.training.deepspeed_config is None
        ):
            accelerator.wait_for_everyone()

        # load normalizer on all processes
        if cfg.task.task_type == "single_dataset":
            normalizer = pickle.load(open(normalizer_path, "rb"))

            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        if (
            cfg.model.policy.action_model_params.predict_action
            and "env_runner" in cfg.task
        ):
            env_runners = load_env_runner(cfg, self.output_dir)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # accelerator
        (
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )

        device = self.model.device

        if self.ema_model is not None:
            self.ema_model.to(device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            print(self.output_dir)

            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):

                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    # resize image
                    batch = resize_image(cfg, batch)
                    # compute loss
                    if (
                        "deepspeed_config" in cfg.training
                        and cfg.training.deepspeed_config is not None
                    ):  # HACK: use autocast for bf16 training
                        with torch.autocast(
                            device_type=str(device), dtype=torch.bfloat16
                        ):
                            raw_loss, (loss_diffusion, loss_action) = self.model(batch)
                    else:
                        raw_loss, (loss_diffusion, loss_action) = self.model(batch)

                    accelerator.backward(raw_loss)

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()

                    # update ema
                    if cfg.training.use_ema:
                        ema.step(accelerator.unwrap_model(self.model))

                    # logging
                    raw_loss_cpu = raw_loss.item()

                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)

                    if cfg.model.policy.autoregressive_model_params.predict_video:
                        loss_diffusion_cpu = loss_diffusion.item()
                    else:
                        loss_diffusion_cpu = 0.0

                    if cfg.model.policy.action_model_params.predict_action:
                        loss_action_cpu = loss_action.item()
                    else:
                        loss_action_cpu = 0.0

                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "diffusion_loss": loss_diffusion_cpu,
                        "action_loss": loss_action_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        accelerator.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (
                        cfg.training.max_train_steps - 1
                    ):
                        break

            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss

            # ========= eval for this epoch ==========
            # policy = self.model
            policy = accelerator.unwrap_model(self.model)
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # ========= evaluate val video generation =========
            if cfg.model.policy.autoregressive_model_params.predict_video:
                fvd_log = test_video_fvd(
                    cfg,
                    policy,
                    val_dataloader,
                    local_epoch_idx,
                    self.output_dir,
                    device,
                )
                step_log.update(fvd_log)

            # ========= evaluate val action error =========
            if (
                cfg.model.policy.action_model_params.predict_action
                and "env_runner" not in cfg.task
            ):
                ## if has similartor, skip this
                act_log = test_action_l2(
                    cfg,
                    policy,
                    val_dataloader,
                    local_epoch_idx,
                    self.output_dir,
                    device,
                )
                step_log.update(act_log)

            # ========= simulator: run rollout =========
            if (
                cfg.model.policy.action_model_params.predict_action
                and "env_runner" in cfg.task
            ):
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_rollout(cfg, env_runners, policy)
                    step_log.update(runner_log)

            # ========= checkpoint =========
            if (
                self.epoch % cfg.training.checkpoint_every
            ) == 0 and accelerator.is_main_process:
                # unwrap the model to save ckpt
                model_ddp = self.model
                self.model = accelerator.unwrap_model(self.model)

                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()

                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace("/", "_")
                    metric_dict[new_key] = value

                # save topk checkpoints
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)

                # recover the DDP model
                self.model = model_ddp

            # ========= eval end for this epoch ==========
            policy.train()
            accelerator.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1

        accelerator.end_training()
