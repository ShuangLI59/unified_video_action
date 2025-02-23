if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pickle
import os
import hydra
import glob
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import shutil
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from unified_video_action.dataset.base_dataset import BaseImageDataset
from unified_video_action.common.checkpoint_util import TopKCheckpointManager
from unified_video_action.common.json_logger import JsonLogger
from unified_video_action.common.pytorch_util import dict_apply, optimizer_to
from unified_video_action.model.diffusion.ema_model import EMAModel
from unified_video_action.model.common.lr_scheduler import get_scheduler
from unified_video_action.utils.load_env import load_env_runner
from unified_video_action.utils.language_model import get_text_model
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        
        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.model.policy, task_name=cfg.task_name, language_emb_model=cfg.task.dataset.language_emb_model)


        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        if cfg.task.dataset.language_emb_model is None:
            self.tokenizer = None
            self.text_model = None
        else:
            self.text_model, self.tokenizer, self.uncond_latents = get_text_model(cfg)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
        self.mixed_precision = cfg.training.mixed_precision
        
        print("Model params: %e" % sum(p.numel() for p in self.model.parameters()))
        # pdb.set_trace()
        

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb', mixed_precision=self.mixed_precision) # default fp16
        cfg.logging.name = self.output_dir.split('/')[-1]
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        # wandb_cfg['resume'] = 'never'
        wandb_cfg['resume'] = 'allow'
        print('-------------------------------------')
        print('wandb_cfg', wandb_cfg)
        print('-------------------------------------')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                # print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))
        
        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))
        # normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))


        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        if cfg.training.debug:
            cfg.task.env_runner.n_envs = 2
            cfg.task.env_runner.n_test = 1
            cfg.task.env_runner.n_train = 1

        ##############################################################################################################################
        # configure env
        ##############################################################################################################################
        env_runners = load_env_runner(cfg, self.output_dir)
    
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )


        if self.text_model is not None:
            train_dataloader, val_dataloader, self.model, self.text_model, self.optimizer, lr_scheduler = accelerator.prepare(
                    train_dataloader, val_dataloader, self.model, self.text_model, self.optimizer, lr_scheduler
                )
            device = self.model.device
            self.text_model.to(device)
            self.text_model.eval()
        else:
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
                    train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
                )
            device = self.model.device
            
        if self.ema_model is not None:
            self.ema_model.to(device)
        
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    
                    print(self.output_dir)
                    for batch_idx, batch in enumerate(tepoch):
                        
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        if 'language_table' in cfg.task.name:
                            B, T, C, H, W = batch['obs']['agentview_rgb'].shape
                            resized_tensor = F.interpolate(batch['obs']['agentview_rgb'].view(B*T, C, H, W), size=(180, 180), mode='bilinear', align_corners=False)
                            batch['obs']['agentview_rgb'] = resized_tensor.view(B, T, C, 180, 180)

                        
                        if train_sampling_batch is None:
                            train_sampling_batch = deepcopy(batch)

                        # compute loss
                        # raw_loss = self.model.compute_loss(batch, self.text_model, self.tokenizer)
                        raw_loss = self.model(batch, self.text_model, self.tokenizer)
                        
                        # loss = raw_loss / cfg.training.gradient_accumulate_every
                        # loss.backward()
                        accelerator.backward(raw_loss)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))
                            # ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            # if not cfg.training.debug:
                            #     wandb_run.log(step_log, step=self.global_step)

                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break



                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                # policy = self.model
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                
                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:    
                    if 'libero' in cfg.task.name:
                        for env_runner in env_runners:
                            runner_log = env_runner.run(policy, text_model=self.text_model, tokenizer=self.tokenizer)
                            step_log.update(runner_log)
                        
                        if cfg.checkpoint.topk.monitor_key=='test_mean_score':
                            assert 'test_mean_score' not in step_log
                            all_test_mean_score = {k:v for k,v in step_log.items() if 'test/' in k and '_mean_score' in k}
                            step_log['test_mean_score'] = np.mean(list(all_test_mean_score.values()))

                            all_train_mean_score = {k:v for k,v in step_log.items() if 'train/' in k and '_mean_score' in k}
                            step_log['train_mean_score'] = np.mean(list(all_train_mean_score.values()))
                    else:
                        env_runner = env_runners
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)

                        step_log['train_mean_score'] = runner_log['train/mean_score']
                        step_log['test_mean_score'] = runner_log['test/mean_score']

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()

                        print(self.output_dir)
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                                loss = self.model(batch, self.text_model, self.tokenizer)

                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                # if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        
                        if cfg.task.dataset.language_emb_model is not None:
                            language_goal = batch['obs']['language']
                            del batch['obs']['language']

                            if cfg.task.dataset.language_emb_model == 'clip':
                                assert (language_goal[:,:-1]-language_goal[:,1:]).sum() < 1e-6
                                # text_tokens = {'input_ids': language_goal[:,0].long()[:,0], 'attention_mask': language_goal[:,0].long()[:,1]}
                                text_tokens = language_goal[:,0].long()[:,0]
                            elif cfg.task.dataset.language_emb_model == 'flant5':
                                text_tokens = language_goal[:,0].long()
                      
                            language_goals = []
                            for item in text_tokens:
                                language_goals.append(self.tokenizer.decode(item, skip_special_tokens=True))
                        else:
                            language_goals = None

                        result = policy.predict_action(obs_dict, language_goal=language_goals, text_model=self.text_model, tokenizer=self.tokenizer)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                # if (self.epoch % cfg.training.checkpoint_every) == 0:
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
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
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    
                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                # if not cfg.training.debug:
                #     wandb_run.log(step_log, step=self.global_step)
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
