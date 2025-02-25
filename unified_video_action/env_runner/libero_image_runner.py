import os
import sys
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from unified_video_action.gym_util.async_vector_env import AsyncVectorEnv
from unified_video_action.gym_util.multistep_wrapper import MultiStepWrapper
from unified_video_action.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)

from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.env_runner.base_image_runner import BaseImageRunner

## here we just use the same env wrapper as robomimic
from unified_video_action.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)
from unified_video_action.env_runner.libero_bddl_mapping import bddl_file_name_dict

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
libero_path = os.path.join(parent_dir, "LIBERO")
sys.path.append(libero_path)
from libero.libero.envs.bddl_base_domain import TASK_MAPPING


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)

    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    if env_meta["bddl_file"] not in bddl_file_name_dict.values():
        print("convert bddl filename")
        print(env_meta["bddl_file"])
        print(env_meta["env_kwargs"]["bddl_file_name"])
        env_meta["bddl_file"] = bddl_file_name_dict[env_meta["bddl_file"]]
        env_meta["env_kwargs"]["bddl_file_name"] = env_meta["bddl_file"]
    else:
        print("use existing bddl file")
        print(env_meta["bddl_file"])
        print(env_meta["env_kwargs"]["bddl_file_name"])

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


class LiberoImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        task_dir,
        output_dir,
        dataset_path,
        shape_meta: dict,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        dataset_path = task_dir
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        rotation_transformer = None
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            from unified_video_action.model.common.rotation_transformer import (
                RotationTransformer,
            )

            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        def env_fn():
            libero_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
            libero_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=libero_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            libero_env = create_env(
                env_meta=env_meta, shape_meta=shape_meta, enable_render=False
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=libero_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        pre_collected_actions = list()

        # train
        with h5py.File(dataset_path, "r") as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f"data/demo_{train_idx}/states"][0]
                pre_collected_action = f[f"data/demo_{train_idx}/actions"][:]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", wv.util.generate_id() + ".mp4"
                        )
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append(
                    "train/%s_" % env_meta["bddl_file"].split("/")[-1][:-5]
                )
                env_init_fn_dills.append(dill.dumps(init_fn))
                pre_collected_actions.append(pre_collected_action)

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/%s_" % env_meta["bddl_file"].split("/")[-1][:-5])
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn, shared_memory=False)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

        if len(pre_collected_actions) > 0:
            self.pre_collected_actions = np.stack(pre_collected_actions)
        self.language_goal = " ".join(task_dir.split("/")[-1][:-10].split("_"))
        self.task_name = env_meta["bddl_file"].split("/")[-1][:-5]

    def run(self, policy: BaseImagePolicy, vis_pred_video=False, **kwargs):
        device = policy.device
        # dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        print("env_runner: ", self.language_goal)
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()

            # past_action = None
            past_action_list = []
            policy.reset()

            # env_name = self.env_meta['env_name']
            env_name = self.env_meta["bddl_file"].split("/")[-1][:-5]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False

            while not done:
                # create obs dict
                # obs = self.convert_obs(obs)
                np_obs_dict = dict(obs)

                if self.past_action:
                    if len(past_action_list) > 1:  ## get 16 actions
                        np_obs_dict["past_action"] = np.concatenate(
                            past_action_list, axis=1
                        )

                # device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(
                        obs_dict,
                        language_goal=[self.language_goal]
                        * obs_dict["agentview_image"].size(0),
                        **kwargs,
                    )

                # device_transfer
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

                action = np_action_dict["action"]  # (1, 8, 10)
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)

                for i in range(len(reward)):
                    if reward[i] == 1:
                        done[i] = True

                done = np.all(done)

                # past_action = action
                past_action_list.append(action)
                if len(past_action_list) > 2:
                    past_action_list.pop(0)

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]

        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
