import numpy as np
import copy

import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation
import collections
import os
import cv2

from robomimic.config import config_factory
import pdb
from diffusion_policy.env_runner.libero_bddl_mapping import bddl_file_name_dict

def write_imge(action, obs, i, vis_path):
    print('write image to', vis_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 10)  # (x, y) coordinates of the text
    font_scale = 0.6
    color = (255, 255, 255)  # White color (BGR format)
    thickness = 2

    text = ''
    for tem in action:
        text += f' {tem:.2f}'
    image = np.transpose(obs['agentview_image'], (1, 2, 0))*255
    image = image[..., ::-1]
    # cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(vis_path, image)
    

class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        ###################################################################################################
        # modality_mapping = collections.defaultdict(list)
        # shape_meta = {'action': {'shape': [10]}, 
        #               'obs': {
        #                     'agentview_image': {'shape': [3, 128, 128], 'type': 'rgb'},
        #                     'ee_ori': {'shape': [3]},
        #                     'ee_pos': {'shape': [3]},
        #                       }
        #               }
        # for key, attr in shape_meta['obs'].items():
        #     modality_mapping[attr.get('type', 'low_dim')].append(key)
        # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        ###################################################################################################
        
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        # if env_meta['bddl_file'] not in bddl_file_name_dict:
        #     print(env_meta['bddl_file'])
        #     print(dataset_path)
        #     pdb.set_trace()
        # exit()

        if env_meta['bddl_file'] not in bddl_file_name_dict.values():
            print('convert bddl filename')
            print(env_meta['bddl_file'])
            print(env_meta['env_kwargs']['bddl_file_name'])
            
            env_meta['bddl_file'] = bddl_file_name_dict[env_meta['bddl_file']]
            env_meta['env_kwargs']['bddl_file_name'] = env_meta['bddl_file']
        else:
            print('use existing bddl file')
            print(env_meta['bddl_file'])
            print(env_meta['env_kwargs']['bddl_file_name'])

        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        
        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert not abs_env.env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')

        
        # self.vis_path = 'debug_convert_action2/%s'%env_meta['bddl_file'].split('/')[-1][:-5]
        # os.makedirs(self.vis_path, exist_ok=True)
        # self.convert_and_eval_idx(0)
    

    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)

        
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(stacked_actions.shape[:-1]+(3,), dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(stacked_actions.shape[:-1]+(3,), dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        for i in range(len(states)):
            _ = env.reset_to({'states': states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(controller.goal_ori).as_rotvec()


        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        print(idx)
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        
        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        # robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        # robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]
        robot0_eef_pos = demo['obs']['ee_pos'][:]
        robot0_eef_quat = demo['obs']['ee_ori'][:]
        # <KeysViewHDF5 ['agentview_rgb', 'ee_ori', 'ee_pos', 'ee_states', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']>
        
        
        # delta_error_info = self.evaluate_rollout_error(
        #     env, states, actions, robot0_eef_pos, robot0_eef_quat, 
        #     metric_skip_steps=eval_skip_steps,
        #     vis_path=self.vis_path)
        
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
            vis_path=self.vis_path)

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info
        

    
    @staticmethod
    def evaluate_rollout_error(env, 
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1,
            vis_path=''):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({'states': states[0]})

        
        for i in range(len(states)):
            obs = env.reset_to({'states': states[i]})
            obs, reward, done, info = env.step(actions[i])

            # dict_keys(['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
            write_imge(actions[i], obs, i, f'{vis_path}/obs_{i}.png')
            
            obs = env.get_observation()

            write_imge(actions[i], obs, i, f'{vis_path}/obs_{i}_2.png')
            
            # rollout_next_states.append(env.get_state()['states'])
            # rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            # rollout_next_eef_quat.append(obs['robot0_eef_quat'])

        
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        # next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
        #     * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        # next_eef_rot_dist = next_eef_rot_diff.magnitude()
        # max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            # 'rot': max_next_eef_rot_dist
        }
        return info
