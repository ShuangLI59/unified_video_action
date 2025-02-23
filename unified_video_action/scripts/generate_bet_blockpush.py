if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import cv2
import os
import click
import pathlib
import numpy as np
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.trajectories.time_step import StepType
# from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.env.block_pushing.block_pushing_multimodal_mpc import BlockPushMultimodal
# from diffusion_policy.env.block_pushing.block_pushing import BlockPush
from diffusion_policy.env.block_pushing.oracles.multimodal_push_oracle import MultimodalOrientedPushOracle
import pdb

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-n', '--n_episodes', default=1000)
@click.option('-c', '--chunk_length', default=-1)
def main(output, n_episodes, chunk_length):
    os.makedirs(output, exist_ok=True)

    buffer = ReplayBuffer.create_empty_numpy()
    
    image_size = (256, 256)
    env = TimeLimit(GymWrapper(BlockPushMultimodal(image_size=image_size)), duration=350)
    # env = TimeLimit(GymWrapper(BlockPushMultimodal()), duration=350)
    for i in tqdm(range(n_episodes)):
        print('image_size', image_size)
        print(i)
        obs_history = list()
        action_history = list()

        env.seed(i)
        policy = MultimodalOrientedPushOracle(env)
        time_step = env.reset()
        policy_state = policy.get_initial_state(1)
        
        count = 0
        episode = list()
        while True:
            action_step = policy.action(time_step, policy_state)
            
            # obs = np.concatenate(list(time_step.observation.values()), axis=-1)
            obs = time_step.observation
            img = obs['rgb']
            del obs['rgb']
            state = np.concatenate(list(obs.values()), axis=-1)
            
            
            action = action_step.action
            # obs_history.append(obs)
            # action_history.append(action)
            
            ##############################################################################
            ## vis rendered images
            ##############################################################################
            if i<10:
                bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{output}/epo{i}_{count}.png", bgr_image)
                count += 1
            ##############################################################################
            ##############################################################################

            
            data = {
                    'img': img,
                    'state': np.float32(state),
                    'action': np.float32(action),
                }
            episode.append(data)

            if time_step.step_type == 2:
                break

            # state = env.wrapped_env().gym.get_pybullet_state()
            time_step = env.step(action)
        
        # obs_history = np.array(obs_history)
        # action_history = np.array(action_history)

        # episode = {
        #     'obs': obs_history,
        #     'action': action_history
        # }
        
        # buffer.add_episode(episode)
        
        
        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack(
                [x[key] for x in episode])
        buffer.add_episode(data_dict, compressors='disk')
    
    buffer.save_to_path(output, chunk_length=chunk_length)
        
if __name__ == '__main__':
    main()
