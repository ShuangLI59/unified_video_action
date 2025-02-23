if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
from diffusion_policy.common.replay_buffer import ReplayBuffer
import pdb
import numpy as np

@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-t', '--target_eef_idx', default=8, type=int)
def main(input, output, target_eef_idx):
    buffer = ReplayBuffer.copy_from_path(input)
    
    # obs = buffer['obs']
    img = buffer['img']
    state = buffer['state']
    action = buffer['action']
    
    print('action shape', action.shape)
    print('action mean', action.mean())
    print('action max', action.max())
    print('action min', action.min())
    print('action abs mean', np.abs(action).mean())
    print('action abs max', np.abs(action).max())
    print('action abs min', np.abs(action).min())
    pdb.set_trace()
    
    prev_eef_target = state[:,target_eef_idx:target_eef_idx+action.shape[1]]
    next_eef_target = prev_eef_target + action
    action[:] = next_eef_target
    buffer.save_to_path(zarr_path=output, chunk_length=-1)

if __name__ == '__main__':
    main()
