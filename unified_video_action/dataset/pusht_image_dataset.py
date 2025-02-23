from typing import Dict
import torch
import numpy as np
import copy
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.common.replay_buffer import ReplayBuffer
from unified_video_action.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.dataset.base_dataset import BaseImageDataset
from unified_video_action.common.normalize_util import get_image_range_normalizer
import torchvision.transforms as transforms
import pdb

class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            dataset_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            language_emb_model=None,
            data_aug=False,
            normalizer_type=None,
            dataset_type='singletask',
            ):
        
        super().__init__()
        
        print('----------------------------------------------------------------------------')
        print('load data from: ', dataset_path)
        print('----------------------------------------------------------------------------')
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.data_aug = data_aug
        self.dataset_type = dataset_type

        
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255
        
        if self.data_aug:
            image_tensor = torch.tensor(image, dtype=torch.float32)
            video_seed = torch.randint(0, 10000, (1,)).item()
            
            if self.dataset_type=='singletask':
                def consistent_augmentations(frame):
                    # Set the random seed for each frame to ensure consistent augmentation
                    torch.manual_seed(video_seed)
                    augmentation = transforms.Compose([
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Color jitter
                        transforms.RandomCrop(size=(80, 80))     # Center crop to 80x80
                    ])
                    return augmentation(frame)
            elif self.dataset_type=='multitask':
                def consistent_augmentations(frame):
                    # Set the random seed for each frame to ensure consistent augmentation
                    torch.manual_seed(video_seed)
                    augmentation = transforms.Compose([
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Color jitter
                    ])
                    return augmentation(frame)
            else:
                raise NotImplementedError

            augmented_images = torch.stack([consistent_augmentations(frame) for frame in image_tensor])
            image = augmented_images.numpy()
        
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(dataset_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
