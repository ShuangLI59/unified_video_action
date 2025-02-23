import json
import os
from typing import Any, Dict, Optional, Union, cast
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset

from unified_video_action.dataset.base_lazy_dataset import BaseLazyDataset, batch_type
from unified_video_action.dataset.umi_lazy_dataset import UmiLazyDataset
from unified_video_action.utils.language_model import get_text_model
from types import SimpleNamespace
from copy import deepcopy
import pdb

class UmiMultiDataset(Dataset[batch_type]):
    """
    Multi-dataset data loader for the official UMI dataset.
    Example structure:

    dataset_0.zarr
    ├── data
    │   ├── camera0_rgb (N, 224, 224, 3) uint8
    │   ├── robot0_demo_end_pose (N, 6) float64
    │   ├── robot0_demo_start_pose (N, 6) float64
    │   ├── robot0_eef_pos (N, 3) float32
    │   ├── robot0_eef_rot_axis_angle (N, 3) float32
    │   └── robot0_gripper_width (N, 1) float32
    └── meta
        └── episode_ends (5,) int64
    dataset_1.zarr
    ├── data
    └── meta
    dataset_2.zarr
    ├── data
    └── meta
    """

    def __init__(
        self,
        dataset_root_dir: str,
        used_episode_indices_file: str,
        dataset_configs: Union[dict[str, dict[str, Any]], DictConfig],
        language_emb_model: Optional[str],
        normalizer_type: Optional[str],
        **base_config: Union[dict[str, Any], DictConfig],
    ):

        self.dataset_root_dir: str = dataset_root_dir

        if isinstance(dataset_configs, DictConfig):
            dataset_configs = cast(
                dict[str, dict[str, Any]], OmegaConf.to_container(dataset_configs)
            )
        self.dataset_configs: dict[str, dict[str, Any]] = dataset_configs
        assert len(self.dataset_configs.keys()) >= 1, "At least one dataset is required"

        if used_episode_indices_file != "":
            assert used_episode_indices_file.endswith(
                ".json"
            ), "used_episode_indices_file must be a json file"
            with open(used_episode_indices_file, "r") as f:
                used_episode_indices_dict: dict[str, list[int]] = json.load(f)
            for name, config in self.dataset_configs.items():
                config["include_episode_indices"] = used_episode_indices_dict[name]
                if "include_episode_num" in config:
                    assert (
                        len(config["include_episode_indices"])
                        == config["include_episode_num"]
                    ), f"include_episode_num {config['include_episode_num']} does not match the length of include_episode_indices {len(config['include_episode_indices'])} for dataset {name}"

        if isinstance(base_config, DictConfig):
            base_config = cast(dict[str, Any], OmegaConf.to_container(base_config))
        self.base_config: dict[str, Any] = base_config

        self.datasets: list[UmiLazyDataset] = []
        for dataset_name, dataset_config in self.dataset_configs.items():
            print(f"Initializing dataset: {dataset_name}")
            config = deepcopy(self.base_config)
            config.update(deepcopy(dataset_config))
            config["zarr_path"] = os.path.join(
                self.dataset_root_dir, dataset_name + ".zarr"
            )
            config["name"] = dataset_name
            dataset = UmiLazyDataset(**config)
            self.datasets.append(dataset)

        self.index_pool: list[tuple[int, int]] = []
        """
        First value: dataset index
        Second value: data index in the corresponding dataset
        """
        self._create_index_pool()

        self.language_emb_model = language_emb_model
        self.language_latents: dict[str, torch.Tensor] = {}

        if self.language_emb_model is not None:
            self.get_language_latent()

        

    def _create_index_pool(self):
        self.index_pool = []
        for dataset_idx, dataset in enumerate(self.datasets):
            self.index_pool.extend((dataset_idx, i) for i in range(len(dataset)))

    def __len__(self):
        return len(self.index_pool)

    def __getitem__(self, idx: int) -> batch_type:
        dataset_idx, data_idx = self.index_pool[idx]
        data_dict = self.datasets[dataset_idx][data_idx]
        data_dict["ids"] = torch.tensor([idx])
        data_dict["language_latents"] = self.language_latents[data_dict["dataset_name"]]
        del data_dict["dataset_name"]
        return data_dict
    
    def get_language_latent(self):
        language_goals = {'cup_arrangement_0': 'pick up an espresso cup and place it onto a saucer with the cup handle oriented to the left of the robot',
                            'towel_folding_0': 'grasp the left edge of the towel and move it to the right, folding it in half',
                            'mouse_arrangement_0': 'pick up the mouse and place it on the mouse pad'}
        
        self.text_model, self.tokenizer, max_length = get_text_model('umi', self.language_emb_model)
    
        with torch.no_grad():    
            for dataset_name, language_goal in language_goals.items():
                language_tokens = self.tokenizer([language_goal], padding='max_length', max_length=max_length, return_tensors="pt")
                self.language_latents[dataset_name] = self.text_model.get_text_features(**language_tokens)[0]

    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: Optional[list[int]] = None,
    ):
        unused_dataset = deepcopy(self)
        unused_dataset.index_pool = []
        unused_dataset.datasets = []
        for dataset_idx, dataset in enumerate(self.datasets):
            unused_single_dataset = dataset.split_unused_episodes(
                remaining_ratio, other_used_episode_indices
            )
            unused_dataset.datasets.append(
                unused_single_dataset
            )
        unused_dataset._create_index_pool()

        return unused_dataset

    def get_dataloader(self):
        return DataLoader(self, **self.base_config["dataloader_cfg"])

    @property
    def transforms(self):
        """Return the transforms of the first dataset. Assuming all datasets have the same transforms."""
        return self.datasets[0].transforms
    
    @property
    def apply_augmentation_in_cpu(self):
        return self.datasets[0].apply_augmentation_in_cpu
    
    def set_datasets_attribute(self, attribute_name: str, attribute_value: Any):
        for dataset in self.datasets:
            setattr(dataset, attribute_name, attribute_value)
        if attribute_name in self.base_config:
            self.base_config[attribute_name] = attribute_value
