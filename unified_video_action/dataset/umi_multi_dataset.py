import json
import os
from typing import Any, Dict, Optional, Union, cast
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset

from unified_video_action.dataset.base_lazy_dataset import BaseLazyDataset, batch_type
from unified_video_action.dataset.umi_lazy_dataset import UmiLazyDataset
from unified_video_action.utils.language_model import get_text_model
import numpy as np
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

        seed = 42
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.language_emb_model = language_emb_model
        self.language_latents: dict[str, list[torch.Tensor]] = {
            "cup_arrangement_0": [],
            "towel_folding_0": [],
            "mouse_arrangement_0": [],
        }

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
        data_dict["language_latents"] = self.rng.choice(
            self.language_latents[data_dict["dataset_name"]], size=1, replace=False
        )[0]
        del data_dict["dataset_name"]
        return data_dict

    def get_language_latent(self):
        # language_goals = {'cup_arrangement_0': 'pick up an espresso cup and place it onto a saucer with the cup handle oriented to the left of the robot',
        #                     'towel_folding_0': 'grasp the left edge of the towel and move it to the right, folding it in half',
        #                     'mouse_arrangement_0': 'pick up the mouse and place it on the mouse pad'}

        language_goals = {
            "cup_arrangement_0": [
                "Pick up an espresso cup and place it onto a saucer with the cup handle oriented to the left of the robot.",
                "Grasp the espresso cup delicately and set it on the saucer, ensuring its handle points to the robot's left.",
                "Lift the small cup and carefully position it on the saucer so that the handle faces left relative to the robot.",
                "Take hold of the espresso cup and gently deposit it onto the saucer, aligning the handle to the left side of the robot.",
                "Place the cup on the saucer with precision, making sure the handle is directed towards the left side of the robot.",
                "Securely pick up the espresso cup and carefully set it down on the saucer with its handle turned leftward.",
                "Gently lift the cup and rest it on the saucer, ensuring the handle points left from the robot's perspective.",
                "Handle the espresso cup with care and place it onto the saucer so that its handle faces to the left of the robot.",
                "Lift the cup and position it on the saucer, orienting the handle to face left relative to the robot.",
                "Pick up the espresso cup and place it onto the saucer.",
                "Grasp the cup and set it on the saucer.",
                "Grab cup and put it on saucer.",
                "Lift the cup and deposit it onto the saucer.",
                "Take the cup and rest it on the saucer.",
                "Hold the cup, then place it on the saucer.",
                "Retrieve the cup and place it neatly on the saucer.",
                "Grab the cup, position it on the saucer.",
                "Lift the cup and align it on the saucer.",
                "Secure the cup and gently set it on the saucer.",
                "Pick up the cup, then carefully put it on the saucer.",
                "Take the espresso cup and set it on the saucer.",
            ],
            "towel_folding_0": [
                "Grasp the left edge of the towel and move it to the right, folding it in half.",
                "Hold the towel by its left side and slide it over to the right, creating a neat, even fold.",
                "Seize the left corner of the towel and pull it rightward to achieve a clean, symmetrical fold.",
                "Take the left edge of the towel and shift it to the right, thereby folding it into two equal parts.",
                "Grab the left side of the towel and fold it over to the right.",
                "Lift the towel from the left and fold it neatly towards the right.",
                "Pick up the towel by its left edge and bring it over to the right to fold it.",
                "Secure the left side of the towel and move it rightward, resulting in a tidy fold.",
                "Fold the towel by grabbing its left side and moving it across to the right.",
                "Take the left portion of the towel and fold it over, forming a perfect half.",
                "Grab the towel’s left side and fold it toward the right side for a smooth fold.",
                "Clasp the left end of the towel and swing it to the right, folding it in half.",
                "Fold the towel.",
                "Fold towel neatly.",
                "Fold the towel over.",
                "Fold towel in half.",
                "Fold towel to the right.",
                "Fold left side of towel.",
                "Fold towel from left to right.",
                "Evenly fold towel.",
                "Fold towel quickly.",
                "Swiftly fold towel.",
            ],
            "mouse_arrangement_0": [
                "Pick up the mouse and place it on the mouse pad.",
                "Grasp the computer mouse and set it down carefully on its mouse pad.",
                "Lift the mouse and gently position it on the designated mouse pad.",
                "Take hold of the mouse and accurately rest it on the mouse pad.",
                "Grab the mouse and put it on the mouse pad.",
                "Lift the mouse and set it neatly on the mouse pad.",
                "Carefully pick up the mouse and deposit it on the mouse pad.",
                "Place the mouse on the pad with precision.",
                "Move the mouse to the mouse pad.",
                "Securely grasp the mouse and lay it on the mouse pad.",
                "Set the mouse down on the mouse pad.",
                "Align the mouse with the mouse pad.",
                "Place the computer mouse onto the pad.",
                "Position the mouse accurately on the mouse pad.",
                "Gently pick up the mouse and rest it on the pad.",
                "Lift the computer mouse and slide it onto the mouse pad.",
                "Grasp the mouse and accurately position it on the pad.",
                "Pick up the mouse and place it correctly on the mouse pad.",
                "Carefully retrieve the mouse and set it on its pad.",
                "Grab the mouse and align it with the mouse pad.",
                "Put mouse on pad.",
                "Move mouse to pad.",
                "Set mouse on pad.",
                "Place mouse on pad.",
                "Rest mouse on pad.",
                "Position mouse on pad.",
                "Slide mouse onto pad.",
                "Shift mouse to pad.",
                "Deposit mouse on pad.",
                "Lay mouse on pad.",
                "Pick up mouse and set on pad.",
                "Grab mouse and place on pad.",
            ],
        }

        self.text_model, self.tokenizer, max_length = get_text_model(
            "umi", self.language_emb_model
        )

        with torch.no_grad():
            for dataset_name, language_goal in language_goals.items():
                for language_goal_text in language_goal:
                    language_tokens = self.tokenizer(
                        [language_goal_text],
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    self.language_latents[dataset_name].append(
                        self.text_model.get_text_features(**language_tokens)[0]
                    )

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
            unused_dataset.datasets.append(unused_single_dataset)
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
