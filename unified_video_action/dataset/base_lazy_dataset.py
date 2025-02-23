import copy
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, cast, Optional, Union
from omegaconf import DictConfig, ListConfig, OmegaConf
import numpy as np
import zarr
import os
import torch
import numpy.typing as npt
import kornia.augmentation as K

from dataclasses import dataclass
from typing import Any
from torch import nn

batch_type = dict[str, Union[dict[str, torch.Tensor], torch.Tensor]]
torch.set_num_threads(1)


@dataclass
class SourceDataMeta:
    name: str
    """The data name from the source dataset."""
    shape: tuple[int, ...]
    """The shape of a single time step of the data."""
    include_indices: list[int]
    """Indices of the data to include in the dataset relative to the current step (0). Negative indices means the data is from the past."""

    def __post_init__(self):
        if isinstance(self.shape, list):
            self.shape = tuple(self.shape)
        if len(self.include_indices) == 0:
            raise ValueError(
                f"include_indices must be a non-empty list in {self.name}."
            )
        for i, index in enumerate(self.include_indices):
            if i < len(self.include_indices) - 1:
                if index > self.include_indices[i + 1]:
                    raise ValueError(
                        f"include_indices must be monotonically increasing, but got {self.include_indices} in {self.name}."
                    )
        if len(self.shape) == 0:
            raise ValueError(f"shape must be a non-empty list in {self.name}.")


@dataclass
class DataMeta:
    name: str
    """The output name to be used for training."""
    shape: tuple[int, ...]
    """The shape of a single time step of the data."""
    data_type: str
    """low_dim or image"""
    length: int
    """The length of the data."""
    normalizer: str
    """identity, range, image"""
    augmentation: list[dict[str, Any]]
    """The augmentation to apply to the data."""
    usage: str
    """global_cond, local_cond, output, meta"""

    def __post_init__(self):

        if isinstance(self.shape, list):
            self.shape = tuple(self.shape)

        if self.data_type not in ["low_dim", "image"]:
            raise ValueError(
                f"data_type must be one of ['low_dim', 'image'] in {self.name}."
            )

        if self.length <= 0:
            raise ValueError(f"length must be greater than 0 in {self.name}.")

        if len(self.shape) == 0:
            raise ValueError(f"shape must be a non-empty list in {self.name}.")

        if self.usage not in [
            "global_cond",
            "local_cond",
            "output",
            "meta",  # For Mujoco dataset
            "obs",
            "action",
        ]:  # For UMI dataset
            raise ValueError(
                f"usage must be one of ['global_cond', 'local_cond', 'output', 'meta', 'obs', 'action'] in {self.name}."
            )

        if self.normalizer not in ["identity", "range"]:
            raise ValueError(
                f"normalizer must be one of ['identity', 'range'] in {self.name}."
            )


def construct_data_meta(
    data_meta: Union[dict[str, dict[str, Any]], DictConfig],
) -> dict[str, DataMeta]:
    if isinstance(data_meta, DictConfig):
        data_meta = cast(
            dict[str, dict[str, Any]], OmegaConf.to_container(data_meta, resolve=True)
        )
    data_meta_dict = {}
    for name, entry_meta_dict in data_meta.items():
        entry_meta_dict.update({"name": name})
        data_meta_dict[name] = DataMeta(**entry_meta_dict)
    return data_meta_dict


def construct_source_data_meta(
    source_data_meta: Union[dict[str, dict[str, Any]], DictConfig],
) -> dict[str, SourceDataMeta]:
    if isinstance(source_data_meta, DictConfig):
        source_data_meta = cast(
            dict[str, dict[str, Any]],
            OmegaConf.to_container(source_data_meta, resolve=True),
        )
    source_data_meta_dict = {}
    for name, entry_meta_dict in source_data_meta.items():
        entry_meta_dict.update({"name": name})
        source_data_meta_dict[name] = SourceDataMeta(**entry_meta_dict)
    return source_data_meta_dict


class SingleFieldLinearNormalizer(nn.Module):
    def __init__(self, meta: DataMeta):
        super().__init__()
        self.meta: DataMeta = meta
        self.scale: nn.Parameter
        self.offset: nn.Parameter
        self.normalizer_type: str = meta.normalizer

    def fit(self, x: torch.Tensor):
        raise NotImplementedError()

    def from_dict(
        self,
        state_dict: dict[
            str, Union[torch.Tensor, npt.NDArray[np.float32], list[float]]
        ],
    ):
        if self.normalizer_type == "identity":
            return
        keys = ["scale", "offset"]

        for key in keys:
            val = state_dict[key]
            assert (
                key in state_dict
            ), f"State dict must contain '{key}' key for {self.meta.name}"
            assert isinstance(
                val, (torch.Tensor, np.ndarray, list)
            ), f"{key} must be a torch tensor, numpy array, or list for {self.meta.name}"

            if isinstance(val, list):
                val = torch.Tensor(val)
            elif isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            else:
                assert isinstance(
                    val, torch.Tensor
                ), f"{key} must be a torch tensor for {self.meta.name}"

            if self.normalizer_type == "range":
                assert (
                    val.shape == self.meta.shape
                ), f"{key} must have the same shape as the data {self.meta.shape} for range normalizer {self.meta.name}"
            else:
                raise ValueError(
                    f"Unknown normalizer {self.normalizer_type} for {self.meta.name}. Valid normalizers are 'identity', 'range'."
                )

            setattr(self, key, nn.Parameter(val))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def as_dict(self, data_class: str) -> dict[str, Any]:
        assert (
            self.scale is not None and self.offset is not None
        ), f"Normalizer for {self.meta.name} is not initialized."
        if data_class == "numpy":
            return {
                "type": self.normalizer_type,
                "scale": self.scale.detach().cpu().numpy(),
                "offset": self.offset.detach().cpu().numpy(),
            }
        elif data_class == "torch":
            return {
                "type": self.normalizer_type,
                "scale": self.scale.detach().cpu(),
                "offset": self.offset.detach().cpu(),
            }
        elif data_class == "list":
            return {
                "type": self.normalizer_type,
                "scale": self.scale.detach().cpu().tolist(),
                "offset": self.offset.detach().cpu().tolist(),
            }
        else:
            raise ValueError(
                f"Unknown data type {data_class} for normalizer {self.meta.name}. Valid types are 'numpy', 'torch', and 'list'."
            )

    def _check_input_shape(self, x: torch.Tensor):
        data_dim = len(self.meta.shape)
        assert (
            x.shape[-data_dim:] == self.meta.shape
        ), f"The last {data_dim} dimensions of {self.meta.name} (shape {x.shape}) must match {self.meta.shape} from meta data"


class IdentityNormalizer(SingleFieldLinearNormalizer):
    def __init__(self, meta: DataMeta):
        super().__init__(meta)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))

    def fit(self, x: torch.Tensor):
        pass

    def load(self, state_dict: dict[str, torch.Tensor]):
        pass

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RangeNormalizer(SingleFieldLinearNormalizer):
    """
    Normalize data to be between -1 and 1.
    """

    def __init__(self, meta: DataMeta):
        super().__init__(meta)
        self.scale = nn.Parameter(torch.nan * torch.ones(meta.shape))
        self.offset = nn.Parameter(torch.nan * torch.ones(meta.shape))

    def fit(self, x: torch.Tensor):
        self._check_input_shape(x)
        x = x.clone().detach().reshape(-1, *self.meta.shape)
        min_val = x.min(dim=0).values
        max_val = x.max(dim=0).values
        self.scale = nn.Parameter((max_val - min_val) / 2 + 1e-7)
        self.offset = nn.Parameter((max_val + min_val) / 2)
        assert tuple(self.scale.shape) == tuple(self.offset.shape) == self.meta.shape

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            not self.scale.isnan().any() and not self.offset.isnan().any()
        ), f"Normalizer for {self.meta.name} is not initialized"
        self._check_input_shape(x)
        return (x - self.offset) / self.scale

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            not self.scale.isnan().any() and not self.offset.isnan().any()
        ), f"Normalizer for {self.meta.name} is not initialized"
        self._check_input_shape(x)
        return x * self.scale + self.offset


class FixedNormalizer(nn.Module):
    """
    Normalizer that is fixed after fitting. Not trainable.
    """

    @torch.no_grad()
    def __init__(
        self,
        data_meta: dict[str, DataMeta],
    ):
        super().__init__()
        self.data_meta: dict[str, DataMeta] = data_meta
        self.normalizers: nn.ModuleDict = nn.ModuleDict()

        for meta in self.data_meta.values():
            if meta.usage not in self.normalizers:
                self.normalizers[meta.usage] = nn.ModuleDict()
            if meta.normalizer == "identity":
                self.normalizers[meta.usage][meta.name] = IdentityNormalizer(meta)
            elif meta.normalizer == "range":
                self.normalizers[meta.usage][meta.name] = RangeNormalizer(meta)
            else:
                raise ValueError(
                    f"Unknown normalizer {meta.normalizer} for {meta.name}"
                )

    @torch.no_grad()
    def fit_range_normalizer(self, data_dict: batch_type):
        for meta in self.data_meta.values():
            if meta.normalizer != "range":
                continue

            assert (
                meta.name in data_dict[meta.usage]
            ), f"Data for {meta.name} not found when fitting normalizer"

            self.normalizers[meta.usage][meta.name].fit(
                data_dict[meta.usage][meta.name]
            )

    @torch.no_grad()
    def from_dict(
        self,
        state_dict: dict[
            str,
            dict[
                str,
                dict[str, Union[torch.Tensor, npt.NDArray[np.float32], list[float]]],
            ],
        ],
    ):
        for meta in self.data_meta.values():
            assert (
                meta.name in state_dict[meta.usage]
            ), f"State dict for {meta.name} not found when loading normalizer"
            if meta.normalizer == "identity":
                continue
            self.normalizers[meta.usage][meta.name].from_dict(
                state_dict[meta.usage][meta.name]
            )

    @torch.no_grad()
    def normalize(self, data_dict: batch_type) -> batch_type:
        for usage, data_dict_ in data_dict.items():
            if usage not in self.normalizers:
                continue
            for name, data in data_dict_.items():
                if name in self.normalizers[usage]:
                    data_dict[usage][name] = self.normalizers[usage][name].normalize(
                        data
                    )

        return data_dict

    @torch.no_grad()
    def unnormalize(self, data_dict: batch_type) -> batch_type:
        for usage, data_dict_ in data_dict.items():
            if usage not in self.normalizers:
                continue
            for name, data in data_dict_.items():
                if name in self.normalizers[usage]:
                    data_dict[usage][name] = self.normalizers[usage][name].unnormalize(
                        data
                    )
        return data_dict

    @torch.no_grad()
    def as_dict(self, data_class: str) -> dict[str, dict[str, Any]]:
        state_dict = {}
        for usage, normalizer_dict in self.normalizers.items():
            if usage not in state_dict:
                state_dict[usage] = {}
            for name, normalizer in normalizer_dict.items():
                state_dict[usage][name] = normalizer.as_dict(data_class)
        return state_dict


class BaseTransforms:
    def __init__(self, data_meta: dict[str, DataMeta]):
        self.transforms: dict[str, K.VideoSequential] = {}
        self.data_meta: dict[str, DataMeta] = data_meta

        for entry_meta in data_meta.values():
            transforms_list = []
            for aug_cfg in entry_meta.augmentation:
                aug_name = aug_cfg["name"]
                if aug_name not in K.__dict__:
                    raise ValueError(
                        f"Augmentation {aug_name} not found in kornia.augmentation. Please implement your own augmentation method."
                    )
                aug_cfg.pop("name")
                transform_cls = K.__dict__[aug_name]
                transforms_list.append(transform_cls(**aug_cfg))
            if len(transforms_list) > 0:
                self.transforms[entry_meta.name] = K.VideoSequential(*transforms_list)

    def to(self, device: Union[torch.device, str]):
        for transform in self.transforms.values():
            transform.to(device)

    def apply(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        for name, data in data_dict.items():
            if isinstance(data, dict):
                data_dict[name] = self.apply(data)
            elif isinstance(data, torch.Tensor):
                if name in self.transforms:
                    data_dim_num = len(self.data_meta[name].shape)
                    data_shape = data.shape
                    new_data_dim_num = len(data.shape)
                    assert (
                        data.shape[-data_dim_num:] == self.data_meta[name].shape
                    ), f"Data {name} shape {data.shape} does not match meta shape {self.data_meta[name].shape}"
                    if new_data_dim_num - data_dim_num == 1:
                        data = data.unsqueeze(0)
                    elif new_data_dim_num - data_dim_num != 2:
                        raise ValueError(
                            f"Data {name} has more than 2 additional dimensions: {data.shape}. Currently only support (traj_len, *shape) or (batch_size, traj_len, *shape)."
                        )
                    data = self.transforms[name](data)
                    data_dict[name] = data.reshape(data_shape)

            else:
                raise ValueError(f"Unknown data type {type(data)} for {name}")
        return data_dict


class BaseLazyDataset(Dataset[batch_type]):
    """
    Base class for all datasets.
    """

    def __init__(
        self,
        zarr_path: str,
        name: str,
        include_episode_num: int,
        include_episode_indices: list[int],
        used_episode_ratio: float,
        index_pool_size_per_episode: int,
        history_padding_length: int,
        future_padding_length: int,
        seed: int,
        source_data_meta: Union[dict[str, dict[str, Any]], DictConfig],
        output_data_meta: Union[dict[str, dict[str, Any]], DictConfig],
        dataloader_cfg: dict[str, Any],
        starting_percentile_max: float,
        starting_percentile_min: float,
        apply_augmentation_in_cpu: bool,
    ):

        self.zarr_path: str = zarr_path
        if name == "":
            name = zarr_path.split("/")[-1].split(".")[0]
        self.name: str = name

        self.include_episode_num: int = include_episode_num
        self.include_episode_indices: list[int] = include_episode_indices
        self.used_episode_ratio: float = used_episode_ratio
        self.index_pool_size_per_episode: int = index_pool_size_per_episode
        self.history_padding_length: int = history_padding_length
        self.future_padding_length: int = future_padding_length
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.dataloader_cfg: dict[str, Any] = dataloader_cfg
        self.starting_percentile_max: float = starting_percentile_max
        self.starting_percentile_min: float = starting_percentile_min
        self.apply_augmentation_in_cpu: bool = apply_augmentation_in_cpu

        zarr_store = zarr.open(self.zarr_path, mode="r")
        assert isinstance(
            zarr_store, zarr.Group
        ), f"zarr store {self.zarr_path} is not a group."
        self.zarr_store: zarr.Group = zarr_store

        assert len(source_data_meta) > 0, "source_data_meta is empty."
        self.source_data_meta: dict[str, SourceDataMeta] = construct_source_data_meta(
            source_data_meta
        )

        assert len(output_data_meta) > 0, "output_data_meta is empty."
        self.output_data_meta: dict[str, DataMeta] = construct_data_meta(
            output_data_meta
        )

        self.max_history_length: int = max(
            0,
            -min(
                entry_meta.include_indices[0]
                for entry_meta in self.source_data_meta.values()
            ),
        )
        self.max_future_length: int = max(
            0,
            max(
                entry_meta.include_indices[-1]
                for entry_meta in self.source_data_meta.values()
            ),
        )
        self.history_padding_length: int = history_padding_length
        self.future_padding_length: int = future_padding_length

        if self.history_padding_length > self.max_history_length:
            raise ValueError(
                f"history_padding_length {self.history_padding_length} is larger than max_history_length {self.max_history_length}. This may cause ambiguity in the data."
            )

        self.normalizer: Optional[FixedNormalizer] = None

        if self.zarr_store.attrs.get("normalizer") is not None:
            self.normalizer = FixedNormalizer(self.output_data_meta)
            self.normalizer.from_dict(self.zarr_store.attrs["normalizer"])
            self.normalizer.to(torch.device("cpu"))
            print(f"Normalizer loaded from zarr store attrs.")

        self.store_episode_num: int
        self.used_episode_indices: list[int]
        self.used_episode_num: int

        self.episode_valid_indices_min: dict[int, int]
        self.episode_valid_indices_max: dict[int, int]

        self.index_pool: list[tuple[int, int]] = []
        """
        index_pool has self.store_episode_num * self.used_episode_ratio * self.index_pool_size_per_episode items.
        Each item contains a tuple of (episode_idx, index), where index means the 0 index of this trajectory in an episode.
        """

        self.transforms: BaseTransforms = BaseTransforms(self.output_data_meta)

    def _check_data_validity(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _create_index_pool(self):
        self.index_pool = []
        for episode_idx in self.used_episode_indices:

            valid_idx_min = self.episode_valid_indices_min[episode_idx]
            valid_idx_max = self.episode_valid_indices_max[episode_idx]
            # valid_idx_min <= sample_idx < valid_idx_max

            zero_idx_max = valid_idx_min + int(
                (valid_idx_max - valid_idx_min) * self.starting_percentile_max
            )  # Exclusive
            zero_idx_min = valid_idx_min + int(
                (valid_idx_max - valid_idx_min) * self.starting_percentile_min
            )  # Inclusive

            if self.index_pool_size_per_episode == -1:
                index_pool_size = zero_idx_max - zero_idx_min
            else:
                assert (
                    self.index_pool_size_per_episode > 0
                ), f"index_pool_size_per_episode must be positive or -1, but got {self.index_pool_size_per_episode}."
                index_pool_size = self.index_pool_size_per_episode

            indices = self.rng.choice(
                range(zero_idx_min, zero_idx_max), size=index_pool_size, replace=False
            )
            indices = np.sort(indices)
            for index in indices:
                self.index_pool.append((episode_idx, index))

    def _update_episode_indices(self):

        if len(self.include_episode_indices) > 0:
            print(
                f"Dataset {self.name}: Using specified episode indices: {self.include_episode_indices}."
            )
            self.include_episode_num: int = len(self.include_episode_indices)
            for episode_idx in self.include_episode_indices:
                assert (
                    episode_idx < self.store_episode_num
                ), f"episode_idx {episode_idx} is out of range. Max is {self.store_episode_num}."
        else:
            if self.include_episode_num > 0:
                assert (
                    self.include_episode_num <= self.store_episode_num
                ), f"include_episode_num {self.include_episode_num} is greater than the number of episodes {self.store_episode_num}."
                self.include_episode_indices = self.rng.choice(
                    self.store_episode_num, size=self.include_episode_num, replace=False
                ).tolist()
                print(
                    f"Dataset {self.name}: Using {self.include_episode_num} episodes from {self.store_episode_num} episodes: {self.include_episode_indices}"
                )
            elif self.include_episode_num == -1:
                self.include_episode_num = self.store_episode_num
                self.include_episode_indices = list(range(self.include_episode_num))
                print(
                    f"Dataset {self.name}: Using all {self.include_episode_num} episodes from {self.store_episode_num}"
                )
            else:
                raise ValueError(
                    f"include_episode_num {self.include_episode_num} is invalid. Must be -1 or a positive integer."
                )

        self.include_episode_indices = sorted(self.include_episode_indices)

        self.used_episode_indices: list[int] = cast(
            list[int],
            self.rng.choice(
                self.include_episode_indices,
                size=int(self.include_episode_num * self.used_episode_ratio),
                replace=False,
            ).tolist(),
        )
        self.used_episode_indices = sorted(self.used_episode_indices)
        self.used_episode_num: int = len(self.used_episode_indices)

    def split_unused_episodes(
        self,
        remaining_ratio: float = 1.0,
        other_used_episode_indices: Optional[list[int]] = None,
    ):
        """
        Split unused episodes from the included episodes.
        """
        print(
            f"Splitting unused data with remaining ratio {remaining_ratio} and other used episode ids {other_used_episode_indices}."
        )
        unused_dataset = copy.deepcopy(self)
        unused_dataset.rng = np.random.default_rng(unused_dataset.seed)
        if other_used_episode_indices is None:
            other_used_episode_indices = []
        unused_episode_indices = [
            episode_idx
            for episode_idx in self.include_episode_indices
            if episode_idx not in self.used_episode_indices
            and episode_idx not in other_used_episode_indices
        ]
        unused_dataset.used_episode_indices = cast(
            list[int],
            self.rng.choice(
                unused_episode_indices,
                size=int(len(unused_episode_indices) * remaining_ratio),
                replace=False,
            ).tolist(),
        )
        unused_dataset.used_episode_ratio = len(
            unused_dataset.used_episode_indices
        ) / len(unused_dataset.include_episode_indices)
        unused_dataset._check_data_validity()
        unused_dataset._create_index_pool()
        assert len(unused_dataset) >= 1, f"Splitted dataset {unused_dataset.name} has no data. Please check the used_data_ratio and the overall dataset size"
        return unused_dataset

    def get_dataloader(self):
        return DataLoader(self, **self.dataloader_cfg)

    def get_all_data(self, entry_names: list[str]) -> batch_type:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def fit_normalizer(self) -> FixedNormalizer:
        self.normalizer = FixedNormalizer(self.output_data_meta)

        range_normalize_entries = [
            entry_meta.name
            for entry_meta in self.output_data_meta.values()
            if entry_meta.normalizer == "range"
        ]

        self.normalizer.fit_range_normalizer(self.get_all_data(range_normalize_entries))
        self.normalizer.to(torch.device("cpu"))

        normalizer_state_dict = self.normalizer.as_dict("list")

        # Temporarily reopen the zarr store in write mode to save the normalizer
        temp_store = zarr.open(self.zarr_path, mode="a")
        temp_store.attrs["normalizer"] = normalizer_state_dict
        print(f"Normalizer dict saved to zarr store attrs.")

        return self.normalizer

    def process_image_data(self, data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if (
            data.shape[-1] <= 4
        ):  # (..., H, W, C) where the color dimension is usually a small number (1 (grayscale), 3 (RGB), or 4 (RGBD))
            dims = len(data.shape)
            data = data.transpose((*range(dims - 3), -1, -3, -2))  # (..., C, H, W)
        if data.dtype == np.uint8:
            return (data / 255.0).astype(np.float32)
        return data.astype(np.float32)

    def __len__(self) -> int:
        return len(self.index_pool)

    def __getitem__(self, idx: int) -> batch_type:
        raise NotImplementedError("This method should be implemented in subclasses.")
