from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import glob
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from unified_video_action.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from unified_video_action.model.common.rotation_transformer import RotationTransformer
from unified_video_action.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from unified_video_action.common.replay_buffer import ReplayBuffer
from unified_video_action.common.sampler import SequenceSampler, get_val_mask
from unified_video_action.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)

register_codecs()
from transformers import T5Tokenizer
from transformers import AutoTokenizer, CLIPModel
import torchvision.transforms as transforms


class LiberoReplayImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        abs_action=False,
        rotation_rep="rotation_6d",  # ignored when abs_action=False
        use_legacy_normalizer=False,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        language_emb_model=None,
        data_aug=False,
        normalizer_type=None,
    ):

        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

        replay_buffer = None
        if use_cache:

            if language_emb_model == "clip":
                cache_zarr_path = dataset_path + "_clip.zarr.zip"
            elif language_emb_model == "flant5":
                cache_zarr_path = dataset_path + "_flant5.zarr.zip"
            else:
                import pdb

                pdb.set_trace()

            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            print("Cache path:", cache_zarr_path)

            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        # store = zarr.DirectoryStore(cache_zarr_path)

                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                            language_emb_model=language_emb_model,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                language_emb_model=language_emb_model,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        self.data_aug = data_aug

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

        self.language_emb_model = language_emb_model

        self.__getitem__(0)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                # dual arm
                this_normalizer = (
                    robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
                )
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("language"):
                continue  ## skip
            else:
                print(key)
                pdb.set_trace()
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = np.rot90(obs_dict[key], k=2, axes=(2, 3)).copy()
            obs_dict[key] = np.flip(obs_dict[key], axis=3).copy()
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        if self.data_aug:
            image_tensor = torch.tensor(obs_dict["agentview_rgb"], dtype=torch.float32)
            video_seed = torch.randint(0, 10000, (1,)).item()

            def consistent_augmentations(frame):
                # Set the random seed for each frame to ensure consistent augmentation
                torch.manual_seed(video_seed)
                augmentation = transforms.Compose(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                        ),  # Color jitter
                    ]
                )
                return augmentation(frame)

            augmented_images = torch.stack(
                [consistent_augmentations(frame) for frame in image_tensor]
            )
            obs_dict["agentview_rgb"] = augmented_images.numpy()

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    language_emb_model=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    file_handles = []  # Store file handles if you need to keep them open
    demos_all = {}
    language_all = {}
    count = 0

    if language_emb_model == "clip":
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    elif language_emb_model == "flant5":
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    else:
        pdb.set_trace()

    if "libero_mixall" in dataset_path:
        dataset_paths = []
        dataset_paths += glob.glob(
            dataset_path.replace("libero_mixall", "libero_10") + "/*.hdf5"
        )
        print(
            f"Loading {dataset_path.replace('libero_mixall', 'libero_10')}",
            len(dataset_paths),
        )
        dataset_paths += glob.glob(
            dataset_path.replace("libero_mixall", "libero_90") + "/*.hdf5"
        )
        print(
            f"Loading {dataset_path.replace('libero_mixall', 'libero_90')}",
            len(dataset_paths),
        )
        dataset_paths += glob.glob(
            dataset_path.replace("libero_mixall", "libero_goal") + "/*.hdf5"
        )
        print(
            f"Loading {dataset_path.replace('libero_mixall', 'libero_goal')}",
            len(dataset_paths),
        )
        dataset_paths += glob.glob(
            dataset_path.replace("libero_mixall", "libero_object") + "/*.hdf5"
        )
        print(
            f"Loading {dataset_path.replace('libero_mixall', 'libero_object')}",
            len(dataset_paths),
        )
        dataset_paths += glob.glob(
            dataset_path.replace("libero_mixall", "libero_spatial") + "/*.hdf5"
        )
        print(
            f"Loading {dataset_path.replace('libero_mixall', 'libero_spatial')}",
            len(dataset_paths),
        )
    elif "libero_90_" in dataset_path:
        dataset_paths = glob.glob(dataset_path.split("_train")[0] + "/*.hdf5")
        dataset_paths_new_scene = [tem for tem in dataset_paths if "STUDY_" in tem]
        dataset_paths_new_object = [
            tem
            for tem in dataset_paths
            if "butter" in tem
            or "wine_bottle" in tem
            or "milk" in tem
            or "chocolate" in tem
        ]
        dataset_paths = [
            tem
            for tem in dataset_paths
            if tem not in dataset_paths_new_scene
            and tem not in dataset_paths_new_object
        ]
        print("there are {} tasks used for training".format(len(dataset_paths)))
    elif "10demos" in dataset_path:
        dataset_paths = glob.glob(dataset_path.replace("_10demos", "") + "/*.hdf5")
    else:
        dataset_paths = glob.glob(dataset_path + "/*.hdf5")

    for dataset_path_each in dataset_paths:
        language_goal = " ".join(dataset_path_each.split("/")[-1][:-10].split("_"))
        print(f"Loading {dataset_path_each}")
        file = h5py.File(
            dataset_path_each, "r"
        )  # Open the file without closing it immediately
        file_handles.append(
            file
        )  # Keep track of the file handle to avoid it being closed
        demos = file["data"]

        for i in range(len(demos)):
            if "10demos" in dataset_path:
                if i >= 10:
                    print(f"there are more than 10 demos in {dataset_path_each}")
                    break

            demo = demos[f"demo_{i}"]
            demos_all[f"demo_{count}"] = demo
            language_all[f"demo_{count}"] = language_goal
            count += 1
    print("Total demos:", count)
    pdb.set_trace()

    if "libero_goal" in dataset_path:
        seq_max_len = 15
    elif "libero_10" in dataset_path:
        seq_max_len = 30
    elif "libero_90" in dataset_path:
        seq_max_len = 30
    elif "libero_mixall" in dataset_path:
        seq_max_len = 30

    if language_emb_model == "clip":
        language_all_tokens = [
            tokenizer(
                language_all[f"demo_{i}"],
                padding="max_length",
                max_length=seq_max_len,
                return_tensors="pt",
            )
            for i in range(len(language_all))
        ]
        language_input_ids = [
            item.input_ids.unsqueeze(1) for item in language_all_tokens
        ]
        language_attention_mask = [
            item.attention_mask.unsqueeze(1) for item in language_all_tokens
        ]
    elif language_emb_model == "flant5":
        language_all_tokens = torch.LongTensor(
            [
                tokenizer(
                    language_all[f"demo_{i}"],
                    padding="max_length",
                    max_length=seq_max_len,
                ).input_ids
                for i in range(len(language_all))
            ]
        )
        language_all_tokens = language_all_tokens.unsqueeze(1)
    else:
        pdb.set_trace()

    demos = demos_all
    episode_ends = list()
    prev_end = 0
    for i in range(len(demos)):
        demo = demos[f"demo_{i}"]
        episode_length = demo["actions"].shape[0]
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    # save lowdim data
    for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
        data_key = "obs/" + key
        if key == "action":
            data_key = "actions"
            this_language_data = list()
        if key == "language":
            continue
        this_data = list()
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            this_data.append(demo[data_key][:].astype(np.float32))

            if key == "action":
                if language_emb_model == "clip":
                    language_tokens = torch.cat(
                        [language_input_ids[i], language_attention_mask[i]], dim=1
                    )
                    this_language_data.append(
                        language_tokens.repeat(this_data[-1].shape[0], 1, 1)
                    )
                elif language_emb_model == "flant5":
                    this_language_data.append(
                        language_all_tokens[i].repeat(this_data[-1].shape[0], 1)
                    )
                else:
                    pdb.set_trace()

        this_data = np.concatenate(this_data, axis=0)

        if key == "action":
            this_data = _convert_actions(
                raw_actions=this_data,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )

            assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])

            this_language_data = np.concatenate(this_language_data, axis=0)
            if language_emb_model == "clip":
                assert this_language_data.shape == (n_steps,) + tuple([2, seq_max_len])
            elif language_emb_model == "flant5":
                assert this_language_data.shape == (n_steps,) + tuple([seq_max_len])
            else:
                pdb.set_trace()
        else:
            assert this_data.shape == (n_steps,) + tuple(
                shape_meta["obs"][key]["shape"]
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

        if key == "action":
            _ = data_group.array(
                name="language",
                data=this_language_data,
                shape=this_language_data.shape,
                chunks=this_language_data.shape,
                compressor=None,
                dtype=this_language_data.dtype,
            )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    with tqdm(
        total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
    ) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = "obs/" + key
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=this_compressor,
                    dtype=np.uint8,
                )

                for episode_idx in range(len(demos)):
                    demo = demos[f"demo_{episode_idx}"]
                    hdf5_arr = demo["obs"][key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError("Failed to encode image!")
                            pbar.update(len(completed))

                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        futures.add(
                            executor.submit(
                                img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                            )
                        )
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(completed))

    # Ensure you close all files when you're done with them
    for file in file_handles:
        file.close()

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )
