import os
import sys
from typing import Any, Union, cast
import zarr
import numpy as np
import torch
import numpy.typing as npt

from unified_video_action.dataset.base_lazy_dataset import BaseLazyDataset
from umi.common.cv_util import draw_predefined_mask
from umi.common.pose_util import pose_to_mat, mat_to_pose10d
from unified_video_action.common.pose_repr_util import convert_pose_mat_rep
from unified_video_action.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

register_codecs()


class UmiLazyDataset(BaseLazyDataset):
    """
    Dataset loader for the official UMI dataset.
    Example structure:
    /
    ├── data
    │   ├── camera0_rgb (2315, 224, 224, 3) uint8
    │   ├── robot0_demo_end_pose (2315, 6) float64
    │   ├── robot0_demo_start_pose (2315, 6) float64
    │   ├── robot0_eef_pos (2315, 3) float32
    │   ├── robot0_eef_rot_axis_angle (2315, 3) float32
    │   └── robot0_gripper_width (2315, 1) float32
    └── meta
        └── episode_ends (5,) int64
    """

    def __init__(
        self,
        robot_num: int,
        use_relative_pose: bool,
        down_sample_steps: int,
        mask_mirror: bool,
        random_img_sampling: bool,
        **kwargs,
    ):

        self.down_sample_steps: int = down_sample_steps
        kwargs["history_padding_length"] = (
            kwargs["history_padding_length"] * down_sample_steps
        )
        kwargs["future_padding_length"] = (
            kwargs["future_padding_length"] * down_sample_steps
        )
        for meta in kwargs["source_data_meta"].values():
            meta["include_indices"] = [
                i * down_sample_steps for i in meta["include_indices"]
            ]

        super().__init__(**kwargs)

        data_store = self.zarr_store["data"]
        assert isinstance(data_store, zarr.Group)

        assert (
            isinstance(robot_num, int) and robot_num >= 1
        ), f"robot_num must be an integer greater than 0, but got {robot_num}."
        self.robot_num: int = robot_num
        self.data_store: zarr.Group = data_store
        self.data_store_keys: list[str] = list(data_store.keys())

        self.episode_ends: npt.NDArray[np.int64] = np.array(
            self.zarr_store["meta"]["episode_ends"]
        )
        self.store_episode_num: int = len(self.episode_ends)
        self.mask_mirror: bool = mask_mirror

        self._update_episode_indices()

        self.episode_starts: npt.NDArray[np.int64] = np.zeros_like(self.episode_ends)
        self.episode_frame_nums: dict[int, int] = {}
        self.episode_valid_indices_min: dict[int, int] = {}
        self.episode_valid_indices_max: dict[int, int] = {}  # Exclusive

        for i, end in enumerate(self.episode_ends):
            if i == 0:
                self.episode_starts[i] = 0
            else:
                self.episode_starts[i] = self.episode_ends[i - 1]

            self.episode_frame_nums[i] = end - self.episode_starts[i]
            self.episode_valid_indices_min[i] = (
                self.max_history_length - self.history_padding_length
            )
            self.episode_valid_indices_max[i] = (
                self.episode_frame_nums[i]
                + self.future_padding_length
                - self.max_future_length
            )
        self.use_relative_pose: bool = use_relative_pose
        self.random_img_sampling: bool = random_img_sampling
        self._create_index_pool()

        print(
            f"Dataset: {self.name}, store_episode_num: {self.store_episode_num}, include_episode_num: {self.include_episode_num}, used_episode_num: {self.used_episode_num}"
        )

    def _check_data_validity(self):
        # No need to check data validity for UMI dataset
        pass

    # @profile
    def _process_source_data(
        self, data_dict: dict[str, npt.NDArray[Any]]
    ) -> dict[str, npt.NDArray[Any]]:
        """
        Will calculate the following data:
            relative poses
            poses wrt episode start
        This step does not include normalization and data augmentation
        Input data_dict:
            camera0_rgb: (..., H, W, 3) uint8
            robot0_demo_start_pose: (1, 6) float64 (optional)
            robot0_eef_pos: (..., 3) float32
            robot0_eef_rot_axis_angle: (..., 3) float32
            robot0_gripper_width: (..., 1) float32
        Output data_dict:
            camera0_rgb: (..., H, W, 3) uint8 # TODO: adjust the frames needed
            robot0_gripper_width: (..., 1) float32 # Truncated to the first few frames based on output_data_meta
            robot0_eef_pos: (..., 3) float32 # Relative to the last frame if use_relative_pose is True
            robot0_eef_rot_axis_angle: (..., 6) float32 # Relative to the last frame if use_relative_pose is True
            robot0_eef_rot_axis_angle_wrt_start: (..., 6) float32 # Relative to the episode start

            action: (..., 10*robot_num) float32 # xyz, rot_6d, gripper_width, all realtive to the first frame
        """

        processed_data_dict: dict[str, npt.NDArray[Any]] = {}

        eef_pos_indices = self.source_data_meta["robot0_eef_pos"].include_indices
        eef_rot_axis_angle_indices = self.source_data_meta[
            "robot0_eef_rot_axis_angle"
        ].include_indices
        assert (
            eef_pos_indices == eef_rot_axis_angle_indices
        ), "eef_pos_indices and eef_rot_axis_angle_indices must be the same"

        eef_pos_length = self.output_data_meta["robot0_eef_pos"].length
        eef_rot_axis_angle_length = self.output_data_meta[
            "robot0_eef_rot_axis_angle"
        ].length
        gripper_width_length = self.output_data_meta["robot0_gripper_width"].length
        assert (
            eef_pos_length == eef_rot_axis_angle_length
        ), "eef_pos_length and eef_rot_axis_angle_length must be the same"

        action_meta = self.output_data_meta["action"]
        action = np.zeros((action_meta.length, *action_meta.shape), dtype=np.float32)

        for i in range(self.robot_num):
            if f"camera{i}_rgb" in data_dict:
                processed_data_dict[f"camera{i}_rgb"] = data_dict[
                    f"camera{i}_rgb"
                ]  # TODO: adjust the frames needed (e.g. random sampling)
                if self.mask_mirror:
                    for j in range(processed_data_dict[f"camera{i}_rgb"].shape[0]):
                        processed_data_dict[f"camera{i}_rgb"][j] = draw_predefined_mask(
                            processed_data_dict[f"camera{i}_rgb"][j],
                            mirror=True,
                            gripper=False,
                            finger=False,
                            use_aa=True,
                        )

            processed_data_dict[f"robot{i}_gripper_width"] = data_dict[
                f"robot{i}_gripper_width"
            ][:gripper_width_length]

            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        data_dict[f"robot{i}_eef_pos"],
                        data_dict[f"robot{i}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )

            if self.use_relative_pose:
                zero_idx = eef_pos_indices.index(0)
                rel_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=pose_mat[zero_idx],
                    pose_rep="relative",
                    backward=False,
                )
                pose = mat_to_pose10d(rel_pose_mat)
            else:
                pose = mat_to_pose10d(pose_mat)

            processed_data_dict[f"robot{i}_eef_pos"] = pose[:eef_pos_length, :3]
            processed_data_dict[f"robot{i}_eef_rot_axis_angle"] = pose[
                :eef_rot_axis_angle_length, 3:
            ]

            action[:, i * 10 : i * 10 + 9] = pose[-action_meta.length :]
            action[:, i * 10 + 9 : (i + 1) * 10] = data_dict[f"robot{i}_gripper_width"][
                -action_meta.length :
            ]

            if f"robot{i}_demo_start_pose" in data_dict:
                # Calculate relative poses wrt episode start
                try:
                    wrt_start_entry_meta = self.output_data_meta[
                        f"robot{i}_eef_rot_axis_angle_wrt_start"
                    ]
                    assert (
                        data_dict[f"robot{i}_demo_start_pose"].shape[0] == 1
                    ), "robot0_demo_start_pose must be (1, 6)"
                    # HACK: add noise to episode start pose
                    start_pose: npt.NDArray[np.float64] = data_dict[
                        f"robot{i}_demo_start_pose"
                    ][0]
                    start_pose += self.rng.normal(
                        scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                        size=start_pose.shape,
                    )
                    start_pose_mat = pose_to_mat(start_pose)
                    rel_pose_mat = convert_pose_mat_rep(
                        pose_mat,
                        base_pose_mat=start_pose_mat,
                        pose_rep="relative",
                        backward=False,
                    )
                    rel_obs_pose = mat_to_pose10d(rel_pose_mat)

                    # Only keep the first wrt_start_entry_meta.length frames
                    processed_data_dict[f"robot{i}_eef_rot_axis_angle_wrt_start"] = (
                        rel_obs_pose[: wrt_start_entry_meta.length, 3:]
                    )

                except ValueError:
                    # No wrt_start_entry_meta, so no relative poses wrt episode start
                    pass

        processed_data_dict["action"] = action
        processed_data_dict["img_indices"] = data_dict["img_indices"][
            :, None
        ]  # [8] -> [8, 1] to satisfy the shape check
        return processed_data_dict

    # @profile
    def __getitem__(self, idx: int):
        """
        output_data_dict:
            obs:
                camera0_rgb: (..., H, W, 3) float32 (0~1)
                robot0_gripper_width: (..., 1) float32
                robot0_eef_pos: (..., 3) float32
                robot0_eef_rot_axis_angle: (..., 6) float32
                robot0_eef_rot_axis_angle_wrt_start: (..., 6) float32
            action: (..., 10*robot_num) float32
        """
        episode_idx, traj_idx = self.index_pool[idx]
        episode_length = self.episode_frame_nums[episode_idx]
        start_idx = self.episode_starts[episode_idx]

        ## select indices
        # T = 32
        # select_timesteps = 4
        # downsample_indices = [i + select_timesteps - 1 for i in range(0, T, T // (select_timesteps * 2))]
        # assert len(downsample_indices) == T // select_timesteps

        source_data_dict: dict[str, Any] = {}
        for entry_meta in self.source_data_meta.values():
            if entry_meta.name not in self.data_store_keys:
                continue

            include_indices = entry_meta.include_indices
            if (
                entry_meta.name in self.output_data_meta
                and self.output_data_meta[entry_meta.name].data_type == "image"
            ):
                if self.random_img_sampling:
                    # Randomly sample 4 frames from [-15, 1)
                    include_indices = (
                        self.rng.choice(np.arange(-15, 1), size=4, replace=False)
                        * self.down_sample_steps
                    ).tolist()
                    include_indices = sorted(include_indices)  # Ascending order
                    include_indices.extend(
                        entry_meta.include_indices[4:]
                    )  # The rest of the frames should still be (4, 8, 12, 16) * down_sample_steps
                source_data_dict["img_indices"] = (
                    np.array(include_indices, dtype=np.int32) / self.down_sample_steps
                    + 15
                )
                # Devide by down_sample_steps to convert the indices to the original scale
                # include_indices are relative to the current frame (0 is the current frame)
                # /self.down_sample_steps+15 to convert the indices to the range of the dataset
            indices = [traj_idx + i for i in include_indices]
            # Crop the indices to the valid range. Will introduce padding if the indices are out of range.
            indices = [
                (0 if i < 0 else episode_length - 1 if i >= episode_length else i)
                for i in indices
            ]
            global_indices = [start_idx + i for i in indices]

            ## select indices
            # if entry_meta.name!='robot0_demo_start_pose':
            #     assert len(global_indices) == T
            #     global_indices = np.array(global_indices)[downsample_indices]

            # HACK: zarr3.0 does not support python 3.9, while zarr2 does not support visiting items using list.
            source_data_dict[entry_meta.name] = np.array(
                [self.data_store[entry_meta.name][idx] for idx in global_indices]
            )
            # source_data_dict[entry_meta.name] = np.zeros((len(global_indices), *entry_meta.shape))

        processed_data_dict = self._process_source_data(source_data_dict)

        output_data_dict: dict[str, Any] = {}
        output_data_dict["obs"] = {}
        output_data_dict["action"] = {}

        for entry_meta in self.output_data_meta.values():
            assert entry_meta.name in processed_data_dict
            processed_data = processed_data_dict[entry_meta.name]
            if isinstance(processed_data, np.ndarray):
                if entry_meta.data_type == "image":
                    processed_data = self.process_image_data(
                        processed_data
                    )  # -> (..., C, H, W), float32 (0~1)
                processed_data = torch.from_numpy(processed_data.astype(np.float32))
            assert processed_data.shape == (
                entry_meta.length,
                *entry_meta.shape,
            ), f"entry_meta: {entry_meta.name}, processed_data.shape: {processed_data.shape}, entry_meta.length: {entry_meta.length}, entry_meta.shape: {entry_meta.shape}"

            output_data_dict[entry_meta.usage][entry_meta.name] = processed_data

        if self.apply_augmentation_in_cpu:
            output_data_dict = self.transforms.apply(output_data_dict)

        if self.normalizer is not None:
            output_data_dict = self.normalizer.normalize(output_data_dict)

        # HACK: action is directly stored:
        output_data_dict["action"] = output_data_dict["action"]["action"]

        output_data_dict["dataset_name"] = self.name

        return output_data_dict
