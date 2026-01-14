
import requests

import math
import argparse
import logging
import os
from functools import partial
from pathlib import Path
import time

import numpy as np
import tensorflow_datasets as tfds
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tensorflow_datasets.core.file_adapters import FileFormat
from tensorflow_datasets.core.utils.lazy_imports_utils import apache_beam as beam
from tensorflow_datasets.rlds import rlds_base
import torch

os.environ["NO_GCE_CHECK"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tfds.core.utils.gcs_utils._is_gcs_disabled = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

url_joint = "http://localhost:8000/pub_joints"

joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

params = []

# -------------------------------------------------
# CONFIG GENERATION (UNCHANGED)
# -------------------------------------------------
def generate_config_from_features(features, encoding_format, **kwargs):
    action_info = {
        **{
            "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
                shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
            )
            for k, v in features.items()
            if "action" in k
        },
    }

    # RLDS allows dict OR tensor, but we will ALWAYS emit dicts
    return dict(
        observation_info={
            **{
                k.split(".")[-1]: tfds.features.Image(
                    shape=v["shape"],
                    dtype=np.uint8,
                    encoding_format=encoding_format,
                    doc=v["names"],
                )
                for k, v in features.items()
                if "observation.image" in k and "depth" not in k
            },
            **{
                k.split(".")[-1]: tfds.features.Tensor(
                    shape=v["shape"][:-1], dtype=np.float32, doc=v["names"]
                )
                for k, v in features.items()
                if "observation.image" in k and "depth" in k
            },
            **{
                "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                )
                for k, v in features.items()
                if "observation.state" in k
            },
        },
        action_info=action_info,
        step_metadata_info={
            "language_instruction": tfds.features.Text(),
        },
        citation=kwargs.get("citation", ""),
        homepage=kwargs.get("homepage", ""),
        overall_description=kwargs.get("overall_description", ""),
        description=kwargs.get("description", ""),
    )

def quat_to_rpy(w, x, y, z):
    """
    Quaternion (w, x, y, z) -> Roll, Pitch, Yaw
    ZYX convention (yaw-pitch-roll), radians
    Assumes quaternion is normalized
    """

    # Roll (x-axis)
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # gimbal lock
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

RIGHT_HAND_OPEN = np.array([
        0.0672,  # thumb0
        0.5666,  # thumb1
        -0.0679,  # thumb2
        -0.0211,  # middle0
        -0.0112,  # middle1
        -0.0162,  # index0
        -0.0283,  # index1
    ])

RIGHT_HAND_CLOSE = np.array([
        -0.03833567723631859,  # thumb0
       -0.36572766304016113,  # thumb1
        -0.024161333218216896,  # thumb2
         0.9473425149917603,  # middle0
        -0.044050849974155426,  # middle1
        0.9455186128616333,  # index0
        -0.06319903582334518,  # index1
    ])

def gripper_state(hand_state: np.ndarray, margin=0.1) -> int:
        d_open = np.linalg.norm(hand_state - RIGHT_HAND_OPEN)
        d_close = np.linalg.norm(hand_state - RIGHT_HAND_CLOSE)
        margin = 0.1
        if d_open + margin < d_close:
            return 0
            # print("OPEN")
        elif d_close + margin < d_open:
            return 1
            # print("CLOSE")
        else:
            return 0 
            print("OPEN")


# -------------------------------------------------
# 🔧 FIXED parse_step (YOUR VERSION)
# -------------------------------------------------
def parse_step(data_item):
    observation_info = {
        **{
            # RGB image: (C,H,W) → (H,W,C), uint8
            k.split(".")[-1]: (v * 255).byte().permute(1, 2, 0).cpu().numpy()
            for k, v in data_item.items()
            if "observation.image" in k and "depth" not in k
        },
        **{
            # Depth image
            k.split(".")[-1]: v.float().squeeze().cpu().numpy()
            for k, v in data_item.items()
            if "observation.image" in k and "depth" in k
        },
        **{
            "_".join(k.split(".")[2:]) or k.split(".")[-1]: v
            for k, v in data_item.items()
            if "observation.state" in k
        },
    }

    # 🔒 ALWAYS return a dict for action
    action_info = {
        "_".join(k.split(".")[2:]) or k.split(".")[-1]: v
        for k, v in data_item.items()
        if "action" in k
    }

    assert isinstance(action_info, dict)
    assert len(action_info) > 0

    return observation_info, action_info, data_item["task"]

##maopping of 28 joints lerobot
# [

#     "kLeftShoulderPitch",
#     "kLeftShoulderRoll",
#     "kLeftShoulderYaw",
#     "kLeftElbow",
#     "kLeftWristRoll",
#     "kLeftWristPitch",
#     "kLeftWristYaw",
#     "kRightShoulderPitch",
#     "kRightShoulderRoll",
#     "kRightShoulderYaw",
#     "kRightElbow",
#     "kRightWristRoll",
#     "kRightWristPitch",
#     "kRightWristYaw",
#     "kLeftHandThumb0",
#     "kLeftHandThumb1",
#     "kLeftHandThumb2",
#     "kLeftHandMiddle0",
#     "kLeftHandMiddle1",
#     "kLeftHandIndex0",
#     "kLeftHandIndex1",
#     "kRightHandThumb0",
#     "kRightHandThumb1",
#     "kRightHandThumb2",
#     "kRightHandIndex0",
#     "kRightHandIndex1",
#     "kRightHandMiddle0",
#     "kRightHandMiddle1"
# ]


#  RIGHT_HAND_OPEN = np.array([
#         0.0672,  # thumb0
#         0.5666,  # thumb1
#         -0.0679,  # thumb2
#         -0.0211,  # middle0
#         -0.0112,  # middle1
#         -0.0162,  # index0
#         -0.0283,  # index1
#     ])

    # RIGHT_HAND_CLOSE = np.array([
    #     -0.03833567723631859,  # thumb0
    #    -0.36572766304016113,  # thumb1
    #     -0.024161333218216896,  # thumb2
    #      0.9473425149917603,  # middle0
    #     -0.044050849974155426,  # middle1
    #     0.9455186128616333,  # index0
    #     -0.06319903582334518,  # index1
    # ])
    
    # [-0.03833567723631859, -0.36572766304016113, -0.024161333218216896, 0.9473425149917603, -0.044050849974155426, 0.9455186128616333, -0.06319903582334518]


# -------------------------------------------------
# DATASET BUILDER (UNCHANGED EXCEPT parse_step EFFECT)
# -------------------------------------------------
class DatasetBuilder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    def __init__(self, raw_dir, name, dataset_config, enable_beam, *, file_format=None, **kwargs):
        self.name = name
        self.VERSION = kwargs["version"]
        self.raw_dir = raw_dir
        self.dataset_config = dataset_config
        self.enable_beam = enable_beam
        self.__module__ = "lerobot2rlds"
        super().__init__(file_format=file_format, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return rlds_base.build_info(
            rlds_base.DatasetConfig(
                name=self.name,
                **self.dataset_config,
            ),
            self,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dl_manager._download_dir.rmtree(missing_ok=True)
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        
        def _generate_examples_beam(episode_index: int, raw_dir: Path):
            episode = []
            dataset = LeRobotDataset("", raw_dir, episodes=[episode_index], load_images=False)

            meta = dataset.meta.episodes[episode_index]
            expected_length = meta["length"]

            print(f"\n=== Episode {episode_index} ===")
            print(f"Expected length: {expected_length}")

            for data_item in dataset:
                obs, act, lang = parse_step(data_item)
                frame_index = data_item["frame_index"].item()
                lang = "Place the red apple in the brown box."

                # ---------------- NEW SIDE-EFFECT LOGIC ----------------

                # Split actions
                action_vec = act["action"]
                first_14 = action_vec[:14]          # (14,)
                right = first_14[7:]         # (7,)
                gripper_act = action_vec[21:]

                # Publish joints
                params = []
                # print("right", right)
                for i in range(len(right)):
                    params.append(("float_arr", right[i].item()))

                resp = requests.get(url_joint, params=params)
                # print(resp.status_code, resp.json())
                params.clear()

                time.sleep(0.1)

                # Query transform
                resp = requests.get("http://localhost:8000/transform")
                # print(resp.status_code)
                # print(resp.json())f

                transform = resp.json()  # NOTE: currently unused (side-effect only)

                # Gripper state (side-effect only)
                grip = gripper_state(gripper_act)
                
                arr = [transform[0], transform[1], transform[2], transform[3], transform[4], transform[5], grip]

                act["action"] = np.asarray(arr, dtype=np.float32)

                print("TRANSFORM", transform)


                # -------------------------------------------------------



                step = {
                    "observation": obs,
                    "action": act,
                    "language_instruction": lang,
                    "is_first": frame_index == 0,
                    "is_last": frame_index == expected_length - 1,
                    "is_terminal": frame_index == expected_length - 1,
                }

                # ---- HARD ASSERTIONS (from original) ----
                assert isinstance(step["observation"], dict)
                assert isinstance(step["action"], dict)

                for k, v in step["action"].items():
                    assert hasattr(v, "shape"), f"Action {k} is not tensor-like"

                assert isinstance(step["is_first"], bool)
                assert isinstance(step["is_last"], bool)

                episode.append(step)

            print(f"Collected steps: {len(episode)}")

            # ---- EPISODE-LEVEL ASSERTIONS (from original) ----
            assert len(episode) == expected_length, (
                f"Length mismatch: got {len(episode)}, expected {expected_length}"
            )

            assert episode[0]["is_first"] is True
            assert episode[-1]["is_last"] is True
            assert episode[-1]["is_terminal"] is True

            return episode_index, {"steps": episode}

        def _generate_examples_regular():
            dataset = LeRobotDataset("", self.raw_dir)
            episode = []
            current_episode_index = 0

            for data_item in dataset:
                if data_item["episode_index"] != current_episode_index:
                    episode[-1]["is_last"] = True
                    episode[-1]["is_terminal"] = True
                    yield f"{current_episode_index}", {"steps": episode}
                    episode = []
                    current_episode_index = data_item["episode_index"]

                obs, act, lang = parse_step(data_item)

                episode.append(
                    {
                        "observation": obs,
                        "action": act,
                        "language_instruction": lang,
                        "is_first": data_item["frame_index"].item() == 0,
                        "is_last": False,
                        "is_terminal": False,
                    }
                )

            episode[-1]["is_last"] = True
            episode[-1]["is_terminal"] = True
            yield f"{current_episode_index}", {"steps": episode}

        if self.enable_beam:
            metadata = LeRobotDatasetMetadata("", self.raw_dir)
            return beam.Create(range(len(metadata.episodes))) | beam.Map(
                partial(_generate_examples_beam, raw_dir=self.raw_dir)
            )
        else:
            return _generate_examples_regular()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main(src_dir, output_dir, task_name, version, encoding_format, enable_beam, **kwargs):
    raw_dataset_meta = LeRobotDatasetMetadata("", root=src_dir)
    dataset_config = generate_config_from_features(raw_dataset_meta.features, encoding_format, **kwargs)

    dataset_builder = DatasetBuilder(
        raw_dir=src_dir,
        name=task_name,
        data_dir=output_dir,
        version=version,
        dataset_config=dataset_config,
        enable_beam=enable_beam,
        file_format=FileFormat.TFRECORD,
    )

    if enable_beam:
        from apache_beam.options.pipeline_options import PipelineOptions
        from apache_beam.runners import create_runner

        beam_options = PipelineOptions(
            direct_running_mode=kwargs["beam_run_mode"],
            direct_num_workers=kwargs["beam_num_workers"],
        )
        beam_runner = create_runner("DirectRunner")
    else:
        beam_options = None
        beam_runner = None

    dataset_builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            try_download_gcs=False,
            verify_ssl=False,
            beam_options=beam_options,
            beam_runner=beam_runner,
        ),
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--enable-beam", action="store_true")
    parser.add_argument("--beam-run-mode", choices=["multi_threading", "multi_processing"], default="multi_processing")
    parser.add_argument("--beam-num-workers", type=int, default=5)
    parser.add_argument("--encoding-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--version", type=str, default="0.1.0")
    parser.add_argument("--citation", type=str, default="")
    parser.add_argument("--homepage", type=str, default="")
    parser.add_argument("--overall-description", type=str, default="")
    parser.add_argument("--description", type=str, default="")

    args = parser.parse_args()
    main(**vars(args))


