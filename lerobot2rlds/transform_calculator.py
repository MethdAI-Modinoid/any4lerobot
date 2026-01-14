#!/usr/bin/env python3

import time
import requests
import argparse
from pathlib import Path
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------- CONFIG ----------------

URL_PUB_JOINTS = "http://localhost:8000/pub_joints"
URL_GET_TRANSFORM = "http://localhost:8000/transform"

SLEEP_S = 0.01   # keep exactly what you were using

# ----------------------------------------

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
            # print("OPEN")



def main(src_dir: Path, output_path: Path):
    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset("", src_dir)

    transforms = {}

    last_episode = None

    for data_item in dataset:
        episode_idx = data_item["episode_index"].item()
        frame_idx = data_item["frame_index"].item()

        # Optional: log episode boundaries
        if last_episode != episode_idx:
            print(f"\n=== Episode {episode_idx} ===")
            last_episode = episode_idx

        # -------------------------------------------------
        # ACTION SPLITTING (your exact logic)
        # -------------------------------------------------
        action_vec = data_item["action"]
        # action_vec = data_item["observation.state"]


        first_14 = action_vec[:14]
        right = first_14[7:]          # 7 joints
        gripper_act = action_vec[21:] # hand state (unused here, kept for parity)

        grip = gripper_state(gripper_act)

        # -------------------------------------------------
        # HTTP SIDE-EFFECTS (SAFE HERE)
        # -------------------------------------------------
        params = [("float_arr", v.item()) for v in right]
        requests.get(URL_PUB_JOINTS, params=params)

        # time.sleep(SLEEP_S)

        resp = requests.get(URL_GET_TRANSFORM)
        resp.raise_for_status()
        transform = resp.json()

        transform.append(grip)

        print(transform)
        

        # -------------------------------------------------
        # CACHE RESULT
        # -------------------------------------------------
        transforms[(episode_idx, frame_idx)] = transform

        print(f"[ep {episode_idx:03d} | frame {frame_idx:04d}] transform cached")

    # -------------------------------------------------
    # SAVE TO DISK
    # -------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, transforms)

    print("\nDone.")
    print(f"Saved {len(transforms)} transforms to:")
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Path to raw LeRobot dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("transforms.npy"),
        help="Output .npy file",
    )

    args = parser.parse_args()
    main(args.src_dir, args.output)
