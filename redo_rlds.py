import logging
from pathlib import Path
import numpy as np
import requests
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch

# -------------------------
# COPY parse_step VERBATIM
# -------------------------
# def parse_step(data_item):
#     observation_info = {
#         **{
#             k.split(".")[-1]: np.array(v * 255, dtype=np.uint8).transpose(1, 2, 0)
#             for k, v in data_item.items()
#             if "observation.image" in k and "depth" not in k
#         },
#         **{
#             k.split(".")[-1]: v.float().squeeze()
#             for k, v in data_item.items()
#             if "observation.image" in k and "depth" in k
#         },
#         **{
#             "_".join(k.split(".")[2:]) or k.split(".")[-1]: v
#             for k, v in data_item.items()
#             if "observation.state" in k
#         },
#     }

#     action_info = {
#         "_".join(k.split(".")[2:]) or k.split(".")[-1]: v
#         for k, v in data_item.items()
#         if "action" in k
#     }

#     action_info = action_info if len(action_info) > 1 else action_info.popitem()[1]

#     return observation_info, action_info, data_item["task"]


##independent node to for ee pose
## add the class here 

url_joint = "http://localhost:8000/pub_joints"
def parse_step(data_item):
    observation_info = {
        **{
            # RGB image: (C,H,W) → (H,W,C), uint8
            k.split(".")[-1]: (v * 255).byte().permute(1, 2, 0).cpu().numpy()
            for k, v in data_item.items()
            if "observation.image" in k and "depth" not in k
        },
        **{
            # depth image
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

    # 🔒 ALWAYS a dict (even if only one action)
    action_info = {
        "_".join(k.split(".")[2:]) or k.split(".")[-1]: v
        for k, v in data_item.items()
        if "action" in k
    }

    assert isinstance(action_info, dict)
    
    assert len(action_info) > 0

    return observation_info, action_info, data_item["task"]

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


# -------------------------
# ISOLATED BEAM FUNCTION
# -------------------------
def generate_episode_beam_like(raw_dir: Path, episode_index: int):
    episode = []
    dataset = LeRobotDataset("", raw_dir, episodes=[episode_index])

    meta = dataset.meta.episodes[episode_index]
    expected_length = meta["length"]

    print(f"\n=== Episode {episode_index} ===")
    print(f"Expected length: {expected_length}")

    for data_item in dataset:
        obs, act, lang = parse_step(data_item)
        frame_index = data_item["frame_index"].item()

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

        # time.sleep(0.001)

        # Query transform
        resp = requests.get("http://localhost:8000/transform")
        # print(resp.status_code)
        # print(resp.json())

        transform = resp.json()  # NOTE: currently unused (side-effect only)

        # Gripper state (side-effect only)
        grip = gripper_state(gripper_act)
        
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



# -------------------------
# MAIN DEBUG DRIVER
# -------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    RAW_DIR = Path("/home/aryan/any4lerobot/apple")

    meta = LeRobotDatasetMetadata("deepansh-methdai/apple_box", RAW_DIR)
    print(f"Total episodes: {len(meta.episodes)}")

    # 🔴 TEST A FEW EPISODES EXPLICITLY
    num_episodes = len(meta.episodes)

    for ep_idx in range(min(3, num_episodes)):
        ep_id, ep_data = generate_episode_beam_like(RAW_DIR, ep_idx)
        steps = ep_data["steps"]
        # print(steps[0].keys())
        # print("Action type:", type(steps[0]["action"]))
        # print("Action keys:", steps[0]["action"].keys())
        # print("Action value shape:", next(iter(steps[0]["action"].values())).shape)
        # print(next(iter(steps[0]["action"].values())))
        # print((steps[0]["language_instruction"]))

