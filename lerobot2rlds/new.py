
#!/usr/bin/env python3

from pathlib import Path
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames_torchcodec

RAW_DIR = Path("/home/aryan/any4lerobot/apple")
EPISODE_INDEX = 0
NUM_STEPS = 5


def main():
    print("=== Loading dataset ===")
    ds = LeRobotDataset("", RAW_DIR, episodes=[EPISODE_INDEX])

    print("\n=== Inspecting dataset timestamps ===")
    timestamps = []

    for i in range(NUM_STEPS):
        item = ds[i]
        ts = item["timestamp"].item()
        timestamps.append(ts)
        print(f"step {i}: timestamp = {ts}")

    timestamps = np.array(timestamps)
    print("\nDataset timestamp stats:")
    print("  min:", timestamps.min())
    print("  max:", timestamps.max())
    print("  diffs:", np.diff(timestamps))

    # ------------------------------------------------------------
    # Now inspect VIDEO timestamps directly
    # ------------------------------------------------------------
    print("\n=== Inspecting video timestamps ===")

    # Pick one video path explicitly
    video_path = next(
        RAW_DIR.glob("videos/observation.images.camera/**/file-*.mp4")
    )

    print("Video file:", video_path)

    # Query a single timestamp (middle one)
    query_ts = np.array([timestamps[0]], dtype=np.float32)

    try:
        frames, loaded_ts = decode_video_frames_torchcodec(
            video_path,
            query_ts,
            tolerance_s=10.0,  # very large on purpose
        )

        print("\nVideo decode result:")
        print("  queried timestamp:", query_ts)
        print("  loaded timestamp :", loaded_ts.numpy())

        diff = loaded_ts.numpy() - query_ts
        print("  difference       :", diff)

    except AssertionError as e:
        print("\n❌ Video decoder assertion triggered:")
        print(e)

    # ------------------------------------------------------------
    # Heuristic unit detection
    # ------------------------------------------------------------
    print("\n=== Heuristic unit analysis ===")

    avg_ts = timestamps.mean()

    if avg_ts > 1e6:
        print("Likely unit: NANOSECONDS → divide by 1e9")
    elif avg_ts > 1e3:
        print("Likely unit: MILLISECONDS → divide by 1e3")
    elif avg_ts > 100:
        print("Possibly FRAMES or SECONDS (check FPS)")
    else:
        print("Likely already SECONDS")

    print("\nDone.")


if __name__ == "__main__":
    main()
