"""
Extract MediaPipe hand landmarks from the ISL-HS videos.

Input : data/raw/Videos/Person1 ... Person6, with files named like "a (1).mov"
Output: data/processed/landmarks.csv, one row per video frame where a hand was found

Run from the project folder:
    python ml/scripts/extract_landmarks.py
"""
import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DYNAMIC_LETTERS = {"j", "x", "z"}                      # letters that involve movement
FILE_NAME = re.compile(r"^([a-z]) \((\d+)\)\.mov$", re.IGNORECASE)


def make_landmarker(model_path):
    """Create a MediaPipe hand landmarker that processes video frames in order."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from ISL-HS videos")
    parser.add_argument("--videos", default="data/raw/Videos", help="folder containing Person1 ... Person6")
    parser.add_argument("--model", default="ml/models/hand_landmarker.task", help="MediaPipe model file")
    parser.add_argument("--out", default="data/processed/landmarks.csv", help="output CSV file")
    parser.add_argument("--step", type=int, default=2, help="use every Nth frame (2 = every second frame)")
    args = parser.parse_args()

    videos = sorted(Path(args.videos).glob("Person*/*.mov"))
    if not videos:
        raise SystemExit(f"No videos found in {args.videos}/Person*/ - check the folder path.")
    if not Path(args.model).exists():
        raise SystemExit(f"Model file not found: {args.model}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Column names: who, which letter, which recording, which frame, then 21 points x 3 values
    header = ["person", "letter", "take", "frame", "is_dynamic", "handedness", "hand_score"]
    header += [f"{axis}{i}" for i in range(21) for axis in "xyz"]

    stats = {}                                           # person -> [frames checked, frames with a hand]
    start = time.time()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for n, path in enumerate(videos, start=1):
            match = FILE_NAME.match(path.name)
            if not match:
                print(f"Skipping unexpected file name: {path}")
                continue
            letter, take = match.group(1).lower(), int(match.group(2))
            person = path.parent.name

            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_ms = max(1, int(1000 / fps))

            # A new landmarker per video, so tracking does not carry over between videos
            with make_landmarker(args.model) as landmarker:
                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % args.step == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # OpenCV uses BGR, MediaPipe expects RGB
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = landmarker.detect_for_video(image, frame_idx * frame_ms)

                        s = stats.setdefault(person, [0, 0])
                        s[0] += 1
                        if result.hand_landmarks:
                            s[1] += 1
                            hand = result.handedness[0][0]
                            row = [person, letter, take, frame_idx, int(letter in DYNAMIC_LETTERS),
                                   hand.category_name, round(hand.score, 4)]
                            for point in result.hand_landmarks[0]:
                                row += [round(point.x, 5), round(point.y, 5), round(point.z, 5)]
                            writer.writerow(row)
                    frame_idx += 1
            cap.release()

            if n % 26 == 0 or n == len(videos):
                print(f"Processed {n}/{len(videos)} videos ({time.time() - start:.0f}s)")

    print("\nFrames with a detected hand:")
    total_checked = total_found = 0
    for person, (checked, found) in sorted(stats.items()):
        total_checked += checked
        total_found += found
        print(f"  {person}: {found}/{checked} ({found / checked:.1%})")
    print(f"  Total: {total_found}/{total_checked} ({total_found / total_checked:.1%})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()