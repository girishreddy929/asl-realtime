"""Gate test: MediaPipe 1.1.0 HolisticLandmarker on this machine.

Run from the project root with the venv active:
    python mp_gate.py
    python mp_gate.py --mjpg      # ask the camera for MJPG (often higher FPS)

Does:
  1) downloads holistic_landmarker.task once (internet needed only for this),
  2) runs on the webcam for RUN_SECONDS,
  3) prints timing and landmark counts.
Press q in the window to stop early.
"""
from __future__ import annotations

import dataclasses
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
MODEL_PATH = Path("artifacts/models/mediapipe/holistic_landmarker.task")
RUN_SECONDS = 20
WARMUP_FRAMES = 10          # excluded from timing statistics
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
# result attribute -> BGR draw color
PARTS: dict[str, tuple[int, int, int]] = {
    "pose_landmarks": (0, 255, 0),
    "left_hand_landmarks": (255, 0, 0),
    "right_hand_landmarks": (0, 0, 255),
    "face_landmarks": (200, 200, 200),
}


def ensure_model() -> bool:
    """Download the model bundle if missing. Returns True when it is usable."""
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        print(f"model present: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
        return True
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - report anything
        print(f"DOWNLOAD FAILED: {type(exc).__name__}: {exc}")
        print("Manual fix: open the MediaPipe docs page 'Holistic landmarker', "
              "download holistic_landmarker.task in your browser, and save it as:")
        print(f"  {MODEL_PATH.resolve()}")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()  # remove partial file
        return False
    print(f"downloaded {MODEL_PATH.stat().st_size / 1e6:.1f} MB")
    return True


def make_landmarker():
    """Create the HolisticLandmarker. Returns (landmarker, uses_video_mode)."""
    if dataclasses.is_dataclass(vision.HolisticLandmarkerOptions):
        names = [f.name for f in dataclasses.fields(vision.HolisticLandmarkerOptions)]
        print("HolisticLandmarkerOptions fields:", names)
    mode_cls = getattr(vision, "RunningMode", None)
    kwargs: dict = {"base_options": BaseOptions(model_asset_path=str(MODEL_PATH))}
    if mode_cls is not None:
        kwargs["running_mode"] = mode_cls.VIDEO   # VIDEO mode tracks across frames
    options = vision.HolisticLandmarkerOptions(**kwargs)
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    print("RunningMode available:", mode_cls is not None)
    return landmarker, mode_cls is not None


def flat_points(value) -> list:
    """Normalize a result field to a flat list of landmarks (handles list-of-lists)."""
    if not value:
        return []
    first = value[0]
    return list(first) if isinstance(first, (list, tuple)) else list(value)


def main() -> int:
    use_mjpg = "--mjpg" in sys.argv
    if not ensure_model():
        return 1
    landmarker, video_mode = make_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("CAMERA FAILED to open (index %d)" % CAMERA_INDEX)
        landmarker.close()
        return 1
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    print(f"camera opened, MJPG requested: {use_mjpg}")
    print("Test sequence: both hands up -> only LEFT hand -> only RIGHT hand -> hands down")

    detect_ms: list[float] = []
    read_ms: list[float] = []
    seen = {name: 0 for name in PARTS}   # frames (after warmup) where the part was found
    frames = 0
    last_ts = -1
    printed_attrs = False
    t_start = time.perf_counter()

    try:
        while time.perf_counter() - t_start < RUN_SECONDS:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            r_ms = (time.perf_counter() - t0) * 1000
            if not ok:
                continue

            # MediaPipe expects RGB, contiguous uint8.
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode needs strictly increasing timestamps in ms.
            ts = max(int((time.perf_counter() - t_start) * 1000), last_ts + 1)
            last_ts = ts

            t1 = time.perf_counter()
            if video_mode:
                result = landmarker.detect_for_video(image, ts)
            else:
                result = landmarker.detect(image)
            d_ms = (time.perf_counter() - t1) * 1000
            frames += 1

            if not printed_attrs:
                attrs = [a for a in dir(result) if not a.startswith("_")]
                print("result attributes:", attrs)
                printed_attrs = True

            points = {name: flat_points(getattr(result, name, None)) for name in PARTS}
            if frames > WARMUP_FRAMES:
                detect_ms.append(d_ms)
                read_ms.append(r_ms)
                for name, pts in points.items():
                    seen[name] += 1 if pts else 0

            # Draw landmarks as small dots.
            h, w = frame.shape[:2]
            for name, color in PARTS.items():
                for lm in points[name]:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, color, -1)

            lines = [
                f"detect {d_ms:5.1f} ms | read {r_ms:5.1f} ms",
                f"pose {len(points['pose_landmarks'])}  "
                f"L-hand {len(points['left_hand_landmarks'])}  "
                f"R-hand {len(points['right_hand_landmarks'])}  "
                f"face {len(points['face_landmarks'])}",
            ]
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
            cv2.imshow("mp_gate (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        elapsed = time.perf_counter() - t_start
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print("\n=========== SUMMARY ===========")
    if not detect_ms:
        print("No timed frames collected (camera returned no frames?).")
        return 1
    d = np.array(detect_ms)
    r = np.array(read_ms)
    n = len(d)
    print(f"frames processed        : {frames} in {elapsed:.1f}s -> loop {frames / elapsed:.1f} FPS")
    print(f"detect ms mean/median/p95: {d.mean():.1f} / {np.median(d):.1f} / {np.percentile(d, 95):.1f}")
    print(f"camera read ms mean     : {r.mean():.1f}")
    for name in PARTS:
        print(f"{name:<22}: found in {100 * seen[name] / n:5.1f}% of frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())