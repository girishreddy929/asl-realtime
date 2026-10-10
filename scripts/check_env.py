"""Environment preflight for asl-realtime (read-only; run from the project root).

Checks: Python/venv, tools on PATH, libraries, MediaPipe Tasks HolisticLandmarker (model file,
hash, functional smoke test), ONNX Runtime (CPU), Torch, PySpark (deferred if no Java), webcam.
Settings come from configs/runtime.yaml. Exit code 1 if any check FAILs.

Usage:  python scripts\\check_env.py [--skip-webcam] [--skip-spark] [--skip-tracker]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as md
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from asl.config import find_project_root, load_config, resolve_path

# Diagnostic-only constants (one-off script, so they may live here).
VERSION_ONLY = {"kaggle": "kaggle"}  # kaggle 2.x import may touch auth, so only read the version
TRACKER_OPTION_KEYS = (
    "min_face_detection_confidence", "min_face_suppression_threshold",
    "min_face_landmarks_confidence", "min_pose_detection_confidence",
    "min_pose_suppression_threshold", "min_pose_landmarks_confidence",
    "min_hand_landmarks_confidence", "output_face_blendshapes", "output_segmentation_mask",
)
RESULT_ATTRS = (
    "face_landmarks", "pose_landmarks", "left_hand_landmarks", "right_hand_landmarks",
    "pose_world_landmarks", "left_hand_world_landmarks", "right_hand_world_landmarks",
)
SMOKE_FRAMES = 6
WEBCAM_WARMUP_READS = 3
LOW_BRIGHTNESS = 40.0


class Report:
    """Prints rows and counts FAIL / WARN / DEFER for the final summary."""

    def __init__(self) -> None:
        self.counts = {"FAIL": 0, "WARN": 0, "DEFER": 0}
        self.notes: list[str] = []

    def section(self, title: str) -> None:
        print(f"\n=== {title} " + "=" * max(3, 62 - len(title)))

    def row(self, label: str, value: Any, status: str = "") -> None:
        tag = f"[{status}] " if status else ""
        print(f"  {label:<26} {tag}{value}")
        if status in self.counts:
            self.counts[status] += 1
            self.notes.append(f"{status:<5} {label}: {value}")


def guarded(rep: Report, title: str, fn: Callable[[], None]) -> None:
    """Run one section; a crash becomes a FAIL row instead of killing the whole report."""
    rep.section(title)
    try:
        fn()
    except Exception as exc:  # diagnostic must keep going
        rep.row("check crashed", f"{type(exc).__name__}: {exc}", "FAIL")


def check_system(rep: Report) -> None:
    rep.row("python", platform.python_version(),
            "" if sys.version_info[:2] == (3, 13) else "WARN")
    rep.row("executable", sys.executable)
    in_venv = sys.prefix != sys.base_prefix
    rep.row("in venv", in_venv, "OK" if in_venv else "FAIL")
    rep.row("os", f"{platform.system()} {platform.release()} {platform.machine()}")
    rep.row("cpu threads", os.cpu_count())
    rep.row("pip", md.version("pip"))
    rep.row("project root", find_project_root())


def check_path(rep: Report) -> None:
    for tool in ("git", "java", "code", "kaggle"):
        found = shutil.which(tool)
        if found:
            rep.row(tool, found)
        else:
            rep.row(tool, "NOT FOUND" + (" (needed in Phase 4)" if tool == "java" else ""))
    for var in ("JAVA_HOME", "HADOOP_HOME", "VIRTUAL_ENV"):
        rep.row(f"env {var}", os.environ.get(var, "(not set)"))
    kj = Path.home() / ".kaggle" / "kaggle.json"
    rep.row("kaggle.json", kj if kj.is_file() else "not found (auth checked in Phase 4)")


def check_libraries(rep: Report, rt: dict) -> None:
    for mod, dist in rt["env_check"]["libraries"].items():
        t0 = time.perf_counter()
        try:
            importlib.import_module(mod)
            rep.row(mod, f"{md.version(dist)} (import {time.perf_counter() - t0:.1f}s)", "OK")
        except Exception as exc:
            rep.row(mod, f"{type(exc).__name__}: {exc}", "FAIL")
    for mod, dist in VERSION_ONLY.items():
        rep.row(f"{mod} (version only)", md.version(dist))
    cv_dists = sorted({d.metadata["Name"].lower() for d in md.distributions()
                       if d.metadata["Name"] and d.metadata["Name"].lower().startswith("opencv")})
    ok = cv_dists == ["opencv-contrib-python"]
    rep.row("opencv dists", ", ".join(cv_dists) or "none", "OK" if ok else "FAIL")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_mediapipe(rep: Report, rt: dict, run_smoke: bool) -> None:
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision

    tr = rt["tracking"]
    ver = md.version("mediapipe")
    rep.row("mediapipe version", ver, "OK" if ver == tr["mediapipe_version"] else "FAIL")
    rep.row("mp.solutions", "present" if hasattr(mp, "solutions")
            else "absent (expected in 1.1.0; we use Tasks)")
    missing = [n for n in ("HolisticLandmarker", "HolisticLandmarkerOptions", "RunningMode")
               if not hasattr(vision, n)]
    rep.row("Tasks classes", "all present" if not missing else f"missing {missing}",
            "OK" if not missing else "FAIL")
    if missing:
        return
    model = resolve_path(tr["model_path"])
    if not model.is_file():
        rep.row("model file", f"MISSING {model} (run: python scripts\\mp_gate.py to download)", "FAIL")
        return
    rep.row("model file", f"{model} ({model.stat().st_size / 1e6:.1f} MB)", "OK")
    rep.row("model sha256", sha256_of(model))
    if not run_smoke:
        rep.row("tracker smoke", "skipped")
        return

    opts = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=getattr(vision.RunningMode, tr["running_mode"]),
        **{k: tr[k] for k in TRACKER_OPTION_KEYS},
    )
    landmarker = vision.HolisticLandmarker.create_from_options(opts)
    try:
        cam = rt["camera"]
        frame = np.zeros((cam["height"], cam["width"], 3), dtype=np.uint8)  # blank RGB frame
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        times, result = [], None
        for i in range(SMOKE_FRAMES):
            t0 = time.perf_counter()
            result = landmarker.detect_for_video(image, (i + 1) * 33)  # strictly increasing ms
            times.append((time.perf_counter() - t0) * 1000)
        absent = [a for a in RESULT_ATTRS if not hasattr(result, a)]
        rep.row("result attributes", "all present" if not absent else f"missing {absent}",
                "OK" if not absent else "FAIL")
        warm = times[1:]
        rep.row("tracker smoke", f"OK, blank frame, first {times[0]:.0f} ms, "
                f"then mean {sum(warm) / len(warm):.0f} ms (not representative of real scenes)", "OK")
    finally:
        landmarker.close()


def check_onnx(rep: Report, rt: dict) -> None:
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper

    providers = rt["onnx"]["providers"]
    rep.row("version", ort.__version__)
    rep.row("available providers", ort.get_available_providers())
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([helper.make_node("Add", ["x", "x"], ["y"])], "t", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    sess = ort.InferenceSession(model.SerializeToString(), providers=providers)
    out = sess.run(None, {"x": np.ones((1, 4), np.float32)})[0]
    good = bool(np.allclose(out, 2.0)) and sess.get_providers() == providers
    rep.row("tiny model on CPU", f"used {sess.get_providers()}, output ok={good}",
            "OK" if good else "FAIL")


def check_torch(rep: Report) -> None:
    import torch
    rep.row("version", torch.__version__)
    rep.row("cuda available", torch.cuda.is_available())
    a = torch.randn(256, 256)
    mean = float((a @ a).abs().mean())
    rep.row("matmul sanity", f"mean={mean:.3f}", "OK" if mean == mean else "FAIL")  # NaN check
    rep.row("threads", torch.get_num_threads())


def check_spark(rep: Report) -> None:
    if not (shutil.which("java") or os.environ.get("JAVA_HOME")):
        rep.row("pyspark session", "no Java found; deferred to Phase 4 (portable JDK)", "DEFER")
        return
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder.master("local[1]").appName("env-check")
             .config("spark.ui.enabled", "false").getOrCreate())
    try:
        n = spark.range(1000).count()
    finally:
        spark.stop()
    rep.row("pyspark session", f"count={n}", "OK" if n == 1000 else "FAIL")


def check_webcam(rep: Report, rt: dict) -> None:
    import cv2

    cam = rt["camera"]
    wanted = int(rt["env_check"]["webcam_frames"])
    apis = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    for name, api in apis.items():
        configured = name == cam["backend"]
        label = name.upper() + (" (configured)" if configured else "")
        cap = cv2.VideoCapture(0, api)
        try:
            if not cap.isOpened():
                rep.row(label, "could not open camera", "FAIL" if configured else "WARN")
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
            for _ in range(WEBCAM_WARMUP_READS):  # first reads are often slow or dark
                cap.read()
            good, brightness, shape = 0, [], None
            t0 = time.perf_counter()
            for _ in range(wanted):
                ok, frame = cap.read()
                if ok and frame is not None:
                    good += 1
                    brightness.append(float(frame.mean()))
                    shape = frame.shape
            fps = good / max(time.perf_counter() - t0, 1e-9)
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
            size_ok = shape is not None and (shape[1], shape[0]) == (cam["width"], cam["height"])
            bright = sum(brightness) / len(brightness) if brightness else 0.0
            status = "OK" if (good == wanted and size_ok) else ("FAIL" if configured else "WARN")
            rep.row(label, f"{good}/{wanted} frames, {shape and shape[1]}x{shape and shape[0]}, "
                    f"{fps:.1f} FPS, fourcc {fourcc}, brightness {bright:.0f}/255", status)
            if brightness and bright < LOW_BRIGHTNESS:
                rep.row(f"{label} light", "image is very dark (LOW_LIGHT risk)", "WARN")
        finally:
            cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description="asl-realtime environment preflight")
    ap.add_argument("--skip-webcam", action="store_true")
    ap.add_argument("--skip-spark", action="store_true")
    ap.add_argument("--skip-tracker", action="store_true")
    args = ap.parse_args()

    rt = load_config("runtime")
    ec = rt["env_check"]
    rep = Report()

    guarded(rep, "SYSTEM / PYTHON", lambda: check_system(rep))
    guarded(rep, "SOFTWARE ON PATH", lambda: check_path(rep))
    guarded(rep, "LIBRARIES", lambda: check_libraries(rep, rt))
    smoke = bool(ec["run_tracker_smoke"]) and not args.skip_tracker
    guarded(rep, "MEDIAPIPE TASKS (HolisticLandmarker)", lambda: check_mediapipe(rep, rt, smoke))
    guarded(rep, "ONNX RUNTIME (CPU)", lambda: check_onnx(rep, rt))
    guarded(rep, "TORCH", lambda: check_torch(rep))
    if ec["run_spark_test"] and not args.skip_spark:
        guarded(rep, "PYSPARK", lambda: check_spark(rep))
    if not args.skip_webcam:
        guarded(rep, "WEBCAM", lambda: check_webcam(rep, rt))

    rep.section("SUMMARY")
    c = rep.counts
    print(f"  FAIL={c['FAIL']}  WARN={c['WARN']}  DEFER={c['DEFER']}")
    for note in rep.notes:
        print(f"  {note}")
    print("\nResult:", "NOT READY (fix FAIL items)" if c["FAIL"] else "READY (DEFER items are for later phases)")
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())