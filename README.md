# asl-realtime

Real-time American Sign Language recognition that runs **fully locally** on a laptop CPU.

Webcam -> MediaPipe Tasks `HolisticLandmarker` -> shared feature normalization ->
ONNX Runtime (CPU) -> words on screen -> local text-to-speech.

- Recognizes isolated signs (words) and fingerspelling (letters).
- Training happens on free cloud GPUs (Kaggle/Colab); **inference is 100% local**, no APIs.
- Research / non-commercial use only (dataset licenses).

## Status

Phase 0 (environment & repo setup) in progress. See the project spec for the phase plan.

## Setup (Windows, cmd)

```cmd
C:\"Program Files"\Python313\python.exe -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements-train.txt
pip install -e .