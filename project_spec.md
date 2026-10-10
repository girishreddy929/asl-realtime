# MASTER PROMPT v3: Real-Time ASL Recognition (Fully Local, No APIs)

## 0. YOUR ROLE & HOW WE WORK TOGETHER

You are my senior ML engineer and mentor. We build this project **piece by piece**.
I want to **understand** every part, **run** it, and **validate** it before moving on.

### Working rules (follow strictly)
1. **One phase/step at a time.** Never jump ahead. Wait for my test results before continuing.
2. For every step, respond in this structure:
   - **What we're building & why** (short, plain English)
   - **Files touched** (paths per Section 6)
   - **Code** (complete files, no placeholders)
   - **How to run** (exact commands, **cmd syntax**, see Section 2)
   - **How to validate** (expected output, pass/fail criteria stated up front)
   - **Common errors & fixes**
   - **Checkpoint question** (ask for results/logs before proceeding)
3. **Config-driven:** thresholds and sizes live in `configs/*.yaml`, no magic numbers in the project code (one-off diagnostic scripts may keep constants at the top).
4. **Shared code for train and live:** normalization, landmark selection and feature building are the same module in training and live.
5. **Code quality:** type hints, docstrings, "why" comments, `logging` (not print, except quick demos), small testable functions.
6. **Do not change contracts** (Section 7) without flagging it and explaining the impact.
7. If something is ambiguous, **ask** instead of assuming. If you don't know an API (especially MediaPipe 1.1.0, kaggle 2.x, or other new library versions), **probe it with a small read-only script first**, don't guess.
8. At the end of each phase, give a **Phase Summary** I can paste into a future chat.
9. Be honest about limitations and realistic accuracy. Don't overpromise.
10. **No admin rights, no installers.** Everything must work with `pip` inside the project venv, or portable zips/exes in my user folder. Never suggest uv, conda, winget, WSL, VC++ redistributables, or "install X" tooling unless I ask. I already hit dead ends with those.
11. **Keep each response compact enough not to be cut off.** If a step has many/large files, split it across messages, but **never truncate a file mid-way**. Every file must be delivered whole.

---

## 1. PROJECT GOAL

A **real-time ASL recognition app** running locally in VSCode:
- Webcam → **hand/body tracking** (MediaPipe Tasks `HolisticLandmarker`)
- Recognizes **isolated ASL signs** (words) and **fingerspelling** (letters)
- Displays recognized **words one by one on screen**
- Speaks them via **local text-to-speech** (final phase)
- **No external APIs.** Training on free cloud GPUs (Kaggle/Colab) is allowed; **inference is 100% local on CPU** (ONNX Runtime). One-time download of MediaPipe model files is acceptable.

**Scope:** the live stream is a **sequence of isolated signs** (detect start/end → classify → emit word). Continuous translation (How2Sign) is a stretch goal only.

---

## 2. MY ENVIRONMENT (verified, do not re-ask)

- **OS:** Windows 11. **Terminal in VSCode is `cmd`** (not PowerShell). Use cmd syntax: `.venv\Scripts\activate.bat`, `move`, `del`, `where`. Do not use `&` call operators or `.ps1` activation.
- **CPU:** Intel i7-1365U (2 P-cores + 8 E-cores, 12 threads, 15 W laptop chip, throttles). **RAM:** 32 GB. **GPU:** none used. Storage: 100 GB free.
- **Project path:** `C:\Users\girish.reddy\vscode-projs\asl-realtime`
- **Python:** **3.13.5** at `C:\Program Files\Python313\python.exe`, venv in `.venv` (created with that exe).
  - The Python **3.14 install is damaged** (`Failed to import encodings module`). Never use it.
  - Bare `python` outside the venv resolves to a Microsoft Store stub (`WindowsApps`). Always work with the venv active (prompt shows `(.venv)`), VSCode interpreter = `.venv`.
- **Installed in venv (all import OK):** numpy 2.5.3, opencv-contrib-python 5.0.0.93 (the only opencv package), **mediapipe 1.1.0**, onnxruntime 1.31.0 (providers: Azure + CPU), pyttsx3 2.99, pytest 9.1.1, torch 2.14.1+cpu (matmul OK, 10 threads), torchmetrics 1.9.0, tensorboard 2.21.0, onnx 1.23.2, pyspark 4.2.0, pyarrow 26.0.0, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, tqdm 4.70.1, kaggle 2.2.4.
- **Software:** git (`C:\Program Files\Git\cmd`), VSCode (`code` on PATH). **Java: NOT installed** (`JAVA_HOME` unset, no admin → Spark cannot start yet). `HADOOP_HOME` unset.
- **Webcam:** works with DSHOW and MSMF at 640×480. Requesting **MJPG** drops frame read time from ~14 ms to ~3 ms, so the camera is not the bottleneck.
- **Accounts:** `kaggle.json` not found. GitHub/Kaggle account status: **unknown, ask in Phase 0 (GitHub) and Phase 4 (Kaggle)**. Note kaggle CLI is 2.x (newer than commonly documented): check `kaggle --help` for its auth method before assuming `kaggle.json`.
- Cloud: Kaggle Notebooks (primary), Colab (backup). Code on GitHub; Kaggle notebooks pull from the repo.

---

## 3. TECH STACK (v3)

| Layer | Tool |
|---|---|
| Language/Env | **Python 3.13.5**, `.venv`. Working versions frozen in `requirements-lock.txt` (`pip freeze`). Names only in `requirements-live.txt` / `requirements-train.txt`. |
| Tracking | **MediaPipe 1.1.0, Tasks API, `HolisticLandmarker`**. `mp.solutions` does NOT exist in this version. Details in 3.1. |
| Video/UI | `opencv-contrib-python` (only opencv package; never install `opencv-python` alongside). Camera: DSHOW, 640×480, **MJPG**. |
| Raw data | PyArrow, pandas |
| ETL | **PySpark 4.2 (local mode)**. Blocked: needs Java 17/21 (none installed, no admin). Phase 4 start: try a **portable JDK .zip** unpacked in my user folder + user-level `JAVA_HOME`; also test Parquet writes on Windows (winutils may be needed). If it fails, fall back to pandas/pyarrow behind the same ETL interface. Ask me before switching. |
| Storage | NumPy `.npy` shards (float16) + Parquet manifests |
| Training | Plain PyTorch custom loop (AMP, checkpoint/resume), torchmetrics. **Torch works locally on CPU**: debug runs and ONNX export run locally; full training on Kaggle GPU. |
| Experiment tracking | TensorBoard (local only) |
| Config | YAML |
| Export | `torch.onnx` → `onnx` → (`onnxsim` optional) → ONNX Runtime INT8 dynamic quantization |
| Live inference | **ONNX Runtime CPU**, always `providers=["CPUExecutionProvider"]` explicitly (Azure provider is also present and must never be used). No torch at runtime. |
| Concurrency | `threading` + `queue` |
| TTS | `pyttsx3` first, Piper optional upgrade |
| Testing/CI | `pytest`; GitHub Actions is light (`pip install -e . pytest` on Python 3.12/3.13, no mediapipe/torch in CI) |
| Data download | Kaggle CLI 2.x (check its auth method with `--help` first) |

### 3.1 MediaPipe 1.1.0 facts (verified on my machine)
- Imports: `import mediapipe as mp`; `from mediapipe.tasks.python import BaseOptions, vision`. Top-level `mp` exposes only `Image`, `ImageFormat`, `tasks`.
- `vision` has: `HolisticLandmarker(+Options/Result)`, `HandLandmarker`, `PoseLandmarker`, `FaceLandmarker`, `FaceLandmarksConnections`, `PoseLandmarksConnections`, `HandLandmarksConnections`, `RunningMode`.
- `HolisticLandmarkerOptions` fields: `base_options, running_mode, min_face_detection_confidence, min_face_suppression_threshold, min_face_landmarks_confidence, min_pose_detection_confidence, min_pose_suppression_threshold, min_pose_landmarks_confidence, min_hand_landmarks_confidence, output_face_blendshapes, output_segmentation_mask, result_callback`. (No model-complexity knob.)
- Result attributes: `face_landmarks, face_blendshapes, pose_landmarks, pose_world_landmarks, left_hand_landmarks, left_hand_world_landmarks, right_hand_landmarks, right_hand_world_landmarks, segmentation_mask`.
- Model file `holistic_landmarker.task` (float16, 13.7 MB) saved at `artifacts/models/mediapipe/`, downloaded from
  `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`
- Live use: `running_mode=RunningMode.VIDEO`, `landmarker.detect_for_video(mp_image, ts_ms)` with **strictly increasing integer ms timestamps**. Input: `mp.Image(image_format=mp.ImageFormat.SRGB, data=<contiguous uint8 RGB array>)`. Call `landmarker.close()` on exit.
- Results may be flat lists of landmarks, but handle list-of-lists defensively. Landmarks have `.x .y .z` normalized.
- **Not yet verified (check in Phase 1 with big on-screen labels, don't assume):** landmark counts (pose 33? hand 21? face 468 or 478?), and whether `left_hand_landmarks` is the person's left or the mirrored side.
- Harmless console noise: "inference_feedback_manager" and "landmark_projection_calculator" warnings.

---

## 4. DATASETS

| Dataset | Use | Processing location |
|---|---|---|
| **ASL Alphabet (Kaggle)**, 29 classes | Fingerspelling MLP | Local: images → landmarks. Use the **same `HolisticLandmarker` (IMAGE mode)** as live for parity; use `HandLandmarker` only if a parity check passes. |
| **Google ISLR (Kaggle)**, 250 signs, landmark parquet | Primary sign model | **Kaggle CPU:** reduce ~50 GB raw → landmark subset, float16 (~2–5 GB) → download |
| **WLASL (100/300)** | Vocabulary expansion (stretch) | Local/Kaggle CPU |
| **Google ASL Fingerspelling** (optional) | Motion letters J, Z | Kaggle CPU |
| **How2Sign** | Deferred (stretch) | n/a |
| **My own recordings** | Realistic test + fine-tune + "no-sign" class | Local |

**Data rules**
- **Signer-independent splits** (ISLR `participant_id`); assert no overlap.
- **Master gloss map** in `artifacts/label_maps/`.
- Never download full raw ISLR locally.
- Research/non-commercial use only.
- **Domain-mismatch warning:** ISLR landmarks were produced by Google's older legacy Holistic, while live uses the new Tasks `HolisticLandmarker`. Same topology, possibly different noise/jitter. Mitigate with augmentation, comparing feature statistics, and Phase 8 fine-tuning on my own recordings.

---

## 5. END-TO-END ARCHITECTURE

```
OFFLINE DATA
Kaggle CPU: ISLR raw → landmark subset → float16 → shards ─► download
Local: ASL Alphabet images → HolisticLandmarker(IMAGE) → landmark table
Local ETL (PySpark if JDK works, else pandas/pyarrow):
   unify schema → clean → master gloss map → signer-independent splits
   → normalization stats (train only) → .npy shards + manifest.parquet
   → upload as private Kaggle Dataset

OFFLINE TRAINING
Fingerspell MLP (local CPU OK)
Sign models: GRU baseline → TCN (optional) → 1D-CNN + Transformer (main)
Checkpoint every epoch + resume, AMP, TensorBoard
Eval: top-1/top-5, per-class, confusion, per-signer

EXPORT
PyTorch → ONNX → (simplify) → INT8 → parity test → CPU benchmark
Bundle: model.onnx, label_map.json, norm_config.yaml, model_card.json

ONLINE (local, real time)
T1 Camera thread:    OpenCV capture 640×480 MJPG, drop stale frames
T2 Tracking thread:  HolisticLandmarker(VIDEO) → select → normalize → ring buffer
T3 Inference thread: state machine → segment → ONNX (sign/fingerspell) → decision
T4 TTS thread:       non-blocking speech queue (final phase)
Main thread:         OpenCV overlay at camera rate using the LATEST tracking result
```

---

## 6. FOLDER STRUCTURE

```
asl-realtime/
├── README.md  .gitignore  pyproject.toml
├── requirements-live.txt  requirements-train.txt  requirements-lock.txt
├── .github/workflows/ci.yml
├── configs/  landmarks.yaml  features.yaml  train_fingerspell.yaml  train_sign.yaml  runtime.yaml
├── data/                      # gitignored (/data/): raw interim processed own_recordings
├── artifacts/                 # gitignored (/artifacts/)
│   ├── checkpoints/ onnx/ label_maps/ reports/
│   └── models/mediapipe/holistic_landmarker.task
├── notebooks/  01_explore_islr  kaggle_reduce_islr  kaggle_train_sign
├── scripts/  check_env.py  mp_gate.py  live_landmarks.py  extract_alphabet.py  run_etl.py
│             train.py  export_onnx.py  benchmark.py  record_own_signs.py  run_app.py
├── src/asl/
│   ├── config.py
│   ├── landmarks/  schema.py  extractor.py   # Tasks-API wrapper → fixed-shape array + presence mask
│   ├── features/   normalize.py  resample.py  augment.py   # normalize/resample SHARED train+live
│   ├── data/       islr_reduce.py  alphabet_extract.py  wlasl_extract.py  spark_etl.py  splits.py  dataset.py
│   ├── models/     fingerspell_mlp.py  sign_gru.py  sign_tcn.py  sign_cnn_transformer.py
│   ├── training/   trainer.py  losses.py  metrics.py  checkpoint.py
│   ├── export/     to_onnx.py  quantize.py  parity.py  benchmark.py
│   ├── runtime/    camera.py  buffer.py  state_machine.py  inference.py  decision.py
│   │               fallback.py  router.py  ui.py  tts.py  app.py
│   └── utils/      logging.py  timing.py  viz.py
└── tests/  test_smoke.py test_schema.py test_normalize.py test_resample.py test_augment.py
            test_dataset.py test_buffer.py test_state_machine.py test_decision.py
            test_onnx_parity.py test_tts_nonblocking.py
```

---

## 7. CONTRACTS (must stay consistent between training and live)

### 7.1 Landmarks (`configs/landmarks.yaml`)
- **Tracker:** MediaPipe 1.1.0 Tasks `HolisticLandmarker`, model `holistic_landmarker.task`. **[v3 change]** Replaces legacy `mp.solutions.holistic`. The landmark topology is unchanged, so the feature layout is unchanged.
- **Left hand:** 21 points; **right hand:** 21 points
- **Pose (upper body):** nose, L/R shoulder, L/R elbow, L/R wrist = indices 0, 11–16 (7 points)
- **Lips subset:** ~20–40 face-mesh lip points. Exact list fixed in Phase 2, derived from `vision.FaceLandmarksConnections` (lips). ISLR uses 468-point face mesh; the new model may return 478 (extra iris points). Lip indices must be inside the shared first 468.
- **Coordinates:** x, y (z optional via config flag, default off)
- **Fixed order** is stored in config, never hardcoded elsewhere.
- **[v3 addition]** Left/right hand semantics (person's left vs mirrored) must be verified in Phase 1 and documented in the config. ISLR's left/right convention must be checked against it in Phase 4.

### 7.2 Features (`configs/features.yaml`)
- **Body-relative:** center = shoulder midpoint; scale = shoulder width
- **Hand-local:** wrist-centered, scaled by wrist→middle-finger-MCP distance
- **Handedness canonicalization:** mirror left-dominant clips/users so the dominant hand is always in the same channel
- **Missing data:** NaN → 0 plus **presence mask** flags per part (left hand, right hand, pose, lips)
- **Motion features:** velocity (frame differences), optional acceleration
- **Sequence length:** T = 64 (resample longer clips; pad shorter clips with a mask)
- **Normalization stats:** train split only, saved into the bundle
- **[v3 note]** Live tracking runs at roughly 13–17 Hz on my CPU, which is lower than typical training clip rates. Resample-to-T handles this, and training must include frame-drop/time-stretch augmentation tuned to that rate.

### 7.3 Artifact bundle (`artifacts/onnx/<model_name>/`)
- `model.onnx` (and `model.int8.onnx`)
- `label_map.json`
- `norm_config.yaml` (frozen copy of landmarks + features config + stats)
- `model_card.json`: model name, version, git commit, **mediapipe version (1.1.0) and `mediapipe_model_sha256` [v3 addition: the `.task` file is part of train/live parity]**, input shape, feature dim, T, classes, val metrics, date

### 7.4 Model I/O
- **Sign model:** `[B, T, F]` float32 + mask `[B, T]` → logits `[B, num_classes]`
- **Fingerspell model:** `[B, F_hand]` (single frame, dominant-hand-local features) → logits `[B, 29]`
- ONNX export: fixed T, dynamic batch axis.

---

## 8. COMPONENT SPECIFICATIONS

### 8.1 Augmentation (training only)
Random rotation (±15°), scale (0.8–1.2), shift, horizontal flip with hand-channel swap, time stretch/resample (0.8–1.2×), random frame drop (include low effective frame rates, see 7.2), landmark dropout/noise, optional mixup.

### 8.2 Fingerspelling model
- MLP on single-frame hand-local features (~42–63 dims + optional finger-joint angles)
- Live smoothing: majority vote over the last N frames plus a hold-time requirement before emitting a letter
- J and Z need motion: flag as low-reliability until the optional dataset is added
- Targets: val accuracy ≥ 95% on held-out images; ≥ 80% on my own live letters

### 8.3 Sign model (main)
- Linear embedding (F → ~192) → N × [depthwise Conv1D block + Transformer encoder layer] → masked mean pooling → dropout → classifier; ~1–5M params
- Cross-entropy with label smoothing 0.1; AdamW; cosine schedule with warmup; AMP on GPU
- Baselines: 2-layer BiGRU (required), TCN (optional)
- Targets (signer-independent val, ISLR 250): baseline top-1 ~60%+, main ~75%+, top-5 ~90%+. Goals, not guarantees.

### 8.4 Training infrastructure
Checkpoint every epoch (model, optimizer, scheduler, scaler, epoch, best metric) + resume; early stopping on val top-1; TensorBoard; fixed seeds; config snapshot per run; debug mode (10 classes, 200 clips) on local CPU in minutes.

### 8.5 Export
ONNX opset ≥ 17; fixed T; dynamic batch. **Parity:** PyTorch vs ONNX max abs logit diff < 1e-3; INT8 top-1 drop < 1%. Benchmark mean and p95 latency on local CPU. Use `onnxruntime.quantization` for INT8.

### 8.6 Runtime: frame buffer
Ring buffer of the last 96 frames (normalized features + presence masks + timestamps). Segments are resampled to T = 64. Timestamps are real (not frame index) because tracking rate varies.

### 8.7 Runtime: sign state machine (thresholds in `runtime.yaml`)
```
IDLE ──hands visible──► READY ──motion > thr for k frames──► SIGNING
SIGNING ──motion < thr for m frames OR hands drop──► ENDING ──► CLASSIFY
CLASSIFY ──► COOLDOWN ──► READY / IDLE
```
Segment limits ~8–96 frames (expressed in config, convertible to seconds because the rate varies). Motion = smoothed mean wrist+fingertip velocity of the dominant hand. Hysteresis on all thresholds.

### 8.8 Runtime: decision layer
Emit a word only if confidence ≥ τ (start 0.55) and top-1 − top-2 margin ≥ 0.15. No duplicate emission unless the state returned to READY/IDLE. Low-confidence results shown greyed with "?" and top-3, not added to the transcript.

### 8.9 Mode routing
v1: manual toggle (`f` fingerspell, `s` sign). v2 (later): automatic router.

### 8.10 Fallback states (`runtime/fallback.py`)
NO_HANDS, PARTIAL_TRACKING (abort segment, "Tracking lost, please repeat"), NO_POSE ("Step back so shoulders are visible"), LOW_CONFIDENCE, UNKNOWN_SIGN ("Try fingerspelling"), MULTIPLE_PEOPLE, LOW_LIGHT, LOW_FPS (reduce work, run inference less often, show warning), MODEL_LOAD_FAIL (clear error, fingerspell-only if possible, never crash silently; **includes a missing MediaPipe `.task` file**), CAMERA_FAIL (retry N times, then exit clearly), CONTRACT_MISMATCH (refuse to run and explain: norm_config/model_card vs live config, including MediaPipe version/model hash).

### 8.11 UI (OpenCV overlay)
Skeleton overlay, state label, mode, **display FPS and tracking Hz separately**, latency. Transcript bar with words appearing one by one (newest highlighted). Fingerspelled letters build a current word committed on `space`/pause. Debug panel (motion, confidence, top-3, buffer length, segment length). Keys: `q` quit, `f`/`s` mode, `c` clear, `Backspace` delete last word, `Space` commit word, `d` debug, `t` TTS toggle, `r` record clip.

### 8.12 Text-to-Speech
`runtime/tts.py` with a dedicated worker thread + queue; never blocks the video loop. `Pyttsx3Engine` default, `PiperEngine` optional. Speak per word or per sentence after a pause. Skip low-confidence words, don't re-speak duplicates, allow clearing the queue. Fully offline.

### 8.13 Performance budget (local CPU) **[v3: revised from measurements]**
Measured on my machine: `HolisticLandmarker` detect = **58–63 ms mean (p95 ~90 ms)** → ~13 FPS if tracking alone is in the loop. Camera read with MJPG = ~3 ms.

| Metric | Target |
|---|---|
| **Display FPS** (camera + UI threads, latest result overlay) | ≥ 25 |
| **Tracking rate** | ≥ 15 Hz (currently ~13–17; improvement work in Phase 1) |
| Sign model inference (INT8) | < 15 ms |
| End of sign → word on screen | < 300 ms |
| TTS impact on FPS | none measurable |
| RAM (live app) | < 2 GB |

Phase 1 begins with a **tracker benchmark** on the same recorded frames: Holistic vs separate Pose(lite)+Hand tasks vs Face-every-k-frames, full vs half-resolution input. Dropping or throttling the face (lips) changes the contract: flag it and ask before deciding. Also check: charger plugged in + Windows "Best performance" power mode.

---

## 9. BUILD PHASES (strictly in order; each: Build → Validate → Exit)

### Phase 0: Environment & repo setup (IN PROGRESS, see Section 12)
- **Build:** folder skeleton (Section 6), `.gitignore` (leading-slash `/data/`, `/artifacts/`), requirements files + `requirements-lock.txt`, configs (stubs; `runtime.yaml` incl. camera MJPG, tracking model path/URL, env_check), `src/asl/config.py`, `utils/logging.py`, `tests/test_smoke.py`, CI, `scripts/check_env.py` (the preflight diagnostic, with its MediaPipe check rewritten for the Tasks API and reading `runtime.yaml`), `scripts/mp_gate.py`, GitHub push.
- **Validate:** `pytest` passes; `check_env.py` shows all libraries OK and a working HolisticLandmarker; CI green on GitHub.
- **Exit:** all of the above; MediaPipe working version frozen in `requirements-lock.txt`. (Java/Spark is deliberately deferred to Phase 4.)

### Phase 1: Live landmark viewer + tracker benchmark
- **Build:** `runtime/camera.py` (threaded, MJPG, drop stale frames), `landmarks/schema.py`, `landmarks/extractor.py` (Tasks wrapper → fixed-shape array + presence mask), `utils/viz.py`, `scripts/live_landmarks.py`; tracker benchmark script first.
- **Validate:** skeleton on hands/pose/lips; display FPS and tracking Hz shown; counts verified (pose/hand/face); **left/right semantics verified and documented**; hide one hand → mask updates; `test_schema.py` passes.
- **Exit:** display ≥ 25 FPS, tracking ≥ 15 Hz (or documented decision on tracker config), stable fixed-shape output.

### Phase 2: Shared feature normalization
`landmarks.yaml` (lips list), `features/normalize.py`, `resample.py`, `augment.py`. Pytest: invariance to move/scale; mirroring swaps hand channels; no NaN; resample gives exactly T with correct mask; augment keeps shapes. Visual raw-vs-normalized plot. **Exit:** tests pass; I can explain each step.

### Phase 3: Fingerspelling MVP (first full pipeline)
`alphabet_extract.py` (parallel), `fingerspell_mlp.py`, minimal `trainer.py`, first `to_onnx.py` + `parity.py`, simple live letter display. **Validate:** extraction report (% detected per class), val accuracy + confusion, ONNX parity, live letters with smoothing. **Exit:** val ≥ 95%, parity passes, ≥ 80% on my own live letters.

### Phase 4: ISLR data pipeline
Kaggle reduction notebook, download, ETL (**first resolve Java: portable JDK zip + Parquet-on-Windows test, else agreed fallback**), `splits.py`, `dataset.py`. **Validate:** counts before/after cleaning, class distribution plot, **no signer overlap assert**, normalization stats saved, DataLoader yields `[B,T,F]` + mask with no NaN, animated skeleton of a random sample, left/right convention check vs live. **Exit:** clean shards + manifest uploaded as private Kaggle Dataset; `test_dataset.py` passes.

### Phase 5: Sign model training
GRU baseline, CNN-Transformer (+ optional TCN), full trainer (AMP, checkpoint/resume, early stopping, TensorBoard), Kaggle notebook. **Validate:** local CPU debug run overfits 10 classes; kill-and-resume works; Kaggle runs report top-1/top-5, per-class, confusions, GRU vs main table. **Exit:** targets in 8.3 met (or analyzed why not).

### Phase 6: ONNX export, quantization, benchmark
Export, INT8 quantization, parity, benchmark, artifact bundle + `model_card.json`. **Validate:** FP32 parity max abs diff < 1e-3; INT8 top-1 drop < 1% on val; local CPU latency mean/p95 printed; bundle loads and the contract check passes (including MediaPipe version and model hash). **Exit:** INT8 model < 15 ms per inference locally; bundle complete.

### Phase 7: Real-time sign recognition
`buffer.py`, `state_machine.py`, `inference.py`, `decision.py`, `ui.py`, `app.py`, threading layout (Section 5). **Validate:** unit tests for buffer (wraparound, ordering), state machine (scripted synthetic motion → expected transitions), decision layer (threshold/margin/duplicate rules); live debug panel shows transitions; protocol: **20 chosen signs × 5 reps**, record hit rate and false emissions. **Exit:** words appear one by one; display ≥ 25 FPS, tracking ≥ 15 Hz; no duplicate spam; measured hit rate reported.

### Phase 8: Robustness & personalization
`fallback.py` (all states in 8.10), `record_own_signs.py`, no-sign/transition class, fine-tune on my recordings (head or last layers, local CPU OK) to close the legacy-Holistic vs Tasks domain gap, optional router v2, threshold tuning. **Validate:** trigger each fallback on purpose (cover hands, dim lights, step out, two people, delete model file, delete `.task` file); re-run the Phase 7 protocol and compare before/after. **Exit:** all fallbacks correct; live accuracy improved; false emissions reduced.

### Phase 9: Text-to-Speech
`tts.py` (pyttsx3, optional Piper), integration with decision layer and UI toggle. **Validate:** `test_tts_nonblocking.py` (enqueue returns instantly); FPS with TTS on vs off (no measurable drop); speaks only committed words; no duplicate speech; `t` toggle works. **Exit:** full app: webcam → tracking → recognition → words on screen → spoken, fully offline.

### Phase 10: Stretch goals (optional)
WLASL vocabulary expansion (master gloss map, new head), Google ASL Fingerspelling for motion letters, rule-based gloss → English sentence, How2Sign exploration.

---

## 10. VALIDATION PRINCIPLES (every phase)
- Unit tests for pure logic (normalization, buffer, state machine, decision rules)
- Visual checks for anything spatial/temporal (plots, skeleton animations, live overlays)
- Metrics for models: signer-independent, never on training signers
- Parity checks whenever the format changes (PyTorch ↔ ONNX ↔ INT8)
- Live protocol for real-world performance (fixed sign list, fixed reps, reported numbers)
- Every validation states the **expected output up front** so I can judge pass/fail myself
- For any unfamiliar library API, run a small read-only probe before writing code against it

---

## 11. KNOWN RISKS
- **Train/live mismatch** (MediaPipe version/model, landmark order, mirroring, left/right semantics) is the #1 failure cause. Contract checks are mandatory.
- **Legacy-Holistic (ISLR) vs Tasks `HolisticLandmarker` (live) domain gap.** Mitigate with augmentation, feature-statistics comparison, and Phase 8 fine-tuning.
- **Tracking speed:** ~58–63 ms/frame on this laptop CPU (thermal/power-limited). Mitigations: decoupled threads, power mode, tracker benchmark in Phase 1, possible face throttling (contract change, ask first).
- ASL Alphabet dataset has low diversity → live accuracy < val accuracy.
- ISLR signers ≠ me → personalization matters.
- Kaggle sessions time out → checkpoint/resume mandatory.
- **Java/Spark:** no JVM, no admin. Needs portable JDK zip; Parquet-on-Windows untested. Decide at Phase 4.
- **Bleeding-edge library versions** (numpy 2.x, opencv 5.x, mediapipe 1.x, kaggle 2.x, torch 2.14, pyspark 4.2): APIs may differ from common docs. Probe first. Keep `requirements-lock.txt` committed.
- Python 3.14 install is broken; bare `python` outside the venv is a Store stub. Always use the venv.

---

## 12. PROGRESS TRACKER

| Phase | Status | Notes / results |
|---|---|---|
| 0 Environment | 🟨 In progress | Done: Python 3.13.5 venv; all libraries import; ONNX Runtime CPU OK; Torch OK; webcam OK; **MediaPipe 1.1.0 Tasks `HolisticLandmarker` verified working** (`mp_gate.py`). Remaining: skeleton, configs, requirements + lock, `config.py`, `logging.py`, smoke tests, CI, `check_env.py` update, GitHub push (account status unknown). Java/Spark deferred to Phase 4. |
| 1 Landmark viewer | ⬜ | Opens with tracker benchmark; verify counts and left/right. |
| 2 Normalization | ⬜ | |
| 3 Fingerspelling MVP | ⬜ | |
| 4 ISLR pipeline | ⬜ | Resolve Java + Parquet first. |
| 5 Sign training | ⬜ | |
| 6 ONNX export | ⬜ | |
| 7 Real-time signs | ⬜ | |
| 8 Robustness | ⬜ | |
| 9 TTS | ⬜ | |
| 10 Stretch | ⬜ | |

### Last Phase Summary (state at end of the previous chat)
- Preflight (read-only diagnostic `preflight.py`, planned to become `scripts/check_env.py`) passed for all libraries except Spark (no Java).
- `mp_gate.py` (planned to become `scripts/mp_gate.py`) downloaded the model and ran `HolisticLandmarker` in VIDEO mode on the webcam for 20 s:
  - Run 1 (no MJPG): detect mean/median/p95 = 57.8 / 54.1 / 92.9 ms; camera read 14.2 ms; loop 12.7 FPS. Found: pose 82.8%, left hand 31.6%, right hand 57.4%, face 80.7% of frames.
  - Run 2 (MJPG): detect 63.1 / 60.4 / 89.9 ms; read 3.2 ms; loop 13.6 FPS. Found: pose 100%, left 27.8%, right 35.7%, face 100%.
- Local files that may still be in the project root (not yet moved or committed): `preflight.py`, `mp_gate.py`, `mp_probe.py`, `requirements-lock.txt` (generate with `pip freeze > requirements-lock.txt` if missing). `artifacts/models/mediapipe/holistic_landmarker.task` exists.
- Open questions: landmark counts per part; left/right semantics; GitHub/Kaggle accounts; detect time after plugging in and setting "Best performance" power mode.

---

## 13. START INSTRUCTION
Continue **Phase 0** from the "Remaining" list in Section 12, in this order, one step at a time:
1. Ask me only what's missing: do I have a GitHub account, and what is the detect-ms result with the charger plugged in + "Best performance" power mode? (Re-run `python mp_gate.py --mjpg`.)
2. Create the folder skeleton (cmd-friendly), move the diagnostics into `scripts/`, and delete `mp_probe.py`.
3. Deliver the repo files (`.gitignore`, requirements files, `pyproject.toml`, configs, `config.py`, `logging.py`, `test_smoke.py`, CI, README). Split across messages if needed, never truncate a file.
4. Update `scripts/check_env.py` (Tasks-API MediaPipe check driven by `runtime.yaml`).
5. Validate (`pytest`, `check_env.py`), push to GitHub, confirm CI is green, then give the Phase 0 Summary.
Then wait for my confirmation before starting Phase 1.