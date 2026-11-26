# Implementation Plan: Low-Power Unified ONNX Proctoring POC (Phase 0.5)

## 1. Objective & Architectural Pivot

This plan refactors the existing `POC-ExamProc` repository to eliminate laptop overheating, remove MediaPipe completely, and solve **still-head eye-glance detection** using a single lightweight **`onnxruntime-web`** runtime.

### Key Design Corrections (Why This Stays Cool & Accurate)
1. **Decouple Capture FPS from Inference Rate:**
   - Webcam capture requests low framerate (`ideal: 6, max: 15` fps, avoiding strict `max: 6` constraints that trigger `OverconstrainedError` on standard UVC webcams that only expose 15/30 fps hardware modes).
   - Inference runs sequentially via `setTimeout` (never `requestAnimationFrame`) at **$1.0\text{–}2.0\text{ Hz}$** for face + gaze, and **$0.25\text{ Hz}$** (every $4\text{ s}$, interleaved) for phone/person detection.
   - Running a $448 \times 448$ CNN at $6\text{ Hz}$ will reintroduce thermal throttling; cheating peeks last $3\text{–}10\text{ s}$, so $1\text{–}2\text{ Hz}$ captures them while letting the CPU/GPU sleep $\sim 85\text{–}90\%$ of every second.
2. **Dual-Resolution Crop Strategy (Avoid the Low-Res Upscaling Trap):**
   - Do **not** drop raw webcam capture to $320 \times 240$ while running a $448 \times 448$ gaze model—a $320 \times 240$ frame produces a blurry $\sim 110 \times 110$ face crop that wastes $448 \times 448$ FLOPs on interpolated blur.
   - Instead, capture at **`480×360`** (or **`640×480`**) at low FPS (`ideal: 6–10 fps`). Downscale to $320 \times 240$ for ultra-fast **YuNet** face detection, then crop the **sharp face ROI from the full captured frame** to feed `mobileone_s0_gaze.onnx`.
3. **Start at $448 \times 448$, Degrade to $224 \times 224$ Only on Empirical Trigger:**
   - Start with the upstream pre-trained `mobileone_s0_gaze.onnx` ($448 \times 448$). Read input dimensions dynamically from the loaded ONNX session so swapping to a $224 \times 224$ re-export later requires **zero TypeScript code changes**.
   - Define an explicit Go/No-Go benchmark gate in Section 7 to decide whether $224 \times 224$ is actually needed.
4. **Zero Continuous Canvas Drawing & Zero Always-On Neural Audio:**
   - Render the raw `<video>` stream only during calibration/debug; disable live landmark mesh `<canvas>` redraws during normal benchmark runs (toggleable for visual debugging only).
   - Replace continuous $31\text{ Hz}$ Silero VAD `AudioWorklet` execution with a **4 Hz native Web Audio bandpass energy gate** that wakes `silero_vad_v5.onnx` only for $5\text{ s}$ bursts when speech-band energy spikes above the adaptive ambient floor.

---

## 2. Unified Model & Library Bill of Materials

All models run under **one runtime (`onnxruntime-web`)**, use commercially permissive licenses, and are self-hosted in `public/models/` and `public/wasm/` (no runtime third-party CDN fallback).

| Role | Model / Package | Source & License | Approx. Size | Cadence |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Runtime** | `onnxruntime-web` (WebGPU primary, single-threaded WASM fallback—no COOP/COEP `SharedArrayBuffer`) | Microsoft (MIT) | Runtime WASM | Shared single worker |
| **Face Detection** | `face_detection_yunet_2026may.onnx` *(dynamic `[1,3,H,W]` input shape; preferred over fixed-640 `2023mar`)* | OpenCV Zoo (MIT) | $\sim 230\text{ KB}$ | $1.0\text{–}2.0\text{ Hz}$ |
| **Still-Head Gaze** | `mobileone_s0_gaze.onnx` ($448 \times 448$; fallback target: $224 \times 224$) | `yakhyo/gaze-estimation` (MIT) | $4.74\text{ MB}$ | $1.0\text{–}2.0\text{ Hz}$ (sequential after YuNet) |
| **Phone & Extra Person** | `yolox_nano.onnx` ($416 \times 416$, COCO classes: `cell phone`, `person`, `book`) | Megvii YOLOX (Apache-2.0) | $\sim 3.5\text{ MB}$ | $0.25\text{ Hz}$ (every $4\text{ s}$, interleaved) |
| **Duty-Cycled Speech** | Native `BiquadFilterNode` + `AnalyserNode` $\rightarrow$ `silero_vad_v5.onnx` | Silero Team (MIT) | $\sim 1.8\text{ MB}$ | Gate: $4\text{ Hz}$ DSP; ONNX: only on speech burst |

> **Packages to Remove from `package.json`:** Remove `@mediapipe/tasks-vision` and `@ricky0123/vad-web` (replace with direct `onnxruntime-web` + lightweight audio gate so there is only one WASM engine and zero continuous `AudioWorklet` ONNX loop).

---

## 3. Repository Refactoring Map (`POC-ExamProc`)

### Files to Delete / Deprecate
*   `src/workers/face-mesh.worker.ts` (replaced by unified worker)
*   `src/workers/object-detector.worker.ts` (merged sequentially into unified worker to prevent concurrent GPU/CPU spikes)
*   `src/lib/math/transform-matrix.ts` (no longer computing MediaPipe 4x4 Euler matrices)
*   `src/lib/math/iris-gaze.ts` (replaced by appearance-based `mobileone_s0_gaze.onnx`)
*   `src/lib/vad.ts` (replaced by two-stage `audio-gate.ts`)

### Files to Add / Update
```text
POC-ExamProc/
├── scripts/
│   └── download-models.mjs          # [UPDATE] Fetch YuNet (2026may), mobileone_s0_gaze, yolox_nano, silero_vad_v5 + copy ort WASM to public/wasm/
├── public/
│   ├── models/
│   │   ├── face_detection_yunet_2026may.onnx
│   │   ├── mobileone_s0_gaze.onnx
│   │   ├── yolox_nano.onnx
│   │   └── silero_vad_v5.onnx
│   └── wasm/                        # Pinned onnxruntime-web .wasm & .mjs files
└── src/
    ├── workers/
    │   └── proctor-onnx.worker.ts   # [NEW] Single sequential ONNX worker (INIT, FRAME, AUDIO_BURST messages)
    ├── lib/
    │   ├── camera.ts                # [UPDATE] Low-FPS stream (480x360 or 640x480 @ ideal 6-10 fps) + setTimeout frame pump (no rAF)
    │   ├── models/
    │   │   ├── yunet.ts             # [NEW] BGR [0..255] NCHW preprocess + 12-head stride decode + NMS
    │   │   ├── mobilegaze.ts        # [NEW] Dynamic-shape RGB ImageNet preprocess + 90-bin softmax degree decode
    │   │   ├── yolox.ts             # [NEW] 416x416 letterbox BGR [0..255] preprocess + stride decode + class-filtered NMS
    │   │   └── silero-vad.ts        # [NEW] Short-window PCM inference wrapper for duty-cycled audio bursts
    │   ├── audio-gate.ts            # [NEW] Stage-1 4 Hz bandpass (300-3400 Hz) adaptive RMS gate + 5s PCM ring capture
    │   ├── browser-events.ts        # [NEW] 3 lightweight listeners: TAB_SWITCH/BLUR, PASTE, FULLSCREEN_EXIT
    │   ├── engine/
    │   │   ├── calibrator.ts        # [UPDATE] Simple 2-second center neutral offset (yaw0, pitch0) + optional 4-corner sanity check
    │   │   ├── state-machine.ts     # [UPDATE] Debounced per-signal state machines (IDLE -> ACTIVE -> RELEASED)
    │   │   └── bucket-scorer.ts     # [NEW] Flat 30-second bucket scoring with 1.5x cross-category multiplier
    │   └── performance.ts           # [UPDATE] Per-stage P95 latency (YuNet, Gaze, YOLOX) + adaptive rate step-down (2 -> 1 -> 0.5 Hz)
    ├── components/
    │   ├── VideoFeed.tsx            # [UPDATE] Raw <video> only by default; optional debug overlay toggle (off during thermal test)
    │   ├── CalibrationModal.tsx     # [UPDATE] 2-second "Look at Start Button" neutral calibration
    │   ├── Gauges.tsx               # [UPDATE] Yaw/Pitch degrees (relative to neutral), face count, phone status, audio gate state
    │   ├── LatencyProfiler.tsx      # [UPDATE] Stage breakdown + duty cycle % + 448x448 vs 224x224 gate indicator
    │   └── BucketTimeline.tsx       # [NEW] Top-3 highest-scoring 30-second buckets + captured snapshot thumbnails
    └── routes/
        ├── index.tsx                # [UPDATE] Main diagnostic playground with A/B thermal preset toggles
        └── test-runner.tsx          # [UPDATE] Scripted tests including "Still-Head Side Glance" & "Close-Up Phone"
```

---

## 4. Exact Model Preprocessing & Postprocessing Contracts

Implementing these exact contracts prevents silent accuracy failures in `onnxruntime-web`.

### 4.1 `yunet.ts` (`face_detection_yunet_2026may.onnx`)
*   **Input:** Resize frame to **`320×240`** on a reusable `OffscreenCanvas`. Convert RGBA `ImageData` to **`BGR` `Float32Array` in `NCHW` layout `[1, 3, 240, 320]` with raw values in $[0, 255]$** (do **not** divide by $255$).
*   **Fallback if `2026may` dynamic model is unavailable:** If using fixed `face_detection_yunet_2023mar.onnx`, letterbox to `[1, 3, 640, 640]` and scale bounding boxes back.
*   **Output Decode:** Decode the 12 per-stride heads (`cls_8/16/32`, `obj_8/16/32`, `bbox_8/16/32`, `kps_8/16/32`):
    $$\text{score} = \sqrt{\text{clamp}(\text{cls}, 0, 1) \cdot \text{clamp}(\text{obj}, 0, 1)}$$
    Filter by `score >= 0.6`, apply IoU NMS (`iouThreshold = 0.3`), and scale boxes back to the original capture resolution (`480×360` or `640×480`).
*   **Outputs emitted:** `faceCount`, primary face `bbox` `[x1, y1, x2, y2]` (largest centered face).

### 4.2 `mobilegaze.ts` (`mobileone_s0_gaze.onnx`)
*   **Dynamic Input Size:** Read `targetH` and `targetW` from the ONNX session input metadata (`448×448` initially; automatically adapts to `224×224` if swapped).
*   **Face Crop:** Take primary `bbox` from YuNet on the **original sharp `ImageBitmap`**, expand by $10\%$ margin, clamp to frame boundaries, and resize to `targetW × targetH` via `OffscreenCanvas`.
*   **Preprocess:** Convert RGBA to **`RGB` `Float32Array` in `NCHW` layout `[1, 3, targetH, targetW]`**, scale to $[0, 1]$, and apply ImageNet normalization:
    $$x_c = \frac{p_c / 255.0 - \mu_c}{\sigma_c}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
*   **Output Decode (CRITICAL — Verify Tensor Order):**
    - Per upstream `yakhyo/gaze-estimation/onnx_inference.py`, the model outputs 2 logit tensors of shape `[1, 90]`: **`outputs[0]` is `yaw` (horizontal)** and **`outputs[1]` is `pitch` (vertical)**.
    - Compute numerically stable softmax over the 90 bins ($i \in \{0, 1, \dots, 89\}$), then map to degrees in $[-180^\circ, 180^\circ]$:
      $$\theta_{\text{deg}} = \left(\sum_{i=0}^{89} i \cdot \text{softmax}(\text{logits})_i\right) \times 4 - 180$$
*   **Temporal Smoothing:** Apply a lightweight EMA ($\alpha = 0.45$) on $(\text{yaw}_{\text{deg}}, \text{pitch}_{\text{deg}})$ before passing to the state machine.
*   **Relative Calibration (`calibrator.ts`):** During the 2-second "Look at the center button" prompt, collect median $(\text{yaw}_0, \text{pitch}_0)$. Runtime deviations are simply:
    $$\Delta\text{yaw} = \text{yaw}_{\text{deg}} - \text{yaw}_0, \quad \Delta\text{pitch} = \text{pitch}_{\text{deg}} - \text{pitch}_0$$
    Starting off-screen thresholds *(assumption—tune in benchmark)*: $|\Delta\text{yaw}| > 25^\circ$ (side peek) or $\Delta\text{pitch} < -22^\circ$ (looking down at phone/notes).

### 4.3 `yolox.ts` (`yolox_nano.onnx`)
*   **Preprocess:** Letterbox the source frame into `[1, 3, 416, 416]` **`BGR` `Float32Array` `NCHW` with raw values $[0, 255]$** (pad with `114`).
*   **Postprocess:** Decode grid strides $\{8, 16, 32\}$ from output `[1, 3549, 85]`, compute `score = obj_conf * max_class_prob`, keep **only** COCO classes `67` (`cell phone`, threshold `0.45`), `0` (`person`, threshold `0.50`), and `73` (`book`, threshold `0.50`), and apply per-class NMS (`iou = 0.45`).
*   *Phase 1 Upgrade Note:* Keep the class map configurable so a custom fine-tuned 1-class phone-in-hand `yolox_nano.onnx` can drop in later without refactoring worker code.

### 4.4 `audio-gate.ts` + `silero-vad.ts` (Two-Stage Duty-Cycled Audio)
*   **Stage 1 (Always-On Native DSP — Main Thread, `setInterval` every $250\text{ ms}$):**
    - Audio graph: `MediaStreamAudioSourceNode` $\rightarrow$ `BiquadFilterNode` (`type: 'bandpass'`, `frequency: 1200`, `Q: 0.7`, covering $\sim 300\text{–}3400\text{ Hz}$) $\rightarrow$ `AnalyserNode` (`fftSize: 1024`).
    - Compute speech-band RMS energy $E_{\text{rms}}$ and update a slow ambient noise floor EMA $E_{\text{floor}}$ whenever energy is below threshold.
    - Keep a tiny circular `Float32Array` PCM ring buffer ($16\text{ kHz}$ downsampled, $5\text{ s}$ max) populated via a lightweight low-overhead tap only when armed, or start capturing for $4\text{ s}$ as soon as $E_{\text{rms}} > E_{\text{floor}} \cdot 2.2$ *(assumption)* persists for $\ge 1.0\text{ s}$.
*   **Stage 2 (Duty-Cycled ONNX Verification):**
    - Post the $4\text{ s}$ $16\text{ kHz}$ PCM snippet (`Float32Array`, transferred zero-copy) to `proctor-onnx.worker.ts`.
    - Run `silero_vad_v5.onnx` across $512$-sample windows ($32\text{ ms}$ each) inside the worker, count windows with speech probability $\ge 0.5$, and emit `SPEECH_DETECTED` if verified speech duration $\ge 2.0\text{ s}$. Immediately return to sleep.

---

## 5. Low-Power Sequential Worker Schedule & Memory Safety

### Main-Thread Frame Pump (`camera.ts`)
1. **Never use `requestAnimationFrame`.** Schedule ticks with `setTimeout` after each worker reply arrives (natural backpressure + guaranteed idle sleep).
2. **Strict `inFlight` Guard:** Never send a new frame while the worker is busy.
3. **Zero-Copy Transfer & Cleanup:** Create an `ImageBitmap` from `<video>` via `createImageBitmap(video)`, transfer `[bitmap]` to the worker, and call **`bitmap.close()` inside a `finally` block** in the worker immediately after preprocessing so GPU/RAM never leaks over a 2-hour exam.
4. **Frozen-Frame Check:** Inside the worker (before YuNet), compute an $8 \times 8$ grayscale thumbnail L1 difference against the previous frame ($< 0.1\text{ ms}$). Flag `CAMERA_FROZEN` if diff $< 0.5$ for $\ge 10\text{ s}$.

### Single-Worker Sequential Execution Order
```text
Tick t (default interval: 500 ms [2 Hz] or 1000 ms [1 Hz]):
  1. Frozen-frame 8x8 L1 check (< 0.1 ms)
  2. Run YuNet (320x240) -> faceCount, primaryBbox
  3. If faceCount >= 1:
       Crop primaryBbox from sharp bitmap -> Run mobileone_s0_gaze
  4. If (now - lastYoloxTime >= 4000 ms):
       Run YOLOX-nano (416x416) -> phone / extra person / book
       (Wait until next tick if Gaze latency already exceeded budget)
  5. Close ImageBitmap, post metrics + per-stage timings to main thread.
```

### Execution Provider & Adaptive Throttling (`performance.ts`)
*   **Explicit `INIT` Message:** Initialize all ONNX sessions once via an explicit `INIT` message before starting the frame loop. Try `executionProviders: ['webgpu', 'wasm']` with `ort.env.wasm.numThreads = 1` (single-threaded WASM avoids COOP/COEP issues and keeps CPU thermal spikes low). Log which provider actually bound per model.
*   **Adaptive Rate Step-Down:** Maintain a rolling 20-sample P95 of `t_yunet + t_gaze`:
    - Target cadence starts at **$1.5\text{ Hz}$** ($667\text{ ms}$ interval).
    - If P95 $> 120\text{ ms}$: step down to **$1.0\text{ Hz}$** ($1000\text{ ms}$).
    - If P95 $> 220\text{ ms}$: step down to **$0.5\text{ Hz}$** ($2000\text{ ms}$) and stretch YOLOX to **$6000\text{ ms}$**.
    - If P95 $> 400\text{ ms}$: surface `DEGRADED_COVERAGE` badge in HUD.

---

## 6. Simplified Signal & Flat 30-Second Bucket Scoring

Replace complex multi-level rules with debounced events fed into flat 30-second buckets (matching our simplified Azure-ready design):

### Debounced Signals (`state-machine.ts`)
| Category | Signal | Trigger Rule | Weight |
| :--- | :--- | :--- | :--- |
| **Vision** | `PHONE_DETECTED` | $\ge 2$ of last $3$ YOLOX samples ($4\text{ s}$ cadence) | 5 |
| **Vision** | `SECOND_PERSON` | `faceCount >= 2` (YuNet) for $\ge 2\text{ s}$ OR YOLOX `person >= 2` in $2$ of $3$ samples | 5 |
| **Vision** | `NO_FACE` | `faceCount === 0` in $90\%$ of samples over $4\text{ s}$ | 2 |
| **Vision** | `GAZE_OFF_SCREEN` | $|\Delta\text{yaw}| > 25^\circ$ for $\ge 3.5\text{ s}$ OR $\Delta\text{pitch} < -22^\circ$ for $\ge 3.0\text{ s}$ | 2 |
| **Vision** | `CAMERA_FROZEN` | $8 \times 8$ L1 diff $< 0.5$ for $\ge 10\text{ s}$ | 5 |
| **Audio** | `SPEECH_DETECTED` | Stage-1 energy spike confirmed by Silero VAD ($\ge 2\text{ s}$ speech) | 2 |
| **Browser** | `TAB_SWITCH` / `PASTE` / `FULLSCREEN_EXIT` | Immediate event (`blur` $> 2\text{ s}$, any `paste`, any fullscreen exit) | 3 each |

### Flat 30-Second Bucket Formula (`bucket-scorer.ts`)
Partition session time into non-overlapping 30-second buckets $B_k$. Inside each bucket, deduplicate identical signal types and compute:
$$\text{score}(B_k) = \left(\sum_{e \in B_k} w_e\right) \times m_k$$
where $m_k = 1.5$ if events from **2 or more distinct categories** (`Vision`, `Audio`, `Browser`) appear in $B_k$, otherwise $m_k = 1.0$.
*   Capture **1 JPEG snapshot** ($\sim 480 \times 360$, quality `0.7`) on the first trigger of any bucket with $\text{score}(B_k) \ge 4$.
*   Rank buckets descending and display the **Top 3 buckets** with their snapshots in `BucketTimeline.tsx`.

---

## 7. Verification Plan & The $448 \times 448 \rightarrow 224 \times 224$ Decision Gate

### Step 1: Automated Unit Tests (`bun test` / `vitest`)
1.  **`mobilegaze.test.ts`:** Verify ImageNet NCHW normalization and 90-bin softmax decode on synthetic one-hot logits (bin `45` $\rightarrow 0^\circ$, bin `0` $\rightarrow -180^\circ$, bin `89` $\rightarrow +176^\circ$, confirm `outputs[0]` = yaw, `outputs[1]` = pitch).
2.  **`yunet.test.ts` & `yolox.test.ts`:** Verify coordinate rescaling, letterbox inverse mapping, class allowlist filtering (`67`, `0`, `73`), and IoU NMS.
3.  **`bucket-scorer.test.ts`:** Verify flat 30-second bucketing, same-type deduplication, and $1.5\times$ cross-category multiplier.
4.  **`audio-gate.test.ts`:** Verify adaptive ambient floor ignores steady fan hum while triggering on sudden speech-band bursts.

### Step 2: Scripted Detection Verification (`test-runner.tsx`)
Run these 5 real-world tests on your laptop after the 2-second neutral center calibration:
1.  **Still-Head Side Peek (The Key Test):** Keep your head completely stationary and shift **only your eyes** to the far left/right edge or a phone beside the monitor for $5\text{ s}$. Verify $\Delta\text{yaw}$ shifts cleanly past $25^\circ$ and fires `GAZE_OFF_SCREEN`.
2.  **Still-Head Downward Peek:** Keep your head level and look down at your lap/desk for $4\text{ s}$. Verify $\Delta\text{pitch}$ drops cleanly and fires `GAZE_OFF_SCREEN`.
3.  **Normal Reading (False-Positive Test):** Read across all four corners of the screen for $60\text{ s}$, blink normally, sip water, and leave a fan/AC running. **Target: $0$ Vision/Audio triggers.**
4.  **Phone & Second Person:** Hold a phone in frame for $8\text{ s}$ (2–3 YOLOX ticks) and have a second person enter frame for $3\text{ s}$. Verify `PHONE_DETECTED` and `SECOND_PERSON` fire and capture snapshots.
5.  **Whisper Test:** Stay silent for $15\text{ s}$ (confirm `silero_vad_v5.onnx` invokes $0$ times), then whisper a sentence for $4\text{ s}$ (confirm Stage-1 wakes Stage-2 and fires `SPEECH_DETECTED`).

### Step 3: 10-Minute Thermal & Latency A/B Gate (Decides $448 \times 448$ vs. $224 \times 224$)
Disable the debug overlay canvas and run the unified pipeline for **10 continuous minutes** at **`480×360 @ ideal 6 fps`** (capture) and **`1.5 Hz`** (inference), recording per-stage P95 latency, worker duty cycle $\left(\frac{t_{\text{infer}}}{t_{\text{interval}}}\right)$, and laptop CPU/package watts (`powertop`/`powermetrics`/Task Manager):

| Metric | Keep $448 \times 448$ (`PASS`) | Trigger $224 \times 224$ Fallback (`DEGRADE`) |
| :--- | :--- | :--- |
| **Gaze P95 Latency (WebGPU)** | $\le 80\text{ ms}$ | $> 120\text{ ms}$ |
| **Gaze P95 Latency (CPU WASM)** | $\le 180\text{ ms}$ | $> 220\text{ ms}$ |
| **Worker Active Duty Cycle** | $\le 20\%$ (asleep $\ge 80\%$ of exam) | $> 30\%$ |
| **10-Min Laptop Thermals** | No sustained fan spin-up; cool chassis | Audible fan ramp / thermal throttling |

If the `DEGRADE` column triggers on your target mid-spec laptop (especially under CPU WASM fallback), re-export `mobileone_s0_gaze` at $224 \times 224$ ($\sim 4\times$ fewer FLOPs) using `yakhyo/gaze-estimation`'s export script and drop the new `.onnx` file into `public/models/`—no frontend code changes required.