# Edge-First Remote Exam Proctoring POC (Phase 0)

A browser-native, client-side proctoring engine built with **React**, **Vite**, **TanStack Router**, and **Google MediaPipe**. 

This Proof of Concept (POC) validates the Phase 0 architecture outlined in [`exam_proctor_plan_v2.md`](file:///home/mohmedh/personal/POC-ExamProc/exam_proctor_plan_v2.md). It runs all computer vision and voice activity inference locally on the candidate's machine, eliminating continuous video stream bandwidth while maintaining high-accuracy signal detection, an adaptive posture tracker, and a decaying risk score.

---

## Table of Contents
- [Key Architectural Highlights](#key-architectural-highlights)
- [How It Works (Under the Hood)](#how-it-works-under-the-hood)
  - [1. Dual Web Worker Concurrency](#1-dual-web-worker-concurrency)
  - [2. Head Pose & Iris Gaze Mathematical Formulation](#2-head-pose--iris-gaze-mathematical-formulation)
  - [3. Dynamic Calibration & Adaptive Posture Drift](#3-dynamic-calibration--adaptive-posture-drift)
  - [4. Three-State Debounce Engine](#4-three-state-debounce-engine)
  - [5. Leaky Bucket Risk Scoring](#5-leaky-bucket-risk-scoring)
  - [6. Performance-Driven Adaptive Throttling](#6-performance-driven-adaptive-throttling)
- [System Requirements & Prerequisites](#system-requirements--prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Diagnostic Playground & UI Walkthrough](#diagnostic-playground--ui-walkthrough)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Project Directory Structure](#project-directory-structure)

---

## Key Architectural Highlights

1. **100% Client-Side Edge Inference**: MediaPipe Tasks Vision (`FaceLandmarker` and `ObjectDetector`) and Silero VAD v5 run directly on the client. No raw video or continuous audio streams leave the device.
2. **Offline-Pinned Assets (No CDN Dependency)**: Model files (`face_landmarker.task`, `efficientdet_lite0.tflite`, `silero_vad_v5.onnx`) and WASM binaries are self-hosted in `public/wasm/` and `public/models/`. Missing assets fail loudly rather than silently falling back to a third-party CDN.
3. **Roll-Invariant Gaze & Head Pose**: Head pose (Pitch, Yaw, Roll) is decomposed algebraically from MediaPipe's 4x4 Metric 3D rigid transform matrix. Iris displacement vectors are projected onto an orthonormal eye-local basis so head tilting (roll) does not corrupt eye tracking.
4. **Adaptive Posture Drift & Occlusion Guards**: Absorbs natural slouching and chair movement over time while freezing updates during blinks, eye rubs, and hand occlusion.
5. **Human-in-the-Loop Philosophy**: The system ranks risk and presents evidence to human proctors rather than automatically disqualifying or penalizing candidates.

---

## How It Works (Under the Hood)

```
                            ┌──────────────────────────────────────────────┐
                            │             Browser Main Thread              │
                            │  getUserMedia (640x480 @ 30fps) + Silero VAD │
                            │     State Machines & Leaky Bucket Engine     │
                            └──────┬────────────────────────────────┬──────┘
                                   │                                │
                       Transferable│ImageBitmap         Transferable│ImageBitmap
                       [5 Hz Loop] │                    [2.5s Loop] │
                                   ▼                                ▼
                 ┌─────────────────────────────────┐ ┌─────────────────────────────────┐
                 │  Worker 1: Face Landmarker      │ │  Worker 2: Object Detector      │
                 │  - 478 3D Landmarks             │ │  - EfficientDet-Lite0           │
                 │  - 4x4 Metric Rigid Transform   │ │  - Allowlist: Phone/Person/Book │
                 │  - Eye-Basis Iris Vector        │ │  - Category Score Filtering     │
                 │  - One-Euro Jitter Smoothing    │ │  - Instant bitmap.close()       │
                 └─────────────────────────────────┘ └─────────────────────────────────┘
```

### 1. Dual Web Worker Concurrency
Frame analysis is separated into two background web workers so inference never blocks the main UI thread:
- **Worker 1 (`face-mesh.worker.ts`) — Fast Loop (5 Hz / 200 ms)**: Runs `FaceLandmarker` with `numFaces: 2` in `VIDEO` mode.
- **Worker 2 (`object-detector.worker.ts`) — Slow Loop (0.4 Hz / 2500 ms)**: Runs `ObjectDetector` (`EfficientDet-Lite0`) scanning for prohibited cell phones, additional persons, and books.
- **Backpressure & Memory Guards**: Incoming frames are transferred as `ImageBitmap` zero-copy buffers. If a worker is already busy (`inFlight === true`), new frames are discarded. `bitmap.close()` is called immediately in a `finally` block to prevent GPU texture memory leaks.

### 2. Head Pose & Iris Gaze Mathematical Formulation

#### 6-DoF Head Pose (Euler Angles)
MediaPipe provides `facialTransformationMatrixes[0].data` as a flat 16-element column-major array representing the 4x4 rigid metric transformation matrix:
$$\mathbf{R} = \begin{bmatrix} R_{00} & R_{01} & R_{02} \\ R_{10} & R_{11} & R_{12} \\ R_{20} & R_{21} & R_{22} \end{bmatrix}, \quad R_{ij} = \text{data}[j \times 4 + i]$$

From this, Euler angles in degrees are extracted:
$$\text{Pitch } (\theta_x) = \arcsin\left(\text{clamp}(-R_{21}, -1, 1)\right) \times \frac{180^\circ}{\pi}$$
$$\text{Yaw } (\theta_y) = \text{atan2}(R_{20}, R_{22}) \times \frac{180^\circ}{\pi}$$
$$\text{Roll } (\theta_z) = \text{atan2}(R_{01}, R_{11}) \times \frac{180^\circ}{\pi}$$

#### Roll-Invariant Iris Gaze Vector
Standard bounding-box gazes break when the head tilts. We project the iris center ($I_{468}, I_{473}$) onto the **orthonormal eye-local coordinate frame** defined by the rigid inner and outer canthus landmarks ($L_{\text{inner}}, L_{\text{outer}}$):
$$\mathbf{e} = L_{\text{right}} - L_{\text{left}}, \quad w_{\text{eye}} = \|\mathbf{e}\|, \quad \hat{u} = \frac{\mathbf{e}}{w_{\text{eye}}}, \quad \hat{w} = \begin{bmatrix} -u_y \\ u_x \end{bmatrix}$$
$$\mathbf{v} = I - \frac{L_{\text{left}} + L_{\text{right}}}{2}$$
$$dx = \frac{\mathbf{v} \cdot \hat{u}}{w_{\text{eye}}}, \quad dy = \frac{\mathbf{v} \cdot \hat{w}}{w_{\text{eye}}}$$

Normalized gaze ratios ($[0, 1]$ where $0.5$ is centered) are calculated with mirrored orientation:
$$H_{\text{ratio}} = \text{clamp}(0.5 - 2.5 \cdot dx, 0, 1), \quad V_{\text{ratio}} = \text{clamp}(0.5 + 3.0 \cdot dy, 0, 1)$$

### 3. Dynamic Calibration & Adaptive Posture Drift

1. **Auto-Baseline Warmup**: On initial page load, the system gathers the first 10 clean frames (~2 seconds) to build an automatic resting baseline before manual calibration is triggered.
2. **5-Point Guided Calibration**: The user focuses on Center, Top-Left, Top-Right, Bottom-Left, and Bottom-Right targets for 2 seconds each. Corner samples establish personal 2D screen bounding boxes:
   $$g_x = -\text{yaw} + (H_{\text{ratio}} - 0.5) \times 40^\circ$$
   $$g_y = -\text{pitch} + (V_{\text{ratio}} - 0.5) \times 35^\circ$$
3. **Adaptive Posture Drift (`AdaptiveDriftTracker`)**: Sitting in a chair inevitably involves natural slouching. When the candidate is looking comfortably inside the screen boundaries, an exponential moving average ($\alpha = 0.015$) slowly shifts the center point. Drift is bounded within $\pm 8^\circ$ horizontally and $\pm 6^\circ$ vertically so intentional off-screen glances can never pull the baseline out of bounds.
4. **Eye-Scratch & Hand-Cover Guard**: If eye openness drops below $0.15$ (blink or closed eye) or impossible single-frame angular jumps ($> 25^\circ$) occur from a hand crossing the face, pose and gaze updates freeze, eliminating transient false-positive spikes.

### 4. Three-State Debounce Engine
Raw computer vision classifications twitch from frame to frame. Every signal implements an **`IDLE` $\rightarrow$ `ACTIVE` $\rightarrow$ `RELEASED`** finite state machine with a mandatory 2.0-second release hysteresis:

| Signal | Family | Weight | Trigger Condition | Onset Requirement |
| :--- | :--- | :--- | :--- | :--- |
| `NO_FACE` | `vision` | 2 | No face detected in $\ge 90\%$ of rolling frames | 4.0 s |
| `EXTRA_FACE` | `vision` | 4 | $\ge 2$ faces detected in $\ge 2$ of 3 consecutive samples | 2.0 s |
| `HEAD_OFF_SCREEN` | `vision` | 1 | Combined gaze / pose projected outside screen boundaries | 4.0 s *(1.0 s in Dev Mode)* |
| `LOOKING_DOWN` | `vision` | 2 | Pitch below drift-compensated floor | 3.0 s *(1.0 s in Dev Mode)* |
| `PHONE` | `vision` | 5 | Cell phone detected with confidence $\ge 0.40$ in $\ge 2$ of last 3 samples | ~5.0 s (2 samples @ 2.5s) |
| `EXTRA_PERSON` | `vision` | 3 | Additional person detected in $\ge 2$ of last 3 samples | ~5.0 s |
| `SPEECH` | `audio` | 2 | Speech detected by Silero VAD for $\ge 8.0\text{ s}$ inside a 60s window | Rolling 60s accumulator |

- **Pattern Bonus**: If `HEAD_OFF_SCREEN` triggers 5 times within a 5-minute rolling window, an additional $+2$ pattern penalty is attached.
- **Sustained Bonus**: If any signal stays active continuously for more than 10 seconds, it accumulates an additional $+0.25 \times \text{weight}$ for every extra 10-second block.

### 5. Leaky Bucket Risk Scoring
All active events flow into a unified risk scoring engine:
- **Continuous Exponential Decay**:
  $$S(t) = S(t_0) \times e^{-\lambda (t - t_0)}, \quad \lambda = \frac{\ln(2)}{120\text{ s}} \approx 0.005776\text{ s}^{-1}$$
  Scores naturally decay with a **2-minute (120 s) half-life**. An accidental glance 5 minutes ago fades away.
- **Cross-Family Compound Multiplier**: If an event from a different modality family (`vision`, `audio`, `browser`) occurs within 30 seconds of an existing active family, its point contribution is multiplied by **$1.5\times$** (e.g. whispering while looking down).
- **Reviewer Tiers**:
  - 🟢 **Green** ($< 4.0$): Normal session; evidence logged locally only.
  - 🟡 **Amber** ($4.0 - 7.99$): Suspicious activity; marks session for post-exam review.
  - 🔴 **Red** ($\ge 8.0$): High risk; immediately escalates to top of human proctor queue.

### 6. Performance-Driven Adaptive Throttling
Maintains a moving 30-sample 95th-percentile (P95) latency window for each worker:
- **Face Loop Target**: $\le 120\text{ ms}$. If P95 exceeds 120 ms, it steps down from $5\text{ Hz} \rightarrow 3\text{ Hz} \rightarrow 2\text{ Hz}$.
- **Object Loop Target**: $\le 1500\text{ ms}$. If P95 exceeds 1500 ms, interval steps up from $2.5\text{ s} \rightarrow 4.0\text{ s} \rightarrow 6.0\text{ s}$.
- **Degraded Coverage Badge**: If processing remains above budget even at the lowest floor rate, a `DEGRADED_COVERAGE` flag is logged on the session so reviewers know gaps were due to hardware limits.

---

## System Requirements & Prerequisites

- **Operating System**: Linux, macOS, or Windows.
- **Node / Runtime**: [Bun](https://bun.sh) (recommended, v1.2+) or Node.js (v20+ LTS).
- **Browser**: Modern Chromium-based browser (Google Chrome, Microsoft Edge, Brave) with WebGL and Web Audio support.

---

## Setup & Installation

1. **Clone the repository and enter the directory**:
   ```bash
   cd /home/mohmedh/personal/POC-ExamProc
   ```

2. **Install project dependencies**:
   ```bash
   bun install
   # or: npm install
   ```

3. **Verify and download local model/WASM binaries**:
   The project includes an automated downloader script that checks, copies, and pins all offline assets:
   ```bash
   bun run download:models
   ```
   *This copies MediaPipe Tasks Vision WASM, Silero VAD v5 ONNX, and ONNX Runtime WASMs to `public/wasm/`, and verifies `face_landmarker.task` (3.58 MB) and `efficientdet_lite0.tflite` (6.92 MB) in `public/models/`.*

---

## Running the Project

1. **Start the local Vite development server**:
   ```bash
   bun dev
   # or: npm run dev
   ```

2. **Open the application**:
   Navigate to **`http://localhost:5173/`** in Chrome or Edge.

3. **Grant Browser Permissions**:
   Allow access to both **Camera** (video feed) and **Microphone** (Silero VAD) when prompted.

---

## Diagnostic Playground & UI Walkthrough

Upon opening the playground, you have access to real-time instrumentation:

- **Live Video Feed**: 640x480 mirrored canvas rendering facial landmark keypoints and detected object bounding boxes (cell phone in red, person in amber).
- **Live Gaze Dot**: A floating tracking dot showing your screen-plane projected gaze:
  - 🟢 **Teal dot**: Looking inside the active screen bounds.
  - 🔴 **Pulsing Red dot**: Gaze has drifted off-screen.
- **Pose & Gaze Gauges**:
  - Head Pitch ($^\circ$), Yaw ($^\circ$), and Roll ($^\circ$).
  - Horizontal & Vertical Iris Ratios ($0\% - 100\%$) normalized against your calibrated baseline.
  - Blink/Closed eye indicator.
- **⚡ Fast Dev (1s) / 🛡️ Exam Mode (3-4s) Toggle**:
  - Found in the top-right header. Defaults to **Fast Dev (1s)** for instant responsiveness during local testing, switchable to standard exam timing at any time.
- **Signal State Machine Grid**: Cards representing all 7 signals showing their current debounce state (`IDLE`, `ACTIVE`, `RELEASED`), base weight, and elapsed active duration.
- **Leaky Bucket Risk Meter**: Real-time decaying point score with visual threshold markers at 4 (Amber) and 8 (Red), complete with a manual reset button.
- **Latency & Worker Profiler**: Displays P95 inference latency, active worker execution delegate (GPU WebGL / CPU fallback), and current sampling FPS.
- **Scripted Accuracy Evaluator**: A protocol test runner with 7 scripted behaviors (e.g. "Looking Down at Desk", "Phone in Camera View", "Reading Aloud") to measure true/false positive detection rates.

---

## Testing & Quality Assurance

Run the comprehensive unit test suite covering transformation matrix algebra, iris projection, One-Euro filtering, debounce state transitions, and leaky bucket decay:

```bash
bun test
# or: npm test
```

### Build Verification
Ensure the production TypeScript bundle compiles cleanly:
```bash
bun run build
# or: npm run build
```

---

## Project Directory Structure

```
POC-ExamProc/
├── exam_proctor_plan_v2.md        # Core technical specifications and architecture design
├── package.json                   # Project configuration and script commands
├── tsconfig.json                  # TypeScript compiler settings
├── vite.config.ts                 # Vite setup with native WASM streaming middleware
├── tailwind.config.js             # Styling configuration for dark-mode proctoring HUD
├── scripts/
│   └── download-models.mjs        # Script ensuring local WASM/model asset presence
├── public/
│   ├── wasm/                      # MediaPipe & ONNX WASM binaries and loaders
│   └── models/                    # Pinned face_landmarker.task and efficientdet_lite0.tflite
├── src/
│   ├── main.tsx                   # Application entrypoint
│   ├── router.tsx                 # TanStack Router setup
│   ├── index.css                  # Tailwind styles and base theme
│   ├── workers/
│   │   ├── face-mesh.worker.ts    # Fast Loop: FaceLandmarker, Euler decomposition, Iris gaze
│   │   └── object-detector.worker.ts # Slow Loop: EfficientDet-Lite0 Object Detection
│   ├── lib/
│   │   ├── camera.ts              # Camera stream capture and frame pump with backpressure
│   │   ├── vad.ts                 # Silero VAD v5 audio worklet manager
│   │   ├── performance.ts         # P95 latency tracking and adaptive rate throttler
│   │   ├── math/
│   │   │   ├── transform-matrix.ts# 4x4 matrix decomposition to Pitch, Yaw, Roll
│   │   │   ├── iris-gaze.ts       # Eye-local basis projection and iris ratio calculation
│   │   │   └── one-euro.ts        # One-Euro adaptive low-pass filter
│   │   └── engine/
│   │       ├── calibrator.ts      # 5-point calibration & AdaptiveDriftTracker
│   │       ├── state-machine.ts   # 3-state debounce engine (Idle/Active/Released)
│   │       └── leaky-bucket.ts    # 120s half-life decaying risk score engine
│   ├── components/
│   │   ├── VideoFeed.tsx          # Mirrored video & canvas overlay with bounding boxes
│   │   ├── Gauges.tsx             # Pitch/Yaw/Roll & Iris ratio gauge meters
│   │   ├── SignalStatusGrid.tsx   # Live cards for all 7 detection signals
│   │   ├── LeakyBucketMeter.tsx   # Risk score bar with Green/Amber/Red thresholds
│   │   ├── LatencyProfiler.tsx    # P95 latency breakdown & delegate indicators
│   │   └── CalibrationModal.tsx   # 5-point guided calibration modal
│   └── routes/
│       ├── index.tsx              # Diagnostic Playground
│       └── test-runner.tsx        # Scripted evaluation harness
└── tests/
    ├── engine.test.ts             # Unit tests for state machines and leaky bucket
    └── math.test.ts               # Unit tests for matrix math, iris vectors, and One-Euro
```

---

## College Lab Recruitment Exam Context (Updated Scope)

For on-campus college computer lab recruitment exams with minimal human invigilators, the threat model narrows to:

1. **Peeking at a neighbor's screen** $\rightarrow$ `GAZE_OFF_SCREEN` / sustained side glance.
2. **Phone under the desk** $\rightarrow$ `LOOKING_DOWN` (sustained downward pitch/gaze).
3. **Tab switching / leaving fullscreen / copy-paste** $\rightarrow$ `TAB_SWITCH`, `FULLSCREEN_EXIT`, `PASTE` (plus optional Safe Exam Browser lockdown).
4. **Additional lab threats:**
   - **Whispering / talking to neighbor** $\rightarrow$ `SPEECH` (bandpass energy gate + Silero VAD).
   - **Neighbor leaning into frame** $\rightarrow$ `EXTRA_FACE` / `EXTRA_PERSON`.
   - **Camera covering / frozen feed** $\rightarrow$ `NO_FACE` / `CAMERA_FROZEN`.

### Branching Strategy & ONNX Stage A Update
- **`approach/ONNX`**: Unified `onnxruntime-web` pipeline (`face_detection_yunet_2023mar.onnx` + `mobileone_s0_gaze.onnx`).
  - **Stage A Parity Verified** (`tests/stage-a-parity.test.ts`): `maxYawDiff = 0.000003°`, `maxPitchDiff = 0.000002°` across 20 frames against upstream `onnx_inference.py`.
- **`approach/FaceDetector`**: Lightweight MediaPipe `FaceDetector` (BlazeFace 6-keypoint, CPU WASM at 1–2 Hz) + Web Audio bandpass gate for ultra-low-power lab PCs.

---

## License

This project is licensed under the Apache-2.0 License. All included pre-trained models (`MediaPipe Tasks Vision`, `Silero VAD`) are distributed under their respective permissive open-source licenses.
