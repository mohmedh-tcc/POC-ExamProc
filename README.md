# Lightweight Client-Side Exam Proctoring POC (`approach/FaceDetector`)

A low-power, browser-native proctoring engine built with **React**, **Vite**, **TanStack Router**, and **Google MediaPipe `FaceDetector` (BlazeFace Short-Range)**.

This branch (`approach/FaceDetector`) is designed for **in-person / computer-lab proctored exams** where each candidate uses a standard webcam without draining battery or overheating laptops. It monitors **head pose, extra faces, looking down, frozen camera feeds, tab switches, and fullscreen exits** 100% on the client—with **zero** audio/VAD, **zero** heavy object detection, and **zero** continuous `requestAnimationFrame` GPU loops.

---

## Key Architectural Highlights

1. **Ultra-Lightweight Edge Inference (~225 KB Model, CPU WASM)**:
   Runs MediaPipe `FaceDetector` (`blaze_face_short_range.tflite`) inside a single Web Worker (`face-detector.worker.ts`) on **CPU WASM** at **1.5 Hz** (with adaptive step-down to 1 Hz / 0.5 Hz).
2. **Low Thermal Footprint**:
   Captures webcam video at `320×240 @ 10–15 fps` and pumps frames via `setTimeout` instead of 60 Hz `requestAnimationFrame` polling, keeping CPU/GPU idle >98% of the time.
3. **6-Keypoint Head Pose & Frozen Feed Detection**:
   Computes coarse **Yaw** and **Pitch** from BlazeFace's 6 facial keypoints (left/right eye, nose tip, mouth center, left/right ear tragions) and detects static/spoofed camera feeds using an 8×8 L1 thumbnail diff (`FROZEN_FEED`).
4. **5-Point Head-Pose Calibration**:
   Guided 5-point calibration (Center + 4 Corners) establishes tight personal exam boundaries (`±14°` yaw, `-10°` pitch down) to catch peeking at neighboring screens or looking down at a phone under the desk.
5. **Browser Lockdown (Fullscreen API + Focus)**:
   Monitors `fullscreenchange`, `visibilitychange`, and `window.blur` events to flag `TAB_SWITCH` and `FULLSCREEN_EXIT` without requiring Safe Exam Browser (SEB).
6. **Three-State Debounce & Leaky Bucket Risk Engine**:
   Smooths raw frame jitter using an `IDLE` → `ACTIVE` → `RELEASED` state machine and scores risk with a 120-second half-life decaying leaky bucket.

---

## Monitored Signals

| Signal | Family | Weight | Trigger Condition |
| :--- | :--- | :--- | :--- |
| `NO_FACE` | `vision` | 2 | No face detected in ≥ 90% of rolling frames |
| `EXTRA_FACE` | `vision` | 4 | ≥ 2 faces detected in ≥ 2 of 3 consecutive samples |
| `HEAD_OFF_SCREEN` | `vision` | 1 | Head yaw/pitch outside calibrated exam screen bounds |
| `LOOKING_DOWN` | `vision` | 2 | Head pitch below calibrated downward floor (e.g., phone under desk) |
| `FROZEN_FEED` | `vision` | 4 | Camera feed 8×8 L1 diff remains static across consecutive samples |
| `TAB_SWITCH` | `browser` | 3 | Candidate switches tab or loses window focus |
| `FULLSCREEN_EXIT` | `browser` | 3 | Candidate exits browser fullscreen mode |

---

## System Requirements & Prerequisites

- **Operating System**: Linux, macOS, or Windows
- **Runtime**: [Bun](https://bun.sh) (v1.2+ recommended) or Node.js (v20+ LTS)
- **Browser**: Modern Chromium-based browser (Google Chrome, Microsoft Edge, Brave) with WebAssembly and Camera permissions

---

## Setup & Installation

1. **Clone the repository and switch to this branch**:
   ```bash
   git clone git@github.com:mohmedh-tcc/POC-ExamProc.git
   cd POC-ExamProc
   git checkout approach/FaceDetector
   ```

2. **Install dependencies**:
   ```bash
   bun install
   # or: npm install
   ```

3. **Download local WASM & model binaries**:
   ```bash
   bun run download:models
   ```
   *This populates `public/wasm/` and `public/models/` (including `blaze_face_short_range.tflite`) for offline-pinned execution.*

---

## Running the Project

1. **Start the Vite development server**:
   ```bash
   bun dev
   # or: npm run dev
   ```

2. **Open the application**:
   Navigate to **`http://localhost:5173/`** in Chrome or Edge and grant **Camera** access when prompted.

---

## Testing & Build Verification

```bash
# Run unit tests (calibration, debounce state machine, leaky bucket, math)
bun test

# Typecheck and build production bundle
bun run build
```

---

## Multi-Account Git Setup (Optional)

If you have both personal and work GitHub accounts on the same machine, configure **this repository only** (without `--global`) so commits and pushes use your work identity:

```bash
git config user.name "Your Work Name"
git config user.email "your-work-email@company.com"
git config core.sshCommand "ssh -i ~/.ssh/your_work_key -o IdentitiesOnly=yes"
```

---

## Project Structure

```
POC-ExamProc/
├── scripts/
│   └── download-models.mjs          # Downloads/copies local WASM & model assets
├── public/
│   ├── wasm/                        # MediaPipe WASM binaries
│   └── models/                      # blaze_face_short_range.tflite
├── src/
│   ├── workers/
│   │   └── face-detector.worker.ts  # 1.5 Hz BlazeFace 6-keypoint + FROZEN_FEED worker
│   ├── lib/
│   │   ├── camera.ts                # 320x240 webcam capture & setTimeout frame pump
│   │   ├── performance.ts           # P95 latency tracking & adaptive throttler
│   │   └── engine/
│   │       ├── calibrator.ts        # 5-point calibration & drift tracker
│   │       ├── state-machine.ts     # 3-state debounce engine
│   │       └── leaky-bucket.ts      # 120s half-life decaying risk engine
│   ├── components/
│   │   ├── CalibrationModal.tsx     # 5-point guided head-pose calibration
│   │   ├── SignalStatusGrid.tsx     # Live signal state cards
│   │   ├── LeakyBucketMeter.tsx     # Risk score bar (Green / Amber / Red)
│   │   └── ...
│   └── routes/
│       ├── index.tsx                # Diagnostic Playground & Fullscreen Exam Mode
│       └── test-runner.tsx          # Scripted evaluation harness
└── tests/
    ├── engine.test.ts               # State machine & leaky bucket unit tests
    └── math.test.ts                 # Math & filter unit tests
```
