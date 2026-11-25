# Remote exam proctoring: plan v2

This revises `system_architecture_technical_design_plan.md`, the plan you uploaded (called "the uploaded plan" below). It keeps what held up, fixes what would fail or mislead, and adds what was missing. Library, license and pricing facts were checked in November 2026. Numbers marked (assumption) are estimates to replace with measurements from a pilot.

Two companion files: `exam_proctor_learning_guide.md` explains why each choice was made, and `plan_review_prompt.md` is the copy-paste error list for the planner model.

## 1. Rules the design follows

1. Detection runs on the candidate's device. The server receives events, a few images and heartbeats.
2. No generative models. Face landmarks, object detection and voice activity detection only.
3. The system never fails or penalizes a candidate. It ranks sessions and hands evidence to a human.
4. Browser-only is the default mode. Safe Exam Browser (SEB) is a per-exam switch for high-stakes papers.
5. Every claim about what gets detected comes with what does not.
6. The candidate's device is untrusted, so the server checks the client's work by sampling.

## 2. What it can and cannot catch

| Threat | Browser-only | SEB mode |
| :--- | :--- | :--- |
| Leaving the tab or fullscreen | Detected and logged | Blocked by SEB |
| Second face in frame | Detected | Detected |
| Phone held up in view | Detected, recall is modest | Same |
| Phone below the desk or off camera | Not detected | Not detected |
| Helper outside camera view | Speech hints only | Same |
| Second monitor | Partly, needs a Chromium permission | Can be restricted in SEB settings |
| Virtual camera or looped video | Weak: liveness prompts, audit, human review | Same |
| Remote desktop or VM | Not detectable | Partly, depends on SEB version and settings |
| OS app switching | Shows up as focus loss | Blocked or restricted by SEB |
| Impersonation | ID photo and selfie checked by a human, plus liveness prompts | Same |
| Phone photo of the screen sent to an AI model | Not detectable | Not detectable |

The last row is a question-design problem. Randomized or parameterized items and question pools help; proctoring does not.

## 3. Architecture

```
Candidate browser (static app served from a CDN)
  main thread : exam shell, voice activity detection, browser events,
                signal engine, offline queue (IndexedDB)
  worker 1    : MediaPipe Face Landmarker   (head pose, iris, face count)
  worker 2    : MediaPipe Object Detector   (phone, person, book)
        |  every 15 s: one request with heartbeat + event batch
        |  on change : exam answers
        |  as needed : snapshots/clips straight to object storage (presigned PUT)
        v
API (Hono, stateless) ---- Postgres: sessions, events, reviews, jobs
        |
        +-- issues presigned URLs --> R2 object storage (evidence)
                                          |
                          Verifier worker (Python, CPU, MediaPipe Tasks)
                                          |
                          Reviewer dashboard (same app, role-gated)

Phase 3 only: live view through a managed SFU
```

The client does the heavy work, so the API stays small and stateless. Postgres doubles as the job queue at pilot scale. The verifier runs the same model files as the browser, so client and server verdicts are comparable.

## 4. Client logic

### 4.1 System check and calibration (about 5 minutes)

1. Feature check: camera, microphone, fullscreen, workers, IndexedDB, persistent storage. Supported browsers in v1: Chrome and Edge. Others get a clear "not supported yet" message.
2. Model download and cache check. Models and WASM files come from your own domain, with versioned file names.
3. Framing and light check: one face, sensible size, enough brightness.
4. Calibration: the candidate looks at the four screen corners and the center for two seconds each. Store their head-pose and iris ranges as a personal baseline.
5. Speed test on live camera frames for 10 seconds. Pass if the 95th-percentile face-landmarker time stays under about 120 ms (assumption). Otherwise offer another device or an exam center.
6. Microphone test and noise floor.
7. Network test: upload one 200 KB test image and time it.
8. Consent screen, then an ID card photo and a selfie. A human compares them later. No automatic face matching in v1.
9. Optional room scan: a slow camera pan, kept as 3 to 5 snapshots.

### 4.2 Signals

All weights are starting values to calibrate in the pilot (assumption).

| Signal | Source | Rate | Raised when | Weight |
| :--- | :--- | :--- | :--- | :--- |
| No face | Face Landmarker | 5 Hz | no face in 90% of samples for 4 s | 2 |
| Extra face | Face Landmarker, 2 faces max | 5 Hz | 2 faces on at least 2 of 3 consecutive samples for 2 s | 4 |
| Head off screen | head pose vs calibrated range | 5 Hz | outside the range for 4 s; pattern bonus of +2 if 5 episodes in 5 minutes | 1 |
| Looking down | pitch below calibrated floor | 5 Hz | 3 s | 2 |
| Phone | Object Detector | one sample every 2.5 s | at least 2 of the last 3 samples, score 0.5 or more | 5 |
| Extra person | Object Detector, person count | same | 2 of 3 samples | 3 |
| Speech | voice activity detector | continuous | 8 s or more of speech inside 60 s | 2 |
| Focus loss | tab hidden or window blur | event | longer than 2 s | 3 |
| Fullscreen exit | fullscreenchange | event | any | 3 |
| Paste attempt | paste event | event | any | 1 |
| Camera tamper | track ended or muted, frozen frames | 1 Hz | 10 s | 5 |
| Heartbeat gap | server | none | no heartbeat for 45 s | 2 |
| Audit mismatch | verifier | none | 3 or more mismatches in a session | 6 |

### 4.3 Debounce and score

Each signal has its own small state machine with three states: idle, active, released. It becomes active only when its "raised when" condition holds, emits one event stamped with the start time, and returns to idle after the condition has been false for 2 seconds. A 40-second glance produces one event, not forty. Fast signals use durations. Slow signals, like the object detector, use sample counts, because "2 seconds" means nothing when samples arrive every 2.5 seconds.

The score works like a leaky bucket. A new episode adds its weight. A sustained episode adds a quarter of its weight again for every extra 10 seconds. The total decays with a two-minute half-life, so an old glance fades. If two different signal families (vision, audio, browser) are active within 30 seconds of each other, the second one counts 1.5 times.

Levels: green below 4, amber from 4 to 8, red from 8 (assumption). Set these from pilot data so that honest candidates rarely reach red.

| Level | What happens |
| :--- | :--- |
| Green | Nothing leaves the device except heartbeats and audit snapshots |
| Amber | Event logged, 3 snapshots uploaded |
| Red | Snapshots uploaded, session moves up the reviewer queue, clip uploaded from phase 2, optional neutral on-screen notice if the exam policy enables it |

Session ranking for reviewers uses the sum of episode weights, the number of red episodes, and any audit mismatch.

### 4.4 Evidence

- **Event snapshots.** Three JPEGs around the event start, about 640 by 480 at quality 0.7, roughly 50 KB each (assumption). Cap at 6 events per 10 minutes.
- **Audit snapshots.** One every 90 to 120 seconds at a random offset, uploaded whether or not anything was flagged, with the client's verdict for that instant attached.
- **Clips (phase 2).** Red events only, 8 to 10 seconds. MediaRecorder chunks cut with a timeslice are not playable on their own, so record with two recorders started 4 seconds apart, each restarting every 8 seconds, and upload the latest finished files. Measure CPU on a weak laptop before switching this on.
- **Audio.** No continuous recording. Speech events upload as markers with start time and duration. Audio clips are an opt-in exam policy.
- **Queueing.** Everything goes into IndexedDB first, uploads in the background, and is deleted locally after the server confirms. Request persistent storage so the browser does not evict the queue.

### 4.5 Liveness prompts

Once or twice per exam, at random moments when the candidate is not typing, the screen asks them to turn their head to one side for three seconds. Head pose verifies it. A miss creates an event with weight 2. It never fails anyone. This catches frozen feeds and simple looped videos. Someone running real-time face-swap software can get past it, so it raises the bar and nothing more.

### 4.6 Performance control

- Two workers, so a slow object pass never delays head pose.
- GPU delegate where it initializes, CPU fallback where it does not.
- A moving 95th-percentile latency per detector. When it exceeds the budget, drop the face rate from 5 to 3 to 2 Hz and stretch the object interval from 2.5 to 4 to 6 seconds.
- Below those floors, mark the session "degraded coverage" so reviewers know the gaps are not clean.
- Models load before the exam starts, never during it.

### 4.7 Offline behavior and resume

- The exam timer lives on the server. The client timer is for display.
- Answers save to IndexedDB on change and sync with sequence numbers. Retried requests are safe.
- A refresh restores the session from the session token.
- Camera lost: 60 seconds to restore it, then an event.
- A scheduled-break button, if the exam policy allows one, pauses the face and gaze rules. The server records how long it lasted.

## 5. Server logic

### 5.1 Session and timer

The server stores each session's start and end time and any time credits. Everything the client shows about time is derived from that.

### 5.2 Heartbeat

Every 15 seconds the client sends one request with its session token, an integer sequence number, the event batch, and a summary of its latest verdict. The server replies with the next expected sequence number and any commands (show a liveness prompt, later "start live view").

- The server accepts the expected sequence number or an exact retry of the previous one, and answers a retry with the same ack.
- A gap in sequence numbers means dropped telemetry. Record it. Raise the heartbeat-gap signal only after 45 seconds of silence.
- A telemetry problem never rejects answers and never locks a candidate out.
- Every request carries an authenticated session token. A bare session ID is not authentication.
- No per-event signatures. They would use a key the candidate can read, so they would add nothing against the people you worry about.

### 5.3 Evidence intake

- Presigned PUT URLs that expire in about 5 minutes, scoped to a per-session key prefix.
- Size fixed when the URL is issued (sign the content length; confirm this works in your SDK), a cap on uploads per session, and CORS limited to your origin.
- R2 supports presigned URLs on its S3 API endpoint only and does not accept HTML form POST uploads, so enforcement happens in your API.
- A lifecycle rule deletes objects after the retention period.

### 5.4 Audit and verification

The verifier picks about 25% of audit snapshots at random, plus every red-event snapshot. It runs the same Face Landmarker and Object Detector models in Python on CPU and compares its verdict with the one the client attached. JPEG compression and resolution cause small differences, so tolerate them. After three persistent mismatches in a session, raise an audit-mismatch signal for the reviewer.

What this proves: a client that suppresses events gets caught. What it does not prove: that the frames come from a real camera. Liveness prompts and human review of random clips cover that part.

### 5.5 SEB mode

- The mode is stored per exam on the server and never read from a request header.
- When it is on, every exam request must carry `X-SafeExamBrowser-RequestHash`, `X-SafeExamBrowser-ConfigKeyHash`, or both. The server computes SHA-256 of the absolute URL (public host and scheme, as the browser saw it, no fragment) followed by each allowed key, and compares in constant time.
- Browser Exam Keys and Config Keys come from SEB's own tools for each SEB version you allow. Store them. Do not derive them by hand. Turn on the option that sends the keys in the SEB config file.
- The server refuses to start without keys. There is no default secret.
- SEB runs on Windows, macOS and iOS. Linux candidates take browser-only or go to a center.
- Before promising SEB mode, test the whole pipeline inside SEB: camera permission, workers, model loading and fullscreen behavior.

### 5.6 Reviewer workflow

A queue sorted by session score. A timeline with thumbnails, signal reasons and the degraded-coverage flag. Decisions: clear, needs follow-up, report. A second reviewer for any adverse outcome. A candidate appeal path with access to the same evidence. A log of who viewed what. Roles: candidate, reviewer, admin.

### 5.7 Behavior analytics and the session report

Computed in a batch after the exam, from JSON only:

- Response time per question against the cohort median, as a z-score.
- Paste attempts, answer changes, long idle gaps followed by fast answers.
- A fast correct answer within 20 seconds of a red vision event, on a question most of the cohort missed.

Cohort statistics need enough candidates; start at 30 or more per paper (assumption). These rows have low weights. They feed the Session Integrity Report: score, ranked reasons, thumbnails, behavior rows, and any challenge results.

## 6. Stack choices

The long reasoning is in the learning guide.

| Layer | Pick | Main caution |
| :--- | :--- | :--- |
| Candidate app | Vite + React + TanStack Router, static files on a CDN | TanStack Start is stable and fine for the dashboard, but SSR gives the exam client nothing |
| API | Hono on Node LTS (Bun is fine, Hono code moves unchanged) | Choose by what your host supports well |
| Database | Postgres + Drizzle | Heartbeat writes are the main load |
| Queue | Postgres jobs table (`FOR UPDATE SKIP LOCKED`) first | Add Redis only when heartbeat load needs it |
| Face and gaze | MediaPipe Face Landmarker | Gaze is coarse; calibrate per person |
| Objects | MediaPipe Object Detector, EfficientDet-Lite0 | Low recall for small, half-hidden phones |
| Object fallback | YOLOX-nano through ONNX Runtime Web (Apache-2.0) | You write letterboxing, decoding and NMS |
| Avoid | Ultralytics YOLO (AGPL-3.0) unless you buy the enterprise license or open-source everything | License |
| Speech | Silero VAD through vad-web | Detects speech, not who is speaking |
| Local storage | `idb` over IndexedDB | Ask for persistent storage |
| Evidence storage | Cloudflare R2 | Presigned URLs on the S3 endpoint only, CORS needed |
| Verifier | Python + MediaPipe Tasks, CPU, scale to zero | Same model files as the browser |
| Live view | Phase 3, managed SFU first | Self-hosting needs TLS, UDP ports, TURN and Redis |
| High stakes | SEB as a per-exam toggle | Windows, macOS, iOS only |

## 7. Hosting, regions and cost

**Regions.** A region is the deployment unit. Each exam cohort is assigned to one region with its own API and Postgres. The app and models sit on the CDN everywhere. Evidence goes to a bucket in the cohort's region. Reviewers query their region. Because only 15-second heartbeats and occasional uploads cross the network, latency barely matters. Regions exist mostly for data residency and to keep one outage from touching every exam. Add a region when law or measured round trips demand it.

**First load.** Plan for 30 MB or more of models and WASM (assumption, measure it). Add a "download exam software" step the day before, cache it with a service worker, and use versioned URLs.

**Cost sketch for 10,000 candidates and one 2-hour exam (all assumptions):**

| Item | Estimate |
| :--- | :--- |
| Static files and models | No egress charge on R2; the CDN carries the rest |
| Evidence storage | About 3.6 MB per candidate, so 36 GB. R2 charges $0.015 per GB-month, so about $0.50 per month |
| Evidence writes | About 84 per candidate, 0.84 million total, inside R2's 1 million free writes |
| Verifier | About 25 frames per candidate, 250,000 frames, about 20 CPU core-hours at 0.3 s per frame |
| API | Up to about 670 heartbeat requests per second if everyone is online at once; a few small instances |
| Reviewers | If 10% of sessions get a 5-minute review, about 83 hours |

Servers are cheap. Reviewer time is the real cost, so the risk score's job is to cut review time. That is also why thresholds need calibration.

## 8. Privacy and compliance

- Consent screen before anything is recorded. State what is collected, why, for how long, and who sees it.
- Collect the minimum. No continuous video or audio upload. No face templates in v1.
- Retention: auto-delete after a fixed period (for example 30 to 90 days) unless a dispute holds it.
- Access: role-based, short-lived signed view URLs, and a log of every view.
- Deletion and access requests have a defined path.
- Data residency follows the cohort's region.
- Face images, ID photos and voice are personal data. Face templates are biometric data, which many laws treat more strictly. Check India's DPDP Act and, for EU candidates, GDPR with someone qualified.

## 9. Evaluation plan

Without numbers, "accurate" and "cost-efficient" are opinions.

- **Data.** 20 to 30 volunteer sessions with scripted behaviors: looking away, phone use, a second person, whispering, normal thinking. Vary lighting, glasses, camera position and laptop class.
- **Metrics.** Precision and recall per signal. False positives per candidate-hour. Time to detect. CPU, memory and battery. Face and object rates actually achieved.
- **Ablations.** Head pose alone against head pose plus iris. Object detector alone against detector plus majority vote.
- **Fairness.** Report metrics by slice: lighting, glasses, skin tone, camera quality. Fix the worst slice before launch.
- **Baselines.** Compare against the approaches in PupilPatrol and the face-mesh gaze paper you found. I have not verified their reported numbers.
- **Load.** Simulate 10,000 heartbeat clients before a real exam.
- **Target (assumption).** An honest candidate sees no more than one amber per 30 minutes and almost never red.

## 10. Roadmap

| Phase | Scope | Exit criterion |
| :--- | :--- | :--- |
| 0 (1 to 2 weeks) | Prototype: webcam to face, object and voice signals, console output, timing on three laptops | Latency numbers and a first false-positive count |
| 1 (4 to 6 weeks) | Exam shell with server timer and answer sync, system check, signals, score, snapshots, heartbeat, audit sampling, reviewer queue | Pilot with real candidates meets the false-positive target |
| 2 | SEB mode, clips, accommodations, session integrity report | SEB pipeline works end to end inside SEB |
| 3 | Live view, optional comprehension challenge, automated identity matching with a permissively licensed model, multi-region, LMS (LTI) integration | Demand and budget justify each item |

## 11. Changes from the uploaded plan

| Uploaded plan | Decision | In v2 |
| :--- | :--- | :--- |
| 1. Philosophy | Kept 1 and 2, changed 3 | Verification is a late, optional module; liveness prompts come first |
| 2.1 Stack matrix | Changed | Section 6 |
| 3. Topology | Changed | Section 3: two workers, Postgres first, audit path added |
| 4.1 Worker | Changed | Explicit init, pinned self-hosted assets, head pose from the transformation matrix, CPU fallback |
| 4.2 Benchmark | Replaced | Live-frame gate plus adaptive throttling |
| 4.3 Risk engine | Replaced | Per-signal state machines and a decaying score |
| 4.4 Active Flow | Demoted | Phase 3, optional, server-graded |
| 4.5 SEB and nonce | Rewritten | Sections 5.2 and 5.5 |
| 5.1 Camera-less | Decision kept | Claim softened; see the capability matrix |
| 5.2 Tauri | Decision kept | Reasoning corrected: capabilities lost are listed |
| 5.3 LMS coupling | Decision kept | LTI integration in phase 3 |
| 5.4 Constant streaming | Decision kept | "98%" removed; self-hosted LiveKit deferred |
| 5.5 Express | Dropped as a decision | Runtime is an operations choice |
| 5.6 Anti-flicker | Decision kept | Answer is question design; add citations |
| Not in the plan | Added | Audit sampling, capability matrix, privacy, evaluation, reviewer workflow, behavior analytics, cost sketch |

## 12. Decisions only you can make

1. Is this commercial or academic? It decides whether the Ultralytics and InsightFace licenses block you.
2. Which exam types? A comprehension challenge needs parameterized questions.
3. Do you need live proctors, or is post-exam review enough?
4. Where are the candidates? That sets regions and residency.
5. Desktop only, or also tablets and phones?
