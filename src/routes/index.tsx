import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { CameraPipeline, FaceDetectionEvent, ObjectDetectionEvent } from '../lib/camera';
import { VADManager } from '../lib/vad';
import {
  CalibrationProfile,
  CalibrationSample,
  AdaptiveDriftTracker,
  buildAutoBaselineProfile,
  mapGazeToScreen,
} from '../lib/engine/calibrator';
import {
  SignalType,
  DebounceStateMachine,
  DebounceState,
} from '../lib/engine/state-machine';
import { LeakyBucketEngine, RiskLevel } from '../lib/engine/leaky-bucket';
import { VideoFeed } from '../components/VideoFeed';
import { Gauges } from '../components/Gauges';
import { SignalStatusGrid } from '../components/SignalStatusGrid';
import { LeakyBucketMeter } from '../components/LeakyBucketMeter';
import { LatencyProfiler } from '../components/LatencyProfiler';
import { CalibrationModal } from '../components/CalibrationModal';
import { TestRunner } from './test-runner';
import { EulerAngles } from '../lib/math/transform-matrix';
import { IrisGazeResult, NormalizedLandmark } from '../lib/math/iris-gaze';
import { Compass } from 'lucide-react';

/**
 * Extended FaceDetectionEvent type supporting optional raw 4x4 transformation matrix
 * forwarded from the FaceLandmarker Web Worker.
 */
type ExtendedFaceDetectionEvent = FaceDetectionEvent & {
  transformMatrix?: number[] | Float32Array;
};

/**
 * Anatomical head-pose fallback limits (degrees from neutral).
 * Beyond +-35 deg yaw, the far eye is occluded by the nasal bridge and monocular iris
 * estimation degrades, so extreme head turns still flag HEAD_OFF_SCREEN as a safety guardrail.
 * Within +-35 deg, the 3D VOR-compensated gaze model decides on-screen vs. off-screen.
 */
const ANATOMICAL_YAW_LIMIT_DEG = 35;
const ANATOMICAL_PITCH_UP_LIMIT_DEG = 28;
const LOOKING_DOWN_GAZE_THRESHOLD = 1.12; // Normalized screen y > 1.12 (below bottom bezel)

export const DiagnosticPlayground: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pipelineRef = useRef<CameraPipeline | null>(null);
  const vadRef = useRef<VADManager | null>(null);

  // Calibration & Dev Mode
  const [calibration, setCalibration] = useState<CalibrationProfile | null>(null);
  const calibrationRef = useRef<CalibrationProfile | null>(null);
  calibrationRef.current = calibration;
  const [isCalibrationOpen, setIsCalibrationOpen] = useState(false);
  const [fastDevMode, setFastDevMode] = useState(true);
  const [gazePoint, setGazePoint] = useState<{ x: number; y: number; isOffScreen: boolean } | null>(null);
  const driftTracker = useMemo(() => new AdaptiveDriftTracker(), []);
  const warmupSamplesRef = useRef<CalibrationSample[]>([]);
  const lastValidAnglesRef = useRef<EulerAngles | null>(null);

  // Workers Ready & Delegate status
  const [workerStatus, setWorkerStatus] = useState({
    faceReady: false,
    faceDelegate: 'Initializing...',
    objectReady: false,
    objectDelegate: 'Initializing...',
    vadReady: false,
  });

  // Detection states
  const [faceCount, setFaceCount] = useState(0);
  const [angles, setAngles] = useState<EulerAngles | null>(null);
  const [transformMatrix, setTransformMatrix] = useState<number[] | undefined>(undefined);
  const [gaze, setGaze] = useState<IrisGazeResult | null>(null);
  const [keyPoints, setKeyPoints] = useState<NormalizedLandmark[] | null>(null);
  const [showMesh, setShowMesh] = useState(true);

  const [objectDetections, setObjectDetections] = useState<Array<{
    categoryName: string;
    score: number;
    boundingBox?: any;
  }>>([]);
  const [hasPhone, setHasPhone] = useState(false);
  const [personCount, setPersonCount] = useState(0);

  // Performance stats
  const [perfStats, setPerfStats] = useState({
    faceP95: 0,
    objectP95: 0,
    faceIntervalMs: 200,
    objectIntervalMs: 2500,
    faceFps: 5,
    isDegraded: false,
  });

  // Engines
  const leakyBucket = useMemo(() => new LeakyBucketEngine(), []);
  const [riskScore, setRiskScore] = useState(0);
  const [riskLevel, setRiskLevel] = useState<RiskLevel>('green');

  // State Machines
  const stateMachines = useRef<Record<SignalType, DebounceStateMachine>>({
    NO_FACE: new DebounceStateMachine('NO_FACE'),
    EXTRA_FACE: new DebounceStateMachine('EXTRA_FACE'),
    HEAD_OFF_SCREEN: new DebounceStateMachine('HEAD_OFF_SCREEN'),
    LOOKING_DOWN: new DebounceStateMachine('LOOKING_DOWN'),
    PHONE: new DebounceStateMachine('PHONE'),
    EXTRA_PERSON: new DebounceStateMachine('EXTRA_PERSON'),
    SPEECH: new DebounceStateMachine('SPEECH'),
  });

  const [signalStatuses, setSignalStatuses] = useState<Record<SignalType, {
    type: SignalType;
    state: DebounceState;
    activeDurationMs: number;
  }>>({
    NO_FACE: { type: 'NO_FACE', state: 'IDLE', activeDurationMs: 0 },
    EXTRA_FACE: { type: 'EXTRA_FACE', state: 'IDLE', activeDurationMs: 0 },
    HEAD_OFF_SCREEN: { type: 'HEAD_OFF_SCREEN', state: 'IDLE', activeDurationMs: 0 },
    LOOKING_DOWN: { type: 'LOOKING_DOWN', state: 'IDLE', activeDurationMs: 0 },
    PHONE: { type: 'PHONE', state: 'IDLE', activeDurationMs: 0 },
    EXTRA_PERSON: { type: 'EXTRA_PERSON', state: 'IDLE', activeDurationMs: 0 },
    SPEECH: { type: 'SPEECH', state: 'IDLE', activeDurationMs: 0 },
  });

  // Rolling history for slow object detector and face count
  const objectPhoneHistory = useRef<boolean[]>([]);
  const objectPersonHistory = useRef<number[]>([]);
  const noFaceHistory = useRef<boolean[]>([]);
  const extraFaceHistory = useRef<boolean[]>([]);

  // Active signals set for test runner
  const activeSignalsSet = useMemo(() => {
    const s = new Set<SignalType>();
    for (const [k, v] of Object.entries(signalStatuses) as [SignalType, typeof signalStatuses[SignalType]][]) {
      if (v.state === 'ACTIVE') s.add(k);
    }
    return s;
  }, [signalStatuses]);

  // Initialize Camera & VAD
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const pipeline = new CameraPipeline();
    pipelineRef.current = pipeline;

    pipeline.init(video, {
      onWorkerReady: (workerName, delegate) => {
        setWorkerStatus((prev) => ({
          ...prev,
          [`${workerName}Ready`]: true,
          [`${workerName}Delegate`]: delegate,
        }));
      },
      onFaceResult: (data: ExtendedFaceDetectionEvent) => {
        setFaceCount(data.faceCount);
        setAngles(data.angles);
        setTransformMatrix(data.transformMatrix);
        setGaze(data.gaze);
        setKeyPoints(data.keyPoints);

        // Update performance stats
        setPerfStats(pipeline.performance.getStats());

        // Update face detection conditions
        handleFaceSignals(data);
      },
      onObjectResult: (data: ObjectDetectionEvent) => {
        setHasPhone(data.hasPhone);
        setPersonCount(data.personCount);
        setObjectDetections(data.detections);

        setPerfStats(pipeline.performance.getStats());

        handleObjectSignals(data);
      },
      onError: (workerName, err) => {
        console.error(`Worker error (${workerName}):`, err);
      },
    }).catch((err) => console.error('Failed to init pipeline:', err));

    // Initialize VAD
    const vad = new VADManager();
    vadRef.current = vad;
    vad.init((isSpeechTriggered) => {
      handleSpeechSignal(isSpeechTriggered);
    }).then(() => {
      setWorkerStatus((prev) => ({ ...prev, vadReady: true }));
    }).catch((err) => console.error('Failed to init VAD:', err));

    return () => {
      pipeline.stop();
      vad.destroy();
    };
  }, []);

  // Sync Fast Dev Mode (1s onset) vs Exam Mode (3s/4s onset)
  useEffect(() => {
    stateMachines.current.LOOKING_DOWN.onsetMs = fastDevMode ? 1000 : 3000;
    stateMachines.current.HEAD_OFF_SCREEN.onsetMs = fastDevMode ? 1000 : 4000;
  }, [fastDevMode]);

  // Evaluates Face Signals (NO_FACE, EXTRA_FACE, HEAD_OFF_SCREEN, LOOKING_DOWN)
  const handleFaceSignals = (data: ExtendedFaceDetectionEvent) => {
    const now = Date.now();

    // 1. NO_FACE: no face in 90% of samples for 4s (at 5Hz = 20 samples)
    noFaceHistory.current.push(data.faceCount === 0);
    if (noFaceHistory.current.length > 20) noFaceHistory.current.shift();
    const noFaceCount = noFaceHistory.current.filter(Boolean).length;
    const noFaceActive = noFaceHistory.current.length >= 15 && (noFaceCount / noFaceHistory.current.length) >= 0.9;

    // 2. EXTRA_FACE: >= 2 faces on >= 2 of 3 consecutive samples for 2s (at 5Hz = 10 samples)
    extraFaceHistory.current.push(data.faceCount >= 2);
    if (extraFaceHistory.current.length > 10) extraFaceHistory.current.shift();
    const extraFaceCount = extraFaceHistory.current.filter(Boolean).length;
    const extraFaceActive = extraFaceHistory.current.length >= 6 && extraFaceCount >= 4;

    updateSignal('NO_FACE', noFaceActive, now);
    updateSignal('EXTRA_FACE', extraFaceActive, now);

    // Auto-baseline warmup on first 10 valid frames if not manually calibrated yet
    let cal = calibrationRef.current;
    if (!cal && data.faceCount === 1 && data.angles && data.gaze && !data.gaze.isBlinking) {
      warmupSamplesRef.current.push({
        yaw: data.angles.yaw,
        pitch: data.angles.pitch,
        roll: data.angles.roll,
        hRatio: data.gaze.hRatio,
        vRatio: data.gaze.vRatio,
        dx: data.gaze.dx,
        dy: data.gaze.dy,
        transformMatrix: data.transformMatrix,
      });
      if (warmupSamplesRef.current.length >= 10) {
        const autoProfile = buildAutoBaselineProfile(warmupSamplesRef.current);
        calibrationRef.current = autoProfile;
        setCalibration(autoProfile);
        cal = autoProfile;
      }
    }

    // Occlusion / Blink / Eye-scratch guard:
    // Freeze pose & gaze updates when both eyes are closed or severely occluded.
    if (
      !data.gaze ||
      !data.angles ||
      data.gaze.isBlinking ||
      (data.gaze.rightOpenness < 0.15 && data.gaze.leftOpenness < 0.15)
    ) {
      return;
    }

    // Outlier spike guard: reject impossible single-frame >25 deg jumps (e.g. hand crossing face)
    if (lastValidAnglesRef.current) {
      const dYaw = Math.abs(data.angles.yaw - lastValidAnglesRef.current.yaw);
      const dPitch = Math.abs(data.angles.pitch - lastValidAnglesRef.current.pitch);
      if (dYaw > 25 || dPitch > 25) {
        lastValidAnglesRef.current = data.angles;
        return;
      }
    }
    lastValidAnglesRef.current = data.angles;

    // Wait until warmup or manual calibration is ready
    if (!cal) return;

    const drift = driftTracker.getDrift();

    // 3D VOR-compensated gaze projection onto normalized screen coordinates [0..1, 0..1]
    const mapped = mapGazeToScreen(
      {
        yaw: data.angles.yaw,
        pitch: data.angles.pitch,
        roll: data.angles.roll,
        hRatio: data.gaze.hRatio,
        vRatio: data.gaze.vRatio,
        dx: data.gaze.dx,
        dy: data.gaze.dy,
        transformMatrix: data.transformMatrix,
      },
      cal,
      drift
    );
    setGazePoint({ x: mapped.x, y: mapped.y, isOffScreen: mapped.isOffScreen });

    // 3. LOOKING_DOWN: triggered by downward screen gaze OR pitch below calibrated floor
    // (Downward gaze droops upper eyelids, so pitch remains a valuable secondary cue;
    //  guard against firing when head is clearly pitched UP above neutral.)
    const pitchFloor = cal.pitchMin + drift.driftPitch;
    const isLookingDownRaw =
      data.angles.pitch < pitchFloor ||
      (mapped.y > LOOKING_DOWN_GAZE_THRESHOLD &&
        data.angles.pitch < cal.neutralPitch + drift.driftPitch + 4);

    // 4. HEAD_OFF_SCREEN:
    // Primary trigger is the 3D VOR-compensated screen-plane gaze (mapped.isOffScreen).
    // Tight head-yaw bounds are intentionally REMOVED so turning the head while fixating
    // on the screen (VOR) does NOT falsely trigger HEAD_OFF_SCREEN, while iris-only
    // glances off-screen ARE caught accurately.
    // Wide anatomical ceilings (+-35 deg yaw, +28 deg pitch up) act only as a fallback
    // when extreme head rotation occludes the eyes.
    const yawFromNeutral = Math.abs(data.angles.yaw - cal.neutralYaw);
    const pitchUpFromNeutral = data.angles.pitch - (cal.neutralPitch + drift.driftPitch);
    const isExtremeHeadTurn =
      yawFromNeutral > ANATOMICAL_YAW_LIMIT_DEG ||
      pitchUpFromNeutral > ANATOMICAL_PITCH_UP_LIMIT_DEG;

    const isHeadOffScreenRaw = mapped.isOffScreen || isExtremeHeadTurn;

    // Slowly adapt posture drift only when gaze is comfortably inside the screen
    driftTracker.update(
      mapped.rawX,
      mapped.rawY,
      data.angles.pitch,
      cal,
      !isHeadOffScreenRaw && !isLookingDownRaw
    );

    updateSignal('LOOKING_DOWN', isLookingDownRaw, now);
    updateSignal('HEAD_OFF_SCREEN', isHeadOffScreenRaw, now);
  };

  // Evaluates Object Signals (PHONE, EXTRA_PERSON)
  const handleObjectSignals = (data: ObjectDetectionEvent) => {
    const now = Date.now();

    // PHONE: detected in >= 2 of last 3 samples
    objectPhoneHistory.current.push(data.hasPhone);
    if (objectPhoneHistory.current.length > 3) objectPhoneHistory.current.shift();
    const phoneCount = objectPhoneHistory.current.filter(Boolean).length;
    const phoneActive = objectPhoneHistory.current.length >= 2 && phoneCount >= 2;

    // EXTRA_PERSON: person count > 1 in >= 2 of last 3 samples
    objectPersonHistory.current.push(data.personCount);
    if (objectPersonHistory.current.length > 3) objectPersonHistory.current.shift();
    const extraPersonCount = objectPersonHistory.current.filter((c) => c > 1).length;
    const extraPersonActive = objectPersonHistory.current.length >= 2 && extraPersonCount >= 2;

    updateSignal('PHONE', phoneActive, now);
    updateSignal('EXTRA_PERSON', extraPersonActive, now);
  };

  const handleSpeechSignal = (isSpeechTriggered: boolean) => {
    const now = Date.now();
    updateSignal('SPEECH', isSpeechTriggered, now);
  };

  const updateSignal = (type: SignalType, conditionActive: boolean, now: number) => {
    const sm = stateMachines.current[type];
    const res = sm.update(conditionActive, now);

    if (res.newActiveEvent) {
      const added = leakyBucket.addWeight(sm.baseWeight + res.patternBonus, sm.family, now);
      console.log(`[Event ${type} ACTIVE] +${added} pts to LeakyBucket`);
    } else if (res.sustainedBonus > 0) {
      const added = leakyBucket.addWeight(res.sustainedBonus, sm.family, now);
      console.log(`[Event ${type} SUSTAINED] +${added} pts to LeakyBucket`);
    }

    setSignalStatuses((prev) => ({
      ...prev,
      [type]: {
        type,
        state: res.state,
        activeDurationMs: res.activeDurationMs,
      },
    }));
  };

  // 1-second interval to decay leaky bucket and refresh UI scores
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      const current = leakyBucket.getScore(now);
      const level = leakyBucket.getLevel(now);
      setRiskScore(current);
      setRiskLevel(level);

      // Also trigger VAD rolling evaluation
      if (vadRef.current) {
        const vadStatus = vadRef.current.evaluateRollingSpeech(now);
        updateSignal('SPEECH', vadStatus.isTriggered, now);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [leakyBucket]);

  const handleCalibrationComplete = useCallback((profile: CalibrationProfile) => {
    calibrationRef.current = profile;
    setCalibration(profile);
    driftTracker.reset();
    stateMachines.current.LOOKING_DOWN.reset();
    stateMachines.current.HEAD_OFF_SCREEN.reset();
    setIsCalibrationOpen(false);
    console.log('Calibration Profile saved:', profile);
  }, [driftTracker]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      {/* Header bar */}
      <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-sky-500/20 border border-sky-400/40 text-sky-400 flex items-center justify-center font-bold">
              v2
            </div>
            <h1 className="text-xl font-bold text-slate-100">Remote Exam Proctor POC — Phase 0</h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Edge-first detection testbench: FaceLandmarker (5 Hz), ObjectDetector (2.5s), Silero VAD (Audio)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setFastDevMode((v) => !v)}
            className={`px-3 py-2 rounded-lg text-xs font-medium border transition ${
              fastDevMode
                ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
            }`}
            title="Toggle between 1s Fast Dev onset and 3-4s Exam Mode onset"
          >
            {fastDevMode ? '⚡ Fast Dev (1s)' : '🛡️ Exam Mode (3-4s)'}
          </button>

          <button
            onClick={() => setIsCalibrationOpen(true)}
            className="px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs transition flex items-center gap-2 shadow-lg shadow-sky-600/20"
          >
            <Compass className="w-4 h-4" />
            {calibration ? 'Re-Calibrate Gaze' : 'Calibrate Gaze (9-Pt + VOR)'}
          </button>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Video Feed & Gauges (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <VideoFeed
            videoRef={videoRef}
            keyPoints={keyPoints}
            objectDetections={objectDetections}
            showMesh={showMesh}
            onToggleMesh={() => setShowMesh(!showMesh)}
            faceCount={faceCount}
          />

          <Gauges
            angles={angles}
            gaze={gaze}
            calibration={calibration}
          />

          <LatencyProfiler
            faceP95={perfStats.faceP95}
            objectP95={perfStats.objectP95}
            faceIntervalMs={perfStats.faceIntervalMs}
            objectIntervalMs={perfStats.objectIntervalMs}
            faceFps={perfStats.faceFps}
            isDegraded={perfStats.isDegraded}
            faceDelegate={workerStatus.faceDelegate}
            objectDelegate={workerStatus.objectDelegate}
          />
        </div>

        {/* Right Column: Leaky Bucket & Signal State Machines (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <LeakyBucketMeter
            score={riskScore}
            level={riskLevel}
            onReset={() => {
              leakyBucket.reset();
              setRiskScore(0);
              setRiskLevel('green');
            }}
          />

          <SignalStatusGrid signals={signalStatuses} />

          <TestRunner activeSignals={activeSignalsSet} />
        </div>
      </div>

      {/* Calibration Modal */}
      <CalibrationModal
        isOpen={isCalibrationOpen}
        onClose={() => setIsCalibrationOpen(false)}
        onComplete={handleCalibrationComplete}
        currentAngles={angles}
        currentGaze={gaze}
        currentTransformMatrix={transformMatrix}
      />

      {/* Live Gaze Dot Overlay (visible after calibration) */}
      {gazePoint && !isCalibrationOpen && (
        <div
          className="fixed pointer-events-none z-40 transition-all duration-75 -translate-x-1/2 -translate-y-1/2"
          style={{
            left: `${Math.max(2, Math.min(98, gazePoint.x * 100))}%`,
            top: `${Math.max(2, Math.min(98, gazePoint.y * 100))}%`,
          }}
        >
          <div
            className={`w-4 h-4 rounded-full border-2 border-white shadow-lg ${
              gazePoint.isOffScreen ? 'bg-rose-500 scale-125 animate-pulse' : 'bg-teal-400/80'
            }`}
          />
        </div>
      )}
    </div>
  );
};