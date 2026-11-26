import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { CameraPipeline, FaceMeshEvent, MouthMotionEvent } from '../lib/camera';
import {
  CalibrationProfile,
  CalibrationSample,
  buildAutoBaselineProfile,
} from '../lib/engine/calibrator';
import {
  SignalType,
  DebounceStateMachine,
  DebounceState,
} from '../lib/engine/state-machine';
import { LeakyBucketEngine, RiskLevel } from '../lib/engine/leaky-bucket';
import { IrisGazeResult } from '../lib/math/iris-gaze';
import { VideoFeed } from '../components/VideoFeed';
import { Gauges } from '../components/Gauges';
import { SignalStatusGrid } from '../components/SignalStatusGrid';
import { ResourceProof } from '../components/ResourceProof';
import { CalibrationModal } from '../components/CalibrationModal';
import { FlagToast, FlagNotification } from '../components/FlagToast';
import { TestRunner } from './test-runner';
import { Compass, Maximize2, RotateCcw, Droplets, ChevronDown, ChevronUp } from 'lucide-react';

interface PerformanceMemory {
  usedJSHeapSize: number;
}

export const DiagnosticPlayground: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pipelineRef = useRef<CameraPipeline | null>(null);

  const [calibration, setCalibration] = useState<CalibrationProfile | null>(null);
  const calibrationRef = useRef<CalibrationProfile | null>(null);
  calibrationRef.current = calibration;

  const [isCalibrationOpen, setIsCalibrationOpen] = useState(false);
  const [fastDevMode, setFastDevMode] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showTestRunner, setShowTestRunner] = useState(false);
  const wasFullscreenRef = useRef(false);

  const [workerDelegate, setWorkerDelegate] = useState('CPU');
  const [faceCount, setFaceCount] = useState(0);
  const [angles, setAngles] = useState<{ pitch: number; yaw: number; roll: number } | null>(null);
  const [transformMatrix, setTransformMatrix] = useState<number[] | undefined>(undefined);
  const [gaze, setGaze] = useState<IrisGazeResult | null>(null);
  const [mouth, setMouth] = useState<MouthMotionEvent | null>(null);
  const [keyPoints, setKeyPoints] = useState<Array<{ x: number; y: number; z: number }> | null>(null);
  const [showMesh, setShowMesh] = useState(false);

  const [perfStats, setPerfStats] = useState({
    faceP95: 0,
    faceAvg: 0,
    faceLast: 0,
    cpuDutyCyclePercent: 0,
    faceIntervalMs: 200,
    faceFps: 5,
    isDegraded: false,
  });
  const [jsHeapMb, setJsHeapMb] = useState<number | null>(null);
  const [mainThreadLagMs, setMainThreadLagMs] = useState(0);

  const leakyBucket = useMemo(() => new LeakyBucketEngine(), []);
  const [leakyBucketEnabled, setLeakyBucketEnabled] = useState(true);
  const [riskScore, setRiskScore] = useState(0);
  const [riskLevel, setRiskLevel] = useState<RiskLevel>('green');

  const [notifications, setNotifications] = useState<FlagNotification[]>([]);
  const nextToastIdRef = useRef(1);

  const pushToast = useCallback((signal: SignalType, points: number) => {
    const id = nextToastIdRef.current++;
    setNotifications((prev) => [
      ...prev.slice(-2),
      { id, signal, points, timestamp: Date.now() },
    ]);
    setTimeout(() => {
      setNotifications((prev) => prev.filter((item) => item.id !== id));
    }, 3500);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setNotifications((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const stateMachines = useRef<Record<SignalType, DebounceStateMachine>>({
    NO_FACE: new DebounceStateMachine('NO_FACE'),
    EXTRA_FACE: new DebounceStateMachine('EXTRA_FACE'),
    HEAD_OFF_SCREEN: new DebounceStateMachine('HEAD_OFF_SCREEN'),
    LOOKING_DOWN: new DebounceStateMachine('LOOKING_DOWN'),
    SPEAKING_VISUAL: new DebounceStateMachine('SPEAKING_VISUAL'),
    FROZEN_FEED: new DebounceStateMachine('FROZEN_FEED'),
    TAB_SWITCH: new DebounceStateMachine('TAB_SWITCH'),
    FULLSCREEN_EXIT: new DebounceStateMachine('FULLSCREEN_EXIT'),
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
    SPEAKING_VISUAL: { type: 'SPEAKING_VISUAL', state: 'IDLE', activeDurationMs: 0 },
    FROZEN_FEED: { type: 'FROZEN_FEED', state: 'IDLE', activeDurationMs: 0 },
    TAB_SWITCH: { type: 'TAB_SWITCH', state: 'IDLE', activeDurationMs: 0 },
    FULLSCREEN_EXIT: { type: 'FULLSCREEN_EXIT', state: 'IDLE', activeDurationMs: 0 },
  });

  const noFaceHistory = useRef<boolean[]>([]);
  const extraFaceHistory = useRef<boolean[]>([]);
  const warmupSamplesRef = useRef<CalibrationSample[]>([]);

  const activeSignalsSet = useMemo(() => {
    const s = new Set<SignalType>();
    for (const [k, v] of Object.entries(signalStatuses) as [SignalType, typeof signalStatuses[SignalType]][]) {
      if (v.state === 'ACTIVE') s.add(k);
    }
    return s;
  }, [signalStatuses]);

  const updateSignal = useCallback((type: SignalType, conditionActive: boolean, now: number) => {
    const sm = stateMachines.current[type];
    const res = sm.update(conditionActive, now);

    if (res.newActiveEvent) {
      const added = leakyBucket.addWeight(sm.baseWeight + res.patternBonus, sm.family, now);
      pushToast(type, added);
    } else if (res.sustainedBonus > 0) {
      leakyBucket.addWeight(res.sustainedBonus, sm.family, now);
    }

    setSignalStatuses((prev) => ({
      ...prev,
      [type]: {
        type,
        state: res.state,
        activeDurationMs: res.activeDurationMs,
      },
    }));
  }, [leakyBucket, pushToast]);

  // Sync Fast Dev Mode (1s onset) vs Exam Mode (3s/4s onset)
  useEffect(() => {
    stateMachines.current.LOOKING_DOWN.onsetMs = fastDevMode ? 1000 : 3000;
    stateMachines.current.HEAD_OFF_SCREEN.onsetMs = fastDevMode ? 1000 : 4000;
    stateMachines.current.SPEAKING_VISUAL.onsetMs = fastDevMode ? 1200 : 2000;
  }, [fastDevMode]);

  // Browser events: TAB_SWITCH & FULLSCREEN_EXIT
  useEffect(() => {
    const checkFocus = () => {
      const lostFocus = document.hidden || !document.hasFocus();
      updateSignal('TAB_SWITCH', lostFocus, Date.now());
    };

    const handleFullscreenChange = () => {
      const active = Boolean(document.fullscreenElement);
      setIsFullscreen(active);
      if (active) {
        wasFullscreenRef.current = true;
        updateSignal('FULLSCREEN_EXIT', false, Date.now());
      } else if (wasFullscreenRef.current) {
        updateSignal('FULLSCREEN_EXIT', true, Date.now());
      }
    };

    document.addEventListener('visibilitychange', checkFocus);
    window.addEventListener('blur', checkFocus);
    window.addEventListener('focus', checkFocus);
    document.addEventListener('fullscreenchange', handleFullscreenChange);

    return () => {
      document.removeEventListener('visibilitychange', checkFocus);
      window.removeEventListener('blur', checkFocus);
      window.removeEventListener('focus', checkFocus);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [updateSignal]);

  // Initialize Camera + FaceLandmarker Worker (CPU WASM @ 5 Hz)
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const pipeline = new CameraPipeline();
    pipelineRef.current = pipeline;

    pipeline.init(video, {
      onWorkerReady: (delegate) => {
        setWorkerDelegate(delegate);
      },
      onFaceResult: (data: FaceMeshEvent) => {
        const now = Date.now();
        setFaceCount(data.faceCount);
        setAngles(data.angles);
        setTransformMatrix(data.transformMatrix);
        setGaze(data.gaze);
        setMouth(data.mouth);
        setKeyPoints(data.keyPoints);
        setPerfStats(pipeline.performance.getStats());

        // 1. NO_FACE (at 4 Hz, 14 frames ~ 3.5s)
        noFaceHistory.current.push(data.faceCount === 0);
        if (noFaceHistory.current.length > 14) noFaceHistory.current.shift();
        const noFaceCount = noFaceHistory.current.filter(Boolean).length;
        const noFaceActive = noFaceHistory.current.length >= 8 && noFaceCount / noFaceHistory.current.length >= 0.9;

        // 2. EXTRA_FACE
        extraFaceHistory.current.push(data.faceCount >= 2);
        if (extraFaceHistory.current.length > 5) extraFaceHistory.current.shift();
        const extraFaceCount = extraFaceHistory.current.filter(Boolean).length;
        const extraFaceActive = extraFaceHistory.current.length >= 3 && extraFaceCount >= 2;

        // 3. FROZEN_FEED
        const isFrozen = data.frameDiff < 0.5;

        // 4. SPEAKING_VISUAL (lip oscillation without mic)
        const isSpeakingVisual = Boolean(data.mouth?.isSpeaking);

        updateSignal('NO_FACE', noFaceActive, now);
        updateSignal('EXTRA_FACE', extraFaceActive, now);
        updateSignal('FROZEN_FEED', isFrozen, now);
        updateSignal('SPEAKING_VISUAL', isSpeakingVisual, now);

        if (data.faceCount === 0 || !data.angles) return;

        // Auto-baseline warmup on first 10 valid frames
        let cal = calibrationRef.current;
        if (!cal && data.faceCount === 1 && data.gaze) {
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

        // 5. HEAD_OFF_SCREEN & LOOKING_DOWN from calibrated 3D head pose (immune to 2px iris noise)
        const neutralYaw = cal?.neutralYaw ?? 0;
        const neutralPitch = cal?.neutralPitch ?? 0;
        const yawMin = cal ? Math.max(cal.yawMin, neutralYaw - 14) : -14;
        const yawMax = cal ? Math.min(cal.yawMax, neutralYaw + 14) : 14;
        const pitchFloor = cal ? Math.max(cal.pitchMin, neutralPitch - 10) : -10;
        const pitchMax = cal ? Math.min(cal.pitchMax, neutralPitch + 16) : 16;

        const isHeadOffScreen =
          data.angles.yaw < yawMin ||
          data.angles.yaw > yawMax ||
          data.angles.pitch > pitchMax;
        const isLookingDown = data.angles.pitch < pitchFloor;

        updateSignal('HEAD_OFF_SCREEN', isHeadOffScreen, now);
        updateSignal('LOOKING_DOWN', isLookingDown, now);
      },
      onError: (err) => {
        console.error('FaceLandmarker worker error:', err);
      },
    }).catch((err) => console.error('Failed to init pipeline:', err));

    return () => {
      pipeline.stop();
    };
  }, [updateSignal]);

  // Poll risk score, tab focus, JS heap, and main thread lag every 500ms
  useEffect(() => {
    let lastTick = performance.now();
    const interval = setInterval(() => {
      const nowMs = Date.now();
      const perfNow = performance.now();
      const lag = Math.max(0, perfNow - lastTick - 500);
      lastTick = perfNow;
      setMainThreadLagMs(lag);

      const mem = (performance as unknown as { memory?: PerformanceMemory }).memory;
      if (mem && typeof mem.usedJSHeapSize === 'number') {
        setJsHeapMb(mem.usedJSHeapSize / (1024 * 1024));
      }

      setRiskScore(leakyBucket.getScore(nowMs));
      setRiskLevel(leakyBucket.getLevel(nowMs));
      const lostFocus = document.hidden || !document.hasFocus();
      updateSignal('TAB_SWITCH', lostFocus, nowMs);
    }, 500);

    return () => clearInterval(interval);
  }, [leakyBucket, updateSignal]);

  const handleCalibrationComplete = useCallback((profile: CalibrationProfile) => {
    calibrationRef.current = profile;
    setCalibration(profile);
    stateMachines.current.LOOKING_DOWN.reset();
    stateMachines.current.HEAD_OFF_SCREEN.reset();
    setIsCalibrationOpen(false);
  }, []);

  const toggleLeakyBucket = () => {
    const next = !leakyBucketEnabled;
    leakyBucket.setEnabled(next, Date.now());
    setLeakyBucketEnabled(next);
  };

  const resetScore = () => {
    leakyBucket.reset();
    setRiskScore(0);
    setRiskLevel('green');
  };

  const toggleFullscreen = async () => {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  };

  const riskBadge = {
    green: { dot: 'bg-emerald-400', text: 'text-emerald-400', pill: 'border-emerald-500/30 bg-emerald-500/10', label: 'Normal' },
    amber: { dot: 'bg-amber-400', text: 'text-amber-400', pill: 'border-amber-500/40 bg-amber-500/10', label: 'Review' },
    red: { dot: 'bg-rose-400', text: 'text-rose-400', pill: 'border-rose-500/50 bg-rose-500/15', label: 'High risk' },
  }[riskLevel];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top navigation bar with embedded scoreboard */}
      <header className="sticky top-0 z-30 bg-slate-950/90 backdrop-blur-md border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 flex items-center justify-center text-xs font-bold">
              FL
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-100 leading-tight">
                Exam proctor
              </h1>
              <p className="text-[11px] text-slate-400 leading-tight">
                FaceLandmarker {perfStats.faceFps} Hz, 480x360 capture
              </p>
            </div>
          </div>

          {/* Scoreboard + Leaky bucket toggle in top nav */}
          <div className="flex items-center gap-2 flex-wrap">
            <div className={`flex items-center gap-2.5 px-3 py-1.5 rounded-lg border ${riskBadge.pill}`}>
              <span className={`w-2 h-2 rounded-full ${riskBadge.dot}`} />
              <span className="text-xs font-mono font-bold text-slate-100">
                {riskScore.toFixed(1)} pts
              </span>
              <span className={`text-[11px] font-medium ${riskBadge.text}`}>
                {riskBadge.label}
              </span>
              <button
                onClick={resetScore}
                className="ml-1 text-slate-400 hover:text-slate-200 transition"
                title="Reset score"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            <button
              onClick={toggleLeakyBucket}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border transition flex items-center gap-1.5 ${
                leakyBucketEnabled
                  ? 'bg-sky-500/15 border-sky-500/40 text-sky-300'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
              }`}
              title="Toggle 120s decay and cross-family multiplier. When off, flag points are halved."
            >
              <Droplets className="w-3.5 h-3.5" />
              {leakyBucketEnabled ? 'Leaky bucket: On' : 'Leaky bucket: Off (0.5x)'}
            </button>
          </div>

          {/* Action controls */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={() => setFastDevMode((v) => !v)}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border transition ${
                fastDevMode
                  ? 'bg-amber-500/15 border-amber-500/40 text-amber-300'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {fastDevMode ? 'Fast dev (1s)' : 'Exam mode (3-4s)'}
            </button>

            <button
              onClick={toggleFullscreen}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border transition flex items-center gap-1.5 ${
                isFullscreen
                  ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
              }`}
            >
              <Maximize2 className="w-3.5 h-3.5" />
              {isFullscreen ? 'Fullscreen' : 'Enter fullscreen'}
            </button>

            <button
              onClick={() => setIsCalibrationOpen(true)}
              className="px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs transition flex items-center gap-1.5"
            >
              <Compass className="w-3.5 h-3.5" />
              {calibration && !calibration.isAutoBaseline ? 'Recalibrate' : 'Calibrate'}
            </button>
          </div>
        </div>
      </header>

      {/* Main uncluttered workspace */}
      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          <div className="lg:col-span-7 space-y-5">
            <VideoFeed
              videoRef={videoRef}
              keyPoints={keyPoints}
              objectDetections={[]}
              showMesh={showMesh}
              onToggleMesh={() => setShowMesh(!showMesh)}
              faceCount={faceCount}
            />

            <ResourceProof
              faceAvg={perfStats.faceAvg}
              faceP95={perfStats.faceP95}
              faceIntervalMs={perfStats.faceIntervalMs}
              faceFps={perfStats.faceFps}
              cpuDutyCyclePercent={perfStats.cpuDutyCyclePercent}
              isDegraded={perfStats.isDegraded}
              delegate={workerDelegate}
              jsHeapMb={jsHeapMb}
              mainThreadLagMs={mainThreadLagMs}
            />

            <Gauges
              angles={angles}
              gaze={gaze}
              mouth={mouth}
              calibration={calibration}
            />
          </div>

          <div className="lg:col-span-5 space-y-5">
            <SignalStatusGrid
              signals={signalStatuses}
              leakyBucketEnabled={leakyBucketEnabled}
            />

            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
              <button
                onClick={() => setShowTestRunner((v) => !v)}
                className="w-full px-4 py-3 flex items-center justify-between text-xs font-semibold text-slate-300 hover:bg-slate-800/40 transition"
              >
                <span>Scripted checks (7)</span>
                {showTestRunner ? (
                  <ChevronUp className="w-4 h-4 text-slate-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-slate-400" />
                )}
              </button>
              {showTestRunner && (
                <div className="border-t border-slate-800">
                  <TestRunner activeSignals={activeSignalsSet} />
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <FlagToast notifications={notifications} onDismiss={dismissToast} />

      <CalibrationModal
        isOpen={isCalibrationOpen}
        onClose={() => setIsCalibrationOpen(false)}
        onComplete={handleCalibrationComplete}
        currentAngles={angles}
        currentGaze={gaze}
        currentTransformMatrix={transformMatrix}
      />
    </div>
  );
};
