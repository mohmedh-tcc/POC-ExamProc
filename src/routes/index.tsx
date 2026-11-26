import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { CameraPipeline, FaceDetectorEvent } from '../lib/camera';
import { CalibrationProfile, CalibrationSample, buildCalibrationProfile } from '../lib/engine/calibrator';
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
import { Compass, Maximize2 } from 'lucide-react';

export const DiagnosticPlayground: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pipelineRef = useRef<CameraPipeline | null>(null);

  const [calibration, setCalibration] = useState<CalibrationProfile | null>(null);
  const calibrationRef = useRef<CalibrationProfile | null>(null);
  calibrationRef.current = calibration;
  const [isCalibrationOpen, setIsCalibrationOpen] = useState(false);
  const [fastDevMode, setFastDevMode] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const wasFullscreenRef = useRef(false);

  const [workerDelegate, setWorkerDelegate] = useState('Initializing...');
  const [faceCount, setFaceCount] = useState(0);
  const [angles, setAngles] = useState<{ pitch: number; yaw: number; roll: number } | null>(null);
  const [keyPoints, setKeyPoints] = useState<Array<{ x: number; y: number; z: number }> | null>(null);
  const [showMesh, setShowMesh] = useState(false); // Off by default to save GPU compositor

  const [perfStats, setPerfStats] = useState({
    faceP95: 0,
    objectP95: 0,
    faceIntervalMs: 667,
    objectIntervalMs: 0,
    faceFps: 1.5,
    isDegraded: false,
  });

  const leakyBucket = useMemo(() => new LeakyBucketEngine(), []);
  const [riskScore, setRiskScore] = useState(0);
  const [riskLevel, setRiskLevel] = useState<RiskLevel>('green');

  const stateMachines = useRef<Record<SignalType, DebounceStateMachine>>({
    NO_FACE: new DebounceStateMachine('NO_FACE'),
    EXTRA_FACE: new DebounceStateMachine('EXTRA_FACE'),
    HEAD_OFF_SCREEN: new DebounceStateMachine('HEAD_OFF_SCREEN'),
    LOOKING_DOWN: new DebounceStateMachine('LOOKING_DOWN'),
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
  }, [leakyBucket]);

  // Sync Fast Dev Mode (1s onset) vs Exam Mode (3s/4s onset)
  useEffect(() => {
    stateMachines.current.LOOKING_DOWN.onsetMs = fastDevMode ? 1000 : 3000;
    stateMachines.current.HEAD_OFF_SCREEN.onsetMs = fastDevMode ? 1000 : 4000;
  }, [fastDevMode]);

  // Browser events: TAB_SWITCH (visibilitychange / blur) & FULLSCREEN_EXIT (Fullscreen API)
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

  // Initialize Camera + BlazeFace FaceDetector Worker
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const pipeline = new CameraPipeline();
    pipelineRef.current = pipeline;

    pipeline.init(video, {
      onWorkerReady: (delegate) => {
        setWorkerDelegate(delegate);
      },
      onFaceResult: (data: FaceDetectorEvent) => {
        const now = Date.now();
        setFaceCount(data.faceCount);
        setAngles({ pitch: data.pitchDeg, yaw: data.yawDeg, roll: 0 });
        setKeyPoints(data.keypoints ? data.keypoints.map((k) => ({ x: k.x, y: k.y, z: 0 })) : null);
        setPerfStats(pipeline.performance.getStats());

        // 1. NO_FACE (at 1.5 Hz, 6 samples ~ 4s)
        noFaceHistory.current.push(data.faceCount === 0);
        if (noFaceHistory.current.length > 6) noFaceHistory.current.shift();
        const noFaceCount = noFaceHistory.current.filter(Boolean).length;
        const noFaceActive = noFaceHistory.current.length >= 4 && noFaceCount / noFaceHistory.current.length >= 0.9;

        // 2. EXTRA_FACE (>= 2 faces in >= 2 of last 3 samples)
        extraFaceHistory.current.push(data.faceCount >= 2);
        if (extraFaceHistory.current.length > 3) extraFaceHistory.current.shift();
        const extraFaceCount = extraFaceHistory.current.filter(Boolean).length;
        const extraFaceActive = extraFaceHistory.current.length >= 2 && extraFaceCount >= 2;

        // 3. FROZEN_FEED
        const isFrozen = data.frameDiff < 0.5;

        updateSignal('NO_FACE', noFaceActive, now);
        updateSignal('EXTRA_FACE', extraFaceActive, now);
        updateSignal('FROZEN_FEED', isFrozen, now);

        // Auto-baseline warmup on first 6 valid frames
        let cal = calibrationRef.current;
        if (!cal && data.faceCount === 1) {
          warmupSamplesRef.current.push({
            yaw: data.yawDeg,
            pitch: data.pitchDeg,
            hRatio: 0.5,
            vRatio: 0.5,
          });
          if (warmupSamplesRef.current.length >= 6) {
            const autoProfile = buildCalibrationProfile(warmupSamplesRef.current);
            calibrationRef.current = autoProfile;
            setCalibration(autoProfile);
            cal = autoProfile;
          }
        }

        if (data.faceCount === 0) return;

        // 4. HEAD_OFF_SCREEN & LOOKING_DOWN from BlazeFace 6-keypoint coarse pose
        // Enforce tight lab-exam focus bounds around neutral (max +-14 deg yaw, -10 deg pitch down)
        const neutralYaw = cal?.neutralYaw ?? 0;
        const neutralPitch = cal?.neutralPitch ?? 0;
        const yawMin = cal ? Math.max(cal.yawMin, neutralYaw - 14) : -14;
        const yawMax = cal ? Math.min(cal.yawMax, neutralYaw + 14) : 14;
        const pitchFloor = cal ? Math.max(cal.pitchMin, neutralPitch - 10) : -10;
        const pitchMax = cal ? Math.min(cal.pitchMax, neutralPitch + 16) : 16;

        const isHeadOffScreen = data.yawDeg < yawMin || data.yawDeg > yawMax || data.pitchDeg > pitchMax;
        const isLookingDown = data.pitchDeg < pitchFloor;

        updateSignal('HEAD_OFF_SCREEN', isHeadOffScreen, now);
        updateSignal('LOOKING_DOWN', isLookingDown, now);
      },
      onError: (err) => {
        console.error('FaceDetector Worker error:', err);
      },
    }).catch((err) => console.error('Failed to init pipeline:', err));

    return () => {
      pipeline.stop();
    };
  }, [updateSignal]);

  // Decay leaky bucket every 500ms & poll tab focus
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      setRiskScore(leakyBucket.getScore(now));
      setRiskLevel(leakyBucket.getLevel(now));
      const lostFocus = document.hidden || !document.hasFocus();
      updateSignal('TAB_SWITCH', lostFocus, now);
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

  const toggleFullscreen = async () => {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-400/40 text-emerald-400 flex items-center justify-center font-bold">
              FD
            </div>
            <h1 className="text-xl font-bold text-slate-100">
              Lab Exam Proctor POC — Lightweight FaceDetector
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            BlazeFace 6-Keypoint (1.5 Hz CPU WASM) • 320×240 Capture • Fullscreen &amp; Tab-Switch API
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={() => setFastDevMode((v) => !v)}
            className={`px-3 py-2 rounded-lg text-xs font-medium border transition ${
              fastDevMode
                ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
            }`}
          >
            {fastDevMode ? '⚡ Fast Dev (1s)' : '🛡️ Exam Mode (3-4s)'}
          </button>

          <button
            onClick={toggleFullscreen}
            className={`px-3 py-2 rounded-lg text-xs font-medium border transition flex items-center gap-1.5 ${
              isFullscreen
                ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <Maximize2 className="w-3.5 h-3.5" />
            {isFullscreen ? 'Fullscreen Active' : 'Enter Fullscreen'}
          </button>

          <button
            onClick={() => setIsCalibrationOpen(true)}
            className="px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs transition flex items-center gap-2 shadow-lg shadow-sky-600/20"
          >
            <Compass className="w-4 h-4" />
            {calibration ? 'Re-Calibrate Pose' : 'Calibrate Pose (5-Point)'}
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 space-y-6">
          <VideoFeed
            videoRef={videoRef}
            keyPoints={keyPoints}
            objectDetections={[]}
            showMesh={showMesh}
            onToggleMesh={() => setShowMesh(!showMesh)}
            faceCount={faceCount}
          />

          <Gauges angles={angles} gaze={null} calibration={calibration} />

          <LatencyProfiler
            faceP95={perfStats.faceP95}
            objectP95={0}
            faceIntervalMs={perfStats.faceIntervalMs}
            objectIntervalMs={0}
            faceFps={perfStats.faceFps}
            isDegraded={perfStats.isDegraded}
            faceDelegate={workerDelegate}
            objectDelegate="DISABLED"
          />
        </div>

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

      <CalibrationModal
        isOpen={isCalibrationOpen}
        onClose={() => setIsCalibrationOpen(false)}
        onComplete={handleCalibrationComplete}
        currentAngles={angles}
        currentGaze={{
          hRatio: 0.5,
          vRatio: 0.5,
          isBlinking: false,
          rightOpenness: 0.3,
          leftOpenness: 0.3,
          dx: 0,
          dy: 0,
          eyeYawRad: 0,
          eyePitchRad: 0,
          eyeVector: [0, 0, 1],
          confidence: 1,
        }}
      />
    </div>
  );
};
