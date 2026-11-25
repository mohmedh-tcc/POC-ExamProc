import React, { useState, useEffect, useRef } from 'react';
import {
  CALIBRATION_STEPS,
  CalibrationSample,
  CalibrationProfile,
  buildCalibrationProfile,
} from '../lib/engine/calibrator';
import { EulerAngles } from '../lib/math/transform-matrix';
import { IrisGazeResult } from '../lib/math/iris-gaze';
import { Target } from 'lucide-react';

interface CalibrationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: (profile: CalibrationProfile) => void;
  currentAngles: EulerAngles | null;
  currentGaze: IrisGazeResult | null;
}

export const CalibrationModal: React.FC<CalibrationModalProps> = ({
  isOpen,
  onClose,
  onComplete,
  currentAngles,
  currentGaze,
}) => {
  const [stepIndex, setStepIndex] = useState(0);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [countdown, setCountdown] = useState(2);
  const samplesRef = useRef<CalibrationSample[]>([]);

  // Keep onComplete in a ref so changes in parent render do not reset our countdown interval
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const currentStep = CALIBRATION_STEPS[stepIndex];

  // Position styles for target indicator
  const targetPositions: Record<string, string> = {
    'center': 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2',
    'top-left': 'top-12 left-12',
    'top-right': 'top-12 right-12',
    'bottom-left': 'bottom-12 left-12',
    'bottom-right': 'bottom-12 right-12',
  };

  useEffect(() => {
    if (!isOpen) {
      setIsCalibrating(false);
      setStepIndex(0);
      setCountdown(2);
      samplesRef.current = [];
    }
  }, [isOpen]);

  // Frame sampling during calibration
  useEffect(() => {
    if (!isCalibrating || !currentAngles || !currentGaze) return;

    if (!currentGaze.isBlinking) {
      samplesRef.current.push({
        yaw: currentAngles.yaw,
        pitch: currentAngles.pitch,
        hRatio: currentGaze.hRatio,
        vRatio: currentGaze.vRatio,
        point: CALIBRATION_STEPS[stepIndex]?.point,
      });
    }
  }, [isCalibrating, currentAngles, currentGaze, stepIndex]);

  // Rock-solid countdown and step progression timer
  useEffect(() => {
    if (!isCalibrating) return;

    let step = 0;
    let timeRemaining = 2; // 2 seconds per point
    setStepIndex(0);
    setCountdown(2);

    const timer = setInterval(() => {
      timeRemaining -= 1;

      if (timeRemaining <= 0) {
        step += 1;
        if (step < CALIBRATION_STEPS.length) {
          setStepIndex(step);
          timeRemaining = 2;
          setCountdown(2);
        } else {
          // All 5 points finished!
          clearInterval(timer);
          setIsCalibrating(false);
          setCountdown(0);
          const profile = buildCalibrationProfile(samplesRef.current);
          onCompleteRef.current(profile);
        }
      } else {
        setCountdown(timeRemaining);
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [isCalibrating]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-4">
      {/* Target visual dot that moves across screen */}
      {isCalibrating && currentStep && (
        <div className={`absolute ${targetPositions[currentStep.point]} flex flex-col items-center pointer-events-none transition-all duration-300`}>
          <div className="relative flex items-center justify-center">
            <span className="w-14 h-14 rounded-full bg-sky-500/30 animate-ping absolute" />
            <span className="w-10 h-10 rounded-full bg-sky-400 border-2 border-white flex items-center justify-center text-slate-950 font-extrabold text-sm shadow-xl">
              {countdown}s
            </span>
          </div>
          <span className="text-xs font-semibold text-sky-300 mt-2 bg-slate-900/90 px-2.5 py-1 rounded-md border border-slate-700 shadow-md">
            Look here
          </span>
        </div>
      )}

      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl text-center space-y-5">
        <div className="w-12 h-12 rounded-xl bg-sky-500/20 border border-sky-500/30 text-sky-400 mx-auto flex items-center justify-center">
          <Target className="w-6 h-6" />
        </div>

        <div>
          <h2 className="text-lg font-bold text-slate-100">
            {isCalibrating ? `Step ${stepIndex + 1} of ${CALIBRATION_STEPS.length}` : 'Personal Gaze & Pose Calibration'}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            {isCalibrating
              ? currentStep?.label
              : 'Sit upright in your normal exam posture. The system will guide you through 5 calibration points (Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right) for 2 seconds each to record your personal normal bounding baseline.'}
          </p>
        </div>

        {isCalibrating && (
          <div className="flex justify-center gap-2">
            {CALIBRATION_STEPS.map((s, idx) => (
              <div
                key={s.point}
                className={`w-3.5 h-3.5 rounded-full transition-all ${
                  idx < stepIndex
                    ? 'bg-emerald-400'
                    : idx === stepIndex
                    ? 'bg-sky-400 ring-2 ring-sky-400/50 scale-125'
                    : 'bg-slate-800'
                }`}
              />
            ))}
          </div>
        )}

        <div className="flex justify-center gap-3 pt-3 border-t border-slate-800">
          {!isCalibrating ? (
            <>
              <button
                onClick={onClose}
                className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300 transition"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  samplesRef.current = [];
                  setStepIndex(0);
                  setCountdown(2);
                  setIsCalibrating(true);
                }}
                className="px-5 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 text-xs font-semibold text-slate-950 transition flex items-center gap-2 shadow-lg shadow-sky-500/20"
              >
                Start 5-Point Calibration
              </button>
            </>
          ) : (
            <button
              onClick={() => {
                setIsCalibrating(false);
                onClose();
              }}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-400 transition"
            >
              Abort Calibration
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
