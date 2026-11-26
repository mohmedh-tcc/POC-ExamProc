import React, { useState, useEffect, useRef } from 'react';
import {
  CALIBRATION_STEPS,
  CALIBRATION_TARGETS,
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
  currentTransformMatrix?: number[];
}

export const CalibrationModal: React.FC<CalibrationModalProps> = ({
  isOpen,
  onClose,
  onComplete,
  currentAngles,
  currentGaze,
  currentTransformMatrix,
}) => {
  const [stepIndex, setStepIndex] = useState(0);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [remainingMs, setRemainingMs] = useState(CALIBRATION_STEPS[0].durationMs);
  const samplesRef = useRef<CalibrationSample[]>([]);

  // Keep onComplete in a ref so parent re-renders never reset our timer
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const currentStep = CALIBRATION_STEPS[stepIndex];
  const targetCoords = currentStep ? CALIBRATION_TARGETS[currentStep.point] : { x: 0.5, y: 0.5 };
  const isVorStep = false;

  useEffect(() => {
    if (!isOpen) {
      setIsCalibrating(false);
      setStepIndex(0);
      setRemainingMs(CALIBRATION_STEPS[0].durationMs);
      samplesRef.current = [];
    }
  }, [isOpen]);

  // Frame sampling during calibration
  useEffect(() => {
    if (!isCalibrating || !currentAngles) return;

    samplesRef.current.push({
      yaw: currentAngles.yaw,
      pitch: currentAngles.pitch,
      roll: currentAngles.roll,
      hRatio: currentGaze?.hRatio ?? 0.5,
      vRatio: currentGaze?.vRatio ?? 0.5,
      dx: currentGaze?.dx,
      dy: currentGaze?.dy,
      transformMatrix: currentTransformMatrix,
      point: CALIBRATION_STEPS[stepIndex]?.point,
    });
  }, [isCalibrating, currentAngles, currentGaze, currentTransformMatrix, stepIndex]);

  // Timer loop for step progression (100ms tick for accurate sub-second durations)
  useEffect(() => {
    if (!isCalibrating) return;

    let step = 0;
    let timeLeft = CALIBRATION_STEPS[0].durationMs;
    setStepIndex(0);
    setRemainingMs(timeLeft);

    const timer = setInterval(() => {
      timeLeft -= 100;

      if (timeLeft <= 0) {
        step += 1;
        if (step < CALIBRATION_STEPS.length) {
          setStepIndex(step);
          timeLeft = CALIBRATION_STEPS[step].durationMs;
          setRemainingMs(timeLeft);
        } else {
          clearInterval(timer);
          setIsCalibrating(false);
          setRemainingMs(0);
          const profile = buildCalibrationProfile(samplesRef.current);
          onCompleteRef.current(profile);
        }
      } else {
        setRemainingMs(timeLeft);
      }
    }, 100);

    return () => clearInterval(timer);
  }, [isCalibrating]);

  if (!isOpen) return null;

  const countdownSec = Math.max(1, Math.ceil(remainingMs / 1000));

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-4">
      {/* Target visual dot positioned at exact normalized screen coordinates */}
      {isCalibrating && currentStep && (
        <div
          className="fixed -translate-x-1/2 -translate-y-1/2 flex flex-col items-center pointer-events-none transition-all duration-300 z-50"
          style={{
            left: `${targetCoords.x * 100}%`,
            top: `${targetCoords.y * 100}%`,
          }}
        >
          <div className="relative flex items-center justify-center">
            <span
              className={`w-14 h-14 rounded-full animate-ping absolute ${
                isVorStep ? 'bg-amber-500/35' : 'bg-sky-500/30'
              }`}
            />
            <span
              className={`w-10 h-10 rounded-full border-2 border-white flex items-center justify-center text-slate-950 font-extrabold text-sm shadow-xl ${
                isVorStep ? 'bg-amber-400' : 'bg-sky-400'
              }`}
            >
              {countdownSec}s
            </span>
          </div>
          <span
            className={`text-xs font-semibold mt-2 bg-slate-900/95 px-2.5 py-1 rounded-md border shadow-md whitespace-nowrap ${
              isVorStep ? 'text-amber-300 border-amber-500/50' : 'text-sky-300 border-slate-700'
            }`}
          >
            {isVorStep ? currentStep.label : 'Look here'}
          </span>
        </div>
      )}

      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl text-center space-y-5">
        <div className="w-12 h-12 rounded-xl bg-sky-500/20 border border-sky-500/30 text-sky-400 mx-auto flex items-center justify-center">
          <Target className="w-6 h-6" />
        </div>

        <div>
          <h2 className="text-lg font-bold text-slate-100">
            {isCalibrating
              ? `Step ${stepIndex + 1} of ${CALIBRATION_STEPS.length}`
              : '5-point gaze & pose calibration'}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            {isCalibrating
              ? currentStep?.label
              : 'Look directly at each of the 5 screen points (center and 4 corners) with natural head posture so the model learns your screen bounds.'}
          </p>
        </div>

        {isCalibrating && (
          <div className="flex justify-center gap-1.5 flex-wrap">
            {CALIBRATION_STEPS.map((s, idx) => (
              <div
                key={s.point}
                className={`w-3 h-3 rounded-full transition-all ${
                  idx < stepIndex
                    ? 'bg-emerald-400'
                    : idx === stepIndex
                    ? s.point.startsWith('vor')
                      ? 'bg-amber-400 ring-2 ring-amber-400/50 scale-125'
                      : 'bg-sky-400 ring-2 ring-sky-400/50 scale-125'
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
                  setRemainingMs(CALIBRATION_STEPS[0].durationMs);
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
