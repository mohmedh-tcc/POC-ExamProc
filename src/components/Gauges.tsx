import React from 'react';
import { EulerAngles } from '../lib/math/transform-matrix';
import { IrisGazeResult } from '../lib/math/iris-gaze';
import { MouthMotionEvent } from '../lib/camera';
import { CalibrationProfile } from '../lib/engine/calibrator';

interface GaugesProps {
  angles: EulerAngles | null;
  gaze: IrisGazeResult | null;
  mouth: MouthMotionEvent | null;
  calibration: CalibrationProfile | null;
}

export const Gauges: React.FC<GaugesProps> = ({ angles, gaze, mouth, calibration }) => {
  const pitch = angles?.pitch ?? 0;
  const yaw = angles?.yaw ?? 0;
  const hRatio = gaze?.hRatio ?? 0.5;
  const vRatio = gaze?.vRatio ?? 0.5;
  const mar = mouth?.mar ?? 0;
  const isSpeaking = mouth?.isSpeaking ?? false;

  const neutralYaw = calibration?.neutralYaw ?? 0;
  const neutralPitch = calibration?.neutralPitch ?? 0;
  const yawMin = calibration ? Math.max(calibration.yawMin, neutralYaw - 14) : -14;
  const yawMax = calibration ? Math.min(calibration.yawMax, neutralYaw + 14) : 14;
  const pitchFloor = calibration ? Math.max(calibration.pitchMin, neutralPitch - 10) : -10;

  const yawOut = yaw < yawMin || yaw > yawMax;
  const pitchDown = pitch < pitchFloor;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3.5">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <h3 className="text-xs font-semibold text-slate-200 tracking-wide">
          Pose, gaze &amp; mouth
        </h3>
        <div className="flex items-center gap-2 text-[11px]">
          <span className={`w-2 h-2 rounded-full ${isSpeaking ? 'bg-pink-400 animate-pulse' : 'bg-emerald-400'}`} />
          <span className="text-slate-400">{isSpeaking ? 'Lip motion' : 'Silent'}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Yaw */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300 font-mono">Yaw: {yaw.toFixed(1)}°</span>
            <span className={`text-[11px] font-mono ${yawOut ? 'text-rose-400' : 'text-slate-500'}`}>
              [{yawMin.toFixed(0)}°, {yawMax.toFixed(0)}°]
            </span>
          </div>
          <div className="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-100 ${yawOut ? 'bg-rose-500' : 'bg-sky-500'}`}
              style={{ width: `${Math.max(0, Math.min(100, ((yaw + 45) / 90) * 100))}%` }}
            />
          </div>
        </div>

        {/* Pitch */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300 font-mono">Pitch: {pitch.toFixed(1)}°</span>
            <span className={`text-[11px] font-mono ${pitchDown ? 'text-rose-400' : 'text-slate-500'}`}>
              Floor: {pitchFloor.toFixed(0)}°
            </span>
          </div>
          <div className="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-100 ${pitchDown ? 'bg-rose-500' : 'bg-indigo-500'}`}
              style={{ width: `${Math.max(0, Math.min(100, ((pitch + 45) / 90) * 100))}%` }}
            />
          </div>
        </div>

        {/* Horizontal Iris */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300 font-mono">Iris H: {(hRatio * 100).toFixed(0)}%</span>
            <span className="text-[11px] font-mono text-slate-500">
              {calibration ? `[${(calibration.hRatioMin * 100).toFixed(0)}%, ${(calibration.hRatioMax * 100).toFixed(0)}%]` : '0-100%'}
            </span>
          </div>
          <div className="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-teal-400 transition-all duration-100"
              style={{ width: `${Math.max(0, Math.min(100, hRatio * 100))}%` }}
            />
          </div>
        </div>

        {/* Mouth Aspect Ratio (MAR) */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300 font-mono">Mouth (MAR): {mar.toFixed(2)}</span>
            <span className={`text-[11px] font-mono ${isSpeaking ? 'text-pink-400' : 'text-slate-500'}`}>
              Cycles: {mouth?.motionScore ?? 0}/3
            </span>
          </div>
          <div className="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-100 ${isSpeaking ? 'bg-pink-500' : 'bg-emerald-500'}`}
              style={{ width: `${Math.max(0, Math.min(100, (mar / 0.5) * 100))}%` }}
            />
          </div>
        </div>
      </div>

      {/* Hidden unused vRatio reference kept cleanly in state */}
      <div className="sr-only">{vRatio}</div>
    </div>
  );
};
