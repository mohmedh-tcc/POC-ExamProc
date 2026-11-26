import React from 'react';
import { EulerAngles } from '../lib/math/transform-matrix';
import { IrisGazeResult } from '../lib/math/iris-gaze';
import { CalibrationProfile } from '../lib/engine/calibrator';

interface GaugesProps {
  angles: EulerAngles | null;
  gaze?: IrisGazeResult | null;
  calibration: CalibrationProfile | null;
}

export const Gauges: React.FC<GaugesProps> = ({ angles, calibration }) => {
  const pitch = angles?.pitch ?? 0;
  const yaw = angles?.yaw ?? 0;

  const neutralYaw = calibration?.neutralYaw ?? 0;
  const neutralPitch = calibration?.neutralPitch ?? 0;
  const yawMin = calibration ? Math.max(calibration.yawMin, neutralYaw - 14) : -14;
  const yawMax = calibration ? Math.min(calibration.yawMax, neutralYaw + 14) : 14;
  const pitchFloor = calibration ? Math.max(calibration.pitchMin, neutralPitch - 10) : -10;

  const yawOut = yaw < yawMin || yaw > yawMax;
  const pitchDown = pitch < pitchFloor;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <h3 className="text-xs font-semibold text-slate-200 tracking-wide">
          Head pose
        </h3>
        <span className="text-[11px] text-slate-400">
          {calibration ? 'Calibrated bounds' : 'Default bounds'}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Yaw (Left/Right) */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300 font-mono">
              Yaw: {yaw.toFixed(1)}°
            </span>
            <span className={`text-[11px] font-mono ${yawOut ? 'text-rose-400' : 'text-slate-500'}`}>
              [{yawMin.toFixed(0)}°, {yawMax.toFixed(0)}°]
            </span>
          </div>
          <div className="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-100 ${yawOut ? 'bg-rose-500' : 'bg-sky-500'}`}
              style={{
                width: `${Math.max(0, Math.min(100, ((yaw + 45) / 90) * 100))}%`,
              }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>-45°</span>
            <span>0°</span>
            <span>+45°</span>
          </div>
        </div>

        {/* Pitch (Up/Down) */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300 font-mono">
              Pitch: {pitch.toFixed(1)}°
            </span>
            <span className={`text-[11px] font-mono ${pitchDown ? 'text-rose-400' : 'text-slate-500'}`}>
              Floor: {pitchFloor.toFixed(0)}°
            </span>
          </div>
          <div className="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-100 ${pitchDown ? 'bg-rose-500' : 'bg-indigo-500'}`}
              style={{
                width: `${Math.max(0, Math.min(100, ((pitch + 45) / 90) * 100))}%`,
              }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>-45° (down)</span>
            <span>0°</span>
            <span>+45° (up)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
