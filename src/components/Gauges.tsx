import React from 'react';
import { EulerAngles } from '../lib/math/transform-matrix';
import { IrisGazeResult } from '../lib/math/iris-gaze';
import { CalibrationProfile } from '../lib/engine/calibrator';

interface GaugesProps {
  angles: EulerAngles | null;
  gaze: IrisGazeResult | null;
  calibration: CalibrationProfile | null;
}

export const Gauges: React.FC<GaugesProps> = ({ angles, gaze, calibration }) => {
  const pitch = angles?.pitch ?? 0;
  const yaw = angles?.yaw ?? 0;
  const roll = angles?.roll ?? 0;

  const hRatio = gaze?.hRatio ?? 0.5;
  const vRatio = gaze?.vRatio ?? 0.5;
  const isBlinking = gaze?.isBlinking ?? false;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          Pose & Gaze Gauges
        </h3>
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${isBlinking ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'}`} />
          <span className="text-xs text-slate-400">{isBlinking ? 'Blink / Closed' : 'Eyes Open'}</span>
        </div>
      </div>

      {/* Head Pose Section */}
      <div className="space-y-3">
        <div className="text-xs font-medium text-slate-400">Head Pose (Euler Angles via 4x4 Metric Matrix)</div>
        
        {/* Pitch (Looking Down/Up) */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300">Pitch (X-Axis): {pitch.toFixed(1)}°</span>
            <span className="text-slate-500">
              {calibration ? `Floor: ${calibration.pitchMin.toFixed(1)}°` : 'Default: -15°'}
            </span>
          </div>
          <div className="relative w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-sky-500 transition-all duration-75"
              style={{
                width: `${Math.max(0, Math.min(100, ((pitch + 45) / 90) * 100))}%`,
              }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
            <span>-45° (Down)</span>
            <span>0° (Center)</span>
            <span>+45° (Up)</span>
          </div>
        </div>

        {/* Yaw (Turning Left/Right) */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300">Yaw (Y-Axis): {yaw.toFixed(1)}°</span>
            <span className="text-slate-500">
              {calibration ? `[${calibration.yawMin.toFixed(0)}°, ${calibration.yawMax.toFixed(0)}°]` : 'Default: [-25°, 25°]'}
            </span>
          </div>
          <div className="relative w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all duration-75"
              style={{
                width: `${Math.max(0, Math.min(100, ((yaw + 45) / 90) * 100))}%`,
              }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
            <span>-45° (Right)</span>
            <span>0° (Center)</span>
            <span>+45° (Left)</span>
          </div>
        </div>

        {/* Roll */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300">Roll (Z-Axis): {roll.toFixed(1)}°</span>
          </div>
          <div className="relative w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-violet-500 transition-all duration-75"
              style={{
                width: `${Math.max(0, Math.min(100, ((roll + 45) / 90) * 100))}%`,
              }}
            />
          </div>
        </div>
      </div>

      {/* Iris Section */}
      <div className="space-y-3 pt-2 border-t border-slate-800">
        <div className="text-xs font-medium text-slate-400">Iris Gaze Displacements (Rigid Intercanthal Normalized)</div>

        {/* Horizontal Iris Ratio */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300">Horizontal Gaze: {(hRatio * 100).toFixed(0)}%</span>
            <span className="text-slate-500">
              {calibration ? `[${(calibration.hRatioMin * 100).toFixed(0)}%, ${(calibration.hRatioMax * 100).toFixed(0)}%]` : 'Default: [25%, 75%]'}
            </span>
          </div>
          <div className="relative w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-teal-400 transition-all duration-75"
              style={{ width: `${Math.max(0, Math.min(100, hRatio * 100))}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
            <span>0% (Look Left)</span>
            <span>50% (Center)</span>
            <span>100% (Look Right)</span>
          </div>
        </div>

        {/* Vertical Iris Ratio */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300">Vertical Gaze: {(vRatio * 100).toFixed(0)}%</span>
            <span className="text-slate-500">
              {calibration ? `[${(calibration.vRatioMin * 100).toFixed(0)}%, ${(calibration.vRatioMax * 100).toFixed(0)}%]` : 'Default: [25%, 75%]'}
            </span>
          </div>
          <div className="relative w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-teal-400 transition-all duration-75"
              style={{ width: `${Math.max(0, Math.min(100, vRatio * 100))}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
            <span>0% (Look Up)</span>
            <span>50% (Center)</span>
            <span>100% (Look Down)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
