import React from 'react';
import { Cpu, Zap, AlertOctagon } from 'lucide-react';

interface LatencyProfilerProps {
  faceP95: number;
  objectP95: number;
  faceIntervalMs: number;
  objectIntervalMs: number;
  faceFps: number;
  isDegraded: boolean;
  faceDelegate: string;
  objectDelegate: string;
}

export const LatencyProfiler: React.FC<LatencyProfilerProps> = ({
  faceP95,
  objectP95,
  faceIntervalMs,
  objectIntervalMs,
  faceFps,
  isDegraded,
  faceDelegate,
  objectDelegate,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
          <Zap className="w-4 h-4 text-sky-400" />
          Runtime Performance & Throttler
        </h3>

        {isDegraded && (
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-500/20 text-rose-400 border border-rose-500/50 text-xs font-bold animate-pulse">
            <AlertOctagon className="w-3.5 h-3.5" />
            DEGRADED_COVERAGE
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Worker 1: Face Mesh */}
        <div className="p-3 bg-slate-800/40 border border-slate-800 rounded-lg space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Worker 1: Face Landmarker</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800 font-mono">
              {faceDelegate.toUpperCase()}
            </span>
          </div>

          <div className="flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono text-slate-100">
              {faceP95.toFixed(1)} <span className="text-xs text-slate-400 font-normal">ms (P95)</span>
            </span>
            <span className="text-xs font-mono text-sky-400 font-medium">
              {faceFps} Hz ({faceIntervalMs}ms)
            </span>
          </div>

          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                faceP95 <= 80 ? 'bg-emerald-400' : faceP95 <= 120 ? 'bg-amber-400' : 'bg-rose-400'
              }`}
              style={{ width: `${Math.min(100, (faceP95 / 150) * 100)}%` }}
            />
          </div>
          <div className="text-[10px] text-slate-500 flex justify-between">
            <span>Budget: &le;120 ms</span>
            <span>Threshold: {faceP95 > 120 ? 'Over budget (throttled)' : 'Healthy'}</span>
          </div>
        </div>

        {/* Worker 2: Object Detector */}
        <div className="p-3 bg-slate-800/40 border border-slate-800 rounded-lg space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-300">Worker 2: Object Detector</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800 font-mono">
              {objectDelegate.toUpperCase()}
            </span>
          </div>

          <div className="flex items-baseline justify-between">
            <span className="text-2xl font-bold font-mono text-slate-100">
              {objectP95.toFixed(0)} <span className="text-xs text-slate-400 font-normal">ms (P95)</span>
            </span>
            <span className="text-xs font-mono text-indigo-400 font-medium">
              Interval: {(objectIntervalMs / 1000).toFixed(1)}s
            </span>
          </div>

          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                objectP95 <= 800 ? 'bg-emerald-400' : objectP95 <= 1500 ? 'bg-amber-400' : 'bg-rose-400'
              }`}
              style={{ width: `${Math.min(100, (objectP95 / 2000) * 100)}%` }}
            />
          </div>
          <div className="text-[10px] text-slate-500 flex justify-between">
            <span>Budget: &le;1500 ms</span>
            <span>{objectP95 > 1500 ? 'Throttling interval' : 'Optimal'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
