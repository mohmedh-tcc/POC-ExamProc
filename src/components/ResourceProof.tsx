import React from 'react';
import { Cpu, AlertOctagon } from 'lucide-react';

interface ResourceProofProps {
  faceAvg: number;
  faceP95: number;
  faceIntervalMs: number;
  faceFps: number;
  cpuDutyCyclePercent: number;
  isDegraded: boolean;
  delegate: string;
  jsHeapMb: number | null;
  mainThreadLagMs: number;
}

export const ResourceProof: React.FC<ResourceProofProps> = ({
  faceAvg,
  faceP95,
  faceIntervalMs,
  faceFps,
  cpuDutyCyclePercent,
  isDegraded,
  delegate,
  jsHeapMb,
  mainThreadLagMs,
}) => {
  const idlePercent = Math.max(0, 100 - cpuDutyCyclePercent);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-emerald-400" />
          <h3 className="text-xs font-semibold text-slate-200 tracking-wide">
            System resource usage
          </h3>
        </div>

        <div className="flex items-center gap-2">
          {isDegraded && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/40 text-[11px] font-semibold">
              <AlertOctagon className="w-3 h-3" />
              Degraded
            </span>
          )}
          <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono">
            {delegate.toUpperCase()} WASM
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-800">
          <div className="text-[11px] text-slate-400">Worker CPU duty</div>
          <div className="text-lg font-bold font-mono text-emerald-400 mt-0.5">
            {cpuDutyCyclePercent.toFixed(1)}%
          </div>
          <div className="text-[10px] text-slate-500">
            {idlePercent.toFixed(1)}% idle
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-800">
          <div className="text-[11px] text-slate-400">Inference time</div>
          <div className="text-lg font-bold font-mono text-slate-100 mt-0.5">
            {faceAvg.toFixed(1)} <span className="text-xs font-normal text-slate-400">ms</span>
          </div>
          <div className="text-[10px] text-slate-500">
            P95: {faceP95.toFixed(1)} ms
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-800">
          <div className="text-[11px] text-slate-400">JS heap</div>
          <div className="text-lg font-bold font-mono text-slate-100 mt-0.5">
            {jsHeapMb !== null ? (
              <>
                {jsHeapMb.toFixed(1)} <span className="text-xs font-normal text-slate-400">MB</span>
              </>
            ) : (
              <span className="text-xs font-normal text-slate-400">N/A</span>
            )}
          </div>
          <div className="text-[10px] text-slate-500">
            {jsHeapMb !== null ? 'Measured heap' : 'Chromium only'}
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-800">
          <div className="text-[11px] text-slate-400">Main thread lag</div>
          <div className="text-lg font-bold font-mono text-slate-100 mt-0.5">
            {mainThreadLagMs.toFixed(0)} <span className="text-xs font-normal text-slate-400">ms</span>
          </div>
          <div className="text-[10px] text-slate-500">
            {faceFps} Hz ({faceIntervalMs} ms)
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-[11px] text-slate-500 font-mono">
        <span>1 worker thread, 0 WebGL contexts, 480x360 capture</span>
        <span>Duty = avg inference / {faceIntervalMs} ms interval</span>
      </div>
    </div>
  );
};
