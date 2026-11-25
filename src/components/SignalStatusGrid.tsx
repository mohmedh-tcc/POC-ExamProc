import React from 'react';
import { SignalType, DebounceState, SIGNAL_DEFINITIONS } from '../lib/engine/state-machine';
import { AlertTriangle, UserX, Users, EyeOff, ArrowDown, Smartphone, Mic } from 'lucide-react';

interface SignalStatusItem {
  type: SignalType;
  state: DebounceState;
  activeDurationMs: number;
}

interface SignalStatusGridProps {
  signals: Record<SignalType, SignalStatusItem>;
}

const ICONS: Record<SignalType, React.ReactNode> = {
  NO_FACE: <UserX className="w-4 h-4 text-rose-400" />,
  EXTRA_FACE: <Users className="w-4 h-4 text-amber-400" />,
  HEAD_OFF_SCREEN: <EyeOff className="w-4 h-4 text-sky-400" />,
  LOOKING_DOWN: <ArrowDown className="w-4 h-4 text-indigo-400" />,
  PHONE: <Smartphone className="w-4 h-4 text-red-500" />,
  EXTRA_PERSON: <Users className="w-4 h-4 text-purple-400" />,
  SPEECH: <Mic className="w-4 h-4 text-emerald-400" />,
};

export const SignalStatusGrid: React.FC<SignalStatusGridProps> = ({ signals }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          Signal State Machines
        </h3>
        <span className="text-xs text-slate-400">3-State Debounced (2s Release)</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {(Object.keys(SIGNAL_DEFINITIONS) as SignalType[]).map((type) => {
          const def = SIGNAL_DEFINITIONS[type];
          const item = signals[type] || { state: 'IDLE', activeDurationMs: 0 };
          const isActive = item.state === 'ACTIVE';
          const isReleased = item.state === 'RELEASED';

          return (
            <div
              key={type}
              className={`p-3 rounded-lg border transition-all ${
                isActive
                  ? 'bg-rose-950/40 border-rose-600/70 shadow-rose-900/20 shadow-md ring-1 ring-rose-500/50'
                  : isReleased
                  ? 'bg-amber-950/30 border-amber-600/50'
                  : 'bg-slate-800/40 border-slate-800 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  {ICONS[type]}
                  <span className="font-semibold text-xs text-slate-200">{type}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
                    wt {def.weight}
                  </span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      isActive
                        ? 'bg-rose-500 text-white animate-pulse'
                        : isReleased
                        ? 'bg-amber-500/30 text-amber-300 border border-amber-500/40'
                        : 'bg-slate-800 text-slate-500'
                    }`}
                  >
                    {item.state}
                  </span>
                </div>
              </div>

              <div className="text-[11px] text-slate-400 leading-tight">
                {def.description}
              </div>

              {isActive && (
                <div className="mt-2 text-[10px] text-rose-300 font-mono flex items-center justify-between">
                  <span>Active for:</span>
                  <span>{(item.activeDurationMs / 1000).toFixed(1)}s</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
