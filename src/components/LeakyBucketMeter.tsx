import React from 'react';
import { RiskLevel } from '../lib/engine/leaky-bucket';
import { ShieldCheck, ShieldAlert, ShieldX, RotateCcw } from 'lucide-react';

interface LeakyBucketMeterProps {
  score: number;
  level: RiskLevel;
  onReset: () => void;
}

export const LeakyBucketMeter: React.FC<LeakyBucketMeterProps> = ({ score, level, onReset }) => {
  const levelStyles = {
    green: {
      badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
      bar: 'bg-emerald-500',
      icon: <ShieldCheck className="w-5 h-5 text-emerald-400" />,
      label: 'GREEN (Normal)',
      desc: 'No suspicious behavior detected. Session clean.',
    },
    amber: {
      badge: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
      bar: 'bg-amber-500',
      icon: <ShieldAlert className="w-5 h-5 text-amber-400" />,
      label: 'AMBER (Suspicious)',
      desc: 'Repeated glances or speech noted. Review threshold warming.',
    },
    red: {
      badge: 'bg-rose-500/20 text-rose-400 border-rose-500/40',
      bar: 'bg-rose-500 animate-pulse',
      icon: <ShieldX className="w-5 h-5 text-rose-400" />,
      label: 'RED (High Risk)',
      desc: 'Escalated to top of human reviewer queue. Evidence logged.',
    },
  }[level];

  // Visual percentage on a scale of 0 to 12 max display
  const barPercent = Math.min(100, (score / 12) * 100);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          {levelStyles.icon}
          <div>
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Risk Score (Leaky Bucket)
            </h3>
            <p className="text-[11px] text-slate-400">120s half-life decay • 1.5x cross-family multiplier</p>
          </div>
        </div>

        <button
          onClick={onReset}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition"
          title="Reset Risk Score"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      <div className="flex items-baseline justify-between">
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-extrabold font-mono text-slate-100">{score.toFixed(2)}</span>
          <span className="text-xs text-slate-500">points</span>
        </div>
        <span className={`px-2.5 py-1 rounded-md text-xs font-bold border ${levelStyles.badge}`}>
          {levelStyles.label}
        </span>
      </div>

      {/* Progress Bar with 4 and 8 thresholds marked */}
      <div className="relative w-full">
        <div className="w-full h-3.5 bg-slate-800 rounded-full overflow-hidden p-0.5">
          <div
            className={`h-full rounded-full transition-all duration-150 ${levelStyles.bar}`}
            style={{ width: `${barPercent}%` }}
          />
        </div>

        {/* Threshold Markers */}
        <div className="absolute top-0 bottom-0 left-[33.33%] w-0.5 bg-amber-400/60 pointer-events-none" />
        <div className="absolute top-0 bottom-0 left-[66.66%] w-0.5 bg-rose-400/60 pointer-events-none" />

        <div className="flex justify-between text-[10px] text-slate-500 mt-1">
          <span>0 (Clean)</span>
          <span className="text-amber-400/80">4 (Amber)</span>
          <span className="text-rose-400/80">8 (Red)</span>
          <span>12+</span>
        </div>
      </div>

      <p className="text-xs text-slate-400">{levelStyles.desc}</p>
    </div>
  );
};
