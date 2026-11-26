import React from 'react';
import { SignalType } from '../lib/engine/state-machine';
import { AlertCircle, X } from 'lucide-react';

export interface FlagNotification {
  id: number;
  signal: SignalType;
  points: number;
  timestamp: number;
}

interface FlagToastProps {
  notifications: FlagNotification[];
  onDismiss: (id: number) => void;
}

const LABELS: Record<SignalType, string> = {
  NO_FACE: 'No face detected',
  EXTRA_FACE: 'Extra person in frame',
  HEAD_OFF_SCREEN: 'Head turned off screen',
  LOOKING_DOWN: 'Looking down',
  SPEAKING_VISUAL: 'Lip movement (speaking)',
  FROZEN_FEED: 'Frozen camera feed',
  TAB_SWITCH: 'Tab switch or focus lost',
  FULLSCREEN_EXIT: 'Exited fullscreen',
};

export const FlagToast: React.FC<FlagToastProps> = ({ notifications, onDismiss }) => {
  if (notifications.length === 0) return null;

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none">
      {notifications.map((n) => (
        <div
          key={n.id}
          className="pointer-events-auto flex items-center gap-2.5 px-3.5 py-2.5 rounded-2xl bg-slate-900/95 border border-rose-500/40 shadow-xl backdrop-blur-md text-slate-100 max-w-xs"
        >
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <div className="min-w-0 flex-1">
            <div className="text-xs font-medium truncate">{LABELS[n.signal]}</div>
          </div>
          <span className="text-[11px] font-mono font-semibold px-1.5 py-0.5 rounded-md bg-rose-500/20 text-rose-300 shrink-0">
            +{n.points.toFixed(1)}
          </span>
          <button
            onClick={() => onDismiss(n.id)}
            className="text-slate-400 hover:text-slate-200 transition p-0.5"
            aria-label="Dismiss notification"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
};
