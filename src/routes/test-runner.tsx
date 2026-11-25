import React, { useState } from 'react';
import { SignalType } from '../lib/engine/state-machine';
import { Play, Check, X, ShieldAlert, Award } from 'lucide-react';

export interface TestScenario {
  id: string;
  title: string;
  instruction: string;
  expectedSignal: SignalType | 'NONE';
  durationSec: number;
}

export const TEST_SCENARIOS: TestScenario[] = [
  {
    id: 'normal-thinking',
    title: 'Normal Screen Reading & Thinking',
    instruction: 'Look at the screen normally, blink naturally, and read silently for 10 seconds.',
    expectedSignal: 'NONE',
    durationSec: 10,
  },
  {
    id: 'looking-down',
    title: 'Looking Down at Desk / Keyboard',
    instruction: 'Tilt head down toward your desk or keyboard continuously for 5 seconds.',
    expectedSignal: 'LOOKING_DOWN',
    durationSec: 5,
  },
  {
    id: 'head-off-screen',
    title: 'Turning Head Away from Screen',
    instruction: 'Turn head sideways away from the screen for 5 seconds.',
    expectedSignal: 'HEAD_OFF_SCREEN',
    durationSec: 5,
  },
  {
    id: 'no-face',
    title: 'No Face in Frame (Cover or Step Away)',
    instruction: 'Cover your webcam or step out of the camera view for 5 seconds.',
    expectedSignal: 'NO_FACE',
    durationSec: 5,
  },
  {
    id: 'phone-detected',
    title: 'Cell Phone in Camera View',
    instruction: 'Hold up a smartphone in front of the camera for 5 seconds.',
    expectedSignal: 'PHONE',
    durationSec: 5,
  },
  {
    id: 'extra-face',
    title: 'Second Person Enters Frame',
    instruction: 'Have a second person look into the camera with you for 4 seconds.',
    expectedSignal: 'EXTRA_FACE',
    durationSec: 4,
  },
  {
    id: 'speech-detected',
    title: 'Continuous Speech / Reading Aloud',
    instruction: 'Read a paragraph or talk aloud continuously for 10 seconds.',
    expectedSignal: 'SPEECH',
    durationSec: 10,
  },
];

interface TestRunnerProps {
  activeSignals: Set<SignalType>;
}

export const TestRunner: React.FC<TestRunnerProps> = ({ activeSignals }) => {
  const [runningIndex, setRunningIndex] = useState<number | null>(null);
  const [countdown, setCountdown] = useState<number>(0);
  const [results, setResults] = useState<Record<string, { passed: boolean; detected: SignalType[] }>>({});
  const [detectedDuringTest, setDetectedDuringTest] = useState<Set<SignalType>>(new Set());

  const startTest = (index: number) => {
    const scenario = TEST_SCENARIOS[index];
    setRunningIndex(index);
    setCountdown(scenario.durationSec);
    setDetectedDuringTest(new Set());

    const startTime = Date.now();
    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      const remaining = scenario.durationSec - elapsed;

      // Track any active signal fired during this window
      activeSignals.forEach((sig) => {
        setDetectedDuringTest((prev) => new Set(prev).add(sig));
      });

      if (remaining <= 0) {
        clearInterval(interval);
        setRunningIndex(null);
        setCountdown(0);

        // Evaluate pass/fail
        const detectedArray = Array.from(detectedDuringTest);
        let passed = false;
        if (scenario.expectedSignal === 'NONE') {
          passed = detectedArray.length === 0;
        } else {
          passed = detectedArray.includes(scenario.expectedSignal);
        }

        setResults((prev) => ({
          ...prev,
          [scenario.id]: { passed, detected: detectedArray },
        }));
      } else {
        setCountdown(remaining);
      }
    }, 200);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Award className="w-5 h-5 text-sky-400" />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Scripted Accuracy & False-Positive Evaluator
          </h3>
        </div>
        <span className="text-xs text-slate-400">Section 9 Protocol</span>
      </div>

      <div className="space-y-3">
        {TEST_SCENARIOS.map((sc, idx) => {
          const isRunning = runningIndex === idx;
          const result = results[sc.id];

          return (
            <div
              key={sc.id}
              className={`p-3.5 rounded-lg border transition-all ${
                isRunning
                  ? 'bg-sky-950/40 border-sky-500 ring-1 ring-sky-500'
                  : result?.passed === true
                  ? 'bg-emerald-950/20 border-emerald-800'
                  : result?.passed === false
                  ? 'bg-rose-950/20 border-rose-800'
                  : 'bg-slate-800/40 border-slate-800'
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                    {sc.title}
                    <span className="text-[10px] text-slate-400 font-normal">
                      (Expect: <span className="font-mono text-sky-300">{sc.expectedSignal}</span>, {sc.durationSec}s)
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">{sc.instruction}</div>
                </div>

                <div className="flex items-center gap-2">
                  {result && (
                    <span
                      className={`flex items-center gap-1 text-[11px] font-bold px-2 py-0.5 rounded ${
                        result.passed
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                          : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                      }`}
                    >
                      {result.passed ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                      {result.passed ? 'PASSED' : 'FAILED'}
                    </span>
                  )}

                  <button
                    onClick={() => startTest(idx)}
                    disabled={runningIndex !== null}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
                      isRunning
                        ? 'bg-sky-500 text-slate-950 font-mono animate-pulse'
                        : 'bg-slate-800 hover:bg-slate-700 text-slate-200 disabled:opacity-40'
                    }`}
                  >
                    {isRunning ? `${countdown}s` : <><Play className="w-3 h-3" /> Test</>}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
