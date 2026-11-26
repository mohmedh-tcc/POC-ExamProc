export type SignalType =
  | 'NO_FACE'
  | 'EXTRA_FACE'
  | 'HEAD_OFF_SCREEN'
  | 'LOOKING_DOWN'
  | 'SPEAKING_VISUAL'
  | 'FROZEN_FEED'
  | 'TAB_SWITCH'
  | 'FULLSCREEN_EXIT';

export type SignalFamily = 'vision' | 'audio' | 'browser';

export interface SignalDefinition {
  type: SignalType;
  family: SignalFamily;
  weight: number;
  description: string;
}

export const SIGNAL_DEFINITIONS: Record<SignalType, SignalDefinition> = {
  NO_FACE: {
    type: 'NO_FACE',
    family: 'vision',
    weight: 2,
    description: 'No face detected in frame',
  },
  EXTRA_FACE: {
    type: 'EXTRA_FACE',
    family: 'vision',
    weight: 4,
    description: 'Extra person / second face detected in frame',
  },
  HEAD_OFF_SCREEN: {
    type: 'HEAD_OFF_SCREEN',
    family: 'vision',
    weight: 1,
    description: 'Head turned away from screen (neighbor peek)',
  },
  LOOKING_DOWN: {
    type: 'LOOKING_DOWN',
    family: 'vision',
    weight: 2,
    description: 'Head tilted down toward desk/lap (phone check)',
  },
  SPEAKING_VISUAL: {
    type: 'SPEAKING_VISUAL',
    family: 'vision',
    weight: 2,
    description: 'Repeated lip movement detected without microphone',
  },
  FROZEN_FEED: {
    type: 'FROZEN_FEED',
    family: 'vision',
    weight: 5,
    description: 'Camera feed frozen or virtual loop detected',
  },
  TAB_SWITCH: {
    type: 'TAB_SWITCH',
    family: 'browser',
    weight: 3,
    description: 'Browser tab hidden or window focus lost > 2s',
  },
  FULLSCREEN_EXIT: {
    type: 'FULLSCREEN_EXIT',
    family: 'browser',
    weight: 3,
    description: 'Exited browser fullscreen mode',
  },
};

export type DebounceState = 'IDLE' | 'ACTIVE' | 'RELEASED';

export interface SignalEvent {
  type: SignalType;
  family: SignalFamily;
  weight: number;
  startTime: number;
  endTime?: number;
  durationMs?: number;
}

export const DEFAULT_ONSET_MS: Record<SignalType, number> = {
  NO_FACE: 0,
  EXTRA_FACE: 0,
  HEAD_OFF_SCREEN: 4000,
  LOOKING_DOWN: 3000,
  SPEAKING_VISUAL: 1500,
  FROZEN_FEED: 10000,
  TAB_SWITCH: 2000,
  FULLSCREEN_EXIT: 0,
};

export class DebounceStateMachine {
  readonly type: SignalType;
  readonly family: SignalFamily;
  readonly baseWeight: number;
  onsetMs: number;

  private state: DebounceState = 'IDLE';
  private onsetStartTime: number | null = null;
  private activeStartTime: number | null = null;
  private releaseStartTime: number | null = null;
  private sustainedTenSecondBlocksCount = 0;

  // History of episode start times for pattern bonus (e.g. HEAD_OFF_SCREEN 5 episodes in 5 mins)
  private episodeHistory: number[] = [];

  private readonly releaseHysteresisMs = 2000; // 2 seconds

  constructor(type: SignalType, onsetMs = DEFAULT_ONSET_MS[type]) {
    this.type = type;
    const def = SIGNAL_DEFINITIONS[type];
    this.family = def.family;
    this.baseWeight = def.weight;
    this.onsetMs = onsetMs;
  }

  getState(): DebounceState {
    return this.state;
  }

  getActiveDuration(now: number): number {
    if (this.state === 'IDLE' || this.activeStartTime === null) return 0;
    return now - this.activeStartTime;
  }

  /**
   * Updates state given whether the raw detection condition holds at time `now`.
   * Returns:
   *  - `newActiveEvent`: true if transitioned IDLE -> ACTIVE (or RELEASED -> ACTIVE without finish)
   *  - `releasedEvent`: true if transitioned RELEASED -> IDLE (episode officially ended)
   *  - `sustainedBonus`: number of extra 10s intervals reached during this turn
   *  - `patternBonus`: extra weight if pattern triggered
   */
  update(conditionActive: boolean, now: number): {
    state: DebounceState;
    newActiveEvent: boolean;
    releasedEvent: boolean;
    sustainedBonus: number;
    patternBonus: number;
    activeDurationMs: number;
  } {
    let newActiveEvent = false;
    let releasedEvent = false;
    let sustainedBonus = 0;
    let patternBonus = 0;

    switch (this.state) {
      case 'IDLE':
        if (conditionActive) {
          if (this.onsetStartTime === null) {
            this.onsetStartTime = now;
          }
          if (now - this.onsetStartTime >= this.onsetMs) {
            this.state = 'ACTIVE';
            this.activeStartTime = this.onsetStartTime;
            this.onsetStartTime = null;
            this.releaseStartTime = null;
            this.sustainedTenSecondBlocksCount = 0;
            newActiveEvent = true;

            // Record episode for pattern detection
            this.episodeHistory.push(now);
            // Clean history older than 5 minutes (300,000 ms)
            this.episodeHistory = this.episodeHistory.filter((t) => now - t <= 300_000);

            if (this.type === 'HEAD_OFF_SCREEN' && this.episodeHistory.length >= 5) {
              patternBonus = 2; // +2 pattern bonus if 5 episodes in 5 minutes
            }
          }
        } else {
          this.onsetStartTime = null;
        }
        break;

      case 'ACTIVE':
        if (conditionActive) {
          // Still active: check sustained 10s increments
          const activeDuration = now - (this.activeStartTime ?? now);
          const currentTenSecondBlocks = Math.floor(activeDuration / 10000);
          if (currentTenSecondBlocks > this.sustainedTenSecondBlocksCount) {
            sustainedBonus = (currentTenSecondBlocks - this.sustainedTenSecondBlocksCount) * (0.25 * this.baseWeight);
            this.sustainedTenSecondBlocksCount = currentTenSecondBlocks;
          }
        } else {
          // Condition dropped: enter RELEASED state
          this.state = 'RELEASED';
          this.releaseStartTime = now;
        }
        break;

      case 'RELEASED':
        if (conditionActive) {
          // Re-triggered within release window: cancel release, resume ACTIVE
          this.state = 'ACTIVE';
          this.releaseStartTime = null;
        } else {
          // Still false: check if 2 seconds have elapsed
          if (now - (this.releaseStartTime ?? now) >= this.releaseHysteresisMs) {
            this.state = 'IDLE';
            releasedEvent = true;
            this.activeStartTime = null;
            this.releaseStartTime = null;
            this.sustainedTenSecondBlocksCount = 0;
          }
        }
        break;
    }

    return {
      state: this.state,
      newActiveEvent,
      releasedEvent,
      sustainedBonus,
      patternBonus,
      activeDurationMs: this.getActiveDuration(now),
    };
  }

  reset(): void {
    this.state = 'IDLE';
    this.onsetStartTime = null;
    this.activeStartTime = null;
    this.releaseStartTime = null;
    this.sustainedTenSecondBlocksCount = 0;
    this.episodeHistory = [];
  }
}
