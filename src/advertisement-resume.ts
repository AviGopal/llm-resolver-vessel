// RESUMING A DE-ADVERTISED COMPLETION PLANE.
//
// When every completion lane is cooling, llm-resolver stops advertising llm_completion, so it receives no traffic and
// nothing arrives to notice a cooldown ending. A timer re-syncs at the cooldown's expiry. The defect this replaces: one
// timer slot kept only the soonest expiry, so when it fired while a later-marked lane still cooled, the sync found no
// quota, de-advertised again and scheduled nothing. The plane stayed dark until a restart (live: llm_completion
// unserved for over half an hour after both cooldowns had ended, every caller answered a routed 404).
//
// Now every sync that finds no quota schedules the next re-sync at the soonest expiry still in the future, so a resume
// that fires early is followed by another. The scheduler keeps the soonest pending timer, as before.

/** How long until the next re-sync should run, or null when nothing will expire (or quota is back). */
export const nextResyncInMs = (quota: boolean, now: number, expiries: Iterable<number>): number | null => {
  if (quota) return null;
  let soonest = Infinity;
  for (const e of expiries) if (e > now && e < soonest) soonest = e;
  return soonest === Infinity ? null : soonest - now;
};

export type ResumeDeps = {
  now: () => number;
  setTimer: (fn: () => void, ms: number) => unknown;
  clearTimer: (handle: unknown) => void;
  sync: () => void;
};

/** One pending re-sync at a time, the soonest kept (a later request never delays an earlier one). */
export const makeResumeScheduler = (d: ResumeDeps): ((ms: number) => void) => {
  let handle: unknown = null;
  let at = 0;
  return (ms: number): void => {
    const fireAt = d.now() + ms + 1000;
    if (handle !== null && fireAt >= at) return;
    if (handle !== null) d.clearTimer(handle);
    at = fireAt;
    handle = d.setTimer(() => { handle = null; d.sync(); }, ms + 1000);
  };
};

/** The log line for an advertisement change, or null when nothing changed. Both directions are said. */
export const advertisementChange = (before: boolean | null, now: boolean, resyncInMs: number | null): string | null => {
  if (before === now) return null;
  return now
    ? before === null ? "advertising llm_completion at start" : "quota returned — advertising llm_completion"
    : `all completion providers cooling — de-advertising llm_completion until quota returns${resyncInMs === null ? " (no cooldown pending: nothing will bring it back but a recovery or a restart)" : `; re-checking in ${Math.round(resyncInMs / 1000)}s`}`;
};
