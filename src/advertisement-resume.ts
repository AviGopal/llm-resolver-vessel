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

/**
 * The cooldown expiries a timer may re-check at: a lane held until served (a billing-class refusal, which no window
 * ends) is left out, so its expiry never re-advertises; it returns on a served probe instead.
 */
export const timedExpiries = (until: Iterable<[string, number]>, held: ReadonlySet<string>): number[] =>
  [...until].filter(([key]) => !held.has(key)).map(([, at]) => at);

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

/**
 * Whether llm_completion can be served: some arm the request path can draw (the policy's arms) is willing now. Not "any
 * keyed model": the client map holds every model a provider is keyed for, and a model nothing draws is never called,
 * never cooled, and so kept the shape advertised while every drawn arm was cooling (live: 22:52-23:00, callers got
 * instant unserved refusals instead of discovery routing them to a peer). The same set the policy handler falls back to.
 */
export const completionQuota = (armModels: string[], willing: (model: string) => boolean): boolean => armModels.some(willing);

/**
 * Whether to advertise: a drawable arm is willing; or none is, no cooldown will end to re-check, and the wider check
 * (any keyed model) says something could answer. Without that last case an arm unwilling for a reason no cooldown ends
 * would de-advertise with nothing scheduled, and a dropped shape receives no call to recover it (qa). A call that then
 * finds nothing to draw is answered unserved, which is the callee's.
 */
export const advertiseQuota = (drawable: boolean, cooldownPending: boolean, wider: () => boolean): boolean =>
  drawable || (!cooldownPending && wider());

/**
 * Latest call wins: each sync takes a ticket when it starts, and after its await goes on only if no newer sync has
 * started. Two syncs whose reads finish out of order would otherwise leave the older one's state advertised, possibly
 * dark with nothing scheduled (deployment).
 */
export const makeLatest = (): { begin: () => number; isLatest: (ticket: number) => boolean } => {
  let last = 0;
  return { begin: () => ++last, isLatest: (ticket: number) => ticket === last };
};
