/**
 * SCALE-TO-ZERO CAPACITY IS NOT "EXPENSIVE" — IT IS ABSENT, AND THE POLICY HAS
 * NO WAY TO SAY SO.
 *
 * A RunPod Serverless endpoint holds no worker when idle. The first request
 * after it sleeps pays an image pull plus a vLLM engine load before the first
 * token: measured at 20-40 minutes on the baked images this substrate ships
 * (~30GB reaching a cold worker at serverless pull bandwidth). While warm, the
 * same endpoint answers at pod speed.
 *
 * The obvious move — register the arm and give it a high `cost_per_mtok` when
 * cold — does not work, and the arithmetic says why. Selection scores an arm as
 *
 *     score = beta_draw - cost_weight * (cost / maxCost)
 *
 * The cost term is bounded by `cost_weight` (0.25 live), while a Beta draw on a
 * near-uniform prior spans [0,1]. So the most expensive arm imaginable is
 * penalised by at most 0.25 and still wins draws regularly — and each of those
 * wins is a 20-40 minute stall that times out, increments the arm's `beta`, and
 * records "this model failed" when what actually happened was "this capacity was
 * asleep". The reach evidence is then wrong in a way no later observation
 * corrects. Worse, a very large cold price becomes `maxCost` and collapses every
 * other arm's cost discount toward zero, so pricing the cold state distorts
 * selection across arms that have nothing to do with the sleeping endpoint.
 *
 * Hence: cold is expressed as NOT ROUTABLE, not as a price. `cost_per_mtok`
 * stays the honest warm price, used only when the endpoint can actually serve.
 * This is the same distinction the vessel already draws between quota
 * (cooldowns, billing) and capability (whether a client exists at all) — a
 * sleeping endpoint is a capability fact.
 *
 * Deliberately pure: the readiness reading is a fact about the control plane, so
 * the parsing and the predicate are separated from the polling and testable
 * without a network. The caller owns the mutable warm flag and the poll loop.
 */

/** Worker counts as reported by RunPod's /health. Fields are optional because
 * the control plane may add states; only the ones meaning "a worker exists and
 * is not terminal" are counted. */
export interface WorkerCounts {
  ready?: number;
  running?: number;
  idle?: number;
  initializing?: number;
  throttled?: number;
  unhealthy?: number;
}

/**
 * Warm iff at least one worker can take work NOW.
 *
 * `initializing` deliberately does NOT count: a worker pulling a 30GB image is
 * exactly the 20-40 minute tail this gate exists to avoid, and RunPod reports
 * `ready` well before vLLM has finished loading, so counting anything earlier
 * would re-admit the stall. `throttled` and `unhealthy` are capacity that
 * cannot serve. A malformed or absent body reads as cold — an unknown control
 * plane is never assumed to be holding capacity.
 */
export function isWarmFromHealth(body: unknown): boolean {
  if (typeof body !== "object" || body === null) return false;
  const workers = (body as { workers?: unknown }).workers;
  if (typeof workers !== "object" || workers === null) return false;
  const w = workers as WorkerCounts;
  const n = (v: number | undefined): number => (typeof v === "number" && Number.isFinite(v) ? v : 0);
  return n(w.ready) > 0 || n(w.running) > 0 || n(w.idle) > 0;
}

/**
 * Whether `model` is served by scale-to-zero capacity that is currently asleep.
 * Models outside `scaleToZeroModels` are never gated, so a substrate with no
 * such endpoint configured behaves exactly as before.
 */
export function isScaleToZeroCold(model: string, scaleToZeroModels: ReadonlySet<string>, warm: boolean): boolean {
  return scaleToZeroModels.has(model) && !warm;
}
