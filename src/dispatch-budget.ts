/**
 * dispatch-budget.ts - per-dispatch input-token budget, enforced at the resource.
 *
 * WHY: the per-turn prompt ceiling (llmModelPolicy.max_input_tokens) bounds ONE provider call.
 * A tool loop re-sends its whole growing message list every turn, and a caller may make several
 * such calls for one dispatch, so every single turn can pass the ceiling while the dispatch as a
 * whole does not stop. Measured 2026-09-30 (llmSpendSummary, 15:40-16:40 UTC): two
 * goal-host:floor_tool_loop calls on gpt-5 sent 3,161,321 input tokens (~1.58M per call, consistent
 * with a many-turn tool loop inside this resolver; inferred, not counted turn by turn) and cost 1.906 USD, most of the fleet's one shared hourly
 * spend envelope, so the autonomous lane skipped selection for the rest of the hour.
 *
 * WHAT: requests that carry a dispatch id (`dispatch_id`, else `execution_id`) are charged, per
 * provider call, the input tokens the provider reported. Before every provider call the tokens
 * already charged to that id plus this call's estimate are compared with the allowance read at use
 * time from llmModelPolicy.max_input_tokens_per_dispatch (law 1: a shaped field, not an env var).
 * Over the allowance, the call is refused BEFORE the provider sees it, with a structured result.
 *
 * The estimate is RESERVED before the call and settled to the reported count after it (refunded
 * if the call throws), so concurrent calls of one dispatch cannot both slip under the allowance.
 *
 * A request with no id is not charged: the per-turn ceiling still applies to it. Such requests are
 * counted by caller so the uncovered traffic stays measurable.
 *
 * The ledger is an in-memory map bounded both by age (entries idle longer than the TTL are
 * dropped) and by size (oldest-first eviction), so it can never grow with the number of dispatches.
 */

export const DISPATCH_LEDGER_TTL_MS = 2 * 3600_000;
export const DISPATCH_LEDGER_MAX_ENTRIES = 2000;

interface LedgerEntry { input_tokens: number; calls: number; last_at: number }

export class DispatchLedger {
  private readonly entries = new Map<string, LedgerEntry>();
  private readonly missing = new Map<string, number>();
  private readonly ttlMs: number;
  private readonly maxEntries: number;
  private readonly now: () => number;

  constructor(opts: { ttlMs?: number; maxEntries?: number; now?: () => number } = {}) {
    this.ttlMs = opts.ttlMs ?? DISPATCH_LEDGER_TTL_MS;
    this.maxEntries = opts.maxEntries ?? DISPATCH_LEDGER_MAX_ENTRIES;
    this.now = opts.now ?? Date.now;
  }

  /** Input tokens charged to `id` and not yet expired. */
  used(id: string): number {
    const e = this.entries.get(id);
    if (!e) return 0;
    if (this.now() - e.last_at > this.ttlMs) { this.entries.delete(id); return 0; }
    return e.input_tokens;
  }

  /** Adds `tokens` (may be negative to settle a reservation) and returns the new total. */
  charge(id: string, tokens: number, countCall = false): number {
    const t = this.now();
    const prev = this.entries.get(id);
    const base = prev && t - prev.last_at <= this.ttlMs ? prev : { input_tokens: 0, calls: 0, last_at: t };
    const next: LedgerEntry = {
      input_tokens: Math.max(0, base.input_tokens + (Number.isFinite(tokens) ? tokens : 0)),
      calls: base.calls + (countCall ? 1 : 0),
      last_at: t,
    };
    // Re-insert so Map order is least-recently-charged first; eviction takes from the front.
    this.entries.delete(id);
    this.entries.set(id, next);
    this.prune(t);
    return next.input_tokens;
  }

  /** Counts a request that carried no dispatch id, by caller. Bounded like the ledger. */
  noteMissingId(caller: string): void {
    const key = caller && caller.length > 0 ? caller : "unknown";
    if (!this.missing.has(key) && this.missing.size >= this.maxEntries) return;
    this.missing.set(key, (this.missing.get(key) ?? 0) + 1);
  }

  get size(): number { return this.entries.size; }

  snapshot(): { tracked_ids: number; ttl_ms: number; max_entries: number; requests_without_id_by_caller: Record<string, number> } {
    this.prune(this.now());
    return {
      tracked_ids: this.entries.size, ttl_ms: this.ttlMs, max_entries: this.maxEntries,
      requests_without_id_by_caller: Object.fromEntries(this.missing),
    };
  }

  private prune(t: number): void {
    for (const [k, e] of this.entries) {
      if (t - e.last_at > this.ttlMs) this.entries.delete(k);
      else break; // insertion order = last_at order, so the rest are younger
    }
    while (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
  }
}

/** The id a request is budgeted under: `dispatch_id`, else `execution_id`; null when neither. */
export function dispatchKeyOf(body: { dispatch_id?: unknown; execution_id?: unknown } | null | undefined): string | null {
  for (const v of [body?.dispatch_id, body?.execution_id]) {
    if (typeof v === "string" && v.trim().length > 0) return v.trim();
  }
  return null;
}

export interface DispatchBudgetVerdict {
  dispatch_id: string;
  used_input_tokens: number;
  estimate_input_tokens: number;
  cap_input_tokens: number;
}

/** Null when the call fits (or there is no id); otherwise the numbers that refused it. */
export function checkDispatchBudget(ledger: DispatchLedger, id: string | null, estimate: number, cap: number): DispatchBudgetVerdict | null {
  if (id === null) return null;
  const used = ledger.used(id);
  if (used + estimate <= cap) return null;
  return { dispatch_id: id, used_input_tokens: used, estimate_input_tokens: estimate, cap_input_tokens: cap };
}

/**
 * The refusal an over-allowance call returns. An ordinary failed llmCompletion, so callers treat it
 * as a failure; `dispatch_budget_exhausted` lets the fallback walk and the arm grader recognise it.
 * The numbers live in the structured `dispatch_budget` field. The prose carries only the allowance
 * (always >= 1000) and the turn, never a count that the provider-error classifiers could read as an
 * HTTP status, and none of their billing/quota phrases, so it never triggers a provider failover.
 */
export function dispatchBudgetRefusal(
  req: { caller?: string; task_type?: string },
  model: string,
  v: DispatchBudgetVerdict,
  turn?: number,
): Record<string, unknown> {
  const turnTag = turn === undefined ? "" : ` turn=${turn}`;
  console.warn(`[llm-resolver-vessel] dispatch budget refused: dispatch_id=${v.dispatch_id} used_input_tokens=${v.used_input_tokens} est_input_tokens=${v.estimate_input_tokens} cap=${v.cap_input_tokens} model=${model} caller=${req.caller ?? "unknown"} task_type=${req.task_type ?? "unknown"}${turnTag}`);
  const where = turn === undefined ? "" : ` at tool-loop turn ${turn}`;
  return {
    resolved: false, shape: "llmCompletion", dispatch_budget_exhausted: true,
    dispatch_budget: { ...v, ...(turn === undefined ? {} : { turn }), policy_field: "llmModelPolicy.max_input_tokens_per_dispatch" },
    error: `dispatch input-token budget spent${where}: the input tokens this dispatch has already sent across its turns and calls, plus this request, would exceed its ${v.cap_input_tokens}-token allowance (llmModelPolicy.max_input_tokens_per_dispatch); refused before the provider call, no fallback`,
  };
}

/**
 * Runs one provider call under the dispatch budget: check, reserve the estimate, call, settle the
 * reservation to the reported input tokens (refund on throw). With no id it is a plain call.
 */
export async function budgetedProviderCall<T>(args: {
  ledger: DispatchLedger;
  id: string | null;
  estimate: number;
  readCap: () => Promise<number>;
  call: () => Promise<T>;
  inputTokensOf: (response: T) => number;
}): Promise<{ refused: DispatchBudgetVerdict } | { response: T }> {
  const { ledger, id, estimate } = args;
  if (id === null) return { response: await args.call() };
  const cap = await args.readCap();
  const verdict = checkDispatchBudget(ledger, id, estimate, cap);
  if (verdict) return { refused: verdict };
  ledger.charge(id, estimate);
  let response: T;
  try {
    response = await args.call();
  } catch (err) {
    ledger.charge(id, -estimate);
    throw err;
  }
  let actual = estimate;
  try {
    const n = Number(args.inputTokensOf(response));
    if (Number.isFinite(n) && n >= 0) actual = n;
  } catch { /* keep the estimate when usage is unreadable */ }
  ledger.charge(id, actual - estimate, true);
  return { response };
}
