/**
 * provider-standing.ts - per-provider CREDIT and FREE-CAP standing, derived from the provider
 * calls this vessel already makes.
 *
 * WHY. The vessel cools a failing MODEL (index.ts modelExhaustedUntil) and walks the fallback
 * chain, so a paid provider answering 402 on every model is absorbed silently: each completion
 * is served by some fallback, the provider-level `cooldown_until_ms` in llmQuotaState stays null
 * (it is only set on the anthropic/openai primary path), and nothing shaped says the provider is
 * out of credit or that the account-wide `:free` daily cap has been hit. Selection cannot read a
 * condition nothing reports.
 *
 * WHAT. One record per provider, kept in process and served additively inside the existing
 * llmQuotaState body. It holds status enums, timestamps, counts and model ids ONLY: the tracker
 * classifies an error and keeps the class, never the error text, so a provider message that
 * echoes a key or a URL cannot reach the served body.
 *
 *  - credit: "exhausted" only once 402-class failures have persisted for longer than one
 *    exhaustion cooldown cycle (a model cooled by a 402 came back and 402'd again). A single 402
 *    that clears within the cycle is a blip, not a standing condition, and reads "ok".
 *  - free_cap: "capped" on the explicit free daily-cap 429 (`free-models-per-day`, or a daily
 *    limit on a `:free` slug). A plain rate-limit 429 changes nothing.
 *  - recovery: a paid model of the provider serving clears credit; a `:free` model serving
 *    clears free_cap. fallback_model names the model that last served in place of one of the
 *    provider's models, and clears when the provider is back in standing.
 *
 * Extracted from index.ts so it can be tested directly (index.ts starts a server at import time),
 * following last-resort, plane-outage and scale-to-zero.
 */

export type StandingKind = "credit" | "free_cap";

export const isFreeSlug = (model: string): boolean => model.endsWith(":free");

/** The standing class of a provider error for `model`, or null when it is neither. */
export function classifyStandingError(model: string, error: unknown): StandingKind | null {
  const m = String(error ?? "").toLowerCase();
  if (m === "") return null;
  // The daily free cap first: OpenRouter's message for it also talks about buying credits.
  if (m.includes("free-models-per-day")) return "free_cap";
  if (isFreeSlug(model) && (m.includes("limit_rpd") || m.includes("daily limit reached"))) return "free_cap";
  if (
    // Standalone status code only, the same guard provider-errors.ts uses, so a token count
    // like "140250" cannot read as a 402.
    /(?:^|[^0-9])402(?:[^0-9]|$)/.test(m) ||
    m.includes("credit balance") ||
    m.includes("insufficient credit") ||
    m.includes("insufficient_quota") ||
    m.includes("exceeded your current quota")
  ) return "credit";
  return null;
}

/** The keys the standing adds to a provider's llmQuotaState entry: an allowlist, pinned in tests. */
export const STANDING_FIELDS = [
  "credit", "since", "credit_402s", "last_402_at",
  "free_cap", "free_cap_since", "fallback_model", "cooling_models",
] as const;

export interface StandingEntry {
  credit: "ok" | "exhausted";
  /** First 402 of the episode, while credit is exhausted; else null. */
  since: string | null;
  /** 402-class failures in the current episode (0 after a paid success). */
  credit_402s: number;
  last_402_at: string | null;
  free_cap: "ok" | "capped";
  free_cap_since: string | null;
  /** The model that last served in place of one of this provider's models. */
  fallback_model: string | null;
  /** This provider's models in cooldown right now (supplied by the caller, which owns cooldowns). */
  cooling_models: string[];
}

interface Row {
  credit402s: number;
  first402At: number | null;
  last402At: number | null;
  freeCapSince: number | null;
  lastFreeCapAt: number | null;
  fallbackModel: string | null;
}

const iso = (ms: number | null): string | null => (ms === null ? null : new Date(ms).toISOString());

export class ProviderStandingTracker {
  private readonly providerOfModel = new Map<string, string>();
  private readonly rows = new Map<string, Row>();

  /** Register a provider's models, in the same order and with the same overwrite rule as the
   * caller's model->client map, so a model resolves to the provider that actually serves it. */
  registerModels(provider: string, models: readonly string[]): void {
    for (const m of models) this.providerOfModel.set(m, provider);
  }

  providerOf(model: string): string | null {
    const registered = this.providerOfModel.get(model);
    if (registered) return registered;
    if (model.startsWith("claude") || model.startsWith("anthropic/")) return "anthropic";
    return null;
  }

  private row(provider: string): Row {
    let r = this.rows.get(provider);
    if (!r) {
      r = { credit402s: 0, first402At: null, last402At: null, freeCapSince: null, lastFreeCapAt: null, fallbackModel: null };
      this.rows.set(provider, r);
    }
    return r;
  }

  /** A provider call for `model` failed with `error`. Returns the class recorded, if any. */
  recordFailure(provider: string, model: string, error: unknown, nowMs: number): StandingKind | null {
    const kind = classifyStandingError(model, error);
    if (kind) this.recordKind(provider, kind, nowMs);
    return kind;
  }

  /** Record a failure whose class the caller already knows (e.g. the anthropic credit-dead path). */
  recordKind(provider: string, kind: StandingKind, nowMs: number): void {
    const r = this.row(provider);
    if (kind === "credit") {
      r.credit402s += 1;
      if (r.first402At === null) r.first402At = nowMs;
      r.last402At = nowMs;
    } else {
      if (r.freeCapSince === null) r.freeCapSince = nowMs;
      r.lastFreeCapAt = nowMs;
    }
  }

  /** `model` of `provider` served a completion. */
  recordServed(provider: string, model: string): void {
    const r = this.rows.get(provider);
    if (!r) return;
    if (isFreeSlug(model)) {
      r.freeCapSince = null;
      r.lastFreeCapAt = null;
    } else {
      r.credit402s = 0;
      r.first402At = null;
      r.last402At = null;
    }
    if (r.credit402s === 0 && r.freeCapSince === null) r.fallbackModel = null;
  }

  /** `servedModel` served a completion in place of `fromModel` (a fallback rescue). */
  recordFallback(fromModel: string, servedModel: string): void {
    const provider = this.providerOf(fromModel);
    if (!provider || fromModel === servedModel) return;
    const r = this.row(provider);
    if (r.credit402s > 0 || r.freeCapSince !== null) r.fallbackModel = servedModel;
  }

  /** The standing fields for `provider`, exactly STANDING_FIELDS. */
  entry(provider: string, cooldownMs: number, coolingModels: readonly string[]): StandingEntry {
    const r = this.rows.get(provider);
    const persistent = !!r && r.credit402s >= 2 && r.first402At !== null && r.last402At !== null
      && r.last402At - r.first402At > cooldownMs;
    return {
      credit: persistent ? "exhausted" : "ok",
      since: persistent ? iso(r!.first402At) : null,
      credit_402s: r?.credit402s ?? 0,
      last_402_at: iso(r?.last402At ?? null),
      free_cap: r && r.freeCapSince !== null ? "capped" : "ok",
      free_cap_since: iso(r?.freeCapSince ?? null),
      fallback_model: r?.fallbackModel ?? null,
      cooling_models: [...coolingModels],
    };
  }
}

/** The vessel's one tracker, shared by index.ts (which feeds and serves it) and the selector. */
export const providerStanding = new ProviderStandingTracker();
