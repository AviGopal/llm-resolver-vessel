/**
 * Classify a provider-level failure — an error about the PROVIDER (billing,
 * quota, reachability), not about the quality of what a model produced.
 *
 * Extracted from index.ts so it can be tested directly. index.ts starts a server
 * at import time, so anything left in it is only testable through a copy of the
 * predicate — and a copy is exactly how a classifier drifts from the thing it is
 * supposed to classify. The vessel's other extracted helpers (last-resort,
 * plane-outage, scale-to-zero) follow the same convention.
 *
 * These predicates already gated FAILOVER and COOLDOWN. They are now also
 * consulted on the LEARNING path: a call the provider refused must not increment
 * a model's beta, because that records an account state as a quality judgement.
 * Measured 2026-08-11 after a credit outage — claude-sonnet-5 at alpha 1 / beta
 * 10.24 and claude-haiku-4-5 at alpha 1.90 / beta 40.05, with all fleet drafting
 * falling through to a 30B free model.
 */
export const isExhaustedProviderError = (e: unknown): boolean => {
  const m = String(e ?? "").toLowerCase();
  return (
    m.includes("credit balance") ||
    m.includes("insufficient_quota") ||
    m.includes("exceeded your current quota") ||
    m.includes("usage cap") ||
    // 402 as a standalone status code only — a bare substring match would
    // false-positive on token counts like "140250" and cool a healthy
    // provider down for the whole window.
    /(?:^|[^0-9])402(?:[^0-9]|$)/.test(m) ||
    m.includes("billing") ||
    m.includes("limit_rpd") ||
    m.includes("limit_rpm") ||
    m.includes("daily limit reached") ||
    m.includes("rate limit") ||
    m.includes("too many requests") ||
    m.includes("free-models-per-day") ||
    // 429 as a standalone status code only (regex-guarded like 402 above so a
    // token count like "14290" cannot false-positive a healthy provider).
    /(?:^|[^0-9])429(?:[^0-9]|$)/.test(m) ||
    // A "free" model that 404s with "unavailable for free / use the paid slug /
    // no endpoints" (openrouter free-tier drift) is dead to US — cool the MODEL
    // so the failover walk fires and the quota-gate de-advertises the arm,
    // instead of sinking the dispatch in ~30ms and keeping a dead arm falsely
    // advertised (which strands a spoke on its dead local arm instead of routing
    // through discovery to a quota-having producer). Narrowed to a standalone 404
    // AND a paid/unavailable phrase so a genuine wrong-model 404 on a healthy
    // provider does not cool the whole lane.
    (/(?:^|[^0-9])404(?:[^0-9]|$)/.test(m) &&
      (m.includes("unavailable") ||
        m.includes("paid") ||
        m.includes("use this slug") ||
        m.includes("no endpoints"))) ||
    m.includes("not a valid model") ||
    m.includes("model_not_found") ||
    m.includes("invalid model id")
  );
};

// Reachability failures — distinct from billing exhaustion. A self-hosted vLLM
// instance on a spot marketplace is frequently down/restarting, so a TCP/DNS
// error, gateway 5xx, or SDK "Connection error" must trigger the SAME failover
// walk (don't hard-error to the caller when another arm can serve) but with a
// short cooldown so the endpoint rejoins rotation quickly once it's back.
export const isUnreachableProviderError = (e: unknown): boolean => {
  const m = String(e ?? "").toLowerCase();
  return (
    m.includes("fetch failed") ||
    m.includes("connection error") ||
    m.includes("econnrefused") ||
    m.includes("econnreset") ||
    m.includes("etimedout") ||
    m.includes("enotfound") ||
    m.includes("eai_again") ||
    m.includes("socket hang up") ||
    m.includes("network error") ||
    // "timed out" rather than "request timed out": Bun's AbortSignal.timeout
    // rejects with `TimeoutError: The operation timed out.`, and the federation
    // ingress surfaces `ingress proxy failed: The operation timed out.` — neither
    // contains the word "request", so BOTH fell through this classifier. Every
    // relay timeout observed on 2026-08-11 was therefore treated as a genuine
    // model failure: no cooldown, no failover credit, and (once this predicate
    // was wired to the learning path) a beta increment charged to whichever model
    // happened to be selected. Caught by pinning the REAL predicate against the
    // literal strings from the logs; the narrower phrase looked correct in review.
    m.includes("timed out") ||
    m.includes("timeouterror") ||
    m.includes("terminated") ||
    // Gateway/edge unavailability (Cloudflare Tunnel origin down, vLLM loading
    // weights) — standalone status codes only, regex-guarded so a token count
    // like "50231" cannot false-positive. Includes Cloudflare's origin-error
    // range 520-527 + 530: a self-hosted pod whose CF tunnel origin is down
    // returns 530 (added 2026-07-31 — a dead q3-32b pod was returning 530, which
    // matched none of these, so it never cooled and the MAB re-sampled it every
    // draw, burning a timeout each time; now it auto-cools like credit exhaustion).
    /(?:^|[^0-9])(?:502|503|504|52[0-7]|530)(?:[^0-9]|$)/.test(m)
  );
};

// A failover-worthy error is either kind; the cooldown length depends on which.
export const isFailoverError = (e: unknown): boolean => isExhaustedProviderError(e) || isUnreachableProviderError(e);
