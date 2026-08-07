/**
 * THE REASONING PLANE CAN GO FULLY DARK WITHOUT ANYTHING FILING A GAP.
 *
 * Observed 2026-08-07. Every arm — llm-resolver-vessel, -opus, -haiku, -google — returned the
 * same body on every call:
 *
 *     "Your credit balance is too low to access the Anthropic API."
 *
 * `llmQuotaState` said it plainly: anthropic present and in cooldown, chutes / groq / mistral /
 * openrouter / google all `present:false`. Quota-gated advertisement then did its job and
 * dropped `llm_completion` from the registry — correctly, so callers route to a producer with
 * quota. But on a substrate where no other arm has a key, "route elsewhere" resolves to
 * nowhere, and every caller degraded in silence: reach grading fell back to deterministic
 * oracles only, command synthesis stopped, and the reach numbers being collected quietly
 * stopped being comparable to the ones before the outage.
 *
 * De-advertising is the correct REACTION and a poor DETECTION. The registry losing a shape is
 * visible to whoever looks; nothing looks. Law 6: every observed bug class gets asked what
 * activity would detect it without an operator — and a reasoning plane failing 100% of calls
 * should be the loudest gap in the store, not an absence.
 *
 * Deliberately pure and LLM-free: the detector for a dead LLM plane must not need the LLM
 * plane. That is the same circularity the bootstrap-tier carve-out exists for — a check cannot
 * be scheduled by the mechanism it exists to recover.
 */

export type ProviderState = { present: boolean; cooldown_until_ms: number | null };

export type PlaneVerdict =
  | { dark: true; id: string; summary: string }
  | { dark: false; reason: string };

/**
 * Is the completion plane dark — no provider that could serve a completion right now?
 *
 * Dark means every provider is either absent (no key configured) or in cooldown. The
 * distinction is load-bearing and belongs in the summary: an ABSENT provider needs a
 * credential and will never recover on its own, while a COOLING one recovers by waiting. A gap
 * that says only "the plane is down" cannot tell the reader which of those they are looking
 * at, and they are not the same problem.
 */
export function classifyPlane(providers: Record<string, ProviderState>, nowMs: number): PlaneVerdict {
  const entries = Object.entries(providers);
  if (entries.length === 0) return { dark: false, reason: "no provider table to judge" };

  const usable = entries.filter(([, s]) => s.present && !(s.cooldown_until_ms !== null && s.cooldown_until_ms > nowMs));
  if (usable.length > 0) {
    return { dark: false, reason: `${usable.length} provider(s) usable: ${usable.map(([id]) => id).join(", ")}` };
  }

  const cooling = entries.filter(([, s]) => s.present && s.cooldown_until_ms !== null && s.cooldown_until_ms > nowMs);
  const absent = entries.filter(([, s]) => !s.present);

  // The recovery story differs entirely between these two, so say which applies.
  const recovery = cooling.length > 0
    ? `${cooling.length} provider(s) are CONFIGURED but cooling (${cooling.map(([id]) => id).join(", ")}) — these recover on their own when the cooldown expires, unless the cause is a credit balance, which does not expire.`
    : `NO provider is configured at all — nothing here recovers without a credential.`;

  const summary = `The LLM completion plane is DARK: no provider can serve a completion. `
    + `cooling=[${cooling.map(([id]) => id).join(", ") || "none"}] absent=[${absent.map(([id]) => id).join(", ") || "none"}]. `
    + `${recovery} `
    + `Quota-gated advertisement has dropped llm_completion from the registry, which is the correct reaction and a poor detection: every caller now degrades silently — reach grading falls back to deterministic oracles only, command synthesis stops, and reach numbers collected in this state are NOT comparable to numbers collected with a live plane. `
    + `This gap closes by itself when any provider becomes usable.`;

  // STABLE id so re-emission upserts one row instead of flooding the store, matching the
  // dedup discipline the capability/reachability gaps already use.
  return { dark: true, id: "llm-completion-plane-dark", summary };
}
