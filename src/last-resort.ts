/**
 * May the LAST-RESORT model be dialled when policy selection produced nothing?
 *
 * WHY THIS EXISTS. `llmCompletionHandler` ends every unpinned path with
 * `body.model ?? DEFAULT_MODEL`. When the policy selector returns no arm, the
 * old code fell straight through to that line and dialled the default BLINDLY.
 * With a credit-dead default that produced a tight loop — hundreds of
 * `credit balance is too low ... cooling down 1800s` lines — while a warm arm
 * sat unused because nothing had routed to it.
 *
 * A default is a LAST RESORT, not a route. It may only be dialled after passing
 * the same willingness check every other arm must pass. Willingness mirrors all
 * three routing branches (wire client, anthropic, openrouter), so refusing here
 * cannot mask a default that IS servable — the only thing it prevents is
 * dialling one that demonstrably is not.
 *
 * Extracted so the decision is testable without booting the vessel's HTTP
 * server, and so the guard is not a re-implemented copy in a test.
 */

export type LastResortChoice =
  | { readonly dial: string }
  | { readonly refuse: string };

export interface LastResortInput {
  /** The caller's pinned model, if they pinned one (absent/"auto" means unpinned). */
  readonly pinnedModel?: string | undefined;
  /** The configured fallback used when the caller pinned nothing. */
  readonly defaultModel: string;
  /** How many policy arms were considered before selection came back empty. */
  readonly armsChecked: number;
  /** Same predicate the selector uses: can this model be routed RIGHT NOW? */
  readonly isWilling: (model: string) => boolean;
  /**
   * Every model this resolver can ROUTE (the wire-client keyspace), regardless of
   * whether it is a policy arm.
   *
   * Policy arms and routable models are NOT the same set: a provider's models can
   * be routable while never having been seeded as arms. Without this, a state
   * where all ARMS are cooling but a routable model is live gets refused — the
   * plane reads as dead while `google/gemini-2.5-flash` answers on the first try.
   * Observed exactly that after tightening this guard.
   */
  readonly routableModels?: readonly string[];
}

/**
 * Decide between dialling the last-resort model and refusing honestly.
 *
 * A pinned model outranks the configured default as the last resort — the
 * caller named it, so it is the more specific intent — but it gets no exemption
 * from the willingness check.
 */
export function decideLastResort(input: LastResortInput): LastResortChoice {
  const { pinnedModel, defaultModel, armsChecked, isWilling, routableModels } = input;
  const pinned =
    typeof pinnedModel === "string" && pinnedModel.length > 0 && pinnedModel !== "auto"
      ? pinnedModel
      : undefined;
  const lastResort = pinned ?? defaultModel;

  if (!lastResort) {
    return {
      refuse:
        `no llm arm is currently servable (${armsChecked} policy arm(s) checked) ` +
        `and no last-resort model is configured — refused rather than guessing one`,
    };
  }

  if (isWilling(lastResort)) return { dial: lastResort };

  // The configured last resort is dry — but "this one model is dry" is not the
  // same claim as "nothing can serve". Before refusing, look for ANY routable
  // model that is willing. Refusing while a live model sits one lookup away is a
  // worse outage than the blind-dial bug this guard exists to prevent.
  const liveAlternative = (routableModels ?? []).find((m) => m !== lastResort && isWilling(m));
  if (liveAlternative) return { dial: liveAlternative };

  {
    return {
      refuse:
        `no llm arm is currently servable (${armsChecked} policy arm(s) checked); ` +
        `last-resort model '${lastResort}' is also unwilling (no key, cooling, or cold) — ` +
        `refused instead of dialling a known-dry model`,
    };
  }
}
