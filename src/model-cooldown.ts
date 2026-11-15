// A COOLED MODEL IS NOT CALLED, AND ITS COOLDOWN DOES NOT SLIDE.
//
// When a model fails over (a 402, a daily cap), it is marked cooling for a while. Two defects let an outage feed
// itself (live, node1, after a credit ran short: 24 of 24 calls failed, the same model re-marked every ~8 s):
// - the request path tried fallbacks first but then called the cooled model anyway (and the pass-through to a catch-all
//   provider never checked at all), so every call paid a round trip and a 402;
// - each re-mark set the expiry to now + the cooldown, so while traffic continued the cooldown never ended, and the
//   plane never fully de-advertised for callers to be routed elsewhere.
// Now a model inside its cooldown is answered `unserved` at once, without a call, once no fallback serves; and a mark
// while a cooldown is still running leaves its expiry as it is (the same outage, seen again).

/** The expiry after a mark at `now` for `ms`: a cooldown still running is left as it is. */
export const markedUntil = (existing: number | undefined, now: number, ms: number): { until: number; changed: boolean } =>
  existing !== undefined && existing > now ? { until: existing, changed: false } : { until: now + ms, changed: true };

/** The answer for a model inside its cooldown: unserved, said at once, naming when it ends. */
export const coolingRefusal = (model: string, until: number) => ({
  resolved: false as const, shape: "llmCompletion" as const, unserved: "llm_completion" as const, requested_model: model,
  cooling_until: new Date(until).toISOString(),
  error: `model '${model}' is cooling until ${new Date(until).toISOString()} (it failed over); not called`,
});
