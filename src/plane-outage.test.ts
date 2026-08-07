import { describe, expect, it } from "bun:test";

import { classifyPlane, type ProviderState } from "./plane-outage";

const NOW = 1_786_100_000_000;
const cooling: ProviderState = { present: true, cooldown_until_ms: NOW + 600_000 };
const expired: ProviderState = { present: true, cooldown_until_ms: NOW - 1 };
const live: ProviderState = { present: true, cooldown_until_ms: null };
const absent: ProviderState = { present: false, cooldown_until_ms: null };

describe("classifyPlane — the detector for a dead plane, which must not need the plane", () => {
  it("OBSERVED LIVE 2026-08-07: anthropic cooling, every other provider absent", () => {
    // The exact llmQuotaState body from the outage. Every arm returned the same credit error,
    // llm_completion left the registry, and nothing filed a gap.
    const v = classifyPlane({
      chutes: absent, groq: absent, mistral: absent, openrouter: absent, google: absent,
      anthropic: { present: true, cooldown_until_ms: 1_786_100_709_043 },
    }, NOW);
    expect(v.dark).toBe(true);
    if (!v.dark) return;
    expect(v.id).toBe("llm-completion-plane-dark");
    expect(v.summary).toMatch(/anthropic/);
    expect(v.summary).toMatch(/groq/);
  });

  it("does NOT fire while any provider can still serve", () => {
    expect(classifyPlane({ anthropic: cooling, groq: live }, NOW).dark).toBe(false);
    // A cooldown in the past is not a cooldown.
    expect(classifyPlane({ anthropic: expired }, NOW).dark).toBe(false);
  });

  it("distinguishes cooling from absent, because only one of them recovers by waiting", () => {
    const coolingOnly = classifyPlane({ anthropic: cooling }, NOW);
    expect(coolingOnly.dark).toBe(true);
    if (coolingOnly.dark) expect(coolingOnly.summary).toMatch(/CONFIGURED but cooling/);

    const absentOnly = classifyPlane({ groq: absent, mistral: absent }, NOW);
    expect(absentOnly.dark).toBe(true);
    if (absentOnly.dark) expect(absentOnly.summary).toMatch(/NO provider is configured/);
  });

  it("warns that numbers gathered during an outage are not comparable", () => {
    // The quiet damage of this outage was not the downtime — it was that reach grading kept
    // producing verdicts from deterministic oracles alone and they looked like ordinary rows.
    const v = classifyPlane({ anthropic: cooling }, NOW);
    if (v.dark) expect(v.summary).toMatch(/NOT comparable/);
  });

  it("abstains rather than firing on an empty provider table", () => {
    // An empty table means the state could not be read, which is not evidence of an outage.
    expect(classifyPlane({}, NOW).dark).toBe(false);
  });
});
