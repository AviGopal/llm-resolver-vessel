import { describe, it, expect } from 'bun:test';
import { readFileSync } from 'node:fs';

/**
 * A RESOLVER THAT REFUSES MUST STOP ADVERTISING WHAT IT REFUSED.
 *
 * decideLastResort() is a pure decision function. When it returns `refuse` it has just
 * established — against the same willingness predicate the router uses — that no arm here can
 * serve a completion. Before this fix that conclusion died in a console.warn: nothing was
 * marked exhausted, hasCompletionQuota() never changed, and llm_completion stayed advertised
 * by a resolver that had just said it could not serve it.
 *
 * WHY THE GATE COULD NOT CLOSE ON ITS OWN. hasCompletionQuota() is:
 *
 *   [...modelClientMap.keys()].some((m) => !inModelCooldown(m)) ||
 *   (anthropic !== null && !inCooldown("anthropic")) ||
 *   (openaiClient !== null && !inCooldown("openai"))
 *
 * Measured on a spoke 2026-08-18: ANTHROPIC_API_KEY set but 401-invalid, OPENROUTER_API_KEY
 * valid but 402. The openrouter models cooled, so clause 1 went false. Clause 2 stayed TRUE
 * forever, because the anthropic client is non-null whenever the KEY IS SET — validity is never
 * checked — and a 401 is neither a quota error (isExhaustedProviderError) nor a reachability
 * error (isUnreachableProviderError). The lane is cooled only on the resolve path, which a
 * refusal never reaches. An invalid credential is MORE disqualifying than exhausted quota, and
 * it was the one failure mode that could never close the gate.
 *
 * ★ THE BLAST RADIUS WAS REMOTE. Both hub-egress fallbacks in development-vessel
 *   (patch-with-tools.ts, llm-completion-dispatch.ts) are gated on "no llm arm is discoverable
 *   locally", and their own comments name the assumption: "the local resolver de-advertises
 *   llm_completion on quota/credit exhaustion". That assumption was false, so the fallbacks
 *   never engaged and every spoke caller was routed into the dead local arm while a funded arm
 *   sat on the hub. A false advertisement here disables failover in a different repo.
 */

const SRC = new URL('./index.ts', import.meta.url).pathname;
const source = () => readFileSync(SRC, 'utf8');

const refusalBlock = (): string => {
  const s = source();
  const i = s.indexOf('if ("refuse" in choice) {');
  expect(i).toBeGreaterThan(-1);
  return s.slice(i, i + 3000);
};

describe('a last-resort refusal de-advertises the completion shapes', () => {
  it('guards the instrument: the refusal branch is findable and is the real one', () => {
    const s = source();
    expect(s).toContain('decideLastResort');
    expect(refusalBlock()).toContain('choice.refuse');
  });

  it('THE REGRESSION: refusing marks the lanes exhausted', () => {
    const b = refusalBlock();
    // Lanes, not models: the refusal is a statement about lanes, and clause 2/3 of
    // hasCompletionQuota() are lane-keyed. Cooling models alone cannot close the gate.
    expect(b).toMatch(/markExhausted\("anthropic"\)/);
    expect(b).toMatch(/markExhausted\("openai"\)/);
  });

  it('the marking happens BEFORE the error returns', () => {
    const b = refusalBlock();
    const mark = b.indexOf('markExhausted("anthropic")');
    const ret = b.indexOf('return { resolved: false');
    expect(mark).toBeGreaterThan(-1);
    expect(ret).toBeGreaterThan(mark);
  });

  it('markExhausted is the mechanism that re-syncs advertisement and schedules resume', () => {
    // The fix relies on markExhausted doing three things. If any is removed, refusing would
    // cool a lane without dropping the shape — silently reintroducing the false advertisement.
    const s = source();
    const i = s.indexOf('const markExhausted =');
    const body = s.slice(i, i + 900);
    expect(body).toContain('syncCompletionAdvertisement()');
    expect(body).toContain('scheduleAdvertisementResume(');
    // Monotonic: a refusal must never SHORTEN a longer window already in force.
    expect(body).toMatch(/if \(\(exhaustedUntil\.get\(key\) \?\? 0\) >= until\) return;/);
  });

  it('RECOVERY IS NOT STRANDED — a returning key re-advertises without traffic', () => {
    // A de-advertised shape receives no completion traffic, so nothing would arrive to clear a
    // passively-expiring cooldown. If this ever regresses, an outage becomes permanent.
    const s = source();
    expect(s).toContain('scheduleAdvertisementResume');
    expect(s).toMatch(/clearExhausted[\s\S]{0,200}syncCompletionAdvertisement\(\)/);
  });

  it('the gate still advertises when SOMETHING is willing', () => {
    // The fix must not de-advertise unconditionally — that would take out a healthy plane.
    // decideLastResort dials rather than refuses whenever a willing model exists, and only the
    // refusal branch marks. Pin that the dial path carries no marking.
    const s = source();
    const i = s.indexOf('if ("refuse" in choice) {');
    const before = s.slice(Math.max(0, i - 1200), i);
    expect(before).not.toContain('markExhausted("anthropic")');
  });

  it('NEGATIVE CONTROL: the assertions reject the pre-fix shape', () => {
    const preFix = `
    if ("refuse" in choice) {
      console.warn(\`[llm-resolver-vessel] \${choice.refuse}\`);
      return { resolved: false, shape: "llmCompletion", error: choice.refuse };
    }`;
    expect(/markExhausted\("anthropic"\)/.test(preFix)).toBe(false);
    expect(/markExhausted\("openai"\)/.test(preFix)).toBe(false);
  });
});
