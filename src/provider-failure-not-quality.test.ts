// Pins the rule that a PROVIDER failure must not move a model's quality posterior.
//
// THE DEFECT: the outcome recorded for an arm was `resolved === true`, so a call
// the provider REFUSED — billing rejection, unreachable host — incremented beta
// exactly like a model that answered badly. Measured 2026-08-11 after a credit
// outage: claude-sonnet-5 alpha 1 / beta 10.24, claude-haiku-4-5 alpha 1.90 /
// beta 40.05, and selection fell through to a 30B free model for ALL fleet
// drafting. The outage taught the learner, not the models.
//
// isFailoverError already classified this class and already gated the cooldown;
// it was simply never consulted on the learning path.
//
// Imports the REAL predicate. An earlier version of this file re-implemented the
// patterns inline and passed while proving nothing — a regex in isolation lies
// about what the function does.
import { describe, expect, test } from "bun:test";
import { isFailoverError, isExhaustedProviderError, isUnreachableProviderError } from "./provider-errors";

describe("the errors that actually poisoned the arms are classified provider-level", () => {
  test("the observed Anthropic billing rejection", () => {
    // Verbatim shape of the error that drove sonnet to beta 10.24.
    expect(
      isFailoverError(
        '400 {"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}',
      ),
    ).toBe(true);
  });

  test("an unreachable provider / dead relay", () => {
    expect(isFailoverError("fetch failed: ECONNREFUSED 127.0.0.1:8401")).toBe(true);
    expect(isFailoverError("ingress proxy failed: The operation timed out.")).toBe(true);
  });

  test("quota and rate-limit exhaustion", () => {
    expect(isExhaustedProviderError("429 free-models-per-day limit reached")).toBe(true);
    expect(isUnreachableProviderError("503 Service Unavailable")).toBe(true);
  });
});

describe("the guard must NOT become a blanket amnesty", () => {
  // If everything counted as provider-level, a genuinely weak arm would never
  // earn beta and the cap on bad models would silently disappear.
  test("a model that answered badly is still graded", () => {
    expect(isFailoverError("model returned an unparseable body")).toBe(false);
    expect(isFailoverError("completion was empty")).toBe(false);
    expect(isFailoverError("the patch did not typecheck")).toBe(false);
  });

  test("token counts containing 402/429/404 do not false-positive", () => {
    // The standalone-status-code regexes exist for exactly this: a bare substring
    // match would cool a healthy provider on a token count.
    expect(isFailoverError("used 14290 input tokens")).toBe(false);
    expect(isFailoverError("used 140250 input tokens")).toBe(false);
  });

  test("empty and nullish errors are not provider-level", () => {
    expect(isFailoverError(undefined)).toBe(false);
    expect(isFailoverError(null)).toBe(false);
    expect(isFailoverError("")).toBe(false);
  });
});
