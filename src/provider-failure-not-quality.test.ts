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
import { isFailoverError, isExhaustedProviderError, isUnreachableProviderError, isUnauthenticatedProviderError } from "./provider-errors";

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

// ---------------------------------------------------------------------------
// A CREDENTIAL FAILURE IS A PROVIDER FAILURE. It was in neither list, so the
// whole failover machinery — which was already built and already correct — never
// engaged. Measured 2026-08-12: 566 401s in one day, 43 hours continuous, zero
// successful completions fleet-wide, while a funded OpenRouter client sat idle.
//
// These are the LITERAL strings from the live journal and from a probe I ran
// against each arm. A paraphrase would pass and prove nothing.
// ---------------------------------------------------------------------------
describe("an invalid credential is classified provider-level, so failover engages", () => {
  const OBSERVED_401 =
    '401 {"type":"error","error":{"type":"authentication_error","message":"API key is invalid."},"request_id":null}';
  const OBSERVED_PEER_UNCONFIGURED =
    "No LLM provider configured. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY.";

  test("the observed Anthropic 401 — the string that ran the fleet dry for 43 hours", () => {
    expect(isUnauthenticatedProviderError(OBSERVED_401)).toBe(true);
    expect(isFailoverError(OBSERVED_401)).toBe(true);
  });

  test("the observed peer arm with no provider configured", () => {
    expect(isUnauthenticatedProviderError(OBSERVED_PEER_UNCONFIGURED)).toBe(true);
    expect(isFailoverError(OBSERVED_PEER_UNCONFIGURED)).toBe(true);
  });

  test("REGRESSION PIN: before this change isFailoverError was false for both", () => {
    // The two pre-existing predicates must still NOT match them — that is the
    // defect being fixed, and if either starts matching, the classes have blurred
    // and cooldownMsFor will hand out the wrong retry window.
    expect(isExhaustedProviderError(OBSERVED_401)).toBe(false);
    expect(isUnreachableProviderError(OBSERVED_401)).toBe(false);
  });

  test("other credential phrasings across providers", () => {
    for (const s of [
      "Incorrect API key provided",
      "invalid x-api-key",
      '{"error":{"code":"invalid_api_key"}}',
      "401 Unauthorized",
      "403 Forbidden",
    ]) expect(isUnauthenticatedProviderError(s)).toBe(true);
  });

  test("CONTROL: a token count containing 401 or 403 must NOT cool a healthy provider", () => {
    // The bare-substring version of this rule is why 402 and 429 are regex-guarded
    // in the same file. A usage line is the exact shape that trips it.
    expect(isUnauthenticatedProviderError("completed: 14012 prompt tokens, 240 output")).toBe(false);
    expect(isUnauthenticatedProviderError("total_tokens: 4403")).toBe(false);
    expect(isUnauthenticatedProviderError("finished in 401ms")).toBe(false);
  });

  test("CONTROL: a genuine model-quality failure is still NOT provider-level", () => {
    expect(isUnauthenticatedProviderError("the model returned malformed JSON")).toBe(false);
    expect(isFailoverError("the model returned malformed JSON")).toBe(false);
  });
});
