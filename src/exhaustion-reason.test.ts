// A cooled lane says why: the provider's status, code and message, redacted, on every exhaustion mark (research; node1's
// paid calls failed after a credit top-up with only "marked exhausted" logged).
import { describe, expect, test } from "bun:test";
import { completionChain, textTurn } from "./completion-chain-harness";
import { loadIndexDeclarations } from "./index-fn-harness";
import { reasonOf, reasonText } from "./exhaustion-reason";

// The shape the OpenAI SDK throws: an Error with the HTTP status and the provider's error body.
const apiError = (status: number, message: string, error: Record<string, unknown>) => Object.assign(new Error(message), { status, error });
const X = "x/paid-arm";
const Y = "y/paid-arm";
const body = { type: "llm_completion", prompt: "write the function" };

describe("the provider's answer, as a reason", () => {
  test("[MUST-FAIL] a 402 with a body: its status, code and message", () => {
    const r = reasonText(reasonOf(apiError(402, "402 Insufficient credits. Add more credits.", { code: 402, message: "Insufficient credits" })));
    expect(r).toBe("provider answered 402 402: 402 Insufficient credits. Add more credits.");
  });

  test("[MUST-FAIL] a credential-shaped string in the provider's message is redacted", () => {
    const r = reasonText(reasonOf(apiError(401, "401 Incorrect API key provided: sk-" + "or-v1-FAKE0123", { code: "invalid_api_key" })));
    expect(r).toContain("401 invalid_api_key");
    expect(r).not.toContain("FAKE0123");
    expect(r).toContain("[redacted]");
  });

  test("[MUST-FAIL] a credential-shaped provider error code is redacted too (qa)", () => {
    const r = reasonText(reasonOf(apiError(401, "401 rejected", { code: "sk-" + "or-v1-FAKEcode" })));
    expect(r).not.toContain("FAKEcode");
    expect(r).toBe("provider answered 401 [redacted]: 401 rejected");
  });

  test("a result with no provider answer says so", () => {
    expect(reasonText(undefined)).toBe("no provider answer recorded");
    expect(reasonText({ status: 500 })).toBe("no provider answer recorded");
  });
});

describe("the exhaustion marks", () => {
  test("[MUST-FAIL] a failed wire call's mark carries the provider's answer", async () => {
    const c = completionChain({ models: [[X, [apiError(402, "402 Insufficient credits", { code: 402 })]], [Y, [textTurn("served by y")]]] });
    const r = await c.fns.llmCompletionHandler({ body: { ...body, model: X } });
    expect(r.resolved).toBe(true);
    expect(c.exhaustionMarks).toEqual([X]);
    expect(c.exhaustionWhys).toEqual(["provider answered 402 402: 402 Insufficient credits"]);
  });

  test("[MUST-FAIL] a FALLBACK model's mark carries that model's own answer, not the drawn one's (qa)", async () => {
    const Z = "z/paid-arm";
    const c = completionChain({ models: [
      [X, [apiError(402, "402 Insufficient credits", { code: 402 })]],
      [Y, [apiError(403, "403 Key limit exceeded", { code: "limit_exceeded" })]],
      [Z, [textTurn("served by z")]],
    ] });
    const r = await c.fns.llmCompletionHandler({ body: { ...body, model: X } });
    expect(r.resolved).toBe(true);
    expect(c.exhaustionMarks).toEqual([X, Y]);
    expect(c.exhaustionWhys[1]).toBe("provider answered 403 limit_exceeded: 403 Key limit exceeded");
  });

  test("[MUST-FAIL] the PRIMARY lane's provider mark carries the provider's answer (qa)", async () => {
    const c = completionChain({ openai: [apiError(429, "429 Rate limit reached", { code: "rate_limit_exceeded" })], models: [[Y, [textTurn("served by y")]]] });
    const r = await c.fns.llmCompletionHandler({ body: { ...body, model: "primary-model" } });
    expect(r.resolved).toBe(true);
    expect(c.exhaustionMarks[0]).toBe("openai");
    expect(c.exhaustionWhys[0]).toBe("provider answered 429 rate_limit_exceeded: 429 Rate limit reached");
  });

  test("[MUST-FAIL] the primary lane's RETRY mark carries the retry's own answer, not the cooled first result's (qa)", async () => {
    // first call: the primary answers 429 and its lane cools; second call: the lane is cooling, nothing else serves, so
    // the primary is tried anyway and answers 402. That mark's why is the 402.
    const c = completionChain({ openai: [apiError(429, "429 Rate limit reached", { code: "rate_limit_exceeded" }),
      apiError(402, "402 Insufficient credits", { code: 402 })], models: [] });
    await c.fns.llmCompletionHandler({ body: { ...body, model: "primary-model" } });
    const r = await c.fns.llmCompletionHandler({ body: { ...body, model: "primary-model" } });
    expect(r.resolved).toBe(false);
    expect(c.exhaustionMarks).toEqual(["openai", "openai"]);
    expect(c.exhaustionWhys[1]).toBe("provider answered 402 402: 402 Insufficient credits");
  });

  test("[MUST-FAIL] the anthropic catch puts the provider's answer on the result (qa)", async () => {
    const fns = loadIndexDeclarations(["resolveWithAnthropicUnobserved"], {
      reasonOf, reasonText,
      console: { log() {}, warn() {}, error() {} },
      anthropic: { messages: { create: async () => { throw apiError(429, "429 Rate limited", { error: { type: "rate_limit_error" } }); } } },
      RETIRED_ANTHROPIC_MODEL_IDS: new Set(), DEFAULT_MODEL: "claude-test", DEFAULT_MAX_TOKENS: 64,
      budgetedProviderCall: async (o: { call: () => Promise<unknown> }) => ({ response: await o.call() }),
      dispatchLedger: {}, dispatchKeyOf: () => "d", estimateInputTokens: () => 1, readMaxInputTokensPerDispatch: async () => null,
      cachedSystem: () => ({}), anthropicCreditFallback: async () => null, anthropicInputTokens: () => 1,
    });
    const r = await fns.resolveWithAnthropicUnobserved({ prompt: "p", model: "claude-test" });
    // (a provider code of 20 or more characters, such as invalid_request_error, reads [redacted]: the shared redaction's
    // length rule; the message still says the cause)
    expect(r.provider_reason).toEqual({ status: 429, code: "rate_limit_error", message: "429 Rate limited" });
  });

  test("[MUST-FAIL] the model mark's log line says why (the shipped markModelExhausted)", () => {
    const lines: string[] = [];
    const fns = loadIndexDeclarations(["markModelExhausted"], {
      console: { warn: (l: string) => lines.push(l), log() {}, error() {} },
      EXHAUSTION_COOLDOWN_MS: 600_000,
      markedUntil: (_prev: unknown, now: number, ms: number) => ({ until: now + ms, changed: true }),
      modelExhaustedUntil: new Map(),
      syncCompletionAdvertisement: async () => {},
      scheduleAdvertisementResume: () => {},
    });
    fns.markModelExhausted(X, "provider answered 402 402: 402 Insufficient credits", 600_000);
    expect(lines).toEqual([`[llm-resolver-vessel] model '${X}' marked exhausted — cooling down 600s — provider answered 402 402: 402 Insufficient credits`]);
  });

  test("[MUST-FAIL] the provider mark's log line says why (the shipped markExhausted)", () => {
    const lines: string[] = [];
    const fns = loadIndexDeclarations(["markExhausted"], {
      console: { warn: (l: string) => lines.push(l), log() {}, error() {} },
      EXHAUSTION_COOLDOWN_MS: 600_000,
      exhaustedUntil: new Map(),
      syncCompletionAdvertisement: async () => {},
      scheduleAdvertisementResume: () => {},
      Date,
    });
    fns.markExhausted("openrouter", "provider answered 403 403: Key limit exceeded", 600_000);
    expect(lines).toEqual(["[llm-resolver-vessel] provider 'openrouter' marked exhausted — cooling down 600s before retry — provider answered 403 403: Key limit exceeded"]);
  });
});
