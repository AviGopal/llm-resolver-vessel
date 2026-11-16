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
