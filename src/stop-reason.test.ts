// A wire completion that reached its token cap is said cut (stop_reason "max_tokens"), whatever finish_reason the
// provider gave: the resolver holds the cap it sent and the output tokens it was billed. The provider's own word is
// kept beside it (provider_stop_reason). Run through the shipped chain (completion-chain-harness.ts), on the
// catch-all OpenRouter lane and on the credit-dead fallback.
import { describe, expect, test } from "bun:test";
import { completionChain } from "./completion-chain-harness";
import { loadIndexDeclarations } from "./index-fn-harness";
import * as empty from "./empty-completion";
import { wireStopReason } from "./stop-reason";

const VENDOR = "vendor/model-x";
const turn = (finish_reason: string, completion_tokens: number) => ({
  choices: [{ index: 0, message: { role: "assistant", content: "{\"half\": \"an obj" }, finish_reason }],
  usage: { prompt_tokens: 10, completion_tokens },
});
const ask = async (t: ReturnType<typeof turn>, extra: Record<string, unknown> = {}) => {
  const c = completionChain({ models: [], openrouter: [t] });
  return (await c.fns.llmCompletionHandler({ body: { type: "llm_completion", prompt: "p", model: VENDOR, ...extra } })) as Record<string, unknown>;
};

describe("a wire completion's stop_reason is checked against the cap it was sent", () => {
  test("[MUST-FAIL] finish_reason stop at the max_tokens sent: max_tokens, the provider's stop beside it", async () => {
    const r = await ask(turn("stop", 512), { max_tokens: 512 });
    expect(r).toMatchObject({ resolved: true, stop_reason: "max_tokens", provider_stop_reason: "stop" });
  });

  test("[MUST-FAIL] no max_tokens in the body: the default sent is the cap, and stop at it is max_tokens", async () => {
    const r = await ask(turn("stop", 4096));
    expect(r).toMatchObject({ resolved: true, stop_reason: "max_tokens", provider_stop_reason: "stop" });
  });

  test("[GUARD] stop below the cap stays stop", async () => {
    const r = await ask(turn("stop", 4095));
    expect(r).toMatchObject({ resolved: true, stop_reason: "stop" });
    expect(await ask(turn("stop", 511), { max_tokens: 512 })).toMatchObject({ stop_reason: "stop" });
  });

  test("[GUARD] length is max_tokens, as before, below the cap too", async () => {
    expect(await ask(turn("length", 100))).toMatchObject({ stop_reason: "max_tokens" });
  });

  test("[GUARD] the rule alone: a missing token count is the provider's word", () => {
    expect(wireStopReason("stop", undefined, 4096)).toEqual({ stop_reason: "stop", provider_stop_reason: "stop" });
    expect(wireStopReason(null, 4096, 4096)).toEqual({ stop_reason: "max_tokens", provider_stop_reason: null });
  });
});

describe("the credit-dead fallback says why its completion stopped, by the same rule", () => {
  const fallback = (t: ReturnType<typeof turn>) => loadIndexDeclarations(["anthropicCreditFallback"], {
    ...empty, wireStopReason,
    console: { log() {}, warn() {}, error() {} },
    VESSEL_ID: "llm-resolver-vessel-test", DEFAULT_MODEL: "claude-test", DEFAULT_MAX_TOKENS: 4096,
    modelClientMap: new Map([[VENDOR, { chat: { completions: { create: async () => structuredClone(t) } } }]]),
    isModelWilling: () => true,
    selectArm: async (_t: string, models: string[]) => (models.length > 0 ? { model: models[0] } : null),
    markExhausted: () => {}, reasonText: (r: unknown) => String(r), reasonOf: (e: unknown) => e,
    budgetedProviderCall: async (o: { call: () => Promise<unknown> }) => ({ response: await o.call() }),
    dispatchLedger: {}, dispatchKeyOf: () => "k", estimateInputTokens: () => 1, readMaxInputTokensPerDispatch: async () => null,
    dispatchBudgetRefusal: () => ({ resolved: false }),
  }).anthropicCreditFallback;
  const dead = new Error("400 Your credit balance is too low to access the API");

  test("[MUST-FAIL] stop at the default cap: max_tokens", async () => {
    const r = await fallback(turn("stop", 4096))(dead, { type: "llm_completion", prompt: "p" });
    expect(r).toMatchObject({ resolved: true, fallback_from: "anthropic-credit", stop_reason: "max_tokens", provider_stop_reason: "stop" });
  });

  test("[MUST-FAIL] stop at the max_tokens sent, below the default: max_tokens", async () => {
    const r = await fallback(turn("stop", 512))(dead, { type: "llm_completion", prompt: "p", max_tokens: 512 });
    expect(r).toMatchObject({ resolved: true, stop_reason: "max_tokens", provider_stop_reason: "stop" });
  });

  test("[MUST-FAIL] stop below the cap: stop, said", async () => {
    const r = await fallback(turn("stop", 100))(dead, { type: "llm_completion", prompt: "p", max_tokens: 512 });
    expect(r).toMatchObject({ resolved: true, stop_reason: "stop", provider_stop_reason: "stop" });
  });
});
