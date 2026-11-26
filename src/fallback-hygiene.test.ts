// FALLBACK HYGIENE (check-first):
// 1. Upstream `finish_reason: "error"` (e.g., OpenRouter safety-tag body "[toxicity=0][toxicity=0]")
//    must NOT return `resolved: true`; it must return an `empty_completion` refusal and fail over
//    without cooling the model (or return `resolved: false` when no fallback can serve).
// 2. Empty/error completion fallback must log an honest `"empty/error completion (...)"` warning
//    (with stop_reason, tokens, reasoning) rather than misleading `"provider exhausted"`, while
//    provider exhaustion (402/429) continues to log `"provider exhausted"`.
// 3. When `llmModelPolicy.fallback_order` is unset or empty, `walkFallbackModels` prefers SOTA
//    fallback models (`DEFAULT_FALLBACK_PREFERENCE`) ahead of legacy registry-head models like
//    `google/gemini-2.5-flash`, while an explicit non-empty `fallback_order` overrides the default.
import { describe, expect, test } from "bun:test";
import { loadIndexDeclarations } from "./index-fn-harness";
import * as providerErrors from "./provider-errors";
import * as emptyCompletion from "./empty-completion";
import * as exhaustionReason from "./exhaustion-reason";
import * as stopReason from "./stop-reason";
import { fallbackWalkOrder } from "./model-policy";
import { dispatchKeyOf } from "./dispatch-budget";

type Turn = Record<string, unknown> | Error;

const makeTurn = (
  content: string | null,
  finish_reason = "stop",
  opts: { completion_tokens?: number; prompt_tokens?: number; extra?: Record<string, unknown> } = {},
): Turn => ({
  choices: [{ index: 0, message: { role: "assistant", content, ...(opts.extra ?? {}) }, finish_reason }],
  usage: { prompt_tokens: opts.prompt_tokens ?? 10, completion_tokens: opts.completion_tokens ?? 7 },
});

interface HygieneChainOptions {
  models: Array<[string, Turn[]]>;
  fallbackOrder?: string[] | undefined;
  openrouter?: Turn[];
}

function hygieneChain(opts: HygieneChainOptions) {
  const calls: string[] = [];
  const warns: string[] = [];
  const exhaustionMarks: string[] = [];
  const cooling = new Set<string>();
  const modelClientMap = new Map<string, unknown>();

  for (const [model, turns] of opts.models) {
    let i = 0;
    modelClientMap.set(model, {
      baseURL: `https://wire.test/${model}`,
      chat: {
        completions: {
          create: async () => {
            calls.push(model);
            const t = turns[Math.min(i++, turns.length - 1)]!;
            if (t instanceof Error) throw t;
            return structuredClone(t);
          },
        },
      },
    });
  }

  let q = 0;
  const openrouterClient = opts.openrouter === undefined ? null : {
    baseURL: "https://openrouter.test",
    chat: {
      completions: {
        create: async () => {
          calls.push("openrouter");
          const t = opts.openrouter![Math.min(q++, opts.openrouter!.length - 1)]!;
          if (t instanceof Error) throw t;
          return structuredClone(t);
        },
      },
    },
  };

  const stubs: Record<string, unknown> = {
    ...providerErrors,
    ...emptyCompletion,
    ...exhaustionReason,
    ...stopReason,
    fallbackWalkOrder,
    dispatchKeyOf,
    console: {
      log() {},
      warn: (...args: unknown[]) => { warns.push(args.map(String).join(" ")); },
      error() {},
    },
    process: { env: {} },
    VESSEL_ID: "llm-resolver-vessel-test",
    DEFAULT_MODEL: "claude-sonnet-5",
    DEFAULT_MAX_TOKENS: 4096,
    openaiClient: null,
    openrouterClient,
    anthropic: null,
    modelClientMap,
    providerStanding: {
      providerOf: () => "openrouter",
      admit: () => true,
      recordFallback: () => {},
      recordServed: () => {},
      recordFailure: () => null,
      recordKind: () => {},
    },
    inModelCooldown: (m: string) => cooling.has(m),
    isDenied: () => false,
    inCooldown: () => false,
    markModelExhausted: (m: string) => { exhaustionMarks.push(m); cooling.add(m); },
    markExhausted: (k: string) => { exhaustionMarks.push(k); },
    modelCooldownMsFor: () => 600_000,
    cooldownMsFor: () => 600_000,
    clearModelEscalation: () => {},
    clearExhausted: () => {},
    providerKeyOf: (c: { baseURL?: string }) => String(c.baseURL),
    wireProviderIdOf: () => "openrouter",
    pickProvider: () => null,
    readGateParams: async () => ({}),
    readFallbackOrder: async () => opts.fallbackOrder,
    readMaxInputTokens: async () => 1_000_000,
    readMaxInputTokensPerDispatch: async () => null,
    estimateInputTokens: () => 1,
    budgetedProviderCall: async (o: { call: () => Promise<unknown> }) => ({ response: await o.call() }),
    dispatchLedger: {},
    promptCeilingRefusal: () => ({ resolved: false, prompt_ceiling: true, error: "prompt too large" }),
    dispatchBudgetRefusal: () => ({ resolved: false, dispatch_budget_exhausted: true, error: "dispatch budget" }),
  };

  const fns = loadIndexDeclarations(
    [
      "holdsUntilServed", "isRateLimitError", "isHardExhaustedError",
      "llmCompletionHandler", "walkFallbackModels", "resolveWithOpenAI", "resolveWithOpenAIUnobserved",
    ],
    stubs,
  );
  return { fns, calls, warns, exhaustionMarks };
}

const ask = (model: string) => ({ type: "llm_completion", prompt: "generate json", model });

describe("Fix 1: finish_reason === 'error' is refused and fails over instead of returning resolved:true", () => {
  test("[MUST-FAIL] non-blank safety-tag body with finish_reason 'error' fails over to the next willing model without cooling", async () => {
    const errTurn = makeTurn("[toxicity=0][toxicity=0]", "error", { completion_tokens: 0 });
    const chain = hygieneChain({
      models: [
        ["deepseek/deepseek-v4-pro-0813", [errTurn]],
        ["google/gemini-3.8-flash", [makeTurn("clean answer")]],
      ],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(true);
    expect(r.content).toBe("clean answer");
    expect(r.served_model).toBe("google/gemini-3.8-flash");
    expect(r.fallback_from).toBe("deepseek/deepseek-v4-pro-0813");
    expect(r.empty_arms).toEqual(["deepseek/deepseek-v4-pro-0813"]);
    expect(chain.calls).toEqual(["deepseek/deepseek-v4-pro-0813", "google/gemini-3.8-flash"]);
    expect(chain.exhaustionMarks).toEqual([]);
  });

  test("[MUST-FAIL] finish_reason 'error' with non-blank body and no fallback returns resolved:false empty_completion", async () => {
    const errTurn = makeTurn("[toxicity=0][toxicity=0]", "error", { completion_tokens: 0 });
    const chain = hygieneChain({
      models: [["deepseek/deepseek-v4-pro-0813", [errTurn]]],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(false);
    expect(r.error).toBe("empty_completion");
    expect(r.stop_reason).toBe("error");
    expect(r.completion_tokens).toBe(0);
    expect(chain.calls).toEqual(["deepseek/deepseek-v4-pro-0813"]);
    expect(chain.exhaustionMarks).toEqual([]);
  });

  test("[CONTROL] normal finish_reason 'stop' with non-blank text still returns resolved:true", async () => {
    const chain = hygieneChain({
      models: [["deepseek/deepseek-v4-pro-0813", [makeTurn("valid output", "stop")]]],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(true);
    expect(r.content).toBe("valid output");
    expect(chain.calls).toEqual(["deepseek/deepseek-v4-pro-0813"]);
  });
});

describe("Fix 2: honest fallback warning log distinguishes empty/error completion from provider exhaustion", () => {
  test("[MUST-FAIL] empty/error completion fallback logs 'empty/error completion' with stop_reason, tokens, reasoning (not 'provider exhausted')", async () => {
    const emptyWithReasoning = makeTurn(null, "length", {
      completion_tokens: 512,
      extra: { reasoning: "hidden chain of thought" },
    });
    const chain = hygieneChain({
      models: [
        ["deepseek/deepseek-v4-pro-0813", [emptyWithReasoning]],
        ["google/gemini-3.8-flash", [makeTurn("rescued")]],
      ],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(true);
    expect(chain.warns.length).toBe(1);
    expect(chain.warns[0]).toContain("empty/error completion (stop_reason=length, tokens=512, reasoning=true)");
    expect(chain.warns[0]).toContain("for 'deepseek/deepseek-v4-pro-0813' — falling back to 'google/gemini-3.8-flash'");
    expect(chain.warns[0]).not.toContain("provider exhausted");
  });

  test("[MUST-FAIL] openrouter catch-all empty completion logs 'empty/error completion' (not 'provider exhausted')", async () => {
    const errTurn = makeTurn("[toxicity=0][toxicity=0]", "error", { completion_tokens: 0 });
    const chain = hygieneChain({
      models: [["google/gemini-3.8-flash", [makeTurn("rescued")]]],
      openrouter: [errTurn],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("vendor/unlisted-model") });
    expect(r.resolved).toBe(true);
    expect(chain.warns.length).toBe(1);
    expect(chain.warns[0]).toContain("empty/error completion (stop_reason=error, tokens=0, reasoning=false)");
    expect(chain.warns[0]).not.toContain("provider exhausted");
  });

  test("[MUST-FAIL] chained empty completion inside walkFallbackModels updates reason for subsequent attempts", async () => {
    const chain = hygieneChain({
      models: [
        ["deepseek/deepseek-v4-pro-0813", [new Error("402 This request requires more credits")]],
        ["google/gemini-3.8-flash", [makeTurn("", "stop", { completion_tokens: 0 })]],
        ["z-ai/glm-5.3", [makeTurn("served by glm")]],
      ],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(true);
    expect(r.served_model).toBe("z-ai/glm-5.3");
    expect(chain.warns.length).toBe(2);
    expect(chain.warns[0]).toContain("provider exhausted for 'deepseek/deepseek-v4-pro-0813' — falling back to 'google/gemini-3.8-flash'");
    expect(chain.warns[1]).toContain("empty/error completion (stop_reason=stop, tokens=0, reasoning=false)");
    expect(chain.warns[1]).toContain("falling back to 'z-ai/glm-5.3'");
    expect(chain.warns[1]).not.toContain("provider exhausted");
  });

  test("[CONTROL] 402 provider exhaustion failover still logs 'provider exhausted'", async () => {
    const chain = hygieneChain({
      models: [
        ["deepseek/deepseek-v4-pro-0813", [new Error("402 This request requires more credits")]],
        ["google/gemini-3.8-flash", [makeTurn("served")]],
      ],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(true);
    expect(chain.warns.length).toBe(1);
    expect(chain.warns[0]).toContain("provider exhausted for 'deepseek/deepseek-v4-pro-0813' — falling back to 'google/gemini-3.8-flash'");
    expect(chain.warns[0]).not.toContain("empty/error completion");
  });
});

describe("Fix 3: SOTA default fallback preference when policy.fallback_order is unset", () => {
  test("[MUST-FAIL] with fallback_order undefined, deepseek empty completion falls back to google/gemini-3.8-flash ahead of google/gemini-2.5-flash", async () => {
    // Legacy google/gemini-2.5-flash is inserted FIRST in modelClientMap (matching registry order).
    const chain = hygieneChain({
      models: [
        ["google/gemini-2.5-flash", [makeTurn("legacy answer")]],
        ["deepseek/deepseek-v4-pro-0813", [makeTurn("", "stop", { completion_tokens: 0 })]],
        ["google/gemini-3.8-flash", [makeTurn("sota answer")]],
        ["z-ai/glm-5.3", [makeTurn("glm answer")]],
      ],
      fallbackOrder: undefined,
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(true);
    expect(r.served_model).toBe("google/gemini-3.8-flash");
    expect(r.content).toBe("sota answer");
    expect(chain.calls).toEqual(["deepseek/deepseek-v4-pro-0813", "google/gemini-3.8-flash"]);
  });

  test("[MUST-FAIL] with fallback_order empty [], SOTA preference still leads ahead of legacy google/gemini-2.5-flash", async () => {
    const chain = hygieneChain({
      models: [
        ["google/gemini-2.5-flash", [makeTurn("legacy answer")]],
        ["deepseek/deepseek-v4-pro-0813", [makeTurn("", "stop", { completion_tokens: 0 })]],
        ["google/gemini-3.8-flash", [makeTurn("", "length", { completion_tokens: 100 })]],
        ["z-ai/glm-5.3", [makeTurn("glm sota answer")]],
      ],
      fallbackOrder: [],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(true);
    expect(r.served_model).toBe("z-ai/glm-5.3");
    expect(chain.calls).toEqual(["deepseek/deepseek-v4-pro-0813", "google/gemini-3.8-flash", "z-ai/glm-5.3"]);
  });

  test("[GUARD] explicit non-empty readFallbackOrder() overrides DEFAULT_FALLBACK_PREFERENCE", async () => {
    const chain = hygieneChain({
      models: [
        ["google/gemini-2.5-flash", [makeTurn("legacy answer")]],
        ["deepseek/deepseek-v4-pro-0813", [makeTurn("", "stop", { completion_tokens: 0 })]],
        ["google/gemini-3.8-flash", [makeTurn("gemini 3.8 answer")]],
        ["qwen/qwen3.8-flash", [makeTurn("qwen explicit answer")]],
      ],
      fallbackOrder: ["qwen/qwen3.8-flash"],
    });
    const r = await chain.fns.llmCompletionHandler({ body: ask("deepseek/deepseek-v4-pro-0813") });
    expect(r.resolved).toBe(true);
    expect(r.served_model).toBe("qwen/qwen3.8-flash");
    expect(r.content).toBe("qwen explicit answer");
    expect(chain.calls).toEqual(["deepseek/deepseek-v4-pro-0813", "qwen/qwen3.8-flash"]);
  });
});
