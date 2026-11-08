// The completion chain of index.ts, run in-process against scripted provider turns (test support).
//
// llmCompletionWithPolicyHandler -> llmCompletionHandler -> walkFallbackModels -> resolveWithOpenAI
// -> resolveWithOpenAIUnobserved are loaded from index.ts's syntax tree (index-fn-harness.ts) and
// call each other for real. Only the edges are stubbed: the provider client (each wire model
// answers from its own script), arm selection, cooldown and standing bookkeeping, and the policy
// write (recordArmOutcome), which is recorded instead of persisted.
import { loadIndexDeclarations } from "./index-fn-harness";
import * as providerErrors from "./provider-errors";
import { servedArmOf, fallbackWalkOrder } from "./model-policy";
import { gateLogLine } from "./provider-standing";
import { dispatchKeyOf } from "./dispatch-budget";

/** One scripted provider turn: an OpenAI chat completion, or an Error the client throws. */
export type Turn = Record<string, unknown> | Error;

export const textTurn = (content: string | null, finish_reason = "stop", extra: Record<string, unknown> = {}): Turn => ({
  choices: [{ index: 0, message: { role: "assistant", content, ...extra }, finish_reason }],
  usage: { prompt_tokens: 10, completion_tokens: 7 },
});

export const toolCallTurn = (n: number): Turn => ({
  choices: [{
    index: 0,
    message: {
      role: "assistant", content: null,
      tool_calls: [{ id: `call_${n}`, type: "function", function: { name: "read_file", arguments: JSON.stringify({ path: `f${n}` }) } }],
    },
    finish_reason: "tool_calls",
  }],
  usage: { prompt_tokens: 10, completion_tokens: 3 },
});

export interface ChainOptions {
  /** Wire models in registry order, each with its scripted turns (consumed in order; the last repeats). */
  models: Array<[string, Turn[]]>;
  /** The arm selection draws (auto mode). */
  selected?: string;
  selectionMeta?: Record<string, unknown>;
  /** Models already in cooldown when the call starts. */
  cooling?: string[];
  /** Replaces the real llmCompletionHandler (to hand the policy handler a fixed result). */
  completionHandler?: (ctx: { body: Record<string, unknown> }) => Promise<Record<string, unknown>>;
}

export interface Chain {
  fns: Record<string, any>;
  /** Provider calls made, by model, in order. */
  calls: string[];
  /** recordArmOutcome(model, ok) calls, in order. */
  grades: Array<{ model: string; ok: boolean }>;
  /** markModelExhausted / markExhausted calls. */
  exhaustionMarks: string[];
}

export function completionChain(opts: ChainOptions): Chain {
  const calls: string[] = [];
  const grades: Array<{ model: string; ok: boolean }> = [];
  const exhaustionMarks: string[] = [];
  const cooling = new Set(opts.cooling ?? []);
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
  const standing = {
    providerOf: () => "wire",
    admit: () => true,
    recordFallback: () => {}, recordServed: () => {}, recordFailure: () => null, recordKind: () => {},
  };
  const stubs: Record<string, unknown> = {
    ...providerErrors,
    servedArmOf, fallbackWalkOrder, gateLogLine, dispatchKeyOf,
    console: { log() {}, warn() {}, error() {} },
    process: { env: {} },
    VESSEL_ID: "llm-resolver-vessel-test",
    DEFAULT_MODEL: "claude-sonnet-5",
    DEFAULT_MAX_TOKENS: 4096,
    DEFAULT_MAX_TOOL_ITERATIONS: 20,
    openaiClient: null, openrouterClient: null, anthropic: null,
    modelClientMap,
    providerStanding: standing,
    inModelCooldown: (m: string) => cooling.has(m),
    inCooldown: () => false,
    markModelExhausted: (m: string) => { exhaustionMarks.push(m); cooling.add(m); },
    markExhausted: (k: string) => { exhaustionMarks.push(k); },
    modelCooldownMsFor: () => 600_000,
    clearModelEscalation: () => {}, clearExhausted: () => {},
    providerKeyOf: (c: { baseURL?: string }) => String(c.baseURL),
    wireProviderIdOf: () => "wire",
    pickProvider: () => null,
    readGateParams: async () => ({}),
    readFallbackOrder: async () => undefined,
    readMaxInputTokens: async () => 1_000_000,
    readMaxInputTokensPerDispatch: async () => null,
    readToolResultLimits: async () => ({ per_result_max_chars: 10_000, total_max_chars: 100_000 }),
    estimateInputTokens: () => 1,
    budgetedProviderCall: async (o: { call: () => Promise<unknown> }) => ({ response: await o.call() }),
    dispatchLedger: {},
    promptCeilingRefusal: () => ({ resolved: false, prompt_ceiling: true, error: "prompt too large" }),
    dispatchBudgetRefusal: () => ({ resolved: false, dispatch_budget_exhausted: true, error: "dispatch budget" }),
    loadPolicy: async () => ({ arms: opts.models.map(([model]) => ({ model, cost_per_mtok: 1, alpha: 1, beta: 1 })) }),
    isModelWilling: () => true,
    selectArm: async () => (opts.selected ? { model: opts.selected, meta: { ...(opts.selectionMeta ?? {}) } } : null),
    decideLastResort: () => ({ refuse: "nothing willing" }),
    recordArmOutcome: async (model: string, ok: boolean) => { grades.push({ model, ok }); },
  };
  // The empty-completion helpers, at a revision that has them.
  try { Object.assign(stubs, require("./empty-completion")); } catch { /* an earlier revision */ }
  if (opts.completionHandler) stubs.llmCompletionHandler = opts.completionHandler;
  const names = [
    "llmCompletionWithPolicyHandler",
    ...(opts.completionHandler ? [] : ["llmCompletionHandler"]),
    "walkFallbackModels", "resolveWithOpenAI", "resolveWithOpenAIUnobserved",
    "chargeEmptyCompletion",
  ];
  const fns = loadIndexDeclarations(names, stubs, ["chargeEmptyCompletion"]);
  return { fns, calls, grades, exhaustionMarks };
}
