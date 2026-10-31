// NEWER OPENROUTER MODELS ARE ROUTABLE ARMS (check-first).
//
// THE DEFECT. isModelWilling admits an OpenRouter model only when its id is in the openrouter
// provider's `models` list in index.ts (OPENROUTER_MODEL_SET, the phantom-arm guard). Newer models
// OpenRouter serves were absent from that list, so a policy arm naming one could never be drawn,
// and an anthropic/ id fell through to the direct-Anthropic prefix rule instead.
//
// THE FIX (pinned). The exact slugs below, each read from OpenRouter's public GET /api/v1/models,
// are appended to that list. The anthropic/ ones need no explicit provider: at startup every model
// of the keyed openrouter provider goes into modelClientMap, and isModelWilling and
// llmCompletionHandler both read the map before the direct-Anthropic prefix rule, so the
// openrouter client serves them under their exact slug.
//
// FALSIFIER. With the openrouter client present, every appended id is willing, and each anthropic/
// slug's willingness follows the openrouter lane, not the direct Anthropic one.
// CONTROLS: an unlisted slash id is still refused (the phantom-arm guard), the ids listed before
// stay listed and willing, and a listed anthropic/ slug is served by its wire client unchanged.
//
// ROUTE-ONLY SLUGS GET NO ARM (second block). Startup seeding gave a global alpha1/beta1 policy arm
// to every model of a keyed provider, so the routable set was the arm set by construction. Six of
// the appended slugs are route-only: they stay routable (modelClientMap, isModelWilling) but the
// seeding loop skips them. The seeded drafter slugs carry their FUNDED_ARM_COST input price, not
// the 0.6 default. The loop is a bare top-level statement, so the test reads it out of index.ts's
// syntax tree (as the harness does for declarations) and runs the shipped statement against stubs.
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";
import { loadIndexDeclarations } from "./index-fn-harness";
import { completionChain, textTurn } from "./completion-chain-harness";

// The appended slugs. Hand-written on purpose: deriving them from the registry would make the
// check vacuous. Extend this table to add a model.
const APPENDED = [
  "anthropic/claude-opus-5.5",
  "moonshotai/kimi-k3.1-pro",
  "openai/gpt-6.1-sol",
  "deepseek/deepseek-v4.1-flash-1017",
  "anthropic/claude-haiku-5.5",
  "deepseek/deepseek-v4.1-flash",
  "z-ai/glm-5.3-flash",
  "openai/gpt-6-luna",
  "minimax/minimax-m3.1-flash",
  "nvidia/nemotron-3.5-lightning",
];
const ANTHROPIC = APPENDED.filter((m) => m.startsWith("anthropic/"));

// The openrouter list before the append.
const EXISTING = [
  "google/gemini-2.5-flash", "openai/gpt-4o-mini", "deepseek/deepseek-chat-v3-0324",
  "nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3-nano-30b-a3b:free",
  "cohere/north-mini-code:free", "google/gemini-2.5-pro", "openai/gpt-5",
];

const OPENROUTER_KEY = "https://openrouter.ai/api/v1";

/** isModelWilling over the shipped registry, with modelClientMap filled the way startup fills it
 * when OPENROUTER_API_KEY is keyed. `anthropic` is the direct client (null = not keyed);
 * `cooling` lists the provider keys in cooldown ("anthropic" or the openrouter baseURL). */
function willingness(anthropic: unknown = null, cooling: string[] = []) {
  const openrouterClient = { baseURL: OPENROUTER_KEY };
  const modelClientMap = new Map<string, unknown>();
  const stubs: Record<string, unknown> = {
    RUNPOD_ENDPOINT_ID: undefined, RUNPOD_MODELS: [],
    modelClientMap, openrouterClient, anthropic,
    isRunpodCold: () => false,
    inModelCooldown: () => false,
    inCooldown: (k: string) => cooling.includes(k),
    providerKeyOf: (c: { baseURL?: string }) => String(c.baseURL),
  };
  const fns = loadIndexDeclarations(["OPENAI_WIRE_PROVIDERS", "OPENROUTER_MODEL_SET", "isModelWilling"], stubs);
  const openrouter = (fns.OPENAI_WIRE_PROVIDERS as Array<{ id: string; models: string[] }>).find((p) => p.id === "openrouter")!;
  for (const m of openrouter.models) modelClientMap.set(m, openrouterClient);
  return {
    isModelWilling: fns.isModelWilling as (m: string) => boolean,
    openrouterModels: openrouter.models, modelClientMap, openrouterClient,
  };
}

describe("openrouter model set", () => {
  test.each(APPENDED)("[MUST-FAIL] %s is listed and willing when the openrouter client is present", (id) => {
    const { isModelWilling, openrouterModels } = willingness();
    expect(openrouterModels).toContain(id);
    expect(isModelWilling(id)).toBe(true);
  });

  test.each(ANTHROPIC)("[MUST-FAIL] %s is routed to the openrouter client, not the direct Anthropic client", (id) => {
    const direct = { messages: {} };
    // The handler's first branch is modelClientMap.get(model): this is the client that serves.
    const w = willingness(direct);
    expect(w.modelClientMap.get(id)).toBe(w.openrouterClient);
    // Willingness follows the openrouter lane: a cooling direct Anthropic client does not stop it,
    expect(willingness(direct, ["anthropic"]).isModelWilling(id)).toBe(true);
    // and a cooling openrouter lane does.
    expect(willingness(direct, [OPENROUTER_KEY]).isModelWilling(id)).toBe(false);
  });

  test("[CONTROL] the phantom-arm guard still refuses an unlisted id", () => {
    const { isModelWilling } = willingness();
    expect(isModelWilling("vendor/not-a-listed-model")).toBe(false);
    expect(isModelWilling("openai/gpt-6.1-terra")).toBe(false);
  });

  test("[CONTROL] the ids listed before are unchanged, in order, and still willing", () => {
    const { isModelWilling, openrouterModels } = willingness();
    expect(openrouterModels.slice(0, EXISTING.length)).toEqual(EXISTING);
    for (const id of EXISTING) expect(isModelWilling(id)).toBe(true);
  });

  test.each(ANTHROPIC)("[CONTROL] a listed %s is served by its wire client under its exact slug", async (id) => {
    const chain = completionChain({ models: [[id, [textTurn("ok")]]] });
    const r = await chain.fns.llmCompletionHandler({ body: { prompt: "p", model: id } });
    expect(r.resolved).toBe(true);
    expect(chain.calls).toEqual([id]);
    expect(r.provider).toBe("openai"); // the OpenAI-wire path; the direct path reports "anthropic"
    expect(r.model).toBe(id); // the wire id sent is the slug, prefix kept
  });
});

// Hand-written, not derived from the registry, so the checks are not vacuous.
const ROUTE_ONLY = [
  "anthropic/claude-haiku-5.5", "deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3-flash",
  "openai/gpt-6-luna", "minimax/minimax-m3.1-flash", "nvidia/nemotron-3.5-lightning",
];
// Every seeded openrouter model and the cost its arm is seeded at.
const SEEDED_COST: Record<string, number> = {
  "google/gemini-2.5-flash": 0.3, "openai/gpt-4o-mini": 0.15, "deepseek/deepseek-chat-v3-0324": 0.3,
  "nvidia/nemotron-3-ultra-550b-a55b:free": 0, "nvidia/nemotron-3-nano-30b-a3b:free": 0,
  "cohere/north-mini-code:free": 0, "google/gemini-2.5-pro": 0.6, "openai/gpt-5": 0.6,
  "anthropic/claude-opus-5.5": 4, "moonshotai/kimi-k3.1-pro": 2, "openai/gpt-6.1-sol": 2,
  "deepseek/deepseek-v4.1-flash-1017": 0.15,
};
const DRAFTERS = ["anthropic/claude-opus-5.5", "moonshotai/kimi-k3.1-pro", "openai/gpt-6.1-sol", "deepseek/deepseek-v4.1-flash-1017"];

/** The text of index.ts's top-level keyed-provider seeding loop (the for-of that calls ensureArmsForModels). */
function seedingLoopText(): string {
  const path = join(import.meta.dir, "index.ts");
  const sf = ts.createSourceFile(path, readFileSync(path, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const loops = sf.statements.filter((st) => ts.isForOfStatement(st) && st.getText(sf).includes("ensureArmsForModels(")
    && st.getText(sf).includes("SEED_SKIP_PROVIDERS"));
  if (loops.length !== 1) throw new Error(`expected one keyed-provider seeding loop in index.ts, found ${loops.length}`);
  return loops[0].getText(sf);
}

/** Run the shipped startup seeding with only `keyed` env keys present; returns the arms seeded, model -> cost. */
async function seedArms(keyed: string[] = ["OPENROUTER_API_KEY"]): Promise<Map<string, number>> {
  const decl = loadIndexDeclarations(["OPENAI_WIRE_PROVIDERS", "FUNDED_ARM_COST", "SEED_SKIP_PROVIDERS"],
    { RUNPOD_ENDPOINT_ID: undefined, RUNPOD_MODELS: [] });
  const seeded = new Map<string, number>();
  const stubs: Record<string, unknown> = {
    ...decl,
    process: { env: Object.fromEntries(keyed.map((k) => [k, "k"])) },
    cleanEnv: (v: string | undefined) => (v && v.trim() ? v.trim() : undefined),
    ensureArmsForModels: async (models: string[], cost: number) => {
      for (const m of models) seeded.set(m, cost);
      return models.length;
    },
    console: { log: () => {}, warn: () => {} },
  };
  const js = ts.transpileModule(`async function __seed() {\n${seedingLoopText()}\n}`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  const scope = new Proxy(stubs, {
    has: (_t, k) => typeof k === "string",
    get: (t, k) => {
      if (typeof k !== "string") return undefined;
      if (k in t) return t[k];
      if (k in globalThis) return (globalThis as Record<string, unknown>)[k];
      throw new ReferenceError(`unstubbed identifier '${k}' in the seeding loop`);
    },
  });
  await new Function("__scope", `with (__scope) {\n${js}\nreturn __seed;\n}`)(scope)();
  return seeded;
}

describe("route-only openrouter models get no policy arm at startup", () => {
  test.each(ROUTE_ONLY)("[MUST-FAIL] %s is not seeded as an arm when openrouter is keyed", async (id) => {
    const seeded = await seedArms();
    expect(seeded.has(id)).toBe(false);
  });

  test.each(DRAFTERS)("[MUST-FAIL] %s is seeded at its FUNDED_ARM_COST price, not the 0.6 default", async (id) => {
    const seeded = await seedArms();
    expect(seeded.get(id)).toBe(SEEDED_COST[id]);
    expect(seeded.get(id)).not.toBe(0.6);
  });

  test("[CONTROL] every other keyed openrouter model is still seeded, at its cost, and nothing else", async () => {
    const seeded = await seedArms();
    expect(Object.fromEntries(seeded)).toEqual(SEEDED_COST);
  });

  test("[CONTROL] another keyed provider's models are still seeded at their costs", async () => {
    const seeded = await seedArms(["GROQ_API_KEY"]);
    expect(Object.fromEntries(seeded)).toEqual({
      "llama-3.3-70b-versatile": 0.6, "moonshotai/kimi-k2-instruct": 1.0, "qwen/qwen3-32b": 0.5,
    });
  });

  test.each(ROUTE_ONLY)("[CONTROL] %s stays routable: in modelClientMap and willing", (id) => {
    const w = willingness();
    expect(w.openrouterModels).toContain(id);
    expect(w.modelClientMap.get(id)).toBe(w.openrouterClient);
    expect(w.isModelWilling(id)).toBe(true);
  });

  test("[CONTROL] the openrouter ids are unchanged, in order", () => {
    expect(willingness().openrouterModels).toEqual([...EXISTING, ...APPENDED]);
  });
});
