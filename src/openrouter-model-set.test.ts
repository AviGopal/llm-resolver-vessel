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
import { describe, expect, test } from "bun:test";
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
