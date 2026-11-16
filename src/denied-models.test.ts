// NO DENIED MODEL ANSWERS A CALL (the USER, 11-15: "No free endpoints"; model-policy.ts denied_models). Run, not read:
// index.ts's own handler and fallback walk, lifted from its syntax tree (completion-chain-harness.ts).
// FALSIFIERS: with every permitted model cooled, a denied model gets 0 calls and the call is unserved; a call asking for
// a denied model is answered by a permitted one or unserved, never by it; with no denial (the row removed), the same
// fallback is walked to. CONTROL: a permitted model is called as before.
import { describe, expect, test } from "bun:test";
import { completionChain, textTurn } from "./completion-chain-harness";
import { deniedBy } from "./model-policy";
import { completionQuota } from "./advertisement-resume";
import { loadIndexDeclarations } from "./index-fn-harness";

const PAID = "vendor/paid-model", PAID2 = "vendor/paid-two", FREE = "vendor/some-model:free";
const ask = (more: Record<string, unknown>) => ({ type: "llm_completion", prompt: "p", task_type: "grading", ...more });
const DENY = [":free"];

describe("denied models", () => {
  test("the owner's entries match by substring; no entries deny nothing", () => {
    expect(deniedBy({ denied_models: DENY }, FREE)).toBe(true);
    expect(deniedBy({ denied_models: DENY }, PAID)).toBe(false);
    expect(deniedBy({}, FREE)).toBe(false);
    expect(deniedBy(null, FREE)).toBe(false);
    expect(deniedBy({ denied_models: [""] }, FREE)).toBe(false);
  });

  test("[MUST-FAIL] all permitted models cooled: the denied fallback gets 0 calls, the call is unserved", async () => {
    const c = completionChain({ models: [[PAID, [new Error("402 can only afford")]], [FREE, [textTurn("from free")]]], denied: DENY });
    const r = await c.fns.llmCompletionHandler({ body: ask({ model: PAID }) });
    expect(c.calls).toEqual([PAID]);
    expect(r.resolved).toBe(false);
  });

  test("[MUST-FAIL] removing the row changes behaviour: with no denial the same fallback is walked to", async () => {
    const c = completionChain({ models: [[PAID, [new Error("402 can only afford")]], [FREE, [textTurn("from free")]]] });
    const r = await c.fns.llmCompletionHandler({ body: ask({ model: PAID }) });
    expect(c.calls).toEqual([PAID, FREE]);
    expect(r.resolved).toBe(true);
  });

  test("[MUST-FAIL] a call asking for a denied model never reaches it: a permitted model answers, or it is unserved", async () => {
    const some = completionChain({ models: [[FREE, [textTurn("never")]], [PAID2, [textTurn("from paid")]]], denied: DENY });
    const r1 = await some.fns.llmCompletionHandler({ body: ask({ model: FREE }) });
    expect(some.calls).toEqual([PAID2]);
    expect(r1.resolved).toBe(true);
    const none = completionChain({ models: [[FREE, [textTurn("never")]]], denied: DENY });
    const r2 = await none.fns.llmCompletionHandler({ body: ask({ model: FREE }) });
    expect(none.calls).toEqual([]);
    expect(r2).toMatchObject({ resolved: false, unserved: "llm_completion", requested_model: FREE });
  });

  // index.ts's own isModelWilling, lifted with a keyed client for each model, the given ones cooling, the given entries denied.
  const willingness = (models: string[], cooling: string[], denied: string[]) => {
    const client = { baseURL: "https://provider.test/v1" };
    const modelClientMap = new Map<string, unknown>(models.map((m) => [m, client]));
    const fns = loadIndexDeclarations(["isModelWilling"], {
      modelClientMap, openrouterClient: null, anthropic: null, OPENROUTER_MODEL_SET: new Set<string>(),
      isRunpodCold: () => false, inModelCooldown: (m: string) => cooling.includes(m), inCooldown: () => false,
      providerKeyOf: (c: { baseURL?: string }) => String(c.baseURL),
      isDenied: (m: string) => denied.some((d) => d !== "" && m.includes(d)),
    });
    return fns.isModelWilling as (m: string) => boolean;
  };

  test("[MUST-FAIL] every permitted arm cooled and a denied arm otherwise willing: not willing, so no quota and no draw (qa)", () => {
    const willing = willingness([PAID, FREE], [PAID], DENY);
    expect(willing(FREE)).toBe(false);
    expect(completionQuota([PAID, FREE], willing)).toBe(false); // de-advertised, so discovery routes to a peer
    // control: with no denial, the same free arm keeps the node advertised
    expect(completionQuota([PAID, FREE], willingness([PAID, FREE], [PAID], []))).toBe(true);
  });
});
