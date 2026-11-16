// NO DENIED MODEL ANSWERS A CALL (the USER, 11-15: "No free endpoints"; model-policy.ts denied_models). Run, not read:
// index.ts's own handler and fallback walk, lifted from its syntax tree (completion-chain-harness.ts).
// FALSIFIERS: with every permitted model cooled, a denied model gets 0 calls and the call is unserved; a call asking for
// a denied model is answered by a permitted one or unserved, never by it; with no denial (the row removed), the same
// fallback is walked to. CONTROL: a permitted model is called as before.
import { describe, expect, test } from "bun:test";
import { completionChain, textTurn } from "./completion-chain-harness";
import { deniedBy } from "./model-policy";

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
});
