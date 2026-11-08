// AN EMPTY COMPLETION IS A REFUSAL, AND THE RESOLVER WALKS ON TO AN ARM THAT ANSWERS (check-first).
//
// THE DEFECT. Every completion path ended `resolved:true, content: <final text> ?? ""`, reading
// neither finish_reason nor reasoning. A reasoning model that spent max_tokens on hidden reasoning
// (`content:null, finish_reason:"length"`), a tool loop that hit its 20-iteration cap with no final
// turn, and a model that answers a tool result with an empty `stop` turn all came back as a
// successful empty answer. The caller's last step then threw ("did not return text or tool_calls"),
// no other arm was tried, and about half of one node's failures were this one string.
//
// THE FIX (pinned). Blank final text (null, "" or whitespace) is `resolved:false,
// error:"empty_completion"` with the facts that tell the causes apart: stop_reason (the provider's
// own: finish_reason on the OpenAI wire), completion_tokens, iterations, reasoning_present (a
// boolean; the reasoning text is never copied), served_model and requested_model, this resolver's
// id and the request's ids. It is a QUALITY failure of the model, not a provider failure: no
// cooldown, no exhaustion mark, but the resolver walks on to the next arm, skipping every arm
// already tried in this dispatch.
//
// FALSIFIERS (red at the parent): the length-truncated reasoning turn, the 20-iteration tool loop,
// and "arm 1 empty, arm 2 answers" served by arm 2. CONTROLS: a normal text answer is unchanged.
//
// The code under test is index.ts's own (completion-chain-harness.ts loads it from the syntax tree).
import { describe, expect, test } from "bun:test";
import { completionChain, textTurn, toolCallTurn } from "./completion-chain-harness";
import { isFailoverError } from "./provider-errors";
import { classifyStandingError } from "./provider-standing";

const X = "x/free-arm:free";
const Y = "y/paid-arm";
const Z = "z/paid-arm";
const TOOLS = [{ name: "read_file", description: "read a file", input_schema: { type: "object", properties: {} } }];
const REASONING = "Let me think about this step by step. The user wants SECRET-REASONING-TEXT";
const body = (extra: Record<string, unknown> = {}) => ({ type: "llm_completion", prompt: "write the function", ...extra });

/** Resolve one call to `model` on its scripted turns through resolveWithOpenAI (the wire path). */
async function resolveOn(model: string, turns: ReturnType<typeof textTurn>[], extra: Record<string, unknown> = {}) {
  const c = completionChain({ models: [[model, turns]] });
  const r = await c.fns.llmCompletionHandler({ body: { ...body(extra), model } });
  return { r: r as Record<string, unknown>, c };
}

describe("MUST-FAIL: blank final text is an empty_completion refusal", () => {
  test("single-shot: content:null with reasoning and finish_reason length", async () => {
    const { r } = await resolveOn(X, [textTurn(null, "length", { reasoning: REASONING })], { execution_id: "exec-1", dispatch_id: "disp-1" });
    expect(r.resolved).toBe(false);
    expect(r.error).toBe("empty_completion");
    expect(r.stop_reason).toBe("length");
    expect(r.reasoning_present).toBe(true);
    expect(r.completion_tokens).toBe(7);
    expect(r.iterations).toBe(1);
    expect(r.served_model).toBe(X);
    expect(r.requested_model).toBe(X);
    expect(r.execution_id).toBe("exec-1");
    expect(r.dispatch_id).toBe("disp-1");
    expect(typeof r.resolver).toBe("string");
    expect(JSON.stringify(r)).not.toContain("SECRET-REASONING-TEXT");
  });

  test("single-shot: \"\" and whitespace-only content are blank too", async () => {
    for (const content of ["", "  \n\t "]) {
      const { r } = await resolveOn(X, [textTurn(content)]);
      expect(r.resolved, JSON.stringify(content)).toBe(false);
      expect(r.error).toBe("empty_completion");
      expect(r.stop_reason).toBe("stop");
      expect(r.reasoning_present).toBe(false);
    }
  });
});

describe("MUST-FAIL: an empty fails over to the next arm, without cooling the empty one", () => {
  test("arm 1 answers empty, arm 2 answers: arm 2 serves the call", async () => {
    const c = completionChain({ models: [[X, [textTurn("")]], [Y, [textTurn("the answer")]]] });
    const r = await c.fns.llmCompletionHandler({ body: { ...body(), model: X } });
    expect(r.resolved).toBe(true);
    expect(r.content).toBe("the answer");
    expect(r.served_model).toBe(Y);
    expect(r.fallback_from).toBe(X);
    expect(r.empty_arms).toEqual([X]);
    expect(c.calls).toEqual([X, Y]);
    expect(c.exhaustionMarks).toEqual([]);
  });

  test("the drawn arm is exhausted and the first fallback answers empty: the next fallback serves", async () => {
    const c = completionChain({ models: [[X, [new Error("429 Rate limit exceeded: free-models-per-day")]], [Y, [textTurn(null, "length")]], [Z, [textTurn("served by z")]]] });
    const r = await c.fns.llmCompletionHandler({ body: { ...body(), model: X } });
    expect(r.resolved).toBe(true);
    expect(r.content).toBe("served by z");
    expect(r.served_model).toBe(Z);
    expect(r.empty_arms).toEqual([Y]);
    expect(c.calls).toEqual([X, Y, Z]);
    expect(c.exhaustionMarks).toEqual([X]); // only the 429 cools; the empty arm does not
  });

  test("every arm answers empty: one call per arm, then the empty refusal", async () => {
    const c = completionChain({ models: [[X, [textTurn("")]], [Y, [textTurn(" ")]], [Z, [textTurn(null, "length")]]] });
    const r = await c.fns.llmCompletionHandler({ body: { ...body(), model: X } });
    expect(r.resolved).toBe(false);
    expect(r.error).toBe("empty_completion");
    expect(r.served_model).toBe(X);
    expect(r.empty_arms).toEqual([X, Y, Z]);
    expect(c.calls).toEqual([X, Y, Z]);
    expect(c.exhaustionMarks).toEqual([]);
  });
});

describe("CONTROLS", () => {
  test("a normal single-shot answer is unchanged", async () => {
    const { r, c } = await resolveOn(Y, [textTurn("the answer")]);
    expect(r).toEqual({
      resolved: true, shape: "llmCompletion", content: "the answer",
      provider: "openai", model: Y,
      usage: { input_tokens: 10, output_tokens: 7 },
      stop_reason: "stop",
    });
    expect(c.calls).toEqual([Y]);
  });

  test("an answering arm is not walked past", async () => {
    const c = completionChain({ models: [[X, [textTurn("x answers")]], [Y, [textTurn("y")]]] });
    const r = await c.fns.llmCompletionHandler({ body: { ...body(), model: X } });
    expect(r.content).toBe("x answers");
    expect(r.fallback_from).toBeUndefined();
    expect(c.calls).toEqual([X]);
  });

  test("empty_completion is not a provider failure: no failover-error class, no standing kind", () => {
    expect(isFailoverError("empty_completion")).toBe(false);
    expect(classifyStandingError(X, "empty_completion")).toBeNull();
  });
});
