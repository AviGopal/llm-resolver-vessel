// AN EMPTY COMPLETION IS A QUALITY FAILURE OF THE ARM THAT SERVED IT (check-first).
//
// Beside provider-failure-not-quality.test.ts, and the opposite ruling. A provider failure (billing,
// quota, unreachable) says nothing about the model, so it is not graded. An EMPTY answer is the
// model's own output: the provider served the call and the model returned nothing usable. It is
// graded ok:false (beta+1) against the arm that SERVED the empty, and never as a success.
//
// THE DEFECT. The policy handler graded `recordArmOutcome(sel.model, resolved === true)`. An empty
// came back resolved:true, so the arm that answered empty gained alpha and was drawn again. After
// the resolve fix an empty is resolved:false, but an empty from a FALLBACK was still invisible to the
// learner: the walk moved on and only the arm that finally served was graded.
//
// THE RULE (pinned):
//   - every arm that answered empty in the dispatch (the result's empty_arms) gets exactly one
//     ok:false, whether it was the drawn arm or a fallback;
//   - the drawn arm is graded on an empty only when IT served the empty: a drawn arm that was
//     exhausted (or probed) and then rescued keeps today's grading, which is none;
//   - a result that claims resolved:true with blank content grades ok:false, whichever path made it;
//   - an empty refusal names the arm that served it (served_model), and servedArmOf honours it;
//   - drafting task types stay ungraded here (their grade is the edit outcome), as before.
//
// The code under test is index.ts's own policy handler (completion-chain-harness.ts).
import { describe, expect, test } from "bun:test";
import { completionChain, textTurn } from "./completion-chain-harness";
import { servedArmOf } from "./model-policy";
import { isFailoverError } from "./provider-errors";

const X = "x/free-arm:free";
const Y = "y/paid-arm";
const Z = "z/paid-arm";
const RATE_LIMIT = () => new Error("429 Rate limit exceeded: free-models-per-day");
const auto = (extra: Record<string, unknown> = {}) => ({ body: { type: "llm_completion", prompt: "summarize this", model: "auto", task_type: "synthesize_content", ...extra } });

describe("MUST-FAIL: an empty answer gains no alpha and costs the serving arm one beta", () => {
  test("an auto-selected arm returning content \"\" gains no alpha", async () => {
    const c = completionChain({ models: [[X, [textTurn("")]]], selected: X });
    const r = await c.fns.llmCompletionWithPolicyHandler(auto());
    expect(r.resolved).toBe(false);
    expect(c.grades.filter((g) => g.ok)).toEqual([]);
    expect(c.grades).toEqual([{ model: X, ok: false }]);
  });

  test("the drawn arm answers empty and a fallback answers: the fallback gains alpha, the drawn arm one beta", async () => {
    const c = completionChain({ models: [[X, [textTurn(null, "length")]], [Y, [textTurn("the summary")]]], selected: X });
    const r = await c.fns.llmCompletionWithPolicyHandler(auto());
    expect(r.content).toBe("the summary");
    expect(c.grades).toEqual([{ model: Y, ok: true }, { model: X, ok: false }]);
  });

  test("every arm answers empty: each gets exactly one beta, none any alpha", async () => {
    const c = completionChain({ models: [[X, [textTurn("")]], [Y, [textTurn(" ")]], [Z, [textTurn(null, "length")]]], selected: X });
    await c.fns.llmCompletionWithPolicyHandler(auto());
    expect(c.grades).toEqual([{ model: X, ok: false }, { model: Y, ok: false }, { model: Z, ok: false }]);
  });

  test("the drawn arm is exhausted and its fallback serves an empty: the fallback gets beta, the drawn arm is ungraded", async () => {
    const c = completionChain({ models: [[X, [RATE_LIMIT()]], [Y, [textTurn("")]]], selected: X });
    await c.fns.llmCompletionWithPolicyHandler(auto());
    expect(c.grades).toEqual([{ model: Y, ok: false }]);
  });

  test("a probe draw whose fallback serves an empty: only the serving arm is graded, never the probed arm", async () => {
    const c = completionChain({
      models: [[X, [RATE_LIMIT()]], [Y, [textTurn("")]], [Z, [textTurn("served by z")]]],
      selected: X, selectionMeta: { probe: "free_cap" },
    });
    const r = await c.fns.llmCompletionWithPolicyHandler(auto());
    expect(r.served_model).toBe(Z);
    expect(c.grades).toEqual([{ model: Z, ok: true }, { model: Y, ok: false }]);
    expect(c.grades.some((g) => g.model === X)).toBe(false);
  });

  test("a resolved:true result with blank content grades ok:false, whatever path produced it", async () => {
    for (const content of ["", "   ", null, undefined]) {
      const c = completionChain({
        models: [[X, [textTurn("unused")]]], selected: X,
        completionHandler: async () => ({ resolved: true, shape: "llmCompletion", content, model: X }),
      });
      await c.fns.llmCompletionWithPolicyHandler(auto());
      expect(c.grades, JSON.stringify(content)).toEqual([{ model: X, ok: false }]);
    }
  });

  test("an empty refusal names the arm that served it, and servedArmOf honours it", () => {
    const refusal = { resolved: false, error: "empty_completion", served_model: Y, requested_model: "claude-sonnet-5" };
    expect(servedArmOf("claude-sonnet-5", refusal)).toBe(Y);
  });
});

describe("CONTROLS", () => {
  test("a real answer still gains alpha, once, for the arm that gave it", async () => {
    const c = completionChain({ models: [[X, [textTurn("the summary")]], [Y, [textTurn("y")]]], selected: X });
    await c.fns.llmCompletionWithPolicyHandler(auto());
    expect(c.grades).toEqual([{ model: X, ok: true }]);
  });

  test("a rescue after an exhausted drawn arm credits the server only, as before", async () => {
    const c = completionChain({ models: [[X, [RATE_LIMIT()]], [Y, [textTurn("y answers")]]], selected: X });
    await c.fns.llmCompletionWithPolicyHandler(auto());
    expect(c.grades).toEqual([{ model: Y, ok: true }]);
  });

  test("an exhausted drawn arm with no rescue is a provider failure: ungraded, as before", async () => {
    const c = completionChain({ models: [[X, [RATE_LIMIT()]]], selected: X });
    await c.fns.llmCompletionWithPolicyHandler(auto());
    expect(c.grades).toEqual([]);
  });

  test("a drafting task type stays ungraded on this path, empty or not", async () => {
    const c = completionChain({ models: [[X, [textTurn("")]], [Y, [textTurn("y")]]], selected: X });
    await c.fns.llmCompletionWithPolicyHandler(auto({ task_type: "feature_compose" }));
    expect(c.grades).toEqual([]);
  });

  test("empty_completion is not a provider failure, so the provider-failure skip cannot swallow it", () => {
    expect(isFailoverError("empty_completion")).toBe(false);
  });

  test("servedArmOf is unchanged for answers and rescues", () => {
    expect(servedArmOf(X, { resolved: true, content: "a" })).toBe(X);
    expect(servedArmOf(X, { resolved: true, fallback_from: X, served_model: Y })).toBe(Y);
    expect(servedArmOf(X, { resolved: true, fallback_from: X })).toBeNull();
  });
});
