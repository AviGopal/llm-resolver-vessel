// A MALFORMED REQUEST IS AN OBSERVATION ABOUT THE CALLER, NOT ABOUT THE MODEL ARM (check-first).
//
// Gap: a-malformed-llm-request-is-graded-as-a-quality-failure-of-whichever-model-arm-was-drawn.
//
// MEASURED, node 1, 2026-10-03. goal-host's walk sent llm_completion_dispatch a pointer with no
// prompt; development-vessel forwarded `prompt: undefined`. Here, llmCompletionHandler refuses
// "body must include non-empty 'prompt' string" before any provider call — correct — but
// llmCompletionWithPolicyHandler had ALREADY drawn an arm (selectArm) and then grades the refusal:
// `resolved !== true`, not a failover error, not a budget refusal, so
// recordArmOutcome(sel.model, false) adds beta to a healthy model for a request it never saw.
//
// CLASS: CALLER FAULT ABSTAINS. A request-validation refusal (malformed input, known before any
// provider call) gives no alpha and no beta to any arm. It is recorded as `malformed_request`,
// attributed to the CALLING step (body.caller + task_type + execution_id), and is visible through
// the llmSpendSummary shape the vessel already serves.
//
// THE SEAM (pinned). Both handlers live in index.ts, whose top level boots the daemon (port,
// discovery registration), so this file cannot import them — same constraint and same remedy as
// refusal-deadvertises.test.ts: a pure module plus source pins.
//   src/request-validation.ts exports
//     validateCompletionRequest(body) -> null | { resolved: false, shape: "llmCompletion",
//         error: "body must include non-empty 'prompt' string", failure_mode: "malformed_request",
//         field: "prompt" }
//     recordMalformedRequest(body) and malformedRequestSummary() -> [{ caller, task_type, count,
//         last_execution_id }]
//   index.ts: llmCompletionWithPolicyHandler validates FIRST — before loadPolicy / selectArm — and on
//   a malformed request records it and returns the refusal, so no arm is ever drawn and
//   "alpha/beta unchanged" is structural, not a grading exception. llmCompletionHandler uses the
//   same predicate (one refusal text). llmSpendSummary carries `malformed_requests`.
//
// UNTESTABLE HERE, BEHAVIOURALLY: "an empty-prompt request through the policy handler leaves every
// arm unchanged" cannot be executed in-process (importing index.ts starts the daemon). It is pinned
// structurally (validation precedes selection) plus the arm-store control below.
//
// CONTROLS (green at base, must stay green): a real model failure (provider answered, answer bad)
// is still graded — isFailoverError does not excuse it and the record branch still calls
// recordArmOutcome — and recordArmOutcome(model, false) adds beta to exactly the drawn arm.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { isFailoverError } from "./provider-errors";

const SRC = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
const policyBlock = (): string => {
  const i = SRC.indexOf("const llmCompletionWithPolicyHandler");
  const j = SRC.indexOf("const runtime = new ExecutionRuntime", i);
  expect(i).toBeGreaterThan(-1);
  expect(j).toBeGreaterThan(i);
  return SRC.slice(i, j);
};
const handlerBlock = (): string => {
  const i = SRC.indexOf("const llmCompletionHandler: ResolverHandler");
  const j = SRC.indexOf("const llmCompletionWithPolicyHandler", i);
  expect(i).toBeGreaterThan(-1);
  return SRC.slice(i, j);
};

type Validation = null | { resolved: false; shape: string; error: string; failure_mode: string; field: string };
async function seam(): Promise<{
  validateCompletionRequest: (b: unknown) => Validation;
  recordMalformedRequest: (b: unknown) => void;
  malformedRequestSummary: () => Array<{ caller: string; task_type: string; count: number; last_execution_id?: string }>;
}> {
  // A non-literal specifier: the module does not exist at base, and a literal would fail `tsc` there.
  const specifier = ["./request", "validation"].join("-");
  const mod = await import(specifier).catch(() => null);
  expect(mod, "src/request-validation.ts must exist (the pure request-validation seam)").not.toBeNull();
  return mod as never;
}

describe("MUST-FAIL — the request validator is the caller-fault classifier", () => {
  test("absent, empty, whitespace and non-string prompts are malformed_request, with the existing refusal text", async () => {
    const { validateCompletionRequest } = await seam();
    for (const prompt of [undefined, "", "   \n", 42, { text: "x" }]) {
      const v = validateCompletionRequest({ type: "llm_completion", ...(prompt === undefined ? {} : { prompt }) });
      expect(v, `prompt=${JSON.stringify(prompt)}`).not.toBeNull();
      expect(v!.resolved).toBe(false);
      expect(v!.failure_mode).toBe("malformed_request");
      expect(v!.field).toBe("prompt");
      expect(v!.error).toContain("body must include non-empty 'prompt' string");
    }
  });

  test("a request with a real prompt is valid", async () => {
    const { validateCompletionRequest } = await seam();
    expect(validateCompletionRequest({ type: "llm_completion", prompt: "Explain Rayleigh scattering" })).toBeNull();
  });

  test("a malformed request is counted against the CALLING step, not a model", async () => {
    const { recordMalformedRequest, malformedRequestSummary } = await seam();
    const caller = `goal-host:walk_llm_completion:${Math.random().toString(36).slice(2)}`;
    recordMalformedRequest({ type: "llm_completion", caller, task_type: "walk_llm_completion", execution_id: "exec-1" });
    recordMalformedRequest({ type: "llm_completion", caller, task_type: "walk_llm_completion", execution_id: "exec-2" });
    const row = malformedRequestSummary().find((r) => r.caller === caller);
    expect(row).toBeDefined();
    expect(row!.count).toBe(2);
    expect(row!.task_type).toBe("walk_llm_completion");
    expect(row!.last_execution_id).toBe("exec-2");
    expect(JSON.stringify(row)).not.toMatch(/model/);
  });
});

describe("MUST-FAIL — the policy handler validates before it draws an arm", () => {
  test("validateCompletionRequest runs before loadPolicy / selectArm / recordArmOutcome in llmCompletionWithPolicyHandler", () => {
    const b = policyBlock();
    const v = b.indexOf("validateCompletionRequest(");
    expect(v, "the policy handler must validate the request").toBeGreaterThan(-1);
    expect(v).toBeLessThan(b.indexOf("selectArm("));
    expect(v).toBeLessThan(b.indexOf("loadPolicy("));
    expect(v).toBeLessThan(b.indexOf("recordArmOutcome("));
  });

  test("the malformed branch records malformed_request and returns before selection", () => {
    const b = policyBlock();
    const v = b.indexOf("validateCompletionRequest(");
    const sel = b.indexOf("selectArm(");
    const branch = v > -1 ? b.slice(v, sel) : "";
    expect(branch.includes("recordMalformedRequest("), "the malformed branch must record the caller fault").toBe(true);
    // The return must belong to the malformed branch itself (the pinned-model branch also returns).
    const r = branch.indexOf("recordMalformedRequest(");
    const blockEnd = r > -1 ? branch.indexOf("\n  }", r) : -1;
    expect(r > -1 && blockEnd > r && /\breturn\s+\w+/.test(branch.slice(r, blockEnd)), "the malformed branch must return before an arm is drawn").toBe(true);
    expect(branch.includes("recordArmOutcome("), "no arm outcome on a caller fault").toBe(false);
  });

  test("llmCompletionHandler refuses with the same predicate (one refusal text, one classifier)", () => {
    expect(handlerBlock().includes("validateCompletionRequest("), "llmCompletionHandler must use validateCompletionRequest").toBe(true);
  });

  test("the caller-fault count is observable through the llmSpendSummary shape", () => {
    const i = SRC.indexOf("const llmSpendSummaryHandler");
    expect(i).toBeGreaterThan(-1);
    const block = SRC.slice(i, SRC.indexOf("};", i));
    expect(block.includes("malformed_requests"), "llmSpendSummary must carry malformed_requests").toBe(true);
  });
});

describe("CONTROL — a real model failure is still graded against the drawn arm", () => {
  test("a bad answer from a reachable provider is not excused as provider-level", () => {
    expect(isFailoverError("model returned an unparseable body")).toBe(false);
    expect(isFailoverError("completion was empty")).toBe(false);
  });

  test("the record branch still grades a non-provider, non-budget failure with recordArmOutcome(sel.model, ...)", () => {
    const b = policyBlock();
    expect(/if \(!DRAFTING_TASK_TYPES\.has\([^)]*\)\)? && !providerLevelFailure && !budgetRefusal\)/.test(b)).toBe(true);
    expect(b).toContain("await recordArmOutcome(sel.model, (result as { resolved?: boolean }).resolved === true, body.task_type)");
  });

  describe("recordArmOutcome(model, false) adds beta to exactly the drawn arm", () => {
    let dir = "";
    beforeEach(() => {
      dir = mkdtempSync(join(tmpdir(), "malformed-policy-"));
      mkdirSync(join(dir, "policies"));
      writeFileSync(join(dir, "policies", "llm-model-policy.json"), JSON.stringify({
        rev: 1, updated_at: new Date().toISOString(), cost_weight: 0.25,
        arms: [
          { model: "drawn/model", cost_per_mtok: 1, alpha: 3, beta: 2, last_updated_at: new Date().toISOString() },
          { model: "other/model", cost_per_mtok: 1, alpha: 5, beta: 1, last_updated_at: new Date().toISOString() },
        ],
      }));
      process.env.WORKSPACE_ROOT = dir;
    });
    afterEach(() => { rmSync(dir, { recursive: true, force: true }); });

    test("beta +1 on the drawn arm for the task; the other arm untouched", async () => {
      // A fresh module instance, so POLICY_PATH resolves under this temp root (see model-policy.test.ts).
      const { recordArmOutcome } = await import(`./model-policy.js?t=${dir}`);
      await recordArmOutcome("drawn/model", false, "walk_llm_completion");
      const arms = JSON.parse(readFileSync(join(dir, "policies", "llm-model-policy.json"), "utf8")).arms as Array<Record<string, any>>;
      const drawn = arms.find((a) => a.model === "drawn/model")!;
      const other = arms.find((a) => a.model === "other/model")!;
      expect(drawn.task_beta.walk_llm_completion).toBeCloseTo(2, 3); // prior 1 + 1
      expect(drawn.task_alpha.walk_llm_completion).toBeCloseTo(1, 3);
      expect(other.task_beta).toBeUndefined();
      expect(other.alpha).toBe(5);
      expect(other.beta).toBe(1);
    });
  });
});
