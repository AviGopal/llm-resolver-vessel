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
import ts from "typescript";
import { isFailoverError } from "./provider-errors";

// ── CROSS-REPO CONTRACT FIXTURE ─────────────────────────────────────────────────────────────────
// Vessels cannot import the super-repo's packages/; this block is copied verbatim from the emitter's
// test so a super-repo check can compare the copies byte for byte.
// CONTRACT-FIXTURE malformed_request BEGIN (emitter: development-vessel test/resolvers/llm-completion-dispatch-refuses-a-missing-prompt.test.ts)
const MALFORMED_REQUEST = "malformed_request";
// CONTRACT-FIXTURE malformed_request END

// ── STRUCTURAL PINS READ THE SYNTAX TREE, NOT THE TEXT ──────────────────────────────────────────
// Presence and order are judged on TypeScript AST positions of calls inside the named handler, so
// whitespace, comments, local renames and line moves do not matter; only the calls and their order do.
const INDEX_PATH = join(import.meta.dir, "index.ts");
let _sf: ts.SourceFile | null = null;
const indexSource = (): ts.SourceFile =>
  (_sf ??= ts.createSourceFile(INDEX_PATH, readFileSync(INDEX_PATH, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS));
function walk(node: ts.Node, visit: (n: ts.Node) => void): void { visit(node); node.forEachChild((c) => walk(c, visit)); }
/** The function bound to `name` (a const arrow/function expression, or a function declaration). */
function fnNamed(name: string): ts.Node {
  let found: ts.Node | undefined;
  walk(indexSource(), (n) => {
    if (found) return;
    if (ts.isFunctionDeclaration(n) && n.name?.text === name) found = n;
    else if (ts.isVariableDeclaration(n) && ts.isIdentifier(n.name) && n.name.text === name && n.initializer
      && (ts.isArrowFunction(n.initializer) || ts.isFunctionExpression(n.initializer))) found = n.initializer;
  });
  expect(found !== undefined, `index.ts must define '${name}'`).toBe(true);
  return found!;
}
const calleeName = (c: ts.CallExpression): string => {
  const e = c.expression;
  return ts.isIdentifier(e) ? e.text : ts.isPropertyAccessExpression(e) ? e.name.text : "";
};
function callsTo(fn: ts.Node, callee: string): ts.CallExpression[] {
  const out: ts.CallExpression[] = [];
  walk(fn, (n) => { if (ts.isCallExpression(n) && calleeName(n) === callee) out.push(n); });
  return out;
}
const firstAt = (fn: ts.Node, callee: string): number => callsTo(fn, callee)[0]?.getStart() ?? Number.POSITIVE_INFINITY;
const identifiersIn = (n: ts.Node): Set<string> => { const s = new Set<string>(); walk(n, (m) => { if (ts.isIdentifier(m)) s.add(m.text); }); return s; };

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
      expect(v!.failure_mode).toBe(MALFORMED_REQUEST);
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
    const fn = fnNamed("llmCompletionWithPolicyHandler");
    const v = firstAt(fn, "validateCompletionRequest");
    expect(Number.isFinite(v), "the policy handler must validate the request").toBe(true);
    for (const later of ["selectArm", "loadPolicy", "recordArmOutcome"]) {
      expect(v < firstAt(fn, later), `validation must precede the first ${later}()`).toBe(true);
    }
  });

  test("the malformed branch records malformed_request and returns before selection", () => {
    const fn = fnNamed("llmCompletionWithPolicyHandler");
    const sel = firstAt(fn, "selectArm");
    let ok = false;
    walk(fn, (n) => {
      if (ok || !ts.isIfStatement(n) || n.getStart() > sel) return;
      const records = callsTo(n.thenStatement, "recordMalformedRequest").length > 0;
      let returns = false;
      walk(n.thenStatement, (m) => { if (ts.isReturnStatement(m)) returns = true; });
      const grades = callsTo(n.thenStatement, "recordArmOutcome").length > 0;
      ok = records && returns && !grades;
    });
    expect(ok, "an `if` before selectArm must record the caller fault and return, grading no arm").toBe(true);
  });

  test("llmCompletionHandler refuses with the same predicate (one refusal text, one classifier)", () => {
    expect(callsTo(fnNamed("llmCompletionHandler"), "validateCompletionRequest").length, "llmCompletionHandler must use validateCompletionRequest").toBeGreaterThan(0);
  });

  test("the caller-fault count is observable through the llmSpendSummary shape", () => {
    let carried = false;
    walk(fnNamed("llmSpendSummaryHandler"), (n) => {
      if ((ts.isPropertyAssignment(n) || ts.isShorthandPropertyAssignment(n)) && (n.name as ts.Identifier).text === "malformed_requests") carried = true;
    });
    expect(carried, "llmSpendSummary must carry malformed_requests").toBe(true);
  });
});

describe("CONTROL — a real model failure is still graded against the drawn arm", () => {
  test("a bad answer from a reachable provider is not excused as provider-level", () => {
    expect(isFailoverError("model returned an unparseable body")).toBe(false);
    expect(isFailoverError("completion was empty")).toBe(false);
  });

  test("the record branch still grades a non-provider, non-budget failure with recordArmOutcome(sel.model, ...)", () => {
    // recordArmOutcome(<sel>.model, ...) under an `if` whose condition still excludes exactly the
    // provider-level and budget refusals (and nothing named for a caller fault slips in as a blanket).
    const fn = fnNamed("llmCompletionWithPolicyHandler");
    let graded = false;
    walk(fn, (n) => {
      if (graded || !ts.isIfStatement(n)) return;
      const cond = identifiersIn(n.expression);
      if (!cond.has("providerLevelFailure") || !cond.has("budgetRefusal")) return;
      graded = callsTo(n.thenStatement, "recordArmOutcome").some((c) => {
        const a0 = c.arguments[0];
        return !!a0 && ts.isPropertyAccessExpression(a0) && a0.name.text === "model";
      });
    });
    expect(graded, "a non-provider, non-budget failure must still call recordArmOutcome(<selection>.model, ...)").toBe(true);
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
