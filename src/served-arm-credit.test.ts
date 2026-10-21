// A FALLBACK-SERVED COMPLETION IS CREDITED TO THE ARM THAT SERVED IT, NEVER TO THE ARM THAT FAILED
// (check-first).
//
// THE DEFECT. llmCompletionWithPolicyHandler graded `recordArmOutcome(sel.model, resolved === true)`.
// When the selected arm A failed for an infra reason (a 402, the `:free` daily-cap 429, a timeout)
// and the fallback walk rescued the call with model B, the result came back resolved:true, so A got
// alpha+1 for a call it did not serve. Measured on a node over three hours: 43 failing calls, all
// rescued (B = one paid model for 42 of them), and the alpha went to the capped `:free` arms that
// had failed. Infra failures are not graded (no beta), so that borrowed alpha was all the evidence
// those arms ever got, and the selector kept drawing them first.
//
// THE FIX (pinned). The fallback results name the arm that served (`served_model`, also response
// metadata for the caller), and the policy handler grades `servedArmOf(sel.model, result)`.
// B gets alpha+1 for the call it served; A's alpha/beta are left byte-unchanged.
//
// FALSIFIER. A completion served by fallback B after selected A 429s gives alpha+1 to B, never to A,
// and A's quality alpha/beta stay unchanged. CONTROL: a direct success by A gives alpha+1 to A.
// Infra failures with no rescue stay ungraded (no beta to A); that was already so, and is pinned.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";
import { isFailoverError } from "./provider-errors";

const A_FREE = "nvidia/nemotron-3-nano-30b-a3b:free";
const A_ANTHROPIC = "claude-sonnet-5";
const B = "google/gemini-2.5-flash";
const NOW = new Date().toISOString();

let dir: string;
let policyPath: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "served-arm-"));
  mkdirSync(join(dir, "policies"));
  policyPath = join(dir, "policies", "llm-model-policy.json");
  process.env.WORKSPACE_ROOT = dir;
  writeFileSync(policyPath, JSON.stringify({
    rev: 3, updated_at: NOW, cost_weight: 0.25, arms: [
      { model: A_FREE, cost_per_mtok: 0.01, alpha: 57, beta: 1, last_updated_at: NOW },
      { model: A_ANTHROPIC, cost_per_mtok: 3, alpha: 9, beta: 4, last_updated_at: NOW },
      { model: B, cost_per_mtok: 0.3, alpha: 3, beta: 2, last_updated_at: NOW },
    ],
  }));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

type Policy = {
  servedArmOf?: (selected: string, result: Record<string, unknown>) => string | null;
  recordArmOutcome: (model: string, ok: boolean, taskType?: string) => Promise<void>;
};
async function policy(): Promise<Policy & { servedArmOf: NonNullable<Policy["servedArmOf"]> }> {
  const mod = (await import(`./model-policy.js?t=${dir}`)) as Policy;
  expect(typeof mod.servedArmOf, "model-policy.ts must export servedArmOf").toBe("function");
  return mod as never;
}
const arm = (model: string): Record<string, unknown> =>
  (JSON.parse(readFileSync(policyPath, "utf-8")).arms as Array<Record<string, unknown>>).find((a) => a.model === model)!;

/** The grading step of llmCompletionWithPolicyHandler for a call that reached it (pinned below). */
async function grade(p: Awaited<ReturnType<typeof policy>>, selected: string, result: Record<string, unknown>): Promise<void> {
  const graded = p.servedArmOf(selected, result);
  if (graded !== null) await p.recordArmOutcome(graded, result.resolved === true);
}

describe("MUST-FAIL: a rescued call credits the server, never the failed arm", () => {
  test("43 free-cap rescues by B: B gains alpha for each, the capped :free arm is byte-unchanged", async () => {
    const p = await policy();
    const before = JSON.stringify(arm(A_FREE));
    const bAlpha = arm(B).alpha as number;
    for (let i = 0; i < 43; i++) {
      await grade(p, A_FREE, { resolved: true, shape: "llmCompletion", content: "ok", model: B, fallback_from: A_FREE, served_model: B });
    }
    expect(JSON.stringify(arm(A_FREE))).toBe(before); // Δα = 0 ≤ 10% of 43 rescues
    expect((arm(B).alpha as number) - bAlpha).toBeGreaterThanOrEqual(0.8 * 43);
    expect(arm(B).beta as number).toBeCloseTo(2, 3);
  });

  test("the anthropic credit-dead fallback is a rescue too: credit the wire model, not the claude arm", async () => {
    const p = await policy();
    const before = JSON.stringify(arm(A_ANTHROPIC));
    const bAlpha = arm(B).alpha as number;
    await grade(p, A_ANTHROPIC, { resolved: true, model: B, provider: "openai-wire", fallback_from: "anthropic-credit", served_model: B });
    expect(JSON.stringify(arm(A_ANTHROPIC))).toBe(before);
    expect(arm(B).alpha as number).toBeCloseTo(bAlpha + 1, 3);
  });

  test("a fallback result that does not name what served grades nothing", async () => {
    const p = await policy();
    const beforeA = JSON.stringify(arm(A_FREE)), beforeB = JSON.stringify(arm(B));
    expect(p.servedArmOf(A_FREE, { resolved: true, fallback_from: A_FREE })).toBeNull();
    await grade(p, A_FREE, { resolved: true, fallback_from: A_FREE });
    expect(JSON.stringify(arm(A_FREE))).toBe(beforeA);
    expect(JSON.stringify(arm(B))).toBe(beforeB);
  });
});

describe("CONTROLS", () => {
  test("a direct success by A gives alpha+1 to A", async () => {
    const p = await policy();
    const aAlpha = arm(A_FREE).alpha as number;
    expect(p.servedArmOf(A_FREE, { resolved: true, model: A_FREE })).toBe(A_FREE);
    await grade(p, A_FREE, { resolved: true, model: A_FREE });
    expect(arm(A_FREE).alpha as number).toBeCloseTo(aAlpha + 1, 3);
  });

  test("a direct quality failure by A still charges A beta", async () => {
    const p = await policy();
    const aBeta = arm(A_FREE).beta as number;
    await grade(p, A_FREE, { resolved: false, error: "completion was empty" });
    expect(arm(A_FREE).beta as number).toBeCloseTo(aBeta + 1, 3);
  });

  test("the infra failures of the live loop are provider-level, which the handler declines to grade", () => {
    for (const e of [
      "429 Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day",
      "402 Insufficient credits",
      "TimeoutError: The operation timed out.",
    ]) expect(isFailoverError(e), e).toBe(true);
  });
});

// ── STRUCTURAL PINS READ THE SYNTAX TREE ────────────────────────────────────────────────────────
const INDEX_PATH = join(import.meta.dir, "index.ts");
let _sf: ts.SourceFile | null = null;
const indexSource = (): ts.SourceFile =>
  (_sf ??= ts.createSourceFile(INDEX_PATH, readFileSync(INDEX_PATH, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS));
function walk(node: ts.Node, visit: (n: ts.Node) => void): void { visit(node); node.forEachChild((c) => walk(c, visit)); }
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
const identifiersIn = (n: ts.Node): Set<string> => { const s = new Set<string>(); walk(n, (m) => { if (ts.isIdentifier(m)) s.add(m.text); }); return s; };
/** Object literals in `fn` that set `fallback_from`. */
function fallbackLiterals(fn: ts.Node): ts.ObjectLiteralExpression[] {
  const out: ts.ObjectLiteralExpression[] = [];
  walk(fn, (n) => {
    if (ts.isObjectLiteralExpression(n) && n.properties.some((p) => p.name && ts.isIdentifier(p.name) && p.name.text === "fallback_from")) out.push(n);
  });
  return out;
}
const propNames = (o: ts.ObjectLiteralExpression): string[] => o.properties.flatMap((p) => (p.name && ts.isIdentifier(p.name) ? [p.name.text] : []));

describe("MUST-FAIL: index.ts grades the served arm and says which model served", () => {
  test("the policy handler grades servedArmOf(sel.model, result); sel.model is graded only when it is what served", () => {
    const h = fnNamed("llmCompletionWithPolicyHandler");
    const served = callsTo(h, "servedArmOf");
    expect(served.length).toBe(1);
    expect(served[0]!.arguments[0]!.getText()).toBe("sel.model");
    const decl = served[0]!.parent;
    expect(ts.isVariableDeclaration(decl) && ts.isIdentifier(decl.name)).toBe(true);
    const servedName = (decl as ts.VariableDeclaration).name.getText();
    const records = callsTo(h, "recordArmOutcome");
    expect(records.some((c) => c.arguments[0]!.getText() === servedName), `some grade passes ${servedName}`).toBe(true);
    for (const c of records) {
      const a0 = c.arguments[0]!.getText();
      if (a0 === servedName) continue;
      // Any other first argument must be sel.model, inside a branch that has established
      // `<served> === sel.model` (the drawn arm itself answered).
      expect(a0).toBe("sel.model");
      let n: ts.Node = c, guarded = false;
      while (n !== h && !guarded) {
        const p = n.parent;
        if (ts.isIfStatement(p) && p.thenStatement === n) {
          const cond = p.expression.getText().replace(/\s+/g, "");
          guarded = cond === `${servedName}===sel.model` || cond === `sel.model===${servedName}`;
        }
        n = p;
      }
      expect(guarded, "recordArmOutcome(sel.model, ...) only where the drawn arm served").toBe(true);
    }
  });

  test("every fallback result names the served model (response metadata)", () => {
    for (const name of ["walkFallbackModels", "anthropicCreditFallback"]) {
      const lits = fallbackLiterals(fnNamed(name));
      expect(lits.length, `${name} returns a fallback result`).toBeGreaterThan(0);
      for (const o of lits) expect(propNames(o), name).toContain("served_model");
    }
  });

  test("CONTROL: the grade is still guarded by the provider-failure, budget and drafting skips", () => {
    const h = fnNamed("llmCompletionWithPolicyHandler");
    let guard: ts.IfStatement | undefined;
    walk(h, (n) => {
      if (!guard && ts.isIfStatement(n) && callsTo(n.thenStatement, "recordArmOutcome").length > 0) guard = n;
    });
    expect(guard).toBeDefined();
    const ids = identifiersIn(guard!.expression);
    for (const id of ["providerLevelFailure", "budgetRefusal", "DRAFTING_TASK_TYPES"]) expect(ids.has(id), id).toBe(true);
  });
});
