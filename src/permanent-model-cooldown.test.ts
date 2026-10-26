// A PAID-ONLY MODEL'S COOLDOWN ESCALATES; A RATE-LIMITED MODEL'S DOES NOT (check-first).
//
// THE DEFECT. A `:free` slug the provider no longer serves for free answers
// `404 ... unavailable for free, use the paid slug`. That matched the exhaustion predicate, so
// cooldownMsFor() gave it the fixed exhaustion window — and when the window lapsed the selector
// drew the same dead arm again, paying one failure and one fallback hop per window, forever.
// Measured on a node: 9 paid-only failures and 10 fallbacks in two hours for one such slug.
//
// THE FIX (pinned). The paid-only / unavailable / unknown-model responses are their own
// permanent class (isPermanentModelError). index.ts keeps a per-model streak of consecutive
// permanent-class errors beside modelExhaustedUntil, and sizes the model's window as the
// exhaustion base doubled per consecutive error, capped at 24h. A success on the model resets
// the streak. Billing (402) and transient rate-limit (429) causes keep exactly today's windows.
//
// FALSIFIER. Three consecutive paid-only 404s on one model: the third window exceeds the base.
// CONTROL. Three consecutive 429s on one model: every window is the rate-limit base, at both ends.
//
// The sizing code lives in index.ts, which starts a server at import time, so the declarations
// that size a model's window are read out of index.ts's syntax tree, transpiled, and run against
// the REAL provider-errors predicates — the code that ships, not a copy of it. At a revision with
// no per-model sizing, the harness falls back to what that revision applies (cooldownMsFor(e)).
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";
import * as pe from "./provider-errors";

const INDEX_PATH = join(import.meta.dir, "index.ts");
const indexSource = (): ts.SourceFile =>
  ts.createSourceFile(INDEX_PATH, readFileSync(INDEX_PATH, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);

const SIZING_DECLS = [
  "EXHAUSTION_COOLDOWN_MS",
  "UNREACHABLE_COOLDOWN_MS",
  "RATE_LIMIT_COOLDOWN_MS",
  "UNAUTHENTICATED_COOLDOWN_MS",
  "PERMANENT_COOLDOWN_DOUBLING",
  "PERMANENT_COOLDOWN_CAP_MS",
  "isHardExhaustedError",
  "isRateLimitError",
  "cooldownMsFor",
  "modelPermanentStreak",
  "modelCooldownMsFor",
  "clearModelEscalation",
] as const;

type Sizing = {
  EXHAUSTION_COOLDOWN_MS: number;
  RATE_LIMIT_COOLDOWN_MS: number;
  PERMANENT_COOLDOWN_CAP_MS?: number;
  cooldownMsFor: (e: unknown) => number;
  modelCooldownMsFor?: (model: string, e: unknown) => number;
  clearModelEscalation?: (model: string) => void;
};

/** A fresh instance of index.ts's cooldown-sizing declarations (fresh streak state each call). */
function loadSizing(): Sizing {
  const sf = indexSource();
  const texts: string[] = [];
  const found: string[] = [];
  for (const st of sf.statements) {
    if (!ts.isVariableStatement(st)) continue;
    const names = st.declarationList.declarations.map((d) => (ts.isIdentifier(d.name) ? d.name.text : ""));
    const hit = names.find((n) => (SIZING_DECLS as readonly string[]).includes(n));
    if (!hit) continue;
    texts.push(st.getText(sf));
    found.push(hit);
  }
  const js = ts.transpileModule(texts.join("\n"), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  const peNames = Object.keys(pe);
  const ret = `return { ${found.join(", ")} };`;
  const factory = new Function("process", ...peNames, `${js}\n${ret}`) as (...args: unknown[]) => Sizing;
  // Empty env: the code's own defaults are what is under test, not this shell's environment.
  return factory({ env: {} }, ...peNames.map((n) => (pe as Record<string, unknown>)[n]));
}

/** The window index.ts applies to `model` after a failover error `e`, at this revision. */
const sizeFor = (s: Sizing) => (model: string, e: unknown): number =>
  s.modelCooldownMsFor ? s.modelCooldownMsFor(model, e) : s.cooldownMsFor(e);
const succeed = (s: Sizing, model: string): void => s.clearModelEscalation?.(model);

const PAID_ONLY_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free";
const PAID_ONLY_404 =
  "404 This model is unavailable for free. Please use the paid slug: nvidia/nemotron-3-nano-30b-a3b";
const RATE_LIMIT_429 = "429 Rate limit exceeded: limit_rpm/nvidia/nemotron-3-nano-30b-a3b:free";
const DAY = 24 * 60 * 60 * 1000;

describe("MUST-FAIL: consecutive paid-only 404s escalate the model's cooldown", () => {
  test("three consecutive paid-only 404s: the third window exceeds the base", () => {
    const s = loadSizing();
    const size = sizeFor(s);
    const w = [1, 2, 3].map(() => size(PAID_ONLY_MODEL, PAID_ONLY_404));
    expect(w[0]).toBe(s.EXHAUSTION_COOLDOWN_MS); // the first is exactly today's window
    expect(w[2]!).toBeGreaterThan(s.EXHAUSTION_COOLDOWN_MS);
    expect(w).toEqual([s.EXHAUSTION_COOLDOWN_MS, 2 * s.EXHAUSTION_COOLDOWN_MS, 4 * s.EXHAUSTION_COOLDOWN_MS]);
  });

  test("the escalation is capped at 24h", () => {
    const s = loadSizing();
    const size = sizeFor(s);
    let last = 0;
    for (let i = 0; i < 40; i++) last = size(PAID_ONLY_MODEL, PAID_ONLY_404);
    expect(s.PERMANENT_COOLDOWN_CAP_MS).toBe(DAY);
    expect(last).toBe(DAY);
  });

  test("the streak is per model: a sibling's first paid-only 404 gets the base window", () => {
    const s = loadSizing();
    const size = sizeFor(s);
    for (let i = 0; i < 3; i++) size(PAID_ONLY_MODEL, PAID_ONLY_404);
    expect(size("qwen/qwen3-coder:free", PAID_ONLY_404)).toBe(s.EXHAUSTION_COOLDOWN_MS);
    expect(size(PAID_ONLY_MODEL, PAID_ONLY_404)).toBe(8 * s.EXHAUSTION_COOLDOWN_MS);
  });

  test("paid-only / unavailable / unknown-model responses are the permanent class", () => {
    expect(typeof pe.isPermanentModelError, "provider-errors.ts must export isPermanentModelError").toBe("function");
    for (const e of [
      PAID_ONLY_404,
      "404 No endpoints found for nvidia/nemotron-3-nano-30b-a3b:free",
      "404 model is unavailable, use this slug instead",
      "400 model_not_found",
      "400 nvidia/x is not a valid model ID",
    ]) expect(pe.isPermanentModelError(e), e).toBe(true);
    for (const e of [RATE_LIMIT_429, "402 Insufficient credits", "404 Not Found", "TimeoutError: The operation timed out."]) {
      expect(pe.isPermanentModelError(e), e).toBe(false);
    }
  });

  test("escalatedCooldownMs doubles per consecutive error and stops at the cap", () => {
    expect(typeof pe.escalatedCooldownMs, "provider-errors.ts must export escalatedCooldownMs").toBe("function");
    const o = { doubling: 2, capMs: DAY };
    expect([1, 2, 3, 4].map((n) => pe.escalatedCooldownMs(600_000, n, o))).toEqual([600_000, 1_200_000, 2_400_000, 4_800_000]);
    expect(pe.escalatedCooldownMs(600_000, 9, o)).toBe(DAY);
    expect(pe.escalatedCooldownMs(600_000, 5000, o)).toBe(DAY);
  });
});

describe("a success resets the escalation", () => {
  test("three paid-only 404s, a success, then a paid-only 404: back to the base window", () => {
    const s = loadSizing();
    const size = sizeFor(s);
    for (let i = 0; i < 3; i++) size(PAID_ONLY_MODEL, PAID_ONLY_404);
    succeed(s, PAID_ONLY_MODEL);
    expect(size(PAID_ONLY_MODEL, PAID_ONLY_404)).toBe(s.EXHAUSTION_COOLDOWN_MS);
  });
});

describe("CONTROLS", () => {
  test("three consecutive 429s keep the rate-limit base window every time", () => {
    const s = loadSizing();
    const size = sizeFor(s);
    const w = [1, 2, 3].map(() => size(PAID_ONLY_MODEL, RATE_LIMIT_429));
    expect(w).toEqual([s.RATE_LIMIT_COOLDOWN_MS, s.RATE_LIMIT_COOLDOWN_MS, s.RATE_LIMIT_COOLDOWN_MS]);
    expect(s.RATE_LIMIT_COOLDOWN_MS).toBe(45 * 1000);
  });

  test("a 429 between paid-only 404s is sized as a 429", () => {
    const s = loadSizing();
    const size = sizeFor(s);
    size(PAID_ONLY_MODEL, PAID_ONLY_404);
    size(PAID_ONLY_MODEL, PAID_ONLY_404);
    expect(size(PAID_ONLY_MODEL, RATE_LIMIT_429)).toBe(s.RATE_LIMIT_COOLDOWN_MS);
  });

  test("a paid-only 404 is still a failover error (the walk and de-advertisement still fire)", () => {
    expect(pe.isFailoverError(PAID_ONLY_404)).toBe(true);
    expect(pe.isExhaustedProviderError(PAID_ONLY_404)).toBe(true);
  });
});

// ── STRUCTURAL PINS: the sizing above is the sizing index.ts applies ─────────────────────────────
function walk(node: ts.Node, visit: (n: ts.Node) => void): void { visit(node); node.forEachChild((c) => walk(c, visit)); }
function calls(root: ts.Node, name: string): ts.CallExpression[] {
  const out: ts.CallExpression[] = [];
  walk(root, (n) => { if (ts.isCallExpression(n) && n.expression.getText() === name) out.push(n); });
  return out;
}

describe("MUST-FAIL: index.ts sizes every model cooldown per model and resets on a served call", () => {
  test("every markModelExhausted(X, …) call sizes its window with modelCooldownMsFor(X, …)", () => {
    const sf = indexSource();
    const marks = calls(sf, "markModelExhausted");
    expect(marks.length).toBeGreaterThanOrEqual(3);
    for (const c of marks) {
      const sized = c.arguments[1];
      expect(sized !== undefined && ts.isCallExpression(sized), c.getText()).toBe(true);
      const sc = sized as ts.CallExpression;
      expect(sc.expression.getText(), c.getText()).toBe("modelCooldownMsFor");
      expect(sc.arguments[0]!.getText(), c.getText()).toBe(c.arguments[0]!.getText());
    }
  });

  test("resolveWithOpenAI clears the model's escalation when the call resolved", () => {
    const sf = indexSource();
    let fn: ts.FunctionDeclaration | undefined;
    walk(sf, (n) => { if (ts.isFunctionDeclaration(n) && n.name?.text === "resolveWithOpenAI") fn = n; });
    expect(fn !== undefined).toBe(true);
    const clears = calls(fn!, "clearModelEscalation");
    expect(clears.length).toBe(1);
    let p: ts.Node | undefined = clears[0]!.parent;
    while (p && !ts.isIfStatement(p)) p = p.parent;
    expect(p !== undefined && (p as ts.IfStatement).expression.getText().replace(/\s+/g, "")).toBe("r.resolved===true");
  });
});
