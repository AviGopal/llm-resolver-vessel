// THE GATE'S DECISIONS REACH THE JOURNAL (check-first).
//
// model_selection.meta (gated / probe / gate_bypassed) and served_model ride on the completion
// result. A reader with only the vessel's journal and its served shapes never sees them, so a live
// check that must exclude gate_bypassed windows, or count rescues by served model, cannot be run.
//
// THE FIX (pinned). llmCompletionWithPolicyHandler logs ONE info line per gate-relevant selection
// (arms held out, a probe, the gate standing aside, or a fallback rescue), built by gateLogLine
// (provider-standing.ts):
//   [llm-gate] provider=<p> model=<m> gated=<bool> probe=<bool> gate_bypassed=<bool> served_model=<id|->
// An ordinary ungated, unrescued selection logs nothing. Ids only: no prompt text, no key material.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";
import * as standingMod from "./provider-standing";

const LINE = /^\[llm-gate\] provider=\S+ model=\S+ gated=(true|false) probe=(true|false) gate_bypassed=(true|false) served_model=\S+$/;
type GateLogLine = (provider: string | null, model: string, meta: Record<string, unknown>, result: Record<string, unknown>) => string | null;
function seam(): GateLogLine {
  const f = (standingMod as Record<string, unknown>).gateLogLine;
  expect(typeof f, "provider-standing.ts must export gateLogLine").toBe("function");
  return f as GateLogLine;
}

let dir: string;
let sfx: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "gate-log-"));
  mkdirSync(join(dir, "policies"));
  process.env.WORKSPACE_ROOT = dir;
  sfx = Math.random().toString(36).slice(2, 8);
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

type Sel = { model: string; meta: Record<string, unknown> };
async function select(arms: Array<Record<string, unknown>>, available: string[]): Promise<Sel> {
  writeFileSync(join(dir, "policies", "llm-model-policy.json"), JSON.stringify({ rev: 1, updated_at: new Date().toISOString(), cost_weight: 0.25, arms }));
  const m = (await import(`./model-policy.js?t=${dir}`)) as unknown as {
    selectArm: (t?: string, a?: string[]) => Promise<Sel | null>; setProviderStanding?: (s: unknown) => void;
  };
  m.setProviderStanding?.(standingMod.providerStanding);
  return (await m.selectArm(undefined, available))!;
}

describe("MUST-FAIL: a gated selection logs exactly one [llm-gate] line", () => {
  test("a free-capped :free arm held out: one line naming provider and model, gated=true", async () => {
    const gateLogLine = seam();
    const OR = `router-${sfx}`, FREE = `vendor/free-${sfx}:free`, PAID = `vendor/paid-${sfx}`;
    standingMod.providerStanding.registerModels(OR, [FREE, PAID]);
    standingMod.providerStanding.recordFailure(OR, FREE, "429 Rate limit exceeded: free-models-per-day", Date.now() - 15 * 60_000);
    const now = new Date().toISOString();
    const sel = await select([
      { model: FREE, cost_per_mtok: 0.01, alpha: 57, beta: 1, last_updated_at: now },
      { model: PAID, cost_per_mtok: 0.3, alpha: 4, beta: 3, last_updated_at: now },
    ], [FREE, PAID]);
    expect(sel.model).toBe(PAID);
    const line = gateLogLine(standingMod.providerStanding.providerOf(sel.model), sel.model, sel.meta,
      { resolved: true, content: "SECRET-PROMPT-ECHO", prompt: "SECRET-PROMPT" });
    expect(line).not.toBeNull();
    expect(line!.split("\n").length).toBe(1);
    expect(line!).toMatch(LINE);
    expect(line!).toBe(`[llm-gate] provider=${OR} model=${PAID} gated=true probe=false gate_bypassed=false served_model=-`);
    expect(line!).not.toContain("SECRET");
  });

  test("a probe, a bypass and a rescue each log their line, with served_model on the rescue", () => {
    const gateLogLine = seam();
    expect(gateLogLine("p", "m:free", { probe: "free_cap" }, { resolved: true })).toBe(
      "[llm-gate] provider=p model=m:free gated=false probe=true gate_bypassed=false served_model=-");
    expect(gateLogLine("p", "m", { gated: [{ model: "m", reason: "credit" }], gate_bypassed: true }, { resolved: true })).toBe(
      "[llm-gate] provider=p model=m gated=true probe=false gate_bypassed=true served_model=-");
    expect(gateLogLine("p", "a:free", {}, { resolved: true, fallback_from: "a:free", served_model: "b-paid" })).toBe(
      "[llm-gate] provider=p model=a:free gated=false probe=false gate_bypassed=false served_model=b-paid");
  });
});

describe("CONTROL: an ungated, non-rescue selection logs nothing", () => {
  test("a real ungated selection gives no line", async () => {
    const gateLogLine = seam();
    const OR = `router-${sfx}`, A = `vendor/a-${sfx}`, B = `vendor/b-${sfx}`;
    standingMod.providerStanding.registerModels(OR, [A, B]);
    const now = new Date().toISOString();
    const sel = await select([
      { model: A, cost_per_mtok: 0.01, alpha: 50, beta: 1, last_updated_at: now },
      { model: B, cost_per_mtok: 0.3, alpha: 4, beta: 3, last_updated_at: now },
    ], [A, B]);
    expect(gateLogLine(OR, sel.model, sel.meta, { resolved: true, model: sel.model })).toBeNull();
    expect(gateLogLine(OR, sel.model, sel.meta, { resolved: false, error: "completion was empty" })).toBeNull();
  });
});

describe("MUST-FAIL: the policy handler writes the line, once, and nothing else carries the prefix", () => {
  const INDEX = join(import.meta.dir, "index.ts");
  test("llmCompletionWithPolicyHandler calls gateLogLine once and logs its result with one console.log", () => {
    const sf = ts.createSourceFile(INDEX, readFileSync(INDEX, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    let h: ts.Node | undefined;
    const find = (n: ts.Node): void => {
      if (!h && ts.isVariableDeclaration(n) && ts.isIdentifier(n.name) && n.name.text === "llmCompletionWithPolicyHandler") h = n.initializer;
      n.forEachChild(find);
    };
    find(sf);
    expect(h).toBeDefined();
    const calls: ts.CallExpression[] = [];
    const visit = (n: ts.Node): void => { if (ts.isCallExpression(n)) calls.push(n); n.forEachChild(visit); };
    visit(h!);
    const gl = calls.filter((c) => ts.isIdentifier(c.expression) && c.expression.text === "gateLogLine");
    expect(gl.length).toBe(1);
    const decl = gl[0]!.parent;
    expect(ts.isVariableDeclaration(decl) && ts.isIdentifier(decl.name)).toBe(true);
    const name = (decl as ts.VariableDeclaration).name.getText();
    const logs = calls.filter((c) => c.expression.getText() === "console.log" && c.arguments.length === 1 && c.arguments[0]!.getText() === name);
    expect(logs.length).toBe(1);
    expect(gl[0]!.arguments[1]!.getText()).toBe("sel.model");
  });

  test("the [llm-gate] prefix appears only in gateLogLine", () => {
    for (const f of ["index.ts", "model-policy.ts"]) expect(readFileSync(join(import.meta.dir, f), "utf8")).not.toContain("[llm-gate]");
    const ps = readFileSync(join(import.meta.dir, "provider-standing.ts"), "utf8");
    expect(ps.split("`[llm-gate]").length - 1).toBe(1);
  });
});
