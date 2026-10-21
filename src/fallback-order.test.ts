// THE FALLBACK WALK ORDER IS A SHAPED POLICY, NOT A CODE CONSTANT (check-first).
//
// THE DEFECT. walkFallbackModels iterated modelClientMap, i.e. the order of OPENAI_WIRE_PROVIDERS
// in index.ts. That order is a cold-start prior written into code ("paid first, :free last"), and
// it is not even that: OpenRouter's capable paid slugs and the whole google provider sit AFTER its
// :free slugs. Preferring a funded paid slug over a capped :free one in the walk meant editing
// index.ts, which the system cannot observe or learn (law 1).
//
// THE FIX (pinned). llmModelPolicy.fallback_order (writable through llmModelPolicy_write, read at
// use time) lists the models the walk tries first, in order; every other routable model follows in
// registry order. fallbackWalkOrder (model-policy.ts) can only reorder: it never adds a model the
// registry cannot route and never drops one it can. The selection half of "prefer funded paid over
// capped :free once the free cap 429s in-window" is the eligibility gate (provider-gate.test.ts),
// and the walk applies the same gate before it tries a model.
//
// FALSIFIER. With fallback_order set, the walk tries those models first, in that order.
// CONTROL: with no fallback_order, the walk order is the registry order, as before.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";

const REGISTRY = [
  "llama-3.3-70b-versatile", "mistral-small-latest",
  "google/gemini-2.5-flash", "openai/gpt-4o-mini",
  "nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3-nano-30b-a3b:free",
  "google/gemini-2.5-pro", "gemini-2.5-flash",
];

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "fallback-order-"));
  mkdirSync(join(dir, "policies"));
  process.env.WORKSPACE_ROOT = dir;
  writeFileSync(join(dir, "policies", "llm-model-policy.json"), JSON.stringify({
    rev: 1, updated_at: new Date().toISOString(), cost_weight: 0.25,
    arms: [{ model: "google/gemini-2.5-flash", cost_per_mtok: 0.3, alpha: 1, beta: 1 }],
  }));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

type Mod = {
  fallbackWalkOrder?: (routable: string[], preferred: unknown) => string[];
  readFallbackOrder?: () => Promise<string[] | undefined>;
  llmModelPolicyWriteHandler: (ctx: { body: unknown }) => Promise<{ resolved: boolean; error?: string }>;
  llmModelPolicyHandler: () => Promise<{ body: Record<string, unknown> }>;
};
async function mod(): Promise<Required<Mod>> {
  const m = (await import(`./model-policy.js?t=${dir}`)) as unknown as Mod;
  expect(typeof m.fallbackWalkOrder, "model-policy.ts must export fallbackWalkOrder").toBe("function");
  expect(typeof m.readFallbackOrder, "model-policy.ts must export readFallbackOrder").toBe("function");
  return m as Required<Mod>;
}

describe("MUST-FAIL: the shaped fallback_order leads the walk", () => {
  test("listed models first, in the listed order; the rest follow in registry order", async () => {
    const { fallbackWalkOrder } = await mod();
    const order = fallbackWalkOrder(REGISTRY, ["google/gemini-2.5-pro", "openai/gpt-4o-mini"]);
    expect(order.slice(0, 2)).toEqual(["google/gemini-2.5-pro", "openai/gpt-4o-mini"]);
    expect(order.slice(2)).toEqual(REGISTRY.filter((m) => m !== "google/gemini-2.5-pro" && m !== "openai/gpt-4o-mini"));
  });

  test("a preference only reorders: unroutable ids, duplicates and non-strings are ignored, nothing is dropped", async () => {
    const { fallbackWalkOrder } = await mod();
    const order = fallbackWalkOrder(REGISTRY, ["not/routable", "gemini-2.5-flash", "gemini-2.5-flash", 7, null]);
    expect(order[0]).toBe("gemini-2.5-flash");
    expect([...order].sort()).toEqual([...REGISTRY].sort());
    expect(order.length).toBe(REGISTRY.length);
  });

  test("fallback_order is written through llmModelPolicy_write and read at use time", async () => {
    const m = await mod();
    expect(await m.readFallbackOrder()).toBeUndefined();
    const w = await m.llmModelPolicyWriteHandler({ body: { fallback_order: ["google/gemini-2.5-pro", "gemini-2.5-flash"] } });
    expect(w.resolved).toBe(true);
    expect(await m.readFallbackOrder()).toEqual(["google/gemini-2.5-pro", "gemini-2.5-flash"]);
    expect((await m.llmModelPolicyHandler()).body.fallback_order).toEqual(["google/gemini-2.5-pro", "gemini-2.5-flash"]);
    expect((await m.llmModelPolicyWriteHandler({ body: { fallback_order: "google/gemini-2.5-pro" } })).resolved).toBe(false);
    expect((await m.llmModelPolicyWriteHandler({ body: { fallback_order: ["", "x"] } })).resolved).toBe(false);
    expect((await m.llmModelPolicyWriteHandler({ body: { fallback_order: null } })).resolved).toBe(true);
    expect(await m.readFallbackOrder()).toBeUndefined();
  });
});

describe("CONTROL: with no fallback_order the walk order is the registry order", () => {
  test("unset, empty or malformed preference gives the registry order unchanged", async () => {
    const { fallbackWalkOrder } = await mod();
    for (const p of [undefined, [], "x", { a: 1 }]) expect(fallbackWalkOrder(REGISTRY, p)).toEqual(REGISTRY);
  });
});

describe("MUST-FAIL: walkFallbackModels walks the shaped order", () => {
  const src = (): ts.SourceFile =>
    ts.createSourceFile("index.ts", readFileSync(join(import.meta.dir, "index.ts"), "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  function walkFn(): ts.Node {
    let found: ts.Node | undefined;
    const visit = (n: ts.Node): void => {
      if (!found && ts.isVariableDeclaration(n) && ts.isIdentifier(n.name) && n.name.text === "walkFallbackModels") found = n.initializer;
      n.forEachChild(visit);
    };
    visit(src());
    expect(found).toBeDefined();
    return found!;
  }

  test("its loop iterates fallbackWalkOrder(..., readFallbackOrder()), not modelClientMap", () => {
    const fn = walkFn();
    const loops: ts.ForOfStatement[] = [];
    const calls: string[] = [];
    const visit = (n: ts.Node): void => {
      if (ts.isForOfStatement(n)) loops.push(n);
      if (ts.isCallExpression(n) && ts.isIdentifier(n.expression)) calls.push(n.expression.text);
      n.forEachChild(visit);
    };
    visit(fn);
    expect(loops.length).toBe(1);
    expect(loops[0]!.expression.getText()).not.toContain("modelClientMap");
    expect(calls).toContain("fallbackWalkOrder");
    expect(calls).toContain("readFallbackOrder");
    // The iterated list is the fallbackWalkOrder result.
    const iter = loops[0]!.expression.getText();
    let assigned = "";
    const find = (n: ts.Node): void => {
      if (ts.isVariableDeclaration(n) && ts.isIdentifier(n.name) && n.name.text === iter && n.initializer) assigned = n.initializer.getText();
      n.forEachChild(find);
    };
    find(fn);
    expect(assigned.startsWith("fallbackWalkOrder(")).toBe(true);
  });
});
