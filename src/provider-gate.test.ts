// A PROVIDER HELD BY ITS STANDING IS NOT RE-PROBED ON EVERY COOLDOWN EXPIRY (check-first).
//
// THE DEFECT. The per-model cooldown keeps a failed arm out for ONE window (10 min), and nothing
// keeps it out after that. A capped `:free` slug therefore came back every 10 minutes, won the
// Thompson draw again (alpha 57-66 at cost ~0, because its infra failures are never charged as
// beta), 429'd again and burned a failover hop: 4.0-4.7 exhaust marks per model per hour; the
// out-of-credit arms did the same at 5.3. "No draws inside the arm's own cooldown" was already
// true, so a test of that would pass with no fix; the defect is the re-probe after expiry.
//
// THE FIX (pinned). While a provider's standing holds (the free cap, or N 402s, measured within a
// window), selectArm gives its affected arms no draw and the fallback walk skips them. When the
// window lapses, ONE draw goes through as a probe and re-arms the window. N and the window are
// read at use time from the shaped llmModelPolicy (provider_gate_credit_402s,
// provider_gate_window_ms), with recorded defaults (2, 2 h).
//
// FALSIFIER. A capped `:free` arm (or a credit-exhausted provider's arm) gets at most one draw per
// window, across cooldown expiries, with alpha AND beta byte-unchanged while gated.
// CONTROLS: a single 429 clearing within a cooldown does not gate; a single 402 does not gate at
// the default N; a recovered provider is drawn again.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";
import { providerStanding } from "./provider-standing";

const MIN = 60_000;
const E_FREE_CAP = "429 Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day";
const E402 = "402 Insufficient credits";
const E_RATE = "429 Too Many Requests: rate limit exceeded, retry in 2s";

let dir: string;
let policyPath: string;
let sfx: string;
let OR: string, GQ: string, FREE1: string, FREE2: string, PAID1: string, PAID2: string, GROQ: string;

type Sel = { model: string; meta: Record<string, unknown> } | null;
type Mod = {
  selectArm: (taskType?: string, available?: string[]) => Promise<Sel>;
  llmModelPolicyWriteHandler: (ctx: { body: unknown }) => Promise<{ resolved: boolean; error?: string; body?: { effective: Record<string, number> } }>;
  llmModelPolicyHandler: () => Promise<{ body: { effective: Record<string, number> } }>;
  setProviderStanding: (s: unknown) => void;
};
/** A fresh model-policy instance wired to the vessel's tracker, as index.ts wires it at startup. */
async function load(key: string): Promise<Mod> {
  const m = (await import(`./model-policy.js?t=${key}`)) as unknown as Mod;
  // Absent at base, where the controls still run against the ungated selector.
  if (typeof m.setProviderStanding === "function") m.setProviderStanding(providerStanding);
  return m;
}
const mod = (): Promise<Mod> => load(dir);

function writePolicy(arms: Array<Record<string, unknown>>, extra: Record<string, unknown> = {}): void {
  writeFileSync(policyPath, JSON.stringify({ rev: 1, updated_at: new Date().toISOString(), cost_weight: 0.25, arms, ...extra }));
}
const now = (): string => new Date().toISOString();
const freeFirstArms = (): Array<Record<string, unknown>> => [
  // The live picture: capped :free arms with borrowed alpha at ~0 cost win every ungated draw.
  { model: FREE1, cost_per_mtok: 0.01, alpha: 57, beta: 1, last_updated_at: now() },
  { model: FREE2, cost_per_mtok: 0.01, alpha: 66, beta: 1, last_updated_at: now() },
  { model: PAID1, cost_per_mtok: 0.3, alpha: 4, beta: 3, last_updated_at: now() },
];
const paidFirstArms = (): Array<Record<string, unknown>> => [
  { model: PAID1, cost_per_mtok: 0.01, alpha: 60, beta: 1, last_updated_at: now() },
  { model: PAID2, cost_per_mtok: 0.01, alpha: 60, beta: 1, last_updated_at: now() },
  { model: GROQ, cost_per_mtok: 0.3, alpha: 4, beta: 3, last_updated_at: now() },
];

async function draws(m: Mod, n: number, available: string[]): Promise<string[]> {
  const out: string[] = [];
  for (let i = 0; i < n; i++) out.push((await m.selectArm(undefined, available))!.model);
  return out;
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "provider-gate-"));
  mkdirSync(join(dir, "policies"));
  policyPath = join(dir, "policies", "llm-model-policy.json");
  process.env.WORKSPACE_ROOT = dir;
  // The tracker is the vessel's one shared instance; fresh provider and model ids per test.
  sfx = Math.random().toString(36).slice(2, 8);
  OR = `router-${sfx}`; GQ = `funded-${sfx}`;
  FREE1 = `vendor/free-one-${sfx}:free`; FREE2 = `vendor/free-two-${sfx}:free`;
  PAID1 = `vendor/paid-one-${sfx}`; PAID2 = `vendor/paid-two-${sfx}`; GROQ = `funded-model-${sfx}`;
  providerStanding.registerModels(OR, [PAID1, PAID2, FREE1, FREE2]);
  providerStanding.registerModels(GQ, [GROQ]);
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

describe("MUST-FAIL: the free cap holds a provider's :free arms out across cooldown expiries", () => {
  for (const ageMin of [11, 61, 119]) {
    test(`a free-cap 429 ${ageMin} min ago (its 10-min model cooldown long expired): no :free draws, alpha/beta byte-unchanged`, async () => {
      writePolicy(freeFirstArms());
      providerStanding.recordFailure(OR, FREE1, E_FREE_CAP, Date.now() - ageMin * MIN);
      const before = readFileSync(policyPath, "utf-8");
      const m = await mod();
      const picked = await draws(m, 60, [FREE1, FREE2, PAID1]);
      expect(picked.filter((x) => x.endsWith(":free"))).toEqual([]);
      expect(new Set(picked)).toEqual(new Set([PAID1]));
      expect(readFileSync(policyPath, "utf-8")).toBe(before);
      const sel = await m.selectArm(undefined, [FREE1, FREE2, PAID1]);
      expect(sel!.meta.gated).toEqual([{ model: FREE1, reason: "free_cap" }, { model: FREE2, reason: "free_cap" }]);
    });
  }

  test("after the window lapses, ONE :free draw goes through as the probe; the rest stay held out", async () => {
    writePolicy(freeFirstArms());
    providerStanding.recordFailure(OR, FREE1, E_FREE_CAP, Date.now() - 121 * MIN);
    const before = readFileSync(policyPath, "utf-8");
    const m = await mod();
    const picked = await draws(m, 60, [FREE1, FREE2, PAID1]);
    expect(picked.filter((x) => x.endsWith(":free")).length).toBe(1); // <= 1 exhaust mark per window
    expect(readFileSync(policyPath, "utf-8")).toBe(before);
  });

  test("a probe that succeeds re-admits the :free arms", async () => {
    writePolicy(freeFirstArms());
    providerStanding.recordFailure(OR, FREE1, E_FREE_CAP, Date.now() - 30 * MIN);
    providerStanding.recordServed(OR, FREE2);
    const picked = await draws(await mod(), 20, [FREE1, FREE2, PAID1]);
    expect(picked.some((x) => x.endsWith(":free"))).toBe(true);
  });
});

describe("MUST-FAIL (F2, selection half): after the free cap 429s in-window, the next draw for that task type is a FUNDED paid slug", () => {
  test("task-typed draw: capped :free and a credit-exhausted provider's paid arm are passed over for the funded one", async () => {
    const TASK = "walk_llm_completion";
    const sfx3 = Math.random().toString(36).slice(2, 8);
    const DRY = `dry-${sfx3}`, DRY_PAID = `dry-model-${sfx3}`;
    providerStanding.registerModels(DRY, [DRY_PAID]);
    const t = (a: number, b: number) => ({ task_alpha: { [TASK]: a }, task_beta: { [TASK]: b } });
    writePolicy([
      { model: FREE1, cost_per_mtok: 0.01, alpha: 1, beta: 1, ...t(57, 1), last_updated_at: now() },
      { model: FREE2, cost_per_mtok: 0.01, alpha: 1, beta: 1, ...t(66, 1), last_updated_at: now() },
      { model: DRY_PAID, cost_per_mtok: 0.05, alpha: 1, beta: 1, ...t(40, 1), last_updated_at: now() },
      { model: GROQ, cost_per_mtok: 0.3, alpha: 1, beta: 1, ...t(4, 3), last_updated_at: now() },
    ]);
    providerStanding.recordFailure(DRY, DRY_PAID, E402, Date.now() - 50 * MIN);
    providerStanding.recordFailure(DRY, DRY_PAID, E402, Date.now() - 30 * MIN);
    providerStanding.recordFailure(OR, FREE2, E_FREE_CAP, Date.now() - 5 * MIN);
    const m = await mod();
    for (let i = 0; i < 30; i++) expect((await m.selectArm(TASK, [FREE1, FREE2, DRY_PAID, GROQ]))!.model).toBe(GROQ);
  });

  test("CONTROL: with no 429 in-window, the task-typed draw is as before (the :free arm wins on its evidence)", async () => {
    const TASK = "walk_llm_completion";
    writePolicy([
      { model: FREE2, cost_per_mtok: 0.01, alpha: 1, beta: 1, task_alpha: { [TASK]: 66 }, task_beta: { [TASK]: 1 }, last_updated_at: now() },
      { model: GROQ, cost_per_mtok: 0.3, alpha: 1, beta: 1, task_alpha: { [TASK]: 4 }, task_beta: { [TASK]: 3 }, last_updated_at: now() },
    ]);
    const m = await mod();
    for (let i = 0; i < 10; i++) expect((await m.selectArm(TASK, [FREE2, GROQ]))!.model).toBe(FREE2);
  });
});

describe("MUST-FAIL: N 402s hold a provider's paid arms out", () => {
  test("two 402s (default N=2) across expiries: the provider's paid arms, failed or not, get no draw", async () => {
    writePolicy(paidFirstArms());
    providerStanding.recordFailure(OR, PAID1, E402, Date.now() - 40 * MIN);
    providerStanding.recordFailure(OR, PAID1, E402, Date.now() - 25 * MIN);
    const before = readFileSync(policyPath, "utf-8");
    const picked = await draws(await mod(), 60, [PAID1, PAID2, GROQ]);
    expect(new Set(picked)).toEqual(new Set([GROQ]));
    expect(readFileSync(policyPath, "utf-8")).toBe(before);
  });

  test("a paid 402 does not hold the provider's :free arms; a 402 FROM a :free slug does", async () => {
    writePolicy([...freeFirstArms(), { model: GROQ, cost_per_mtok: 0.3, alpha: 4, beta: 3, last_updated_at: now() }]);
    providerStanding.recordFailure(OR, PAID1, E402, Date.now() - 40 * MIN);
    providerStanding.recordFailure(OR, PAID1, E402, Date.now() - 25 * MIN);
    const m = await mod();
    expect((await draws(m, 10, [FREE1, FREE2, PAID1])).some((x) => x.endsWith(":free"))).toBe(true);
    providerStanding.recordFailure(OR, FREE1, E402, Date.now() - 20 * MIN);
    // FREE1/FREE2 now held (negative balance), PAID1 held: every candidate is out, so the gate
    // stands aside rather than empty the field, and says so.
    const sel = await m.selectArm(undefined, [FREE1, FREE2, PAID1]);
    expect(sel!.meta.gate_bypassed).toBe(true);
    expect((await draws(m, 20, [FREE1, FREE2, PAID1, GROQ])).every((x) => x === GROQ)).toBe(true);
  });
});

describe("MUST-FAIL: N and the window are read at use time from the shaped policy", () => {
  test("provider_gate_credit_402s=1 gates on one 402; provider_gate_window_ms=60s lets a 2-min-old cap probe", async () => {
    writePolicy(paidFirstArms());
    const m = await mod();
    const w = await m.llmModelPolicyWriteHandler({ body: { provider_gate_credit_402s: 1, provider_gate_window_ms: 60_000 } });
    expect(w.resolved).toBe(true);
    expect(w.body!.effective.provider_gate_credit_402s).toBe(1);
    expect(w.body!.effective.provider_gate_window_ms).toBe(60_000);
    providerStanding.recordFailure(OR, PAID1, E402, Date.now() - 10_000);
    expect(new Set(await draws(m, 30, [PAID1, PAID2, GROQ]))).toEqual(new Set([GROQ]));
    // A 2-minute-old 402 is outside a 60 s window: one probe, then held out again.
    const sfx2 = Math.random().toString(36).slice(2, 8);
    const P3 = `vendor/paid-three-${sfx2}`;
    providerStanding.registerModels(`router-${sfx2}`, [P3]);
    providerStanding.recordFailure(`router-${sfx2}`, P3, E402, Date.now() - 2 * MIN);
    writePolicy([{ model: P3, cost_per_mtok: 0.01, alpha: 60, beta: 1, last_updated_at: now() }, { model: GROQ, cost_per_mtok: 0.3, alpha: 4, beta: 3, last_updated_at: now() }],
      { provider_gate_credit_402s: 1, provider_gate_window_ms: 60_000 });
    const m2 = await load(`${dir}-2`);
    expect((await draws(m2, 30, [P3, GROQ])).filter((x) => x === P3).length).toBe(1);
  });

  test("the defaults are recorded and served as effective; invalid writes are refused", async () => {
    writePolicy(paidFirstArms());
    const m = await mod();
    const eff = (await m.llmModelPolicyHandler()).body.effective;
    expect(eff.provider_gate_credit_402s).toBe(2);
    expect(eff.provider_gate_window_ms).toBe(7_200_000);
    for (const bad of [0, 1.5, -1]) expect((await m.llmModelPolicyWriteHandler({ body: { provider_gate_credit_402s: bad } })).resolved).toBe(false);
    expect((await m.llmModelPolicyWriteHandler({ body: { provider_gate_window_ms: 500 } })).resolved).toBe(false);
  });
});

describe("CONTROLS", () => {
  test("a single 429 that clears within a cooldown does not gate the provider", async () => {
    writePolicy(freeFirstArms());
    providerStanding.recordFailure(OR, FREE1, E_RATE, Date.now() - 1 * MIN);
    const picked = await draws(await mod(), 20, [FREE1, FREE2, PAID1]);
    expect(picked.every((x) => x.endsWith(":free"))).toBe(true);
  });

  test("a single 402 does not gate at the default N", async () => {
    writePolicy(paidFirstArms());
    providerStanding.recordFailure(OR, PAID1, E402, Date.now() - 1 * MIN);
    const picked = await draws(await mod(), 20, [PAID1, PAID2, GROQ]);
    expect(picked.some((x) => x === PAID1 || x === PAID2)).toBe(true);
  });

  test("a paid success clears the credit hold", async () => {
    writePolicy(paidFirstArms());
    providerStanding.recordFailure(OR, PAID1, E402, Date.now() - 40 * MIN);
    providerStanding.recordFailure(OR, PAID1, E402, Date.now() - 25 * MIN);
    providerStanding.recordServed(OR, PAID2);
    const picked = await draws(await mod(), 20, [PAID1, PAID2, GROQ]);
    expect(picked.some((x) => x === PAID1 || x === PAID2)).toBe(true);
  });
});

describe("MUST-FAIL: the fallback walk asks the same gate", () => {
  test("admit: held out inside the window; one probe after it lapses, then held again", () => {
    const params = { creditFailures: 2, windowMs: 2 * 60 * MIN };
    const t = Date.now();
    providerStanding.recordFailure(OR, FREE1, E_FREE_CAP, t - 30 * MIN);
    expect(providerStanding.admit(FREE2, t, params)).toBe(false);
    expect(providerStanding.admit(FREE1, t + 91 * MIN, params)).toBe(true); // the probe
    expect(providerStanding.admit(FREE2, t + 92 * MIN, params)).toBe(false);
    expect(providerStanding.admit(PAID1, t, params)).toBe(true); // the free cap holds :free only
  });

  test("model-policy.ts takes the provider standing", async () => {
    expect(typeof (await mod()).setProviderStanding).toBe("function");
  });

  test("index.ts hands selection the tracker it feeds", () => {
    const text = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
    const src = ts.createSourceFile("index.ts", text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    const topLevel = src.statements.filter((st) => ts.isExpressionStatement(st) && ts.isCallExpression(st.expression)
      && ts.isIdentifier(st.expression.expression) && st.expression.expression.text === "setProviderStanding");
    expect(topLevel.length).toBe(1);
    expect((topLevel[0] as ts.ExpressionStatement).expression.getText()).toBe("setProviderStanding(providerStanding)");
  });

  test("walkFallbackModels skips a model the gate does not admit, before it tries it", () => {
    const src = ts.createSourceFile("index.ts", readFileSync(join(import.meta.dir, "index.ts"), "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    let walkFn: ts.Node | undefined;
    const visit = (n: ts.Node): void => {
      if (!walkFn && ts.isVariableDeclaration(n) && ts.isIdentifier(n.name) && n.name.text === "walkFallbackModels") walkFn = n.initializer;
      n.forEachChild(visit);
    };
    visit(src);
    expect(walkFn).toBeDefined();
    const calls: Array<{ name: string; at: number }> = [];
    const collect = (n: ts.Node): void => {
      if (ts.isCallExpression(n)) {
        const e = n.expression;
        calls.push({ name: ts.isIdentifier(e) ? e.text : ts.isPropertyAccessExpression(e) ? e.name.text : "", at: n.getStart() });
      }
      n.forEachChild(collect);
    };
    collect(walkFn!);
    const first = (name: string): number => Math.min(...calls.filter((c) => c.name === name).map((c) => c.at));
    expect(Number.isFinite(first("admit"))).toBe(true);
    expect(first("readGateParams")).toBeLessThan(first("admit"));
    expect(first("admit")).toBeLessThan(first("resolveWithOpenAI"));
  });
});
