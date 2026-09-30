// Pins the per-dispatch input-token budget (llmModelPolicy.max_input_tokens_per_dispatch).
//
// THE DEFECT: the prompt ceiling (llmModelPolicy.max_input_tokens) is PER TURN. A tool loop
// re-sends its growing message list every turn, so each turn passes the ceiling while the
// dispatch as a whole never stops. Measured 2026-09-30 15:40-16:40 UTC: goal-host:floor_tool_loop
// sent 3,161,321 input tokens in 2 calls on gpt-5 (~1.58M per call) and spent 1.906 USD of the
// fleet's single 2 USD/h spend envelope; the autonomous lane skipped selection for the hour.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  DispatchLedger, budgetedProviderCall, checkDispatchBudget, dispatchBudgetRefusal, dispatchKeyOf,
} from "./dispatch-budget";
import { isFailoverError, isExhaustedProviderError, isUnreachableProviderError, isUnauthenticatedProviderError } from "./provider-errors";

/** A provider stand-in that counts its calls and reports a fixed input-token usage. */
function mockProvider(reportedInputTokens: number) {
  const state = { calls: 0 };
  const call = async () => { state.calls += 1; return { usage: { prompt_tokens: reportedInputTokens } }; };
  return { state, call, inputTokensOf: (r: { usage: { prompt_tokens: number } }) => r.usage.prompt_tokens };
}
const capOf = (n: number) => async () => n;

describe("accumulation per dispatch id", () => {
  test("charges the provider-reported input tokens across turns of one call and across separate calls", async () => {
    const ledger = new DispatchLedger();
    const p = mockProvider(30_000);
    // Three turns of a tool loop, then a second call (a new floor iteration) with the same id.
    for (let turn = 0; turn < 3; turn++) {
      const r = await budgetedProviderCall({ ledger, id: "d-1", estimate: 25_000, readCap: capOf(400_000), call: p.call, inputTokensOf: p.inputTokensOf });
      expect("response" in r).toBe(true);
    }
    expect(ledger.used("d-1")).toBe(90_000); // settled to the REPORTED count, not the estimate
    const again = await budgetedProviderCall({ ledger, id: "d-1", estimate: 1_000, readCap: capOf(400_000), call: p.call, inputTokensOf: p.inputTokensOf });
    expect("response" in again).toBe(true);
    expect(ledger.used("d-1")).toBe(120_000);
    expect(p.state.calls).toBe(4);
  });

  test("a runaway loop is stopped: the 1.58M-token shape halts near the allowance", async () => {
    const ledger = new DispatchLedger();
    const p = mockProvider(80_000); // ~20 turns x ~80k reproduces the observed ~1.58M per-call total (shape inferred)
    let refusedAt = -1;
    for (let turn = 1; turn <= 20; turn++) {
      const r = await budgetedProviderCall({ ledger, id: "floor", estimate: 80_000, readCap: capOf(400_000), call: p.call, inputTokensOf: p.inputTokensOf });
      if ("refused" in r) { refusedAt = turn; break; }
    }
    expect(refusedAt).toBe(6);
    expect(p.state.calls).toBe(5);
    expect(ledger.used("floor")).toBe(400_000);
  });
});

describe("refusal happens before the provider call", () => {
  test("over the allowance: the provider is never called and the numbers are reported", async () => {
    const ledger = new DispatchLedger();
    ledger.charge("d-2", 390_000, true);
    const p = mockProvider(50_000);
    const r = await budgetedProviderCall({ ledger, id: "d-2", estimate: 20_000, readCap: capOf(400_000), call: p.call, inputTokensOf: p.inputTokensOf });
    expect("refused" in r).toBe(true);
    expect(p.state.calls).toBe(0);
    if ("refused" in r) {
      expect(r.refused).toEqual({ dispatch_id: "d-2", used_input_tokens: 390_000, estimate_input_tokens: 20_000, cap_input_tokens: 400_000 });
    }
    expect(ledger.used("d-2")).toBe(390_000); // a refusal charges nothing
  });

  test("the reservation is refunded when the provider call throws", async () => {
    const ledger = new DispatchLedger();
    await expect(budgetedProviderCall({
      ledger, id: "d-3", estimate: 10_000, readCap: capOf(400_000),
      call: async () => { throw new Error("boom"); }, inputTokensOf: () => 0,
    })).rejects.toThrow("boom");
    expect(ledger.used("d-3")).toBe(0);
  });

  test("the reservation is held DURING the call, so a concurrent call of the same dispatch sees it", async () => {
    const ledger = new DispatchLedger();
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    const slow = budgetedProviderCall({
      ledger, id: "d-4", estimate: 300_000, readCap: capOf(400_000),
      call: async () => { await gate; return { usage: { prompt_tokens: 300_000 } }; },
      inputTokensOf: (r) => r.usage.prompt_tokens,
    });
    await new Promise((r) => setTimeout(r, 0)); // let the first call reach its provider await
    const p = mockProvider(300_000);
    const second = await budgetedProviderCall({ ledger, id: "d-4", estimate: 300_000, readCap: capOf(400_000), call: p.call, inputTokensOf: p.inputTokensOf });
    expect("refused" in second).toBe(true);
    expect(p.state.calls).toBe(0);
    release();
    await slow;
  });

  test("the refusal is a structured, non-failover llmCompletion failure", () => {
    const refusal = dispatchBudgetRefusal({ caller: "goal-host:floor_tool_loop", task_type: "floor_tool_loop" }, "gpt-5",
      { dispatch_id: "d-5", used_input_tokens: 402, estimate_input_tokens: 429, cap_input_tokens: 400_000 }, 7);
    expect(refusal.resolved).toBe(false);
    expect(refusal.shape).toBe("llmCompletion");
    expect(refusal.dispatch_budget_exhausted).toBe(true);
    expect((refusal.dispatch_budget as Record<string, unknown>).cap_input_tokens).toBe(400_000);
    expect((refusal.dispatch_budget as Record<string, unknown>).turn).toBe(7);
    expect(String(refusal.error)).toContain("llmModelPolicy.max_input_tokens_per_dispatch");
    // Must never be classified as a provider failure: that would cool a healthy lane and walk fallbacks.
    for (const cls of [isFailoverError, isExhaustedProviderError, isUnreachableProviderError, isUnauthenticatedProviderError]) {
      expect(cls(refusal.error)).toBe(false);
    }
  });
});

describe("scope of the budget", () => {
  test("a different dispatch id is unaffected", async () => {
    const ledger = new DispatchLedger();
    ledger.charge("spent", 400_000, true);
    const p = mockProvider(10_000);
    const r = await budgetedProviderCall({ ledger, id: "fresh", estimate: 10_000, readCap: capOf(400_000), call: p.call, inputTokensOf: p.inputTokensOf });
    expect("response" in r).toBe(true);
    expect(p.state.calls).toBe(1);
    expect(ledger.used("fresh")).toBe(10_000);
    expect(ledger.used("spent")).toBe(400_000);
  });

  test("no id means no per-dispatch cap and no charge", async () => {
    const ledger = new DispatchLedger();
    const p = mockProvider(900_000);
    let capReads = 0;
    const r = await budgetedProviderCall({ ledger, id: null, estimate: 900_000, readCap: async () => { capReads++; return 1_000; }, call: p.call, inputTokensOf: p.inputTokensOf });
    expect("response" in r).toBe(true);
    expect(p.state.calls).toBe(1);
    expect(capReads).toBe(0);
    expect(ledger.size).toBe(0);
    expect(checkDispatchBudget(ledger, null, 10_000_000, 1_000)).toBeNull();
  });

  test("the id is dispatch_id, else execution_id", () => {
    expect(dispatchKeyOf({ dispatch_id: "a", execution_id: "b" })).toBe("a");
    expect(dispatchKeyOf({ execution_id: "b" })).toBe("b");
    expect(dispatchKeyOf({ dispatch_id: "  ", execution_id: "" })).toBeNull();
    expect(dispatchKeyOf({})).toBeNull();
    expect(dispatchKeyOf(undefined)).toBeNull();
  });

  test("requests without an id are counted by caller", () => {
    const ledger = new DispatchLedger();
    ledger.noteMissingId("development-vessel:feature_compose");
    ledger.noteMissingId("development-vessel:feature_compose");
    ledger.noteMissingId("");
    expect(ledger.snapshot().requests_without_id_by_caller).toEqual({ "development-vessel:feature_compose": 2, unknown: 1 });
  });
});

describe("the ledger is bounded", () => {
  test("entries idle longer than the TTL expire", () => {
    let now = 1_000_000;
    const ledger = new DispatchLedger({ ttlMs: 60_000, now: () => now });
    ledger.charge("old", 100_000, true);
    now += 30_000;
    ledger.charge("young", 5_000, true);
    expect(ledger.used("old")).toBe(100_000);
    now += 31_000; // "old" is 61s idle, "young" 31s
    expect(ledger.used("old")).toBe(0);
    ledger.charge("other", 1, true); // any charge prunes expired entries
    expect(ledger.size).toBe(2);
    expect(ledger.used("young")).toBe(5_000);
  });

  test("size never exceeds maxEntries; the least recently charged is evicted first", () => {
    const ledger = new DispatchLedger({ maxEntries: 3 });
    for (const id of ["a", "b", "c"]) ledger.charge(id, 10, true);
    ledger.charge("a", 10, true); // a becomes most recent
    ledger.charge("d", 10, true);
    expect(ledger.size).toBe(3);
    expect(ledger.used("b")).toBe(0);
    expect(ledger.used("a")).toBe(20);
    for (let i = 0; i < 500; i++) ledger.charge(`x${i}`, 1, true);
    expect(ledger.size).toBe(3);
  });

  test("the no-id counter is bounded too", () => {
    const ledger = new DispatchLedger({ maxEntries: 2 });
    for (let i = 0; i < 10; i++) ledger.noteMissingId(`c${i}`);
    expect(Object.keys(ledger.snapshot().requests_without_id_by_caller).length).toBe(2);
  });
});

describe("the allowance is read from the llmModelPolicy shape", () => {
  let dir: string;
  let policyPath: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "dispatch-budget-"));
    mkdirSync(join(dir, "policies"));
    policyPath = join(dir, "policies", "llm-model-policy.json");
    writeFileSync(policyPath, JSON.stringify({ rev: 3, updated_at: "2026-09-30T00:00:00.000Z", cost_weight: 0.25, arms: [{ model: "gpt-5", cost_per_mtok: 1, alpha: 1, beta: 1 }] }));
    process.env.WORKSPACE_ROOT = dir;
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("default 400k when the stored policy predates the field, and it is visible on read", async () => {
    const mp = await import(`./model-policy.js?t=${dir}`);
    expect(await mp.readMaxInputTokensPerDispatch()).toBe(400_000);
    expect(mp.POLICY_DEFAULTS.max_input_tokens_per_dispatch).toBe(400_000);
    const read = await mp.llmModelPolicyHandler();
    expect(read.body.effective.max_input_tokens_per_dispatch).toBe(400_000);
  });

  test("writing the policy field changes whether the same call is refused", async () => {
    const mp = await import(`./model-policy.js?t=${dir}`);
    const ledger = new DispatchLedger();
    ledger.charge("d-p", 4_000, true);
    const p = mockProvider(1);
    const run = () => budgetedProviderCall({ ledger, id: "d-p", estimate: 2_000, readCap: mp.readMaxInputTokensPerDispatch, call: p.call, inputTokensOf: () => 0 });

    const w1 = await mp.llmModelPolicyWriteHandler({ body: { max_input_tokens_per_dispatch: 5_000 } });
    expect(w1.resolved).toBe(true);
    expect(JSON.parse(readFileSync(policyPath, "utf-8")).max_input_tokens_per_dispatch).toBe(5_000);
    expect("refused" in (await run())).toBe(true);
    expect(p.state.calls).toBe(0);

    await mp.llmModelPolicyWriteHandler({ body: { max_input_tokens_per_dispatch: 50_000 } });
    expect("response" in (await run())).toBe(true);
    expect(p.state.calls).toBe(1);

    // null reverts to the default; the arms are untouched by a cap-only write
    await mp.llmModelPolicyWriteHandler({ body: { max_input_tokens_per_dispatch: null } });
    expect(await mp.readMaxInputTokensPerDispatch()).toBe(400_000);
    expect(JSON.parse(readFileSync(policyPath, "utf-8")).arms.length).toBe(1);
  });

  test("an invalid value is rejected by the writer and ignored by the reader", async () => {
    const mp = await import(`./model-policy.js?t=${dir}`);
    const bad = await mp.llmModelPolicyWriteHandler({ body: { max_input_tokens_per_dispatch: 429 } });
    expect(bad.resolved).toBe(false);
    expect(mp.effectiveMaxInputTokensPerDispatch({ max_input_tokens_per_dispatch: 999 })).toBe(400_000);
    expect(mp.effectiveMaxInputTokensPerDispatch({ max_input_tokens_per_dispatch: Number.NaN })).toBe(400_000);
    expect(mp.effectiveMaxInputTokensPerDispatch({ max_input_tokens_per_dispatch: 250_000 })).toBe(250_000);
  });
});

describe("source: every provider call in index.ts is budgeted, and the loops check before each turn's call", () => {
  const src = readFileSync(join(import.meta.dir, "index.ts"), "utf-8");

  test("both tool-use loops run the budget before the provider call on every turn", () => {
    const loops = [...src.matchAll(/for \(let iter = 1; iter <= maxIter; iter\+\+\) \{/g)].map((m) => m.index!);
    expect(loops.length).toBe(2);
    for (const start of loops) {
      const body = src.slice(start, start + 4000);
      const budget = body.indexOf("budgetedProviderCall(");
      const create = body.search(/\.(messages|chat\.completions)\.create\(/);
      expect(budget).toBeGreaterThan(-1);
      expect(create).toBeGreaterThan(budget);
      // the create is the budgeted call itself (inside `call: () =>`), not a separate call after it
      expect(body.slice(budget, create)).toMatch(/call: \(\) => (anthropic!|client)$/);
      expect(body.slice(budget, create)).toContain("readCap: readMaxInputTokensPerDispatch");
      expect(body.slice(budget, create)).toContain("dispatchKeyOf(body)");
    }
  });

  test("no provider create call in index.ts escapes the budget", () => {
    const creates = [...src.matchAll(/\.(messages|chat\.completions)\.create\(/g)];
    expect(creates.length).toBe(5);
    for (const m of creates) {
      const before = src.slice(Math.max(0, m.index! - 40), m.index!);
      expect(before).toMatch(/call: \(\) => (anthropic!|client)$/);
    }
  });

  test("the entry check refuses a spent dispatch before model selection, and refusals stop the fallback walk and are not graded", () => {
    const entry = src.indexOf("const llmCompletionMeteredHandler");
    const sel = src.indexOf("await llmCompletionWithPolicyHandler(ctx)", entry);
    const check = src.indexOf("checkDispatchBudget(dispatchLedger, entryKey", entry);
    expect(check).toBeGreaterThan(entry);
    expect(sel).toBeGreaterThan(check);
    expect(src).toContain("if (fb.dispatch_budget_exhausted === true) return fb;");
    expect(src).toMatch(/!providerLevelFailure && !budgetRefusal/);
  });
});
