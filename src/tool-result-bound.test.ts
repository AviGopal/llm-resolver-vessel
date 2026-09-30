// Pins the tool-result bound (llmModelPolicy.tool_result_max_chars / tool_results_total_max_chars).
//
// THE DEFECT: both tool loops appended every tool result VERBATIM and re-sent the whole message
// list each turn. The per-turn prompt ceiling then refused 20+ goal-host:floor_tool_loop turns
// estimated at 202k to 1,673,516 input tokens, but only after the earlier turns were paid for:
// one append moved a turn from under 200k tokens to 1.6M.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { POINTER_ARGS_MAX_CHARS, ToolResultBudget, toolResultText } from "./tool-result-bound";

const LIMITS = { per_result_max_chars: 16_000, total_max_chars: 64_000 };
/** Upper bound on a record's own text (marker + JSON + separator line), independent of the result. */
const RECORD_OVERHEAD_MAX = 1_200;

describe("one oversized result", () => {
  test("a 525k-char result enters the message list at <= cap plus a record", () => {
    const b = new ToolResultBudget(LIMITS);
    const big = "x".repeat(525_000);
    const content = b.bound("source_code", { filePath: "repos/goal-host-vessel/src/index.ts" }, { ok: true, result: big });
    expect(content.length).toBeLessThanOrEqual(LIMITS.per_result_max_chars + RECORD_OVERHEAD_MAX);
    expect(content.startsWith("[TOOL RESULT TRUNCATED] ")).toBe(true);
    const record = JSON.parse(content.slice("[TOOL RESULT TRUNCATED] ".length, content.indexOf("\n")));
    expect(record).toMatchObject({
      shape: "toolResultExcerpt", truncated: true, producer: "source_code", result_kind: "text",
      size_chars: 525_000, shown_chars: 16_000,
      pointer: { tool: "source_code", args: { filePath: "repos/goal-host-vessel/src/index.ts" } },
    });
    expect(content.endsWith("x".repeat(16_000))).toBe(true); // the head excerpt is there
  });

  test("an object result is measured and excerpted as the JSON the loop used to send", () => {
    const b = new ToolResultBudget(LIMITS);
    const obj = { rows: Array.from({ length: 5_000 }, (_, i) => ({ i, v: "abcdefghij" })) };
    const content = b.bound("shell", { command: "cat big.json" }, { ok: true, result: obj });
    expect(content.length).toBeLessThanOrEqual(LIMITS.per_result_max_chars + RECORD_OVERHEAD_MAX);
    const record = JSON.parse(content.slice("[TOOL RESULT TRUNCATED] ".length, content.indexOf("\n")));
    expect(record.result_kind).toBe("object");
    expect(record.size_chars).toBe(JSON.stringify(obj).length);
  });

  test("huge arguments are compacted in the pointer, so the record stays bounded", () => {
    const b = new ToolResultBudget(LIMITS);
    const content = b.bound("shell", { command: "y".repeat(50_000) }, { ok: true, result: "z".repeat(100_000) });
    expect(content.length).toBeLessThanOrEqual(LIMITS.per_result_max_chars + RECORD_OVERHEAD_MAX);
    const record = JSON.parse(content.slice("[TOOL RESULT TRUNCATED] ".length, content.indexOf("\n")));
    expect(String(record.pointer.args).length).toBeLessThan(POINTER_ARGS_MAX_CHARS + 40);
  });
});

describe("small results are unchanged", () => {
  test("strings, objects and errors under the cap are byte-identical to the old content", () => {
    const b = new ToolResultBudget(LIMITS);
    const cases = [
      { ok: true, result: "hello" },
      { ok: true, result: { a: 1, b: [1, 2, 3] } },
      { ok: false, result: null, error: "HTTP 500: boom" },
      { ok: true, result: undefined },
    ];
    for (const r of cases) {
      const old = typeof r.result === "string" ? r.result : JSON.stringify(r.result ?? r.error ?? null);
      expect(b.bound("t", {}, r)).toBe(old);
    }
    expect(toolResultText({ ok: true, result: "x" })).toBe("x");
  });

  test("a result exactly at the cap is unchanged", () => {
    const b = new ToolResultBudget(LIMITS);
    const s = "q".repeat(LIMITS.per_result_max_chars);
    expect(b.bound("t", {}, { ok: true, result: s })).toBe(s);
  });
});

describe("the per-request total is bounded", () => {
  test("30 turns x 8 calls of 525k-char results add at most total + one record each", () => {
    const b = new ToolResultBudget(LIMITS);
    let sum = 0, calls = 0, excerpts = 0;
    for (let turn = 0; turn < 30; turn++) {
      for (let c = 0; c < 8; c++) {
        const content = b.bound("source_code", { filePath: `f${turn}-${c}` }, { ok: true, result: "x".repeat(525_000) });
        sum += content.length; calls++;
        if (content.includes("\n--- head")) excerpts++;
      }
    }
    expect(sum).toBeLessThanOrEqual(LIMITS.total_max_chars + calls * RECORD_OVERHEAD_MAX);
    expect(excerpts).toBe(4); // 64k total / 16k per result
    expect(b.remainingChars).toBe(0);
  });

  test("once the total is spent even a small result becomes a record without excerpt", () => {
    const b = new ToolResultBudget({ per_result_max_chars: 16_000, total_max_chars: 20_000 });
    expect(b.bound("t", {}, { ok: true, result: "a".repeat(15_000) })).toBe("a".repeat(15_000));
    const partial = b.bound("t", {}, { ok: true, result: "b".repeat(10_000) });
    expect(partial).toContain('"shown_chars":5000');
    expect(partial).toContain("tool_results_total_max_chars");
    const none = b.bound("t", {}, { ok: true, result: "c".repeat(2_000) });
    expect(none).toContain('"shown_chars":0');
    expect(none).not.toContain("--- head");
  });

  test("the re-sent list stays bounded over turns (what each turn re-sends never exceeds total + records)", () => {
    const b = new ToolResultBudget(LIMITS);
    const messages: string[] = [];
    for (let turn = 1; turn <= 20; turn++) {
      messages.push(b.bound("shell", { command: `cat part${turn}` }, { ok: true, result: "r".repeat(40_000) }));
      const resent = messages.reduce((n, m) => n + m.length, 0);
      expect(resent).toBeLessThanOrEqual(LIMITS.total_max_chars + turn * RECORD_OVERHEAD_MAX);
    }
  });
});

describe("the allowances are read from the llmModelPolicy shape", () => {
  let dir: string;
  let policyPath: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "tool-result-bound-"));
    mkdirSync(join(dir, "policies"));
    policyPath = join(dir, "policies", "llm-model-policy.json");
    writeFileSync(policyPath, JSON.stringify({ rev: 1, updated_at: "2026-09-30T00:00:00.000Z", cost_weight: 0.25, arms: [{ model: "m", cost_per_mtok: 1, alpha: 1, beta: 1 }] }));
    process.env.WORKSPACE_ROOT = dir;
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  test("defaults when unset; a policy write changes the excerpt size; null reverts", async () => {
    const mp = await import(`./model-policy.js?t=${dir}`);
    expect(await mp.readToolResultLimits()).toEqual({ per_result_max_chars: 16_000, total_max_chars: 64_000 });
    expect((await mp.llmModelPolicyHandler()).body.effective).toMatchObject({ tool_result_max_chars: 16_000, tool_results_total_max_chars: 64_000 });

    const w = await mp.llmModelPolicyWriteHandler({ body: { tool_result_max_chars: 2_000, tool_results_total_max_chars: 3_000 } });
    expect(w.resolved).toBe(true);
    expect(JSON.parse(readFileSync(policyPath, "utf-8")).tool_result_max_chars).toBe(2_000);
    const b = new ToolResultBudget(await mp.readToolResultLimits());
    expect(b.bound("t", {}, { ok: true, result: "x".repeat(10_000) })).toContain('"shown_chars":2000');

    await mp.llmModelPolicyWriteHandler({ body: { tool_result_max_chars: null } });
    expect((await mp.readToolResultLimits()).per_result_max_chars).toBe(16_000);
    expect((await mp.llmModelPolicyWriteHandler({ body: { tool_results_total_max_chars: 12 } })).resolved).toBe(false);
  });
});

describe("source: both tool loops bound every tool result through the helper", () => {
  const src = readFileSync(join(import.meta.dir, "index.ts"), "utf-8");

  test("no tool result is appended verbatim any more", () => {
    expect(src).not.toMatch(/content: typeof r\.result === "string" \? r\.result/);
  });

  test("each loop creates one budget from the policy BEFORE its first turn and bounds at its push site", () => {
    const loops = [...src.matchAll(/for \(let iter = 1; iter <= maxIter; iter\+\+\) \{/g)].map((m) => m.index!);
    expect(loops.length).toBe(2);
    const creations = [...src.matchAll(/const toolResultBudget = new ToolResultBudget\(await readToolResultLimits\(\)\);/g)].map((m) => m.index!);
    expect(creations.length).toBe(2);
    for (let i = 0; i < 2; i++) {
      expect(creations[i]).toBeLessThan(loops[i]!);
      if (i > 0) expect(creations[i]).toBeGreaterThan(loops[i - 1]!);
    }
    const anthropicPush = src.slice(loops[0]!, loops[1]!);
    expect(anthropicPush).toContain('toolResults.push({ type: "tool_result", tool_use_id: tu.id, content: toolResultBudget.bound(tu.name, tu.input, r)');
    const openaiPush = src.slice(loops[1]!, loops[1]! + 6000);
    expect(openaiPush).toContain("content: toolResultBudget.bound(tc.function.name, toolInput, r),");
  });
});
