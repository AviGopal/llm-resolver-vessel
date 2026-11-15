// EVERY COMPLETION CALL IS COUNTED BY OUTCOME AND PRODUCER (call-outcome.ts). FALSIFIER: a resolved call, an empty
// one, a strict pin's unserved refusal and a failure each land in their own outcome, under the model that served (else
// the one asked for); the metered handler counts every call, before the spend accounting that skips calls without usage.
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { countOutcome, outcomeOf, producerOf } from "./call-outcome";
import { EMPTY_COMPLETION } from "./empty-completion";

describe("call outcome", () => {
  test("[MUST-FAIL] each outcome is told apart, under the model that served", () => {
    const c: Record<string, number> = {};
    countOutcome(c, { resolved: true, model: "m-a" }, "m-a");
    countOutcome(c, { resolved: true, model: "m-b", served_model: "m-b", fallback_from: "m-a" }, "m-a");
    countOutcome(c, { resolved: false, error: EMPTY_COMPLETION, served_model: "m-a" }, "m-a");
    countOutcome(c, { resolved: false, unserved: "llm_completion", requested_model: "m-g" }, "m-g");
    countOutcome(c, { resolved: false, error: "402" }, "m-a");
    countOutcome(c, { resolved: true, model: "m-a" }, "m-a");
    expect(c).toEqual({ "resolved|m-a": 2, "resolved|m-b": 1, "empty|m-a": 1, "unserved|m-g": 1, "failed|m-a": 1 });
  });

  test("an answer naming no model is counted under the one asked for, else unknown", () => {
    expect(producerOf({ resolved: false }, "m-x")).toBe("m-x");
    expect(producerOf(undefined, undefined)).toBe("unknown");
    expect(outcomeOf(undefined)).toBe("failed");
  });

  test("[MUST-FAIL] the metered handler counts every call, before the spend accounting, and the window logs them", () => {
    const src = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
    const call = src.indexOf("const result = await llmCompletionWithPolicyHandler(ctx);");
    const count = src.indexOf("countOutcome(w.outcomes ??= {}, result,");
    const spend = src.indexOf("spend accounting failed (non-fatal)");
    expect(call).toBeGreaterThan(0);
    expect(count).toBeGreaterThan(call);
    expect(count).toBeLessThan(spend);
    expect(src).toContain("[llm-outcomes] window");
  });
});
