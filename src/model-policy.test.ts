import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// model-policy resolves its file from WORKSPACE_ROOT at IMPORT time, so the env
// has to be set before the dynamic import below.
let dir: string;
let policyPath: string;

const RUNPOD = "Qwen/Qwen3-Coder-Next-FP8";
const ALWAYS_ON = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8";

function writePolicy(arms: unknown[]): void {
  writeFileSync(policyPath, JSON.stringify({ rev: 10, updated_at: "2026-08-08T00:00:00.000Z", cost_weight: 0.25, arms }));
}
function readArms(): Array<{ model: string; cost_per_mtok?: number; note?: string; alpha?: number; beta?: number }> {
  return JSON.parse(readFileSync(policyPath, "utf-8")).arms;
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "policy-"));
  mkdirSync(join(dir, "policies"));
  policyPath = join(dir, "policies", "llm-model-policy.json");
  process.env.WORKSPACE_ROOT = dir;
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

describe("repriceSeededArm — repair the cost-blind seeding without stomping the operator", () => {
  it("OBSERVED LIVE 2026-08-08: repairs the RunPod arm the generic vLLM path seeded at 0", async () => {
    // The generic self-hosted path won the race and registered metered
    // GPU-second capacity at cost 0; ensureArmsForModels then refused to
    // correct it because the arm already existed.
    writePolicy([{ model: RUNPOD, cost_per_mtok: 0, alpha: 7, beta: 3, note: "self-hosted vLLM" }]);
    const { repriceSeededArm } = await import(`./model-policy.js?t=${dir}`);
    expect(await repriceSeededArm(RUNPOD, 0.35, "self-hosted vLLM", "runpod serverless")).toBe(true);
    const [arm] = readArms();
    expect(arm.cost_per_mtok).toBe(0.35);
    expect(arm.note).toBe("runpod serverless");
    // The price was wrong; the evidence was not.
    expect(arm.alpha).toBe(7);
    expect(arm.beta).toBe(3);
  });

  it("is one-shot — rewriting the note means it cannot fire twice", async () => {
    writePolicy([{ model: RUNPOD, cost_per_mtok: 0, alpha: 1, beta: 1, note: "self-hosted vLLM" }]);
    const { repriceSeededArm } = await import(`./model-policy.js?t=${dir}`);
    expect(await repriceSeededArm(RUNPOD, 0.35, "self-hosted vLLM", "runpod serverless")).toBe(true);
    expect(await repriceSeededArm(RUNPOD, 0.35, "self-hosted vLLM", "runpod serverless")).toBe(false);
  });

  it("REGRESSION: never touches an operator-tuned price, even at the same note", async () => {
    // An operator who set 0.9 deliberately must not have it reset on every boot.
    writePolicy([{ model: RUNPOD, cost_per_mtok: 0.9, alpha: 1, beta: 1, note: "self-hosted vLLM" }]);
    const { repriceSeededArm } = await import(`./model-policy.js?t=${dir}`);
    expect(await repriceSeededArm(RUNPOD, 0.35, "self-hosted vLLM", "runpod serverless")).toBe(false);
    expect(readArms()[0].cost_per_mtok).toBe(0.9);
  });

  it("REGRESSION: an always-on self-hosted arm at 0 is CORRECT and must survive", async () => {
    // Qwen3-30B-A3B is the q3-30b.syzygy.host tunnel box — already paid for, so
    // 0 is right for it. Scoping lives with the caller, so a caller that passes
    // the wrong model would corrupt it; this pins that the guard is the model.
    writePolicy([{ model: ALWAYS_ON, cost_per_mtok: 0, alpha: 4, beta: 1, note: "self-hosted vLLM" }]);
    const { repriceSeededArm } = await import(`./model-policy.js?t=${dir}`);
    // The RunPod caller only ever asks about RUNPOD models, so this is a no-op.
    expect(await repriceSeededArm(RUNPOD, 0.35, "self-hosted vLLM", "runpod serverless")).toBe(false);
    expect(readArms()[0].cost_per_mtok).toBe(0);
    expect(readArms()[0].model).toBe(ALWAYS_ON);
  });

  it("no-ops on an unknown model rather than inventing an arm", async () => {
    writePolicy([{ model: ALWAYS_ON, cost_per_mtok: 0, alpha: 1, beta: 1, note: "self-hosted vLLM" }]);
    const { repriceSeededArm } = await import(`./model-policy.js?t=${dir}`);
    expect(await repriceSeededArm("nope/not-a-model", 0.35, "self-hosted vLLM", "x")).toBe(false);
    expect(readArms()).toHaveLength(1);
  });
});
