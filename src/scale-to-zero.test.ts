import { describe, expect, it } from "bun:test";

import { isScaleToZeroCold, isWarmFromHealth } from "./scale-to-zero";

const MODELS = new Set(["Qwen/Qwen3-Coder-Next-FP8"]);

describe("isWarmFromHealth — can this endpoint take work right now", () => {
  it("OBSERVED LIVE 2026-08-08: 3 standby workers idle, no jobs in flight", () => {
    // The exact /health body from the live vllm-qwen3-coder-next-fp8 endpoint.
    // workersStandby=3 holds workers warm, so the arm IS servable — the 29,491
    // completed jobs had long since stopped and every worker sat idle.
    expect(isWarmFromHealth({
      jobs: { completed: 29491, failed: 21, inProgress: 0, inQueue: 0, retried: 0 },
      workers: { idle: 3, initializing: 0, ready: 3, running: 0, throttled: 0, unhealthy: 0 },
    })).toBe(true);
  });

  it("scaled to zero — every count absent or zero", () => {
    expect(isWarmFromHealth({ workers: { idle: 0, ready: 0, running: 0 } })).toBe(false);
    expect(isWarmFromHealth({ workers: {} })).toBe(false);
  });

  it("a worker still PULLING THE IMAGE is not warm", () => {
    // The whole point of the gate: `initializing` is the 20-40 min tail itself.
    expect(isWarmFromHealth({ workers: { initializing: 1, ready: 0, idle: 0, running: 0 } })).toBe(false);
  });

  it("throttled or unhealthy capacity cannot serve", () => {
    expect(isWarmFromHealth({ workers: { throttled: 3, unhealthy: 1 } })).toBe(false);
  });

  it("a busy endpoint is warm — running counts even with nothing idle", () => {
    expect(isWarmFromHealth({ workers: { idle: 0, ready: 0, running: 2 } })).toBe(true);
  });

  it("fails closed on anything it cannot read", () => {
    // A control-plane outage, a 401 body, a truncated response: unknown is never
    // read as "holding capacity", or the gate would leak traffic into a sleeper.
    for (const body of [null, undefined, {}, "warm", 42, { workers: null }, { workers: "3" },
                        { status: 401, title: "Unauthorized" }, { workers: { ready: "3" } }]) {
      expect(isWarmFromHealth(body)).toBe(false);
    }
  });

  it("ignores NaN/Infinity rather than treating them as capacity", () => {
    // Non-finite counts are nonsense from the control plane, not evidence of
    // workers — they read as 0 and the endpoint stays gated.
    expect(isWarmFromHealth({ workers: { ready: NaN } })).toBe(false);
    expect(isWarmFromHealth({ workers: { ready: Infinity } })).toBe(false);
  });
});

describe("isScaleToZeroCold — gating is scoped to the metered lane", () => {
  it("gates the scale-to-zero model only while cold", () => {
    expect(isScaleToZeroCold("Qwen/Qwen3-Coder-Next-FP8", MODELS, false)).toBe(true);
    expect(isScaleToZeroCold("Qwen/Qwen3-Coder-Next-FP8", MODELS, true)).toBe(false);
  });

  it("never gates an always-on arm, cold or warm", () => {
    for (const warm of [true, false]) {
      expect(isScaleToZeroCold("claude-haiku-4-5-20251001", MODELS, warm)).toBe(false);
      expect(isScaleToZeroCold("gemini-2.5-flash", MODELS, warm)).toBe(false);
    }
  });

  it("with nothing configured, no model is ever gated — the no-op default", () => {
    const none: ReadonlySet<string> = new Set();
    expect(isScaleToZeroCold("Qwen/Qwen3-Coder-Next-FP8", none, false)).toBe(false);
  });

  it("REGRESSION: the same model id served by a self-hosted arm must not be gated", () => {
    // Qwen/Qwen3-Coder-Next-FP8 is also what a VLLM_ENDPOINTS instance serves.
    // The caller must pass an EMPTY set when no RunPod endpoint is configured,
    // or an always-on self-hosted arm serving that id would be gated on the
    // readiness of an endpoint that does not exist. Caught pre-merge.
    const noRunpodConfigured: ReadonlySet<string> = new Set();
    expect(isScaleToZeroCold("Qwen/Qwen3-Coder-Next-FP8", noRunpodConfigured, false)).toBe(false);
    expect(isScaleToZeroCold("Qwen/Qwen3-Coder-Next-FP8", noRunpodConfigured, true)).toBe(false);
  });
});
