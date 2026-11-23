// A lane refused at the provider level (billing, an unreachable host, a refused credential) is held until a probe to
// it is served; a rate-limited lane returns by its timer. Run against the shipped declarations of index.ts with a fake
// clock and fake timers: syncCompletionAdvertisement, the marks, willingness and the probe call each other for real.
import { describe, expect, test } from "bun:test";
import { loadIndexDeclarations } from "./index-fn-harness";
import * as ad from "./advertisement-resume";
import { markedUntil } from "./model-cooldown";
import { escalatedCooldownMs, isPermanentModelError, isUnauthenticatedProviderError, isUnreachableProviderError } from "./provider-errors.js";

const PAID = "vendor/paid-model";
const FREE = "vendor/free-model:free";
const CLAUDE = "claude-test";
const MIN = 60_000;
const BILLING = "402 Insufficient credits";
const RATE = "429 Too Many Requests: rate limit";
const UNREACHABLE = "Connection error: fetch failed";
const refused = (error: string) => ({ resolved: false, shape: "llmCompletion", error, provider_reason: { message: error } });
const served = { resolved: true, shape: "llmCompletion", text: "OK" };

const NAMES = [
  "exhaustedUntil", "heldUntilServed", "holdsUntilServed", "providerKeyOf", "inCooldown", "modelExhaustedUntil", "inModelCooldown",
  "modelPermanentStreak", "isHardExhaustedError", "isRateLimitError", "UNAUTHENTICATED_COOLDOWN_MS", "cooldownMsFor", "modelCooldownMsFor",
  "isModelWilling", "hasCompletionQuota", "completionAdvertised", "syncTurn", "syncCompletionAdvertisement", "scheduleAdvertisementResume",
  "markExhausted", "clearExhausted", "markModelExhausted", "PROBE_BODY", "probeTimers", "scheduleProbe", "probeHeldLane",
];
const NEW = ["heldUntilServed", "holdsUntilServed", "PROBE_BODY", "probeTimers", "scheduleProbe", "probeHeldLane"];

/** The shipped functions over two wire models (and the anthropic lane, when `claude`), a fake clock and fake timers. */
function world(o: { claude?: boolean; cold?: boolean } = {}) {
  let clock = 1_000_000;
  const timers: Array<{ at: number; fn: () => void; live: boolean }> = [];
  const shapes: string[][] = [];
  const probes: Array<Record<string, unknown>> = [];
  const answers = new Map<string, Record<string, unknown>>();
  const client = { baseURL: "https://wire.example/v1" };
  const fns = loadIndexDeclarations(NAMES, {
    Date: { now: () => clock },
    setTimeout: (fn: () => void, ms: number) => { const t = { at: clock + ms, fn, live: true }; timers.push(t); return t; },
    clearTimeout: (t: { live: boolean }) => { t.live = false; },
    console: { warn() {}, log() {}, error() {} },
    EXHAUSTION_COOLDOWN_MS: 10 * MIN, RATE_LIMIT_COOLDOWN_MS: 45_000, UNREACHABLE_COOLDOWN_MS: 30_000,
    PERMANENT_COOLDOWN_DOUBLING: 2, PERMANENT_COOLDOWN_CAP_MS: 24 * 60 * MIN,
    ...ad, markedUntil, escalatedCooldownMs, isPermanentModelError, isUnauthenticatedProviderError, isUnreachableProviderError,
    LLM_PINNED_PROVIDER: null, OPENAI_WIRE_PROVIDERS: [],
    modelClientMap: new Map([[PAID, client], [FREE, client]]),
    anthropic: o.claude ? {} : null, openaiClient: null, openrouterClient: null, OPENROUTER_MODEL_SET: new Set(),
    isDenied: () => false, isRunpodCold: () => o.cold === true,
    loadPolicy: async () => ({ arms: [PAID, FREE, ...(o.claude ? [CLAUDE] : [])].map((model) => ({ model })) }),
    daemon: { setShapes: async (s: string[]) => { shapes.push([...s]); } },
    reasonText: (r: unknown) => JSON.stringify(r ?? null),
    resolveWithOpenAI: async (body: Record<string, unknown>) => { probes.push(body); return answers.get(String(body.model)) ?? refused(BILLING); },
    resolveWithAnthropic: async (body: Record<string, unknown>) => {
      probes.push({ ...body, lane: "anthropic" });
      const a = answers.get("anthropic") as unknown;
      return typeof a === "function" ? await (a as () => Promise<Record<string, unknown>>)() : (a as Record<string, unknown> | undefined) ?? refused(BILLING);
    },
  }, NEW);
  const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
  /** Move the clock to `ms` past the start, firing each timer due on the way, in order. */
  const start = clock;
  const at = async (ms: number) => {
    for (;;) {
      const due = timers.filter((t) => t.live && t.at <= start + ms).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      due.live = false;
      clock = Math.max(clock, due.at);
      due.fn();
      await flush();
    }
    clock = start + ms;
    await flush();
  };
  // A lane's class, as the call sites mark it (a revision without the class marks every lane for its window only).
  const holds = (e: string): boolean => (typeof fns.holdsUntilServed === "function" ? fns.holdsUntilServed(e) : false);
  const refuseModel = async (m: string, e: string) => { fns.markModelExhausted(m, e, fns.modelCooldownMsFor(m, e), holds(e)); await flush(); };
  const advertised = () => shapes.at(-1)?.includes("llm_completion") === true;
  return { fns, at, shapes, probes, answers, refuseModel, advertised, holds, flush };
}

describe("a lane refused at the provider level is held until a probe to it is served", () => {
  test("[MUST-FAIL] a 402 on every lane: past every window llm_completion stays dropped, and nothing is re-sent but on a mark", async () => {
    const w = world();
    await w.fns.syncCompletionAdvertisement();
    expect(w.advertised()).toBe(true);
    await w.refuseModel(PAID, BILLING);
    await w.refuseModel(FREE, BILLING);
    expect(w.advertised()).toBe(false);
    const dropped = w.shapes.length;
    await w.at(10 * MIN - 1);
    // nothing re-sent while the lanes are held: each re-send re-registers with discovery
    expect(w.shapes.length).toBe(dropped);
    await w.at(35 * MIN);
    expect(w.shapes.slice(dropped - 1).some((s) => s.includes("llm_completion"))).toBe(false);
    // the probes at 10, 20 and 30 minutes were refused again: one re-send for each lane's re-mark, none else
    expect(w.shapes.length).toBe(dropped + 3 * 2);
    expect([w.fns.isModelWilling(PAID), w.fns.isModelWilling(FREE)]).toEqual([false, false]);
  });

  test("[MUST-FAIL] then a probe the lane serves brings it back: advertised again, that lane willing, the other still held", async () => {
    const w = world();
    await w.fns.syncCompletionAdvertisement();
    await w.refuseModel(PAID, BILLING);
    await w.refuseModel(FREE, BILLING);
    w.answers.set(FREE, served);
    await w.at(11 * MIN);
    expect(w.advertised()).toBe(true);
    expect([w.fns.isModelWilling(PAID), w.fns.isModelWilling(FREE)]).toEqual([false, true]);
    expect(w.probes.map((p) => [p.model, p.max_tokens])).toEqual([[PAID, 16], [FREE, 16]]);
  });

  test("[MUST-FAIL] an unreachable lane is held too, past its short window", async () => {
    const w = world();
    await w.fns.syncCompletionAdvertisement();
    await w.refuseModel(PAID, UNREACHABLE);
    await w.refuseModel(FREE, UNREACHABLE);
    await w.at(5 * MIN);
    expect(w.advertised()).toBe(false);
    expect(w.probes.length).toBeGreaterThan(0);
  });

  test("[MUST-FAIL] a probe refused by a rate limit returns the lane by its timer, not at once", async () => {
    const w = world();
    await w.fns.syncCompletionAdvertisement();
    await w.refuseModel(PAID, BILLING);
    await w.refuseModel(FREE, BILLING);
    w.answers.set(PAID, refused(RATE));
    w.answers.set(FREE, refused(RATE));
    await w.at(10 * MIN + 30_000);
    expect(w.advertised()).toBe(false);
    await w.at(10 * MIN + 50_000);
    expect(w.advertised()).toBe(true);
  });

  test("[MUST-FAIL] the anthropic lane held for credit: a probe answered only by a fallback leaves it held", async () => {
    const w = world({ claude: true });
    await w.fns.syncCompletionAdvertisement();
    w.fns.markExhausted("anthropic", "credit balance too low", 30 * MIN, true);
    await w.flush();
    w.answers.set("anthropic", { ...served, fallback_from: "anthropic-credit", model: FREE });
    await w.at(45 * MIN);
    expect(w.probes.some((p) => p.lane === "anthropic")).toBe(true);
    expect([w.fns.inCooldown("anthropic"), w.fns.isModelWilling(CLAUDE)]).toEqual([true, false]);
    w.answers.set("anthropic", served);
    await w.at(65 * MIN);
    expect([w.fns.inCooldown("anthropic"), w.fns.isModelWilling(CLAUDE)]).toEqual([false, true]);
  });

  test("[MUST-FAIL] the anthropic lane held: it stays cooling while its probe is out, and nothing is re-sent but on a mark", async () => {
    const w = world({ claude: true });
    await w.fns.syncCompletionAdvertisement();
    w.fns.markExhausted("anthropic", "credit balance too low", 30 * MIN, true);
    await w.flush();
    let answer: (r: Record<string, unknown>) => void = () => {};
    w.answers.set("anthropic", (() => new Promise((res) => { answer = res; })) as never);
    await w.at(31 * MIN);
    expect(w.probes.filter((p) => p.lane === "anthropic").length).toBe(1);
    expect([w.fns.inCooldown("anthropic"), w.fns.isModelWilling(CLAUDE)]).toEqual([true, false]);
    answer(refused(BILLING));
    await w.at(39 * MIN);
    // the start, the mark, the probe's re-mark: no timer re-sync in between
    expect(w.shapes.length).toBe(3);
    expect(w.fns.inCooldown("anthropic")).toBe(true);
  });

  test("[GUARD] a 429 on every lane re-advertises at its expiry", async () => {
    const w = world();
    await w.fns.syncCompletionAdvertisement();
    await w.refuseModel(PAID, RATE);
    await w.refuseModel(FREE, RATE);
    expect(w.advertised()).toBe(false);
    await w.at(50_000);
    expect(w.advertised()).toBe(true);
    expect(w.probes).toEqual([]);
  });

  test("[GUARD] mixed lanes, a 402 and a 429: advertised at the 429's expiry, the 402 lane not drawn", async () => {
    const w = world();
    await w.fns.syncCompletionAdvertisement();
    await w.refuseModel(PAID, BILLING);
    await w.refuseModel(FREE, RATE);
    await w.at(50_000);
    expect(w.advertised()).toBe(true);
    expect([w.fns.isModelWilling(PAID), w.fns.isModelWilling(FREE)]).toEqual([false, true]);
  });

  test("[GUARD] no arm drawable for a reason no cooldown ends: advertised on the wider check, as before", async () => {
    const w = world({ cold: true });
    await w.fns.syncCompletionAdvertisement();
    expect(w.advertised()).toBe(true);
  });

  test("[GUARD] a call the lane serves on its own path (clearExhausted) releases it too", async () => {
    const w = world({ claude: true });
    await w.fns.syncCompletionAdvertisement();
    w.fns.markExhausted("anthropic", "credit balance too low", 30 * MIN, w.holds("credit balance too low"));
    await w.flush();
    w.fns.clearExhausted("anthropic");
    expect([w.fns.inCooldown("anthropic"), w.fns.isModelWilling(CLAUDE)]).toEqual([false, true]);
  });
});

describe("every mark for a provider's refusal passes the refusal's class", () => {
  test("[MUST-FAIL] each markExhausted / markModelExhausted call on a provider_reason says whether it holds until served", async () => {
    const src = await Bun.file(new URL("./index.ts", import.meta.url).pathname).text();
    const sites = src.split("\n").filter((l) => /\bmark(Model)?Exhausted\(/.test(l) && l.includes("provider_reason"));
    expect(sites.length).toBe(5);
    expect(sites.filter((l) => !/holdsUntilServed\((fb|r|result|retry)\.error\)\)/.test(l))).toEqual([]);
  });
});
