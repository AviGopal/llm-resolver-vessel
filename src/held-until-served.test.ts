// A lane refused at the provider level (billing, an unreachable host, a refused credential) is held until a probe to
// it is served; a rate-limited lane returns by its timer. Run against the shipped declarations of index.ts with a fake
// clock and fake timers: syncCompletionAdvertisement, the marks, willingness and the probe call each other for real.
import { describe, expect, test } from "bun:test";
import { loadIndexDeclarations } from "./index-fn-harness";
import * as ad from "./advertisement-resume";
import { markedUntil } from "./model-cooldown";
import { escalatedCooldownMs, isFailoverError, isPermanentModelError, isUnauthenticatedProviderError, isUnreachableProviderError } from "./provider-errors.js";
import { emptyArmsOf, isEmptyCompletion, withEmptyArms } from "./empty-completion";
import { coolingRefusal } from "./model-cooldown";

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
  "anthropicCreditFallback", "walkFallbackModels", "llmCompletionHandler",
];
const NEW = ["heldUntilServed", "holdsUntilServed", "PROBE_BODY", "probeTimers", "scheduleProbe", "probeHeldLane"];

/** The shipped functions over two wire models (and the anthropic lane, when `claude`), a fake clock and fake timers. */
function world(o: { claude?: boolean; cold?: boolean; wire?: boolean; openai?: boolean; openrouter?: boolean } = {}) {
  let clock = 1_000_000;
  const timers: Array<{ at: number; fn: () => void; live: boolean }> = [];
  const shapes: string[][] = [];
  const probes: Array<Record<string, unknown>> = [];
  const answers = new Map<string, Record<string, unknown>>();
  const client = { baseURL: "https://wire.example/v1" };
  const openai = { baseURL: "https://openai.example/v1" };
  const fns = loadIndexDeclarations(NAMES, {
    Date: { now: () => clock },
    setTimeout: (fn: () => void, ms: number) => { const t = { at: clock + ms, fn, live: true }; timers.push(t); return t; },
    clearTimeout: (t: { live: boolean }) => { t.live = false; },
    console: { warn() {}, log() {}, error() {} },
    EXHAUSTION_COOLDOWN_MS: 10 * MIN, RATE_LIMIT_COOLDOWN_MS: 45_000, UNREACHABLE_COOLDOWN_MS: 30_000,
    PERMANENT_COOLDOWN_DOUBLING: 2, PERMANENT_COOLDOWN_CAP_MS: 24 * 60 * MIN,
    ...ad, markedUntil, escalatedCooldownMs, isPermanentModelError, isUnauthenticatedProviderError, isUnreachableProviderError,
    LLM_PINNED_PROVIDER: null, OPENAI_WIRE_PROVIDERS: [],
    modelClientMap: new Map(o.wire === false ? [] : [[PAID, client], [FREE, client]]),
    anthropic: o.claude ? {} : null, openaiClient: o.openai ? openai : null, openrouterClient: o.openrouter ? client : null, OPENROUTER_MODEL_SET: new Set(),
    isDenied: () => false, isRunpodCold: () => o.cold === true,
    loadPolicy: async () => ({ arms: [...(o.wire === false ? [] : [PAID, FREE]), ...(o.claude ? [CLAUDE] : [])].map((model) => ({ model })) }),
    isFailoverError, isEmptyCompletion, emptyArmsOf, withEmptyArms, coolingRefusal, DEFAULT_MODEL: CLAUDE,
    readGateParams: async () => ({}), readFallbackOrder: async () => [], fallbackWalkOrder: (keys: string[]) => keys,
    providerStanding: { admit: () => true, recordFallback() {} },
    selectArm: async (_t: string, models: string[]) => (models.length > 0 ? { model: models[0] } : null),
    reasonOf: (e: unknown) => ({ message: String(e) }),
    pickProvider: (m: string) => (m.startsWith("claude") ? (o.claude ? "anthropic" : null) : o.openai ? "openai" : null),
    daemon: { setShapes: async (s: string[]) => { shapes.push([...s]); } },
    reasonText: (r: unknown) => JSON.stringify(r ?? null),
    resolveWithOpenAI: async (body: Record<string, unknown>, c: unknown = openai) => {
      probes.push({ ...body, client: c === openai ? "openai" : "wire" });
      return answers.get(String(body.model ?? "openai")) ?? refused(BILLING);
    },
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

  test("[GUARD] a call the lane serves on its own path (clearExhausted) releases it too; its pending probe then calls nothing", async () => {
    const w = world({ claude: true });
    await w.fns.syncCompletionAdvertisement();
    w.fns.markExhausted("anthropic", "credit balance too low", 30 * MIN, w.holds("credit balance too low"));
    await w.flush();
    w.fns.clearExhausted("anthropic");
    expect([w.fns.inCooldown("anthropic"), w.fns.isModelWilling(CLAUDE)]).toEqual([false, true]);
    await w.at(35 * MIN);
    expect(w.probes.filter((p) => p.lane === "anthropic")).toEqual([]);
    expect([w.fns.inCooldown("anthropic"), w.fns.isModelWilling(CLAUDE)]).toEqual([false, true]);
  });

  test("[MUST-FAIL] a held lane whose window a later mark lengthened is probed once, at the window's end, and is never stranded", async () => {
    const w = world({ claude: true });
    await w.fns.syncCompletionAdvertisement();
    w.answers.set("anthropic", refused(UNREACHABLE));
    w.fns.markExhausted("anthropic", UNREACHABLE, 30_000, true);
    await w.at(10_000);
    // a last-resort retry refused for billing lengthens the window while the first probe is pending
    w.fns.markExhausted("anthropic", BILLING, 10 * MIN, true);
    await w.at(10 * MIN);
    expect(w.probes.filter((p) => p.lane === "anthropic").length).toBe(0);
    // the probe's answer is still out when every timer due at the window's end has fired: one probe, not two
    let answer: (r: Record<string, unknown>) => void = () => {};
    w.answers.set("anthropic", (() => new Promise((res) => { answer = res; })) as never);
    await w.at(10 * MIN + 10_001);
    expect(w.probes.filter((p) => p.lane === "anthropic").length).toBe(1);
    // refused unreachable again: held for that window, its probe pending; then served, released
    answer(refused(UNREACHABLE));
    await w.flush();
    w.answers.set("anthropic", served);
    await w.at(12 * MIN);
    expect(w.probes.filter((p) => p.lane === "anthropic").length).toBe(2);
    expect([w.fns.inCooldown("anthropic"), w.fns.isModelWilling(CLAUDE)]).toEqual([false, true]);
  });

  test("[MUST-FAIL] a provider's probe due half a second before its lane's window ends calls nothing, and fires at the end", async () => {
    const w = world({ claude: true });
    await w.fns.syncCompletionAdvertisement();
    w.answers.set("anthropic", served);
    w.fns.markExhausted("anthropic", UNREACHABLE, 30_000, true);
    await w.at(500);
    // a rate limit marked half a second later moves the window's end 500 ms past the pending probe
    w.fns.markExhausted("anthropic", RATE, 30_000, false);
    await w.at(30_000);
    expect(w.probes.filter((p) => p.lane === "anthropic").length).toBe(0);
    expect(w.fns.inCooldown("anthropic")).toBe(true);
    await w.at(30_500);
    expect(w.probes.filter((p) => p.lane === "anthropic").length).toBe(1);
    expect(w.fns.inCooldown("anthropic")).toBe(false);
  });

  test("[MUST-FAIL] a held model's probe that comes due inside the model's own window calls nothing; the one probe is at its end", async () => {
    const w = world();
    await w.fns.syncCompletionAdvertisement();
    await w.refuseModel(PAID, BILLING);
    await w.at(5 * MIN);
    // the model's window, not the provider map, decides: a probe run while the model still cools is moved, not sent
    await w.fns.probeHeldLane(PAID);
    await w.flush();
    expect(w.probes.filter((p) => p.model === PAID).length).toBe(0);
    expect(w.fns.inModelCooldown(PAID)).toBe(true);
    w.answers.set(PAID, served);
    await w.at(10 * MIN + 1);
    expect(w.probes.filter((p) => p.model === PAID).length).toBe(1);
    expect(w.fns.inModelCooldown(PAID)).toBe(false);
  });

  test("[MUST-FAIL] a held mark that leaves a longer window in force still holds the lane and leaves its probe pending", async () => {
    const w = world({ claude: true });
    await w.fns.syncCompletionAdvertisement();
    w.fns.markExhausted("anthropic", RATE, 20 * MIN, false);
    w.fns.markExhausted("anthropic", BILLING, 10 * MIN, true);
    await w.at(25 * MIN);
    expect(w.fns.inCooldown("anthropic")).toBe(true);
    expect(w.probes.filter((p) => p.lane === "anthropic").length).toBe(1);
    w.answers.set("anthropic", served);
    await w.at(36 * MIN);
    expect(w.fns.inCooldown("anthropic")).toBe(false);
  });
});

describe("each call site that cools a lane for a provider's refusal holds it", () => {
  const ask = (w: ReturnType<typeof world>, model: string) => w.fns.llmCompletionHandler({ body: { type: "llm_completion", prompt: "p", model } });

  test("[MUST-FAIL] anthropic credit-dead with no willing fallback: llm_completion stays dark past 30 minutes; a served probe brings it back", async () => {
    const w = world({ claude: true, wire: false });
    await w.fns.syncCompletionAdvertisement();
    expect(w.advertised()).toBe(true);
    expect(await w.fns.anthropicCreditFallback(new Error("400 Your credit balance is too low to access the API"), { type: "llm_completion", prompt: "p" })).toBeNull();
    await w.at(45 * MIN);
    expect(w.advertised()).toBe(false);
    w.answers.set("anthropic", served);
    await w.at(51 * MIN);
    expect(w.advertised()).toBe(true);
  });

  test("[MUST-FAIL] the primary provider refused for billing is held (the primary path)", async () => {
    const w = world({ claude: true, wire: false });
    w.answers.set("anthropic", refused(BILLING));
    await ask(w, CLAUDE);
    await w.at(15 * MIN);
    expect(w.fns.inCooldown("anthropic")).toBe(true);
  });

  test("[MUST-FAIL] the primary refused for billing on its last-resort retry is held (the retry path)", async () => {
    const w = world({ claude: true, wire: false });
    w.fns.markExhausted("anthropic", "rate limited", 45_000, false);
    w.answers.set("anthropic", refused(BILLING));
    await ask(w, CLAUDE);
    expect(w.probes.filter((p) => p.lane === "anthropic").length).toBe(1);
    await w.at(15 * MIN);
    expect(w.fns.inCooldown("anthropic")).toBe(true);
  });

  test("[MUST-FAIL] the openai provider held: its probe goes to the openai client and, served, releases it", async () => {
    const w = world({ openai: true, wire: false });
    w.answers.set("gpt-test", refused(BILLING));
    w.answers.set("openai", served);
    await ask(w, "gpt-test");
    expect(w.fns.inCooldown("openai")).toBe(true);
    await w.at(11 * MIN);
    expect(w.probes.at(-1)).toMatchObject({ client: "openai", max_tokens: 16 });
    expect(w.probes.at(-1)?.model).toBeUndefined();
    expect(w.fns.inCooldown("openai")).toBe(false);
  });

  test("[MUST-FAIL] a wire model refused for billing is held (the wire path)", async () => {
    const w = world();
    w.answers.set(PAID, refused(BILLING));
    w.answers.set(FREE, served);
    await ask(w, PAID);
    await w.at(15 * MIN);
    expect(w.fns.inModelCooldown(PAID)).toBe(true);
  });

  test("[MUST-FAIL] a model refused for billing in the fallback walk is held (the walk)", async () => {
    const w = world();
    w.answers.set(PAID, refused(RATE));
    w.answers.set(FREE, refused(BILLING));
    await ask(w, PAID);
    await w.at(15 * MIN);
    expect([w.fns.inModelCooldown(PAID), w.fns.inModelCooldown(FREE)]).toEqual([false, true]);
  });

  test("[MUST-FAIL] a model routed through openrouter refused for billing is held (the openrouter path)", async () => {
    const w = world({ openrouter: true });
    const other = "other-vendor/model";
    w.answers.set(other, refused(BILLING));
    w.answers.set(PAID, served);
    await ask(w, other);
    await w.at(15 * MIN);
    expect(w.fns.inModelCooldown(other)).toBe(true);
  });

  test("[GUARD] each path refused by a rate limit returns by its timer", async () => {
    const w = world({ claude: true });
    w.answers.set("anthropic", refused(RATE));
    w.answers.set(PAID, refused(RATE));
    w.answers.set(FREE, refused(RATE));
    await ask(w, PAID);
    await ask(w, CLAUDE);
    await w.at(2 * MIN);
    expect([w.fns.inCooldown("anthropic"), w.fns.inModelCooldown(PAID), w.fns.inModelCooldown(FREE)]).toEqual([false, false, false]);
  });
});
