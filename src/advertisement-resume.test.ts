// A DE-ADVERTISED COMPLETION PLANE COMES BACK WHEN ITS LAST COOLDOWN ENDS (advertisement-resume.ts).
// FALSIFIER (research): two overlapping cooldowns, the earlier expiring first. The resume that fires at the first
// expiry finds the second still cooling; it must schedule another, and the plane is advertised once the later ends.
// CONTROL: with nothing cooling, nothing is scheduled. index.ts's wiring is pinned by source (the sync is not exported).
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { advertiseQuota, advertisementChange, completionQuota, makeLatest, makeResumeScheduler, nextResyncInMs } from "./advertisement-resume";

// A fake clock and timer queue, and a sync built like index.ts's: quota when no lane cools; with none, the next re-sync.
const world = () => {
  let t = 0;
  const timers: Array<{ at: number; fn: () => void; dead: boolean }> = [];
  const until = new Map<string, number>();
  const log: string[] = [];
  let advertised: boolean | null = null;
  const schedule = makeResumeScheduler({
    now: () => t,
    setTimer: (fn, ms) => { const h = { at: t + ms, fn, dead: false }; timers.push(h); return h; },
    clearTimer: (h) => { (h as { dead: boolean }).dead = true; },
    sync: () => sync(),
  });
  const sync = (): void => {
    const quota = [...until.values()].every((u) => u <= t);
    const next = nextResyncInMs(quota, t, until.values());
    if (next !== null) schedule(next);
    const line = advertisementChange(advertised, quota, next);
    if (line !== null) log.push(`${t}: ${line}`);
    advertised = quota;
  };
  const mark = (lane: string, ms: number): void => { until.set(lane, t + ms); sync(); schedule(ms); };
  // Bounded: a regression that re-arms a timer for a moment already past must fail here, not stall the suite (qa).
  const runUntil = (end: number): void => {
    for (let fired = 0; ; fired++) {
      if (fired > 1000) throw new Error("more than 1000 timers fired: a re-sync keeps re-arming for a moment already past");
      const due = timers.filter((h) => !h.dead && h.at <= end).sort((a, b) => a.at - b.at)[0];
      if (due === undefined) break;
      due.dead = true; t = due.at; due.fn();
    }
    t = end;
  };
  return { mark, runUntil, advertised: () => advertised, log, sync, at: (x: number) => { t = x; } };
};

describe("advertisement resume", () => {
  test("[MUST-FAIL] two overlapping cooldowns: re-advertised when the LATER one ends, not left dark (research)", () => {
    const w = world();
    w.sync(); // boot: advertised
    w.mark("model-a", 600_000); // t=0, cools until 600 s
    w.at(60_000);
    w.mark("model-b", 600_000); // t=60 s, cools until 660 s
    w.runUntil(630_000); // the first resume fired at ~601 s and found b still cooling
    expect(w.advertised()).toBe(false);
    w.runUntil(700_000);
    expect(w.advertised()).toBe(true);
    expect(w.log.at(-1)).toContain("advertising llm_completion");
  });

  test("the next re-sync is the soonest expiry still ahead; none when quota is back or nothing cools", () => {
    expect(nextResyncInMs(false, 100, [50, 300, 200])).toBe(100);
    expect(nextResyncInMs(true, 100, [300])).toBeNull();
    expect(nextResyncInMs(false, 100, [50])).toBeNull();
  });

  test("[MUST-FAIL] both directions are logged, once per change", () => {
    expect(advertisementChange(true, false, 5000)).toContain("de-advertising llm_completion until quota returns; re-checking in 5s");
    expect(advertisementChange(false, true, null)).toBe("quota returned — advertising llm_completion");
    // A start is not a resume: null → true says so (research).
    expect(advertisementChange(null, true, null)).toBe("advertising llm_completion at start");
    expect(advertisementChange(true, true, null)).toBeNull();
    expect(advertisementChange(false, false, 5000)).toBeNull();
  });

  test("[MUST-FAIL] index.ts reschedules from the sync itself when it finds no quota, and logs each change", () => {
    const src = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
    const body = src.slice(src.indexOf("async function syncCompletionAdvertisement()"), src.indexOf("async function llmQuotaStateHandler"));
    expect(body).toContain("nextResyncInMs(drawable, Date.now(), [...exhaustedUntil.values(), ...modelExhaustedUntil.values()])");
    expect(body).toContain("if (resyncIn !== null) scheduleAdvertisementResume(resyncIn);");
    expect(body).toContain("advertisementChange(completionAdvertised, quota, resyncIn)");
  });

  test("[MUST-FAIL] every drawable arm cooling means no quota, whatever other keyed models are idle (research)", () => {
    const cooling = new Set(["arm/a", "arm/b"]);
    const willing = (m: string) => !cooling.has(m);
    // "keyed/idle" is in the client map but no arm: nothing draws it, so it never cools and must not count
    expect(completionQuota(["arm/a", "arm/b"], willing)).toBe(false);
    expect(completionQuota(["arm/a", "arm/c"], willing)).toBe(true);
    expect(completionQuota([], willing)).toBe(false);
  });

  test("[MUST-FAIL] the sync reads quota from the policy's arms and the willingness predicate, not from the client map", () => {
    const src = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
    const body = src.slice(src.indexOf("async function syncCompletionAdvertisement()"), src.indexOf("async function llmQuotaStateHandler"));
    expect(body).toContain("completionQuota(p.arms.map((a) => a.model), isModelWilling)");
  });

  test("[MUST-FAIL] no drawable arm and no cooldown pending: still advertised on the wider check, never stranded (qa)", () => {
    expect(advertiseQuota(false, false, () => true)).toBe(true); // e.g. every arm cold, nothing to expire
    expect(advertiseQuota(false, true, () => true)).toBe(false); // a cooldown will end: de-advertise and re-check then
    expect(advertiseQuota(false, false, () => false)).toBe(false); // nothing keyed at all
    expect(advertiseQuota(true, true, () => false)).toBe(true);
  });

  test("[MUST-FAIL] the sync decides with advertiseQuota after the resync is known, and a warmth change re-syncs", () => {
    const src = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
    const body = src.slice(src.indexOf("async function syncCompletionAdvertisement()"), src.indexOf("async function llmQuotaStateHandler"));
    expect(body.indexOf("advertiseQuota(drawable, resyncIn !== null")).toBeGreaterThan(body.indexOf("nextResyncInMs(drawable,"));
    const warm = src.slice(src.indexOf("if (was !== runpodWarm) {"), src.indexOf("if (was !== runpodWarm) {") + 400);
    expect(warm).toContain("void syncCompletionAdvertisement();");
  });

  test("[MUST-FAIL] two syncs whose reads finish out of order: the newer state is the one that stays (deployment)", async () => {
    const turn = makeLatest();
    let advertised = null as string | null;
    // a sync built like index.ts's: ticket, await a read, stop if no longer latest, then set
    const sync = async (read: Promise<string>) => {
      const ticket = turn.begin();
      const state = await read;
      if (!turn.isLatest(ticket)) return;
      advertised = state;
    };
    let releaseOld!: (v: string) => void;
    const old = sync(new Promise<string>((r) => { releaseOld = r; })); // started first, its read finishes last
    const newer = sync(Promise.resolve("dark"));
    await newer;
    releaseOld("advertised");
    await old;
    expect(advertised).toBe("dark");
  });

  test("[MUST-FAIL] index.ts's sync takes a ticket before its read and stops after it when no longer latest", () => {
    const src = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
    const body = src.slice(src.indexOf("async function syncCompletionAdvertisement()"), src.indexOf("async function llmQuotaStateHandler"));
    const ticket = body.indexOf("const ticket = syncTurn.begin();");
    const read = body.indexOf("await loadPolicy()");
    const stop = body.indexOf("if (!syncTurn.isLatest(ticket)) return;");
    const set = body.indexOf("await daemon.setShapes(shapes)");
    expect(ticket).toBeGreaterThan(0);
    expect(read).toBeGreaterThan(ticket);
    expect(stop).toBeGreaterThan(read);
    expect(set).toBeGreaterThan(stop);
  });
});
