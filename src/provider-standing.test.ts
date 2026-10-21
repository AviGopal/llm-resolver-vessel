// A PAID PROVIDER OUT OF CREDIT, OR THE FREE DAILY CAP HIT, MUST BE A SHAPED SIGNAL (check-first).
//
// THE DEFECT. index.ts cools a failing MODEL and walks the fallback chain, so a provider answering
// 402 on every model is absorbed silently: every completion is served by some fallback, and the
// provider's llmQuotaState entry stays { present: true, cooldown_until_ms: null } (the provider
// cooldown is set only on the anthropic/openai primary path). The same holds for OpenRouter's
// account-wide `:free` daily cap: each capped slug 429s, cools for one window, comes back, 429s.
// Nothing shaped says either condition, so selection cannot read it.
//
// THE FIX (pinned). src/provider-standing.ts keeps one standing record per provider, fed from the
// two provider-call wrappers in index.ts, and llmQuotaStateHandler spreads it into each existing
// provider entry. Additive: `present` and `cooldown_until_ms` are unchanged.
//
// FALSIFIER. With the paid provider answering 402 on every model for longer than one cooldown
// cycle, its llmQuotaState entry reads credit "exhausted", names the fallback in use, and lists its
// cooling models, while another provider serves. CONTROLS: every provider serving reads ok; a single
// 429 that clears within the cooldown leaves no record; a free-cap 429 cleared by a `:free` success
// reads ok again.
//
// The served body carries status enums, timestamps, counts and model ids ONLY: the keys are an
// allowlist and no value may look like key material, whatever the provider's error text said.
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";

type Entry = Record<string, unknown> & {
  credit: string; since: string | null; credit_402s: number; last_402_at: string | null;
  free_cap: string; free_cap_since: string | null; fallback_model: string | null; cooling_models: string[];
};
type Tracker = {
  registerModels(provider: string, models: string[]): void;
  providerOf(model: string): string | null;
  recordFailure(provider: string, model: string, error: unknown, nowMs: number): string | null;
  recordKind(provider: string, kind: "credit" | "free_cap", nowMs: number): void;
  recordServed(provider: string, model: string): void;
  recordFallback(fromModel: string, servedModel: string): void;
  entry(provider: string, cooldownMs: number, coolingModels: string[]): Entry;
};
type Mod = { ProviderStandingTracker: new () => Tracker; STANDING_FIELDS: readonly string[]; classifyStandingError: (m: string, e: unknown) => string | null };

async function seam(): Promise<Mod> {
  // A non-literal specifier: the module does not exist at base, and a literal would fail `tsc` there.
  const specifier = ["./provider", "standing"].join("-");
  const mod = await import(specifier).catch(() => null);
  expect(mod, "src/provider-standing.ts must exist (the provider-standing seam)").not.toBeNull();
  return mod as Mod;
}

const COOLDOWN = 10 * 60 * 1000; // index.ts EXHAUSTION_COOLDOWN_MS default
const T0 = Date.parse("2026-10-21T06:00:00.000Z");
const PAID_A = "google/gemini-2.5-flash";
const PAID_B = "openai/gpt-4o-mini";
const FREE_A = "nvidia/nemotron-3-nano-30b-a3b:free";
const GROQ = "llama-3.3-70b-versatile";
// Error texts in the shapes the providers send them.
const E402 = "402 Insufficient credits. Add more using https://billing.example.test/credits";
const E_FREE_CAP = "429 Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day";
const E_RATE = "429 Too Many Requests: rate limit exceeded, retry in 2s";

async function fresh(): Promise<Tracker> {
  const { ProviderStandingTracker } = await seam();
  const t = new ProviderStandingTracker();
  t.registerModels("openrouter", [PAID_A, PAID_B, FREE_A]);
  t.registerModels("groq", [GROQ]);
  return t;
}

describe("MUST-FAIL: a paid provider 402ing on every model past one cooldown cycle reads credit exhausted", () => {
  test("the falsifier: exhausted, since the first 402, naming the fallback, while groq serves", async () => {
    const t = await fresh();
    // Cycle 1: both paid slugs 402, groq rescues.
    t.recordFailure("openrouter", PAID_A, E402, T0);
    t.recordFailure("openrouter", PAID_B, E402, T0 + 1_000);
    t.recordServed("groq", GROQ);
    t.recordFallback(PAID_A, GROQ);
    // Cycle 2: the cooled slug came back after the cooldown and 402'd again.
    t.recordFailure("openrouter", PAID_A, E402, T0 + COOLDOWN + 5_000);
    t.recordServed("groq", GROQ);
    t.recordFallback(PAID_A, GROQ);

    const or = t.entry("openrouter", COOLDOWN, [PAID_A, PAID_B]);
    expect(or.credit).toBe("exhausted");
    expect(or.since).toBe(new Date(T0).toISOString());
    expect(or.credit_402s).toBe(3);
    expect(or.last_402_at).toBe(new Date(T0 + COOLDOWN + 5_000).toISOString());
    expect(or.fallback_model).toBe(GROQ);
    expect(or.cooling_models).toEqual([PAID_A, PAID_B]);
    expect(or.free_cap).toBe("ok");

    const groq = t.entry("groq", COOLDOWN, []);
    expect(groq.credit).toBe("ok");
    expect(groq.fallback_model).toBeNull();
  });

  test("it is ONE record per provider: more 402s update it, they do not multiply it", async () => {
    const t = await fresh();
    t.recordFailure("openrouter", PAID_A, E402, T0);
    t.recordFailure("openrouter", PAID_A, E402, T0 + COOLDOWN + 1);
    const first = t.entry("openrouter", COOLDOWN, []);
    for (let i = 2; i < 12; i++) t.recordFailure("openrouter", PAID_A, E402, T0 + i * COOLDOWN);
    const later = t.entry("openrouter", COOLDOWN, []);
    expect(later.credit).toBe("exhausted");
    expect(later.since).toBe(first.since);
    expect(later.credit_402s).toBe(12);
  });

  test("the free daily cap reads capped on the provider, and leaves credit alone", async () => {
    const t = await fresh();
    t.recordFailure("openrouter", FREE_A, E_FREE_CAP, T0);
    const e = t.entry("openrouter", COOLDOWN, [FREE_A]);
    expect(e.free_cap).toBe("capped");
    expect(e.free_cap_since).toBe(new Date(T0).toISOString());
    expect(e.credit).toBe("ok");
    expect(e.credit_402s).toBe(0);
  });

  test("recovery: a paid success clears credit, a :free success clears the free cap", async () => {
    const t = await fresh();
    t.recordFailure("openrouter", PAID_A, E402, T0);
    t.recordFailure("openrouter", PAID_A, E402, T0 + COOLDOWN + 1);
    t.recordFailure("openrouter", FREE_A, E_FREE_CAP, T0 + COOLDOWN + 2);
    t.recordFallback(PAID_A, GROQ);
    t.recordServed("openrouter", FREE_A);
    let e = t.entry("openrouter", COOLDOWN, []);
    expect(e.free_cap).toBe("ok");
    expect(e.credit).toBe("exhausted"); // a free slug answering says nothing about paid credit
    expect(e.fallback_model).toBe(GROQ);
    t.recordServed("openrouter", PAID_B);
    e = t.entry("openrouter", COOLDOWN, []);
    expect(e.credit).toBe("ok");
    expect(e.credit_402s).toBe(0);
    expect(e.since).toBeNull();
    expect(e.fallback_model).toBeNull();
  });

  test("the anthropic credit-dead answer (a 400, not a 402) is classified credit", async () => {
    const { classifyStandingError } = await seam();
    expect(classifyStandingError("claude-sonnet-5",
      '400 {"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}',
    )).toBe("credit");
  });
});

describe("CONTROLS: no record where there is no standing condition", () => {
  const untouched = { credit: "ok", since: null, credit_402s: 0, last_402_at: null, free_cap: "ok", free_cap_since: null, fallback_model: null, cooling_models: [] };

  test("every provider serving: nothing exhausted, nothing capped", async () => {
    const t = await fresh();
    t.recordServed("openrouter", PAID_A);
    t.recordServed("openrouter", FREE_A);
    t.recordServed("groq", GROQ);
    expect(t.entry("openrouter", COOLDOWN, [])).toEqual(untouched);
    expect(t.entry("groq", COOLDOWN, [])).toEqual(untouched);
  });

  test("a single 429 that clears within the cooldown writes no exhaustion record", async () => {
    const t = await fresh();
    expect(t.recordFailure("openrouter", PAID_A, E_RATE, T0)).toBeNull();
    t.recordFallback(PAID_A, GROQ);
    t.recordServed("openrouter", PAID_A);
    expect(t.entry("openrouter", COOLDOWN, [])).toEqual(untouched);
  });

  test("a single 402 cleared by a paid success within the cooldown is a blip, not exhaustion", async () => {
    const t = await fresh();
    t.recordFailure("openrouter", PAID_A, E402, T0);
    expect(t.entry("openrouter", COOLDOWN, []).credit).toBe("ok"); // one cycle is not persistent
    t.recordServed("openrouter", PAID_B);
    expect(t.entry("openrouter", COOLDOWN, [])).toEqual(untouched);
  });

  test("402s inside ONE cooldown cycle are not yet persistent", async () => {
    const t = await fresh();
    t.recordFailure("openrouter", PAID_A, E402, T0);
    t.recordFailure("openrouter", PAID_B, E402, T0 + COOLDOWN - 1);
    const e = t.entry("openrouter", COOLDOWN, []);
    expect(e.credit).toBe("ok");
    expect(e.credit_402s).toBe(2);
  });

  test("a free-cap 429 cleared by a :free success within the cooldown reads ok", async () => {
    const t = await fresh();
    t.recordFailure("openrouter", FREE_A, E_FREE_CAP, T0);
    t.recordServed("openrouter", FREE_A);
    expect(t.entry("openrouter", COOLDOWN, [])).toEqual(untouched);
  });

  test("token counts and model failures are not standing errors", async () => {
    const { classifyStandingError } = await seam();
    expect(classifyStandingError(PAID_A, "used 140250 input tokens")).toBeNull();
    expect(classifyStandingError(PAID_A, "completion was empty")).toBeNull();
    expect(classifyStandingError(PAID_A, E_RATE)).toBeNull();
    expect(classifyStandingError(PAID_A, undefined)).toBeNull();
  });
});

describe("MUST-FAIL: the served standing is an allowlist with no key material", () => {
  // Fake key-shaped strings, as a provider error might echo them back.
  const FAKE_KEY = "sk-or-v1-" + "0123456789abcdef".repeat(4);
  const LEAKY = `402 Insufficient credits for key ${FAKE_KEY} (Authorization: Bearer ${FAKE_KEY}) at https://router.example.test/v1`;
  const KEY_LIKE = [/sk-[a-z0-9-]{8,}/i, /bearer/i, /api[_-]?key/i, /authorization/i, /https?:\/\//i, /[A-Za-z0-9_\-]{40,}/];
  const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;

  test("keys are exactly STANDING_FIELDS, and every value is an enum, a timestamp, a count or a model id", async () => {
    const { STANDING_FIELDS } = await seam();
    const t = await fresh();
    t.recordFailure("openrouter", PAID_A, LEAKY, T0);
    t.recordFailure("openrouter", PAID_A, LEAKY, T0 + COOLDOWN + 1);
    t.recordFailure("openrouter", FREE_A, `${E_FREE_CAP} ${FAKE_KEY}`, T0 + 2);
    t.recordFallback(PAID_A, GROQ);
    const e = t.entry("openrouter", COOLDOWN, [PAID_A]);
    expect(Object.keys(e).sort()).toEqual([...STANDING_FIELDS].sort());
    expect(["ok", "exhausted"]).toContain(e.credit);
    expect(["ok", "capped"]).toContain(e.free_cap);
    for (const ts of [e.since, e.last_402_at, e.free_cap_since]) if (ts !== null) expect(ts).toMatch(ISO);
    expect(Number.isInteger(e.credit_402s)).toBe(true);
    const models = new Set([PAID_A, PAID_B, FREE_A, GROQ]);
    if (e.fallback_model !== null) expect(models.has(e.fallback_model)).toBe(true);
    for (const m of e.cooling_models) expect(models.has(m)).toBe(true);
    const body = JSON.stringify(e);
    expect(body).not.toContain(FAKE_KEY);
    for (const re of KEY_LIKE) expect(body).not.toMatch(re);
  });
});

// ── STRUCTURAL PINS READ THE SYNTAX TREE ────────────────────────────────────────────────────────
// index.ts boots the daemon at import, so its wiring is pinned on AST calls, not executed.
const INDEX_PATH = join(import.meta.dir, "index.ts");
let _sf: ts.SourceFile | null = null;
const indexSource = (): ts.SourceFile =>
  (_sf ??= ts.createSourceFile(INDEX_PATH, readFileSync(INDEX_PATH, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TS));
function walk(node: ts.Node, visit: (n: ts.Node) => void): void { visit(node); node.forEachChild((c) => walk(c, visit)); }
function fnNamed(name: string): ts.Node {
  let found: ts.Node | undefined;
  walk(indexSource(), (n) => {
    if (found) return;
    if (ts.isFunctionDeclaration(n) && n.name?.text === name) found = n;
    else if (ts.isVariableDeclaration(n) && ts.isIdentifier(n.name) && n.name.text === name && n.initializer
      && (ts.isArrowFunction(n.initializer) || ts.isFunctionExpression(n.initializer))) found = n.initializer;
  });
  expect(found !== undefined, `index.ts must define '${name}'`).toBe(true);
  return found!;
}
const calleeName = (c: ts.CallExpression): string => {
  const e = c.expression;
  return ts.isIdentifier(e) ? e.text : ts.isPropertyAccessExpression(e) ? e.name.text : "";
};
function callsTo(fn: ts.Node, callee: string): ts.CallExpression[] {
  const out: ts.CallExpression[] = [];
  walk(fn, (n) => { if (ts.isCallExpression(n) && calleeName(n) === callee) out.push(n); });
  return out;
}
const identifiersIn = (n: ts.Node): Set<string> => { const s = new Set<string>(); walk(n, (m) => { if (ts.isIdentifier(m)) s.add(m.text); }); return s; };

describe("MUST-FAIL: index.ts serves and feeds the standing", () => {
  test("llmQuotaStateHandler spreads the standing entry into each provider, keeping present and cooldown_until_ms", () => {
    const h = fnNamed("llmQuotaStateHandler");
    expect(callsTo(h, "entry").length).toBeGreaterThanOrEqual(2); // wire providers + anthropic
    const ids = identifiersIn(h);
    expect(ids.has("present")).toBe(true);
    expect(ids.has("cooldown_until_ms")).toBe(true);
    expect(ids.has("EXHAUSTION_COOLDOWN_MS")).toBe(true); // persistence is judged against the real cycle
  });

  test("both provider-call wrappers record served and failed calls", () => {
    for (const name of ["resolveWithOpenAI", "resolveWithAnthropic"]) {
      const f = fnNamed(name);
      expect(callsTo(f, "recordServed").length, `${name} records served`).toBeGreaterThan(0);
      expect(callsTo(f, "recordFailure").length, `${name} records failures`).toBeGreaterThan(0);
    }
  });

  test("the fallback walk names the model that served in place of the failed one", () => {
    expect(callsTo(fnNamed("walkFallbackModels"), "recordFallback").length).toBeGreaterThan(0);
  });

  test("the wire client loop registers each provider's models", () => {
    expect(callsTo(indexSource(), "registerModels").length).toBeGreaterThan(0);
  });
});
