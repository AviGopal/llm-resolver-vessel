// A COOLED MODEL IS NOT CALLED, AND ITS COOLDOWN DOES NOT SLIDE (model-cooldown.ts).
// FALSIFIERS (qa, agf_kax2rdyrj5ui): a mark inside a running cooldown leaves the expiry unchanged; a cooled model's
// answer is unserved, said at once; both request branches (the model's own client, and the catch-all pass-through)
// return it, without a call, once no fallback serves. The branches are pinned by source (the handler is not exported).
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { coolingRefusal, markedUntil } from "./model-cooldown";

describe("model cooldown", () => {
  test("[MUST-FAIL] a re-mark while the model still cools leaves the expiry where it was", () => {
    expect(markedUntil(1_600, 1_000, 600)).toEqual({ until: 1_600, changed: false }); // 8 s later, the same outage
    expect(markedUntil(undefined, 1_000, 600)).toEqual({ until: 1_600, changed: true });
    expect(markedUntil(900, 1_000, 600)).toEqual({ until: 1_600, changed: true }); // an expired one is renewed
  });

  test("a cooled model's answer is unserved and says until when", () => {
    expect(coolingRefusal("vendor/m", Date.UTC(2026, 0, 1))).toMatchObject({
      resolved: false, unserved: "llm_completion", requested_model: "vendor/m", cooling_until: "2026-01-01T00:00:00.000Z",
    });
  });

  test("[MUST-FAIL] both request branches answer a cooled model without calling it, after the fallbacks", () => {
    const src = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
    const h = src.slice(src.indexOf("const llmCompletionHandler: ResolverHandler"), src.indexOf("const provider = pickProvider(model, body.provider);"));
    // the model's own client: the cooled branch returns before the call
    const wire = h.indexOf("if (wireClient) {");
    const wireCool = h.indexOf("return walked ?? coolingRefusal(model,", wire);
    const wireCall = h.indexOf("await resolveWithOpenAI(body, wireClient)");
    expect(wireCool).toBeGreaterThan(wire);
    expect(wireCall).toBeGreaterThan(wireCool);
    // the pass-through: checked too, before its call
    const pass = h.indexOf("if (openrouterClient && model.includes(\"/\")");
    const passCool = h.indexOf("return walked ?? coolingRefusal(model,", pass);
    const passCall = h.indexOf("await resolveWithOpenAI(body, openrouterClient)");
    expect(pass).toBeGreaterThan(0);
    expect(passCool).toBeGreaterThan(pass);
    expect(passCall).toBeGreaterThan(passCool);
  });

  test("[MUST-FAIL] the mark keeps a running cooldown and is silent when it does", () => {
    const src = readFileSync(join(import.meta.dir, "index.ts"), "utf8");
    const m = src.slice(src.indexOf("const markModelExhausted ="), src.indexOf("const markModelExhausted =") + 600);
    expect(m).toContain("markedUntil(modelExhaustedUntil.get(m), Date.now(), ms)");
    expect(m.indexOf("if (!changed) return;")).toBeLessThan(m.indexOf("console.warn"));
  });
});
