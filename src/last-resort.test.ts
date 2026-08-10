// Pins the last-resort guard.
//
// THE DEFECT: when policy selection returned no arm, the handler fell through to
// `body.model ?? DEFAULT_MODEL` and dialled the default without checking it was
// servable. Observed live: hundreds of
// `credit balance is too low ... cooling down 1800s` in a tight loop against a
// dead provider, while a warm arm sat unused.
//
// Imports the REAL module — a re-implemented copy would pass while the shipped
// guard rots.
import { describe, expect, test } from "bun:test";
import { decideLastResort } from "./last-resort";

const never = () => false;
const always = () => true;

describe("decideLastResort", () => {
  test("refuses rather than dialling a dry default (the shipped defect)", () => {
    const d = decideLastResort({
      defaultModel: "claude-sonnet-5",
      armsChecked: 0,
      isWilling: never,
    });
    expect("refuse" in d).toBe(true);
    if ("refuse" in d) {
      expect(d.refuse).toContain("claude-sonnet-5");
      expect(d.refuse).toContain("0 policy arm(s) checked");
    }
  });

  test("dials the default when it IS servable — this must not become an outage", () => {
    // The over-correction guard. Refusing a willing default would take the plane
    // down harder than the bug ever did.
    const d = decideLastResort({
      defaultModel: "claude-sonnet-5",
      armsChecked: 3,
      isWilling: always,
    });
    expect(d).toEqual({ dial: "claude-sonnet-5" });
  });

  test("a pinned model outranks the default as last resort", () => {
    const d = decideLastResort({
      pinnedModel: "groq/llama-3.3-70b",
      defaultModel: "claude-sonnet-5",
      armsChecked: 1,
      isWilling: (m) => m === "groq/llama-3.3-70b",
    });
    expect(d).toEqual({ dial: "groq/llama-3.3-70b" });
  });

  test("a pinned model gets NO exemption from the willingness check", () => {
    // A pin is a preference, not a mandate — the vessel already neutralises
    // hardcoded pins at the choke point, and the last resort must agree.
    const d = decideLastResort({
      pinnedModel: "claude-sonnet-5",
      defaultModel: "some-other",
      armsChecked: 2,
      isWilling: never,
    });
    expect("refuse" in d).toBe(true);
    if ("refuse" in d) expect(d.refuse).toContain("claude-sonnet-5");
  });

  test('"auto" and empty are not pins, so the default is the last resort', () => {
    for (const pinnedModel of ["auto", "", undefined]) {
      const d = decideLastResort({
        pinnedModel,
        defaultModel: "the-default",
        armsChecked: 0,
        isWilling: always,
      });
      expect(d).toEqual({ dial: "the-default" });
    }
  });

  test("no configured default at all → refuse rather than guess a model", () => {
    // The operator's stated preference is that there be no default. When there
    // is none, the honest outcome is a refusal naming that fact.
    const d = decideLastResort({ defaultModel: "", armsChecked: 0, isWilling: always });
    expect("refuse" in d).toBe(true);
    if ("refuse" in d) expect(d.refuse).toContain("no last-resort model is configured");
  });

  test("the refusal names how many arms were checked, so a 0 is diagnosable", () => {
    // "no arm servable" with 0 checked means the policy is EMPTY; with 12 checked
    // it means every arm is cooling. Those are different operator actions.
    const d = decideLastResort({ defaultModel: "x", armsChecked: 12, isWilling: never });
    if ("refuse" in d) expect(d.refuse).toContain("12 policy arm(s) checked");
  });
});

describe("decideLastResort — a dry default must not blackhole a live plane", () => {
  test("dials a live ROUTABLE model when the default is dry", () => {
    // The regression this prevents: all policy ARMS cooling, default dry, but
    // `google/gemini-2.5-flash` routable and answering. Refusing there reads as
    // a dead plane while a model is one lookup away.
    const d = decideLastResort({
      defaultModel: "claude-sonnet-5",
      armsChecked: 0,
      isWilling: (m) => m === "google/gemini-2.5-flash",
      routableModels: ["nvidia/nemotron-3-nano-30b-a3b:free", "google/gemini-2.5-flash"],
    });
    expect(d).toEqual({ dial: "google/gemini-2.5-flash" });
  });

  test("still refuses when NOTHING routable is willing either", () => {
    const d = decideLastResort({
      defaultModel: "claude-sonnet-5",
      armsChecked: 4,
      isWilling: () => false,
      routableModels: ["a/b", "c/d"],
    });
    expect("refuse" in d).toBe(true);
  });

  test("prefers the last resort over an alternative when it IS willing", () => {
    const d = decideLastResort({
      defaultModel: "claude-sonnet-5",
      armsChecked: 0,
      isWilling: () => true,
      routableModels: ["some/other"],
    });
    expect(d).toEqual({ dial: "claude-sonnet-5" });
  });

  test("never returns the dry last-resort as its own alternative", () => {
    const d = decideLastResort({
      defaultModel: "dry/model",
      armsChecked: 0,
      isWilling: (m) => m === "dry/model" ? false : true,
      routableModels: ["dry/model"],
    });
    expect("refuse" in d).toBe(true);
  });
});
