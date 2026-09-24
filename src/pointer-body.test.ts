import { describe, expect, test } from "bun:test";
import { unwrapPointerBody } from "./pointer-body";

describe("unwrapPointerBody", () => {
  test("a flat body is unchanged", () => {
    const b = { type: "llm_completion", prompt: "hi", max_tokens: 8 };
    expect(unwrapPointerBody(b)).toEqual(b);
  });
  test("discovery's { pointer } form yields the flat fields", () => {
    expect(unwrapPointerBody({ pointer: { type: "llm_completion", prompt: "hi" } }))
      .toEqual({ type: "llm_completion", prompt: "hi" });
  });
  test("the { impulse: { pointer } } form yields the flat fields", () => {
    expect(unwrapPointerBody({ impulse: { pointer: { type: "llmCompletion", prompt: "hi", model: "m" } } }))
      .toEqual({ type: "llmCompletion", prompt: "hi", model: "m" });
  });
  test("top-level fields win over pointer fields", () => {
    expect(unwrapPointerBody({ pointer: { prompt: "inner", model: "a" }, model: "b" }))
      .toEqual({ prompt: "inner", model: "b" });
  });
  test("non-objects pass through", () => {
    expect(unwrapPointerBody(null)).toBeNull();
    expect(unwrapPointerBody("x")).toBe("x");
  });
});
