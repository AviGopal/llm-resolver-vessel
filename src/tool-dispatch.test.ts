import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { offeredToolNames, toolPointer, dispatchToolCall } from "./tool-dispatch";

// The dispatcher, not the model, owns a tool call's routing (2026-09-30): the pointer was built as
// { type: toolName, ...toolInput }, so a model-written "type" in the input replaced the tool it called — a
// web_search call arrived at :8230 as shell — and a tool name the request never offered was dispatched anyway.
describe("a tool pointer's type is the dispatched tool's", () => {
  test("a web_search call whose input carries type: shell still sends web_search", () => {
    const p = toolPointer("web_search", { query: "news today", type: "shell", command: "id" });
    expect(p.type).toBe("web_search");
    expect(p.query).toBe("news today");
  });

  test("the input's other fields are carried unchanged", () => {
    expect(toolPointer("shell", { command: "ls", execution_id: "e1" })).toEqual({ command: "ls", execution_id: "e1", type: "shell" });
  });

  test("a non-object input (a parsed array or scalar) carries no fields", () => {
    expect(toolPointer("web_search", ["type", "shell"] as unknown as Record<string, unknown>)).toEqual({ type: "web_search" });
    expect(toolPointer("web_search", null as unknown as Record<string, unknown>)).toEqual({ type: "web_search" });
  });
});

describe("only a tool the request offered is dispatchable", () => {
  test("client-side tools (no type, or custom) are offered; provider-run server tools are not", () => {
    const names = offeredToolNames([
      { name: "web_search" },
      { name: "read_file", type: "custom" },
      { name: "web_search_server", type: "web_search_20250305" },
    ]);
    expect([...names].sort()).toEqual(["read_file", "web_search"]);
  });

  test("a name outside the offered set is not in it", () => {
    expect(offeredToolNames([{ name: "web_search" }]).has("shell")).toBe(false);
  });

  test("no tools offers nothing", () => {
    expect(offeredToolNames(undefined).size).toBe(0);
  });
});

describe("both tool-use loops dispatch through the guard", () => {
  const src = readFileSync(join(import.meta.dir, "index.ts"), "utf8");

  // The dispatch body moved to tool-dispatch.ts (dispatchToolCall); index.ts's dispatchTool delegates to it.
  const dispatchSrc = readFileSync(join(import.meta.dir, "tool-dispatch.ts"), "utf8");

  test("the pointer is never built with the model's input spread over the tool type", () => {
    expect(src).not.toMatch(/\{\s*type:\s*toolName\s*,\s*\.\.\./);
    // code only: tool-dispatch.ts's header comment quotes the old form
    const dispatchCode = dispatchSrc.split("\n").filter((l) => !/^\s*(\*|\/\/|\/\*)/.test(l)).join("\n");
    expect(dispatchCode).not.toMatch(/\{\s*type:\s*toolName\s*,\s*\.\.\./);
    expect(dispatchSrc).toContain("toolPointer(toolName, toolInput)");
    expect(src).toContain("dispatchToolCall(resolveToolEndpoint,");
  });

  test("dispatchTool refuses an unoffered tool before resolving its endpoint", async () => {
    const body = dispatchSrc.slice(dispatchSrc.indexOf("export async function dispatchToolCall("));
    const refuse = body.indexOf("!offered.has(toolName)");
    expect(refuse).toBeGreaterThan(-1);
    expect(refuse).toBeLessThan(body.indexOf("resolveToolEndpoint("));
    // and by behaviour: the endpoint lookup never runs for an unoffered tool
    let looked = 0;
    const r = await dispatchToolCall(async () => { looked++; return { url: "http://x", via: "fallback" as const }; }, { endpoint: "http://x", nodeKey: "k", apiKey: "k" }, "shell", {}, new Set(["web_search"]));
    expect(r.ok).toBe(false);
    expect(looked).toBe(0);
  });

  test("the Anthropic and OpenAI loops each pass the request's offered set", () => {
    const calls = src.match(/await dispatchTool\([^;]*\);/g) ?? [];
    expect(calls.length).toBe(2);
    for (const c of calls) expect(c).toContain("offered");
  });
});
