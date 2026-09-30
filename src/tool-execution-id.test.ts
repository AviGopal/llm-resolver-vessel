import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { withDispatchExecutionId } from "./tool-dispatch";

// A model-supplied execution_id used to win over the dispatch's (`!("execution_id" in input)`), so a tool
// call could be attributed to another dispatch, or escape goal-host's floor-lineage recursion guard.
describe("the dispatch's execution_id always wins on a tool call", () => {
  test("a model-written execution_id is overridden by the dispatch's", () => {
    expect(withDispatchExecutionId({ command: "ls", execution_id: "someone-else" }, "d-1")).toEqual({ command: "ls", execution_id: "d-1" });
  });
  test("absent in the input, the dispatch's is set", () => {
    expect(withDispatchExecutionId({ command: "ls" }, "d-1")).toEqual({ command: "ls", execution_id: "d-1" });
  });
  test("with no dispatch id, a model-written one is dropped, not trusted", () => {
    expect(withDispatchExecutionId({ command: "ls", execution_id: "forged" }, undefined)).toEqual({ command: "ls" });
    expect(withDispatchExecutionId({ command: "ls" }, "")).toEqual({ command: "ls" });
  });
  test("both tool loops (Anthropic and OpenAI) dispatch through it, and neither keeps the old model-wins form", () => {
    const src = readFileSync(join(import.meta.dir, "index.ts"), "utf-8");
    expect(src).toContain("dispatchTool(dispatchEndpoint, dispatchApiKey, tu.name, withDispatchExecutionId(tu.input, (body as { execution_id?: unknown }).execution_id), offered)");
    expect(src).toContain("dispatchTool(dispatchEndpoint, dispatchApiKey, tc.function.name, withDispatchExecutionId(toolInput, (body as { execution_id?: unknown }).execution_id), offered)");
    expect(src).not.toContain('!("execution_id" in tu.input)');
    expect(src).not.toContain('!("execution_id" in toolInput)');
  });
});
