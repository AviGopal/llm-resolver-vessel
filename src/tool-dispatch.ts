/**
 * The dispatcher, not the model, owns where a tool call is routed.
 *
 * A tool call's pointer was built as `{ type: toolName, ...toolInput }`. The input is model output, so a
 * model-written `"type"` replaced the tool it called: a web_search call whose arguments carried
 * `type: "shell"` reached the shell resolver (discovery serves both from the same endpoint). And any tool
 * name the model emitted was dispatched, whether or not the request offered it.
 *
 * So the pointer's `type` is always the dispatched tool's, set last, and only a tool the request offered is
 * dispatchable at all.
 */

/** The resolve pointer for a tool call: the input's fields, with `type` fixed to the tool being dispatched. */
export function toolPointer(toolName: string, toolInput: Record<string, unknown>): Record<string, unknown> {
  const input = typeof toolInput === "object" && toolInput !== null && !Array.isArray(toolInput) ? toolInput : {};
  const { type: _modelType, ...args } = input;
  return { ...args, type: toolName };
}

/**
 * The tool names this vessel may dispatch for a request: its client-side tools (no `type`, or `custom`).
 * Provider-run server tools are executed by the provider, never dispatched here, so they are not included.
 */
export function offeredToolNames(tools: ReadonlyArray<{ name: string; type?: string }> | undefined): Set<string> {
  return new Set((tools ?? []).filter((t) => !t.type || t.type === "custom").map((t) => t.name));
}

/**
 * The dispatcher, not the model, owns a tool call's attribution too. A tool call's execution_id was the
 * dispatch's only when the model had not written one (`!("execution_id" in input)`), so a model-supplied
 * value won: a call could be attributed to another dispatch, or escape goal-host's floor-lineage
 * recursion guard, which reads it. The dispatch's id now always wins; with no dispatch id, a
 * model-written one is dropped rather than trusted.
 */
export function withDispatchExecutionId(toolInput: Record<string, unknown>, dispatchExecutionId: unknown): Record<string, unknown> {
  const input = typeof toolInput === "object" && toolInput !== null && !Array.isArray(toolInput) ? toolInput : {};
  const { execution_id: _modelId, ...args } = input;
  return typeof dispatchExecutionId === "string" && dispatchExecutionId ? { ...args, execution_id: dispatchExecutionId } : args;
}
