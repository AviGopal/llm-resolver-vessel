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

// ─── Where a tool call goes, and with which credential ─────────────────────────────────────────────────
// Moved here from index.ts unchanged, so the routing and the credential are testable without booting the
// vessel (index.ts starts timers and a server at import).

/** The request's tool-dispatch route: the endpoint to fall back to and the key to send. */
export function toolDispatchRoute(
  body: { tool_dispatch_endpoint?: string; tool_dispatch_api_key?: string },
  configuredEndpoint: string,
  nodeKey: string | undefined,
): { endpoint: string; apiKey: string } {
  return {
    endpoint: body.tool_dispatch_endpoint ?? configuredEndpoint,
    apiKey: body.tool_dispatch_api_key ?? nodeKey ?? "",
  };
}

/** A per-tool endpoint lookup through discovery, falling back to `fallback` (cached per tool name). */
export function makeToolEndpointResolver(discoveryEndpoint: string, discoveryKey: string | undefined) {
  const cache = new Map<string, string>();
  return async function resolveToolEndpoint(toolName: string, fallback: string): Promise<string> {
    const cached = cache.get(toolName);
    if (cached) return cached;
    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (discoveryKey) headers.Authorization = `ApiKey ${discoveryKey}`;
      const res = await fetch(`${discoveryEndpoint}/resolve`, {
        method: "POST",
        headers,
        body: JSON.stringify({ pointer: { type: "vesselCapability", shape: toolName } }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) {
        cache.set(toolName, fallback);
        return fallback;
      }
      const body = (await res.json()) as { content?: { vessels?: Array<{ endpoint?: string; resolve_endpoint?: string }> } };
      const vessel = body?.content?.vessels?.[0];
      if (!vessel || !vessel.endpoint || !vessel.resolve_endpoint) {
        cache.set(toolName, fallback);
        return fallback;
      }
      const re = vessel.resolve_endpoint;
      const url = re.startsWith("http://") || re.startsWith("https://")
        ? re
        : `${vessel.endpoint.replace(/\/$/, "")}${re.startsWith("/") ? re : "/" + re}`;
      cache.set(toolName, url);
      return url;
    } catch {
      cache.set(toolName, fallback);
      return fallback;
    }
  };
}

/** Dispatch one tool call: refuse an unoffered tool, resolve its endpoint, POST the pointer. */
export async function dispatchToolCall(
  resolveToolEndpoint: (toolName: string, fallback: string) => Promise<string>,
  endpoint: string,
  apiKey: string,
  toolName: string,
  toolInput: Record<string, unknown>,
  offered: ReadonlySet<string>,
): Promise<{ ok: boolean; result: unknown; error?: string }> {
  // Only a tool the request offered is dispatched, and its type is never the model's to choose.
  if (!offered.has(toolName)) return { ok: false, result: null, error: `tool '${toolName}' was not offered in this request` };
  const pointer = toolPointer(toolName, toolInput);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 60_000);
  try {
    const target = await resolveToolEndpoint(toolName, endpoint);
    const res = await fetch(target, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `ApiKey ${apiKey}`,
      },
      body: JSON.stringify({ impulse: { pointer } }),
      signal: ctrl.signal,
    });
    clearTimeout(timer);
    const text = await res.text();
    if (!res.ok) return { ok: false, result: null, error: `HTTP ${res.status}: ${text.slice(0, 300)}` };
    let parsed: unknown;
    try { parsed = JSON.parse(text); } catch { parsed = text; }
    if (parsed && typeof parsed === "object" && "body" in parsed) {
      const env = parsed as Record<string, unknown>;
      if (env.shape === "structuredError") return { ok: false, result: env.body, error: JSON.stringify(env.body).slice(0, 300) };
      return { ok: true, result: env.body };
    }
    return { ok: true, result: parsed };
  } catch (err) {
    clearTimeout(timer);
    return { ok: false, result: null, error: err instanceof Error ? err.message : String(err) };
  }
}
