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
// index.ts starts timers and a server at import, so this lives here to be testable.
//
// THE NODE KEY NEVER FOLLOWS A CALLER-SUPPLIED ENDPOINT. A request may name `tool_dispatch_endpoint` (used
// when discovery has no producer for the tool) and `tool_dispatch_api_key`. The rule:
//   - a configured or discovered endpoint gets this vessel's node key, as before;
//   - a caller-supplied endpoint gets only the caller's own key, or no Authorization at all;
//   - the caller's key is honoured only with that override: it never goes to a configured or discovered
//     endpoint;
//   - a request's override is never cached as the tool's endpoint, so it cannot redirect a later request
//     (which would carry the node key) to the caller's URL.
// A blank override is unset.

export type ToolDispatchRoute = {
  /** Where a tool goes when discovery has no producer: the caller's override, or the configured endpoint. */
  endpoint: string;
  /** The caller's override and the caller's own key for it (possibly ""), when the request supplied one. */
  override?: { url: string; key: string };
  /** This vessel's key, for configured and discovered endpoints only. */
  nodeKey: string;
  /** A credential is available for the route at all (the loops refuse client-side tools without one). */
  apiKey: string;
};

/** The request's tool-dispatch route. */
export function toolDispatchRoute(
  body: { tool_dispatch_endpoint?: string; tool_dispatch_api_key?: string },
  configuredEndpoint: string,
  nodeKey: string | undefined,
): ToolDispatchRoute {
  const raw = body.tool_dispatch_endpoint;
  const overrideUrl = typeof raw === "string" && raw.trim() !== "" ? raw : undefined;
  const node = nodeKey ?? "";
  if (overrideUrl === undefined) return { endpoint: configuredEndpoint, nodeKey: node, apiKey: node };
  const callerKey = typeof body.tool_dispatch_api_key === "string" ? body.tool_dispatch_api_key : "";
  return { endpoint: overrideUrl, override: { url: overrideUrl, key: callerKey }, nodeKey: node, apiKey: callerKey || node };
}

/**
 * A per-tool endpoint lookup through discovery. Only a discovered endpoint is cached: the fallback is the
 * request's (possibly its caller's override), so caching it would hand that URL to later requests.
 */
export function makeToolEndpointResolver(discoveryEndpoint: string, discoveryKey: string | undefined) {
  const cache = new Map<string, string>();
  return async function resolveToolEndpoint(toolName: string, fallback: string): Promise<{ url: string; via: "discovered" | "fallback" }> {
    const cached = cache.get(toolName);
    if (cached) return { url: cached, via: "discovered" };
    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (discoveryKey) headers.Authorization = `ApiKey ${discoveryKey}`;
      const res = await fetch(`${discoveryEndpoint}/resolve`, {
        method: "POST",
        headers,
        body: JSON.stringify({ pointer: { type: "vesselCapability", shape: toolName } }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) return { url: fallback, via: "fallback" };
      const body = (await res.json()) as { content?: { vessels?: Array<{ endpoint?: string; resolve_endpoint?: string }> } };
      const vessel = body?.content?.vessels?.[0];
      if (!vessel || !vessel.endpoint || !vessel.resolve_endpoint) return { url: fallback, via: "fallback" };
      const re = vessel.resolve_endpoint;
      const url = re.startsWith("http://") || re.startsWith("https://")
        ? re
        : `${vessel.endpoint.replace(/\/$/, "")}${re.startsWith("/") ? re : "/" + re}`;
      cache.set(toolName, url);
      return { url, via: "discovered" };
    } catch {
      return { url: fallback, via: "fallback" };
    }
  };
}

/** The key a tool call to `target` carries: see the rule above. "" means no Authorization. */
export function toolDispatchKey(route: ToolDispatchRoute, target: { url: string; via: "discovered" | "fallback" }): string {
  if (target.via === "fallback" && route.override && target.url === route.override.url) return route.override.key;
  return route.nodeKey;
}

/** Dispatch one tool call: refuse an unoffered tool, resolve its endpoint, POST the pointer. */
export async function dispatchToolCall(
  resolveToolEndpoint: (toolName: string, fallback: string) => Promise<{ url: string; via: "discovered" | "fallback" }>,
  route: ToolDispatchRoute,
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
    const target = await resolveToolEndpoint(toolName, route.endpoint);
    const key = toolDispatchKey(route, target);
    const res = await fetch(target.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(key ? { Authorization: `ApiKey ${key}` } : {}),
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
