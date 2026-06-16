/**
 * llm-resolver-vessel — standalone LLM resolver vessel (port 8220).
 *
 * Spec: openspec/changes/2026-05-23-substrate-explicit-vessels Phase 2, tasks 2.1–2.3.
 *
 * Advertises shape: llmCompletion
 * Registers with discovery-vessel at http://127.0.0.1:8100
 *
 * Resolver contract:
 *   POST /resolve
 *   { "type": "llm_completion", "prompt": string, "model"?: string,
 *     "max_tokens"?: number, "system"?: string }
 *
 *   Response:
 *   { "resolved": true, "shape": "llmCompletion",
 *     "content": string, "usage": { "input_tokens": number, "output_tokens": number } }
 *
 * ANTHROPIC_API_KEY is read from the environment (supplied via
 * EnvironmentFile=/etc/substrate/env in the systemd unit).
 * The vessel starts even without an API key — it logs a warning and
 * returns an error response on any resolve call.
 */

import Anthropic from "@anthropic-ai/sdk";
import {
  ActivityExecutor,
  ExecutionRuntime,
  VesselDaemon,
} from "@avigopal/ias-executor-ts";
import type { ResolverHandler } from "@avigopal/ias-executor-ts";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.PORT ?? "8220", 10);
const VESSEL_ID = process.env.LLM_RESOLVER_VESSEL_ID ?? process.env.VESSEL_ID ?? "llm-resolver-vessel";
const DISCOVERY_ENDPOINT = process.env.DISCOVERY_VESSEL_ENDPOINT ?? "http://127.0.0.1:8100";
const API_KEY = process.env.LLM_RESOLVER_VESSEL_API_KEY ?? process.env.METABOB_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const DEFAULT_MODEL = process.env.LLM_DEFAULT_MODEL ?? "claude-sonnet-4-20250514";
const DEFAULT_MAX_TOKENS = 4096;

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic client (lazy — warn at startup if key absent; fail at resolve time)
// ─────────────────────────────────────────────────────────────────────────────

let anthropic: Anthropic | null = null;

if (!ANTHROPIC_API_KEY) {
  console.warn(
    "[llm-resolver-vessel] ANTHROPIC_API_KEY is not set. " +
    "llm_completion resolver will return errors until the key is provided.",
  );
} else {
  anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
}

// ─────────────────────────────────────────────────────────────────────────────
// llm_completion resolver
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tool definitions accepted by /resolve. Two forms:
 *
 * 1. Client-side custom tool: `{ name, description, input_schema }`. The
 *    model emits `tool_use` blocks; this vessel dispatches them to
 *    `tool_dispatch_endpoint` (default dev-vessel /v2/impulses/resolve).
 *
 * 2. Server-side built-in tool: `{ type: "web_search_20250305", name,
 *    max_uses? }` and similar. Anthropic executes these server-side; the
 *    response carries `server_tool_use` + `web_search_tool_result` blocks
 *    inline and the loop continues to a final text block without any
 *    client dispatch. No `description`/`input_schema` for these.
 *
 * The SDK accepts either via the typed-tools union; we keep our request
 * type permissive and forward verbatim.
 */
interface AnthropicToolDef {
  type?: string;
  name: string;
  description?: string;
  input_schema?: Record<string, unknown>;
  // web_search-specific knobs (max_uses, allowed_domains, etc.) flow through
  // via index access — keep the interface open rather than enumerating every
  // server-tool field.
  [k: string]: unknown;
}

interface LlmCompletionRequest {
  type: "llm_completion";
  prompt: string;
  model?: string;
  max_tokens?: number;
  system?: string;
  // Tool-use mode (#121): when `tools` is non-empty, run an iterative tool-use
  // loop. Each tool_use block is dispatched to `tool_dispatch_endpoint`
  // (default: dev-vessel /v2/impulses/resolve) by wrapping the tool's `name`
  // + `input` as a pointer `{type: <name>, ...input}`. Append tool_result,
  // call Anthropic again, repeat until stop_reason != tool_use or
  // max_tool_iterations reached.
  tools?: AnthropicToolDef[];
  tool_dispatch_endpoint?: string;
  tool_dispatch_api_key?: string;
  max_tool_iterations?: number;
}

interface ToolCallTraceEntry {
  iteration: number;
  tool_name: string;
  tool_input: unknown;
  tool_output: unknown;
  duration_ms: number;
}

const DEFAULT_TOOL_DISPATCH_ENDPOINT =
  process.env.LLM_TOOL_DISPATCH_ENDPOINT ?? "http://127.0.0.1:8090/v2/impulses/resolve";
const DEFAULT_MAX_TOOL_ITERATIONS = parseInt(
  process.env.LLM_MAX_TOOL_ITERATIONS ?? "8",
  10,
);

async function dispatchTool(
  endpoint: string,
  apiKey: string,
  toolName: string,
  toolInput: Record<string, unknown>,
): Promise<{ ok: boolean; result: unknown; error?: string }> {
  const pointer = { type: toolName, ...toolInput };
  const reqBody = JSON.stringify({ impulse: { pointer } });
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 60_000);
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `ApiKey ${apiKey}`,
      },
      body: reqBody,
      signal: ctrl.signal,
    });
    clearTimeout(timer);
    const text = await res.text();
    if (!res.ok) {
      return { ok: false, result: null, error: `HTTP ${res.status}: ${text.slice(0, 300)}` };
    }
    let parsed: unknown;
    try { parsed = JSON.parse(text); } catch { parsed = text; }
    if (parsed && typeof parsed === "object" && "body" in parsed) {
      const env = parsed as Record<string, unknown>;
      if (env.shape === "structuredError") {
        return { ok: false, result: env.body, error: JSON.stringify(env.body).slice(0, 300) };
      }
      return { ok: true, result: env.body };
    }
    return { ok: true, result: parsed };
  } catch (err) {
    clearTimeout(timer);
    return { ok: false, result: null, error: err instanceof Error ? err.message : String(err) };
  }
}

const llmCompletionHandler: ResolverHandler = async (ctx) => {
  if (!anthropic) {
    return {
      resolved: false,
      shape: "llmCompletion",
      error: "ANTHROPIC_API_KEY not configured on llm-resolver-vessel",
    };
  }

  const body = ctx.body as LlmCompletionRequest;

  if (!body.prompt || typeof body.prompt !== "string") {
    return {
      resolved: false,
      shape: "llmCompletion",
      error: "Request body must include a non-empty 'prompt' string field",
    };
  }

  // Normalize model id: callers send "anthropic/claude-..." (provider-prefixed)
  // but the Anthropic SDK expects bare names like "claude-haiku-4-5-20251001".
  const rawModel = body.model ?? DEFAULT_MODEL;
  const model = rawModel.startsWith("anthropic/") ? rawModel.slice("anthropic/".length) : rawModel;
  const maxTokens = body.max_tokens ?? DEFAULT_MAX_TOKENS;

  // Plain completion path (no tools)
  if (!body.tools || body.tools.length === 0) {
    try {
      const response = await anthropic.messages.create({
        model,
        max_tokens: maxTokens,
        ...(body.system ? { system: body.system } : {}),
        messages: [{ role: "user", content: body.prompt }],
      });

      const content = response.content
        .filter((block) => block.type === "text")
        .map((block) => (block as { type: "text"; text: string }).text)
        .join("");

      return {
        resolved: true,
        shape: "llmCompletion",
        content,
        usage: {
          input_tokens: response.usage.input_tokens,
          output_tokens: response.usage.output_tokens,
        },
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[llm-resolver-vessel] llm_completion error:", message);
      return {
        resolved: false,
        shape: "llmCompletion",
        error: message,
      };
    }
  }

  // Tool-use loop path (#121, +web_search 2026-06-09)
  const dispatchEndpoint = body.tool_dispatch_endpoint ?? DEFAULT_TOOL_DISPATCH_ENDPOINT;
  const dispatchApiKey = body.tool_dispatch_api_key ?? process.env.METABOB_API_KEY ?? "";
  const maxIter = Math.max(1, Math.min(body.max_tool_iterations ?? DEFAULT_MAX_TOOL_ITERATIONS, 20));

  // Client-side tools require a dispatch key (each tool_use block is POSTed
  // to dev-vessel). Server-side tools (web_search_20250305 and similar) run
  // inside Anthropic and need no dispatch. Only enforce the key when there
  // is at least one client-side custom tool. A custom tool is identified by
  // the absence of a `type` field (server tools always carry one like
  // "web_search_20250305").
  const hasClientSideTools = body.tools.some(
    (t) => !t.type || (typeof t.type === "string" && t.type === "custom"),
  );
  if (hasClientSideTools && !dispatchApiKey) {
    return {
      resolved: false,
      shape: "llmCompletion",
      error: "tool-use requested with client-side tools but neither tool_dispatch_api_key nor METABOB_API_KEY is set",
    };
  }

  const messages: Array<{ role: "user" | "assistant"; content: unknown }> = [
    { role: "user", content: body.prompt },
  ];
  const toolCalls: ToolCallTraceEntry[] = [];
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let finalText = "";

  for (let iter = 1; iter <= maxIter; iter++) {
    let response;
    try {
      response = await anthropic.messages.create({
        model,
        max_tokens: maxTokens,
        tools: body.tools as unknown as Anthropic.Messages.Tool[],
        ...(body.system ? { system: body.system } : {}),
        messages: messages as unknown as Anthropic.Messages.MessageParam[],
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        resolved: false,
        shape: "llmCompletion",
        error: `anthropic call (iter ${iter}): ${message}`,
        tool_calls: toolCalls,
      };
    }

    totalInputTokens += response.usage.input_tokens;
    totalOutputTokens += response.usage.output_tokens;

    messages.push({ role: "assistant", content: response.content });

    if (response.stop_reason !== "tool_use") {
      finalText = response.content
        .filter((b) => b.type === "text")
        .map((b) => (b as { type: "text"; text: string }).text)
        .join("");
      break;
    }

    const toolUses = response.content.filter((b) => b.type === "tool_use") as Array<{
      type: "tool_use"; id: string; name: string; input: Record<string, unknown>;
    }>;

    const toolResults: Array<{ type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean }> = [];
    for (const tu of toolUses) {
      const start = Date.now();
      const r = await dispatchTool(dispatchEndpoint, dispatchApiKey, tu.name, tu.input);
      const duration = Date.now() - start;
      toolCalls.push({
        iteration: iter,
        tool_name: tu.name,
        tool_input: tu.input,
        tool_output: r.ok ? r.result : { error: r.error },
        duration_ms: duration,
      });
      toolResults.push({
        type: "tool_result",
        tool_use_id: tu.id,
        content: typeof r.result === "string"
          ? r.result
          : JSON.stringify(r.result ?? r.error ?? null),
        ...(r.ok ? {} : { is_error: true }),
      });
    }

    messages.push({ role: "user", content: toolResults });
  }

  return {
    resolved: true,
    shape: "llmCompletion",
    content: finalText,
    tool_calls: toolCalls,
    iterations: toolCalls.length > 0 ? toolCalls[toolCalls.length - 1]!.iteration : 0,
    usage: {
      input_tokens: totalInputTokens,
      output_tokens: totalOutputTokens,
    },
  };
};

// ─────────────────────────────────────────────────────────────────────────────
// ActivityExecutor (minimal — llm-resolver-vessel does not run activities
// itself; the executor is required by VesselDaemon's constructor but the
// /run-goal endpoint is not used by this vessel)
// ─────────────────────────────────────────────────────────────────────────────

const runtime = new ExecutionRuntime({
  attachedVessels: [
    { id: VESSEL_ID, kind: "custom" as never, resolverIds: ["llm_completion"] },
  ],
});

const executor = new ActivityExecutor(runtime);

// ─────────────────────────────────────────────────────────────────────────────
// VesselDaemon
// ─────────────────────────────────────────────────────────────────────────────

const resolvers = new Map<string, ResolverHandler>([
  ["llm_completion", llmCompletionHandler],
]);

const daemon = new VesselDaemon({
  port: PORT,
  vesselId: VESSEL_ID,
  vesselName: "LLM Resolver Vessel",
  // Advertise BOTH names. The handler is registered as `llm_completion`
  // (snake_case, line 145) so that's what discovery clients must POST to
  // /resolve with `pointer.type`. Older callers query discovery for the
  // camelCase `llmCompletion` — keep both advertised so neither shape-name
  // convention breaks. Without this dual advertisement, discovery-routed
  // LLM calls have been silently failing for the entire substrate
  // (53 traces with 0/0 tokens during goal[7]'s exec_x19p0558 confirmed
  // the gap — every llm-touching task across slot-binding,
  // create-shape-provider-goal, validator-dispatch, draft-gap-closing-
  // activity, and try-direct-answer was reaching no LLM).
  shapes: ["llm_completion", "llmCompletion"],
  executor,
  resolvers,
  discoveryEndpoint: DISCOVERY_ENDPOINT,
  apiKey: API_KEY,
  version: "0.1.0",
  enforceCompositionChain: false,
});

await daemon.start();

console.log(
  `[llm-resolver-vessel] started on port ${PORT}` +
  ` | discovery: ${DISCOVERY_ENDPOINT}` +
  ` | anthropic: ${anthropic ? "configured" : "MISSING KEY"}`,
);

// ─────────────────────────────────────────────────────────────────────────────
// Iteration 9 of the cross-vessel OOM hunt — periodic Bun.gc(true) workaround.
// See: concept_T-CTTOEl97IM (description), concept_s9ye5GKLw2L8 (signature),
//      concept_9ldsmRgqSTd5 (iter-6 derivation in goal-host-vessel).
//
// Hypothesis: Bun 1.3.14 retains heap-arena pages after free; affected vessels
// show RSS growth disconnected from heapUsed. goal-host hit OOM first because
// of its event volume; per iter-9 we apply the same workaround substrate-wide.
// A periodic forced full GC bounds RSS without changing semantics.
//
// .unref() so the timer doesn't prevent process exit.
// ─────────────────────────────────────────────────────────────────────────────
const GC_INTERVAL_MS = parseInt(process.env.LLM_RESOLVER_GC_INTERVAL_MS ?? "30000", 10);
interface BunGlobal { Bun?: { gc?: (force: boolean) => number } }
const bunGlobal = globalThis as unknown as BunGlobal;
setInterval(() => {
  const gc = bunGlobal.Bun?.gc;
  if (typeof gc === "function") {
    try {
      const freed = gc(true);
      const rssMB = (process.memoryUsage().rss / 1024 / 1024).toFixed(1);
      console.log(`[gc-tick] vessel=llm-resolver-vessel freed=${freed}B rss_after=${rssMB}MB`);
    } catch (err) {
      console.warn(`[gc-tick] Bun.gc failed: ${(err as Error).message}`);
    }
  }
}, GC_INTERVAL_MS).unref();
