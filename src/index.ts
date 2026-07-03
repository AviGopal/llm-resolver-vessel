/**
 * llm-resolver-vessel — multi-provider LLM resolver vessel (port 8220).
 *
 * Spec: openspec/changes/2026-05-23-substrate-explicit-vessels Phase 2, tasks 2.1–2.3.
 *
 * Advertises shapes: llm_completion, llmCompletion
 * Registers with discovery-vessel at http://127.0.0.1:8100
 *
 * Resolver contract:
 *   POST /resolve
 *   { "type": "llm_completion",
 *     "prompt": string,
 *     "model"?: string,          // e.g. "claude-sonnet-4-6", "gpt-4o", "llama3.2"
 *     "provider"?: string,       // explicit override: "anthropic" | "openai" | "auto"
 *     "max_tokens"?: number,
 *     "system"?: string,
 *     "tools"?: AnthropicToolDef[] }
 *
 * Provider routing (auto mode):
 *   ANTHROPIC_API_KEY set → Anthropic for claude-* models
 *   OPENAI_API_KEY set    → OpenAI-compatible for everything else
 *     OPENAI_BASE_URL override enables:
 *       Ollama:     http://localhost:11434/v1  (OPENAI_API_KEY=ollama)
 *       LM Studio:  http://localhost:1234/v1   (OPENAI_API_KEY=lm-studio)
 *       Groq:       https://api.groq.com/openai/v1
 *       Together:   https://api.together.xyz/v1
 *       vLLM:       http://<host>:8000/v1
 *       Any other OpenAI-compatible endpoint
 */

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
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

// Provider config
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL; // default: OpenAI; set for Ollama/Groq/etc.
const LLM_PROVIDER = (process.env.LLM_PROVIDER ?? "auto") as "anthropic" | "openai" | "auto";

const DEFAULT_MODEL = process.env.LLM_DEFAULT_MODEL || "claude-sonnet-5";
const DEFAULT_MAX_TOKENS = 4096;

// Retired Anthropic model ids that callers still hardcode and that 404 at the API.
const RETIRED_ANTHROPIC_MODEL_IDS = new Set(["claude-sonnet-4-20250514"]);

// Model prefixes that always route to the OpenAI-compatible path.
const OPENAI_MODEL_PREFIXES = [
  "gpt-", "o1-", "o3-", "o4-",
  "llama", "mistral", "mixtral", "gemma", "phi-",
  "qwen", "deepseek", "yi-", "command-", "nova-",
  "whisper-", "tts-", "dall-e-",
];

// ─────────────────────────────────────────────────────────────────────────────
// Client initialisation
// ─────────────────────────────────────────────────────────────────────────────

let anthropic: Anthropic | null = null;
let openaiClient: OpenAI | null = null;

if (ANTHROPIC_API_KEY) {
  anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
} else {
  console.warn("[llm-resolver-vessel] ANTHROPIC_API_KEY not set — Anthropic provider unavailable");
}

if (OPENAI_API_KEY) {
  openaiClient = new OpenAI({
    apiKey: OPENAI_API_KEY,
    ...(OPENAI_BASE_URL ? { baseURL: OPENAI_BASE_URL } : {}),
  });
  const endpoint = OPENAI_BASE_URL ?? "https://api.openai.com/v1";
  console.log(`[llm-resolver-vessel] OpenAI-compatible provider: ${endpoint}`);
} else if (!ANTHROPIC_API_KEY) {
  console.warn("[llm-resolver-vessel] Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY set. All resolve calls will fail.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider routing
// ─────────────────────────────────────────────────────────────────────────────

function pickProvider(model: string, explicitProvider?: string): "anthropic" | "openai" | null {
  if (explicitProvider === "anthropic") return anthropic ? "anthropic" : null;
  if (explicitProvider === "openai") return openaiClient ? "openai" : null;

  if (LLM_PROVIDER === "anthropic") return anthropic ? "anthropic" : null;
  if (LLM_PROVIDER === "openai") return openaiClient ? "openai" : null;

  // auto: route by model name
  const lower = model.toLowerCase();
  if (lower.startsWith("claude") || lower.startsWith("anthropic/claude")) {
    return anthropic ? "anthropic" : (openaiClient ? "openai" : null);
  }
  for (const prefix of OPENAI_MODEL_PREFIXES) {
    if (lower.startsWith(prefix)) {
      return openaiClient ? "openai" : null;
    }
  }
  // Default: prefer Anthropic if available, else OpenAI
  if (anthropic) return "anthropic";
  if (openaiClient) return "openai";
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool dispatch (shared between both provider paths)
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_TOOL_DISPATCH_ENDPOINT =
  process.env.LLM_TOOL_DISPATCH_ENDPOINT ?? "http://127.0.0.1:8090/v2/impulses/resolve";
const DEFAULT_MAX_TOOL_ITERATIONS = parseInt(process.env.LLM_MAX_TOOL_ITERATIONS ?? "8", 10);

async function dispatchTool(
  endpoint: string,
  apiKey: string,
  toolName: string,
  toolInput: Record<string, unknown>,
): Promise<{ ok: boolean; result: unknown; error?: string }> {
  const pointer = { type: toolName, ...toolInput };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 60_000);
  try {
    const res = await fetch(endpoint, {
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

// ─────────────────────────────────────────────────────────────────────────────
// Request types
// ─────────────────────────────────────────────────────────────────────────────

interface AnthropicToolDef {
  type?: string;
  name: string;
  description?: string;
  input_schema?: Record<string, unknown>;
  [k: string]: unknown;
}

interface LlmCompletionRequest {
  type: "llm_completion";
  prompt: string;
  model?: string;
  provider?: "anthropic" | "openai" | "auto";
  max_tokens?: number;
  system?: string;
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

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic resolution path (existing, preserved)
// ─────────────────────────────────────────────────────────────────────────────

async function resolveWithAnthropic(body: LlmCompletionRequest): Promise<Record<string, unknown>> {
  if (!anthropic) {
    return { resolved: false, shape: "llmCompletion", error: "ANTHROPIC_API_KEY not configured" };
  }
  const rawModel = body.model ?? DEFAULT_MODEL;
  const stripped = rawModel.startsWith("anthropic/") ? rawModel.slice("anthropic/".length) : rawModel;
  const model = RETIRED_ANTHROPIC_MODEL_IDS.has(stripped) ? DEFAULT_MODEL : stripped;
  const maxTokens = body.max_tokens ?? DEFAULT_MAX_TOKENS;

  if (!body.tools || body.tools.length === 0) {
    try {
      const response = await anthropic.messages.create({
        model, max_tokens: maxTokens,
        ...(body.system ? { system: body.system } : {}),
        messages: [{ role: "user", content: body.prompt }],
      });
      const content = response.content
        .filter((b) => b.type === "text")
        .map((b) => (b as { type: "text"; text: string }).text)
        .join("");
      return {
        resolved: true, shape: "llmCompletion", content,
        provider: "anthropic", model,
        usage: { input_tokens: response.usage.input_tokens, output_tokens: response.usage.output_tokens },
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[llm-resolver-vessel] anthropic error:", message);
      return { resolved: false, shape: "llmCompletion", error: message };
    }
  }

  // Tool-use loop (Anthropic)
  const dispatchEndpoint = body.tool_dispatch_endpoint ?? DEFAULT_TOOL_DISPATCH_ENDPOINT;
  const dispatchApiKey = body.tool_dispatch_api_key ?? process.env.METABOB_API_KEY ?? "";
  const maxIter = Math.max(1, Math.min(body.max_tool_iterations ?? DEFAULT_MAX_TOOL_ITERATIONS, 20));
  const hasClientSideTools = body.tools.some((t) => !t.type || t.type === "custom");
  if (hasClientSideTools && !dispatchApiKey) {
    return { resolved: false, shape: "llmCompletion", error: "tool-use with client-side tools requires METABOB_API_KEY" };
  }

  const messages: Array<{ role: "user" | "assistant"; content: unknown }> = [
    { role: "user", content: body.prompt },
  ];
  const toolCalls: ToolCallTraceEntry[] = [];
  let totalInputTokens = 0, totalOutputTokens = 0, finalText = "";

  for (let iter = 1; iter <= maxIter; iter++) {
    let response;
    try {
      response = await anthropic.messages.create({
        model, max_tokens: maxTokens,
        tools: body.tools as unknown as Anthropic.Messages.Tool[],
        ...(body.system ? { system: body.system } : {}),
        messages: messages as unknown as Anthropic.Messages.MessageParam[],
      });
    } catch (err) {
      return { resolved: false, shape: "llmCompletion", error: `anthropic (iter ${iter}): ${err instanceof Error ? err.message : String(err)}`, tool_calls: toolCalls };
    }
    totalInputTokens += response.usage.input_tokens;
    totalOutputTokens += response.usage.output_tokens;
    messages.push({ role: "assistant", content: response.content });
    if (response.stop_reason !== "tool_use") {
      finalText = response.content.filter((b) => b.type === "text").map((b) => (b as { type: "text"; text: string }).text).join("");
      break;
    }
    const toolUses = response.content.filter((b) => b.type === "tool_use") as Array<{ type: "tool_use"; id: string; name: string; input: Record<string, unknown> }>;
    const toolResults: Array<{ type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean }> = [];
    for (const tu of toolUses) {
      const start = Date.now();
      const r = await dispatchTool(dispatchEndpoint, dispatchApiKey, tu.name, tu.input);
      toolCalls.push({ iteration: iter, tool_name: tu.name, tool_input: tu.input, tool_output: r.ok ? r.result : { error: r.error }, duration_ms: Date.now() - start });
      toolResults.push({ type: "tool_result", tool_use_id: tu.id, content: typeof r.result === "string" ? r.result : JSON.stringify(r.result ?? r.error ?? null), ...(r.ok ? {} : { is_error: true }) });
    }
    messages.push({ role: "user", content: toolResults });
  }
  return {
    resolved: true, shape: "llmCompletion", content: finalText,
    provider: "anthropic", model,
    tool_calls: toolCalls,
    iterations: toolCalls.length > 0 ? toolCalls[toolCalls.length - 1]!.iteration : 0,
    usage: { input_tokens: totalInputTokens, output_tokens: totalOutputTokens },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAI-compatible resolution path (OpenAI, Ollama, Groq, Together, vLLM, …)
// ─────────────────────────────────────────────────────────────────────────────

async function resolveWithOpenAI(body: LlmCompletionRequest): Promise<Record<string, unknown>> {
  if (!openaiClient) {
    return { resolved: false, shape: "llmCompletion", error: "OPENAI_API_KEY not configured" };
  }

  const rawModel = body.model ?? DEFAULT_MODEL;
  const model = rawModel.startsWith("openai/") ? rawModel.slice("openai/".length) : rawModel;
  const maxTokens = body.max_tokens ?? DEFAULT_MAX_TOKENS;

  const systemMessages: OpenAI.Chat.ChatCompletionMessageParam[] = body.system
    ? [{ role: "system", content: body.system }]
    : [];

  if (!body.tools || body.tools.length === 0) {
    try {
      const response = await openaiClient.chat.completions.create({
        model,
        max_tokens: maxTokens,
        messages: [...systemMessages, { role: "user", content: body.prompt }],
      });
      const content = response.choices[0]?.message?.content ?? "";
      return {
        resolved: true, shape: "llmCompletion", content,
        provider: "openai", model,
        usage: {
          input_tokens: response.usage?.prompt_tokens ?? 0,
          output_tokens: response.usage?.completion_tokens ?? 0,
        },
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[llm-resolver-vessel] openai error:", message);
      return { resolved: false, shape: "llmCompletion", error: message };
    }
  }

  // Tool-use loop (OpenAI format)
  const dispatchEndpoint = body.tool_dispatch_endpoint ?? DEFAULT_TOOL_DISPATCH_ENDPOINT;
  const dispatchApiKey = body.tool_dispatch_api_key ?? process.env.METABOB_API_KEY ?? "";
  const maxIter = Math.max(1, Math.min(body.max_tool_iterations ?? DEFAULT_MAX_TOOL_ITERATIONS, 20));
  if (!dispatchApiKey) {
    return { resolved: false, shape: "llmCompletion", error: "tool-use requires METABOB_API_KEY" };
  }

  // Convert Anthropic-style tool defs to OpenAI format
  const oaiTools: OpenAI.Chat.ChatCompletionTool[] = body.tools
    .filter((t) => !t.type || t.type === "custom")
    .map((t) => ({
      type: "function" as const,
      function: {
        name: t.name,
        description: t.description ?? "",
        parameters: (t.input_schema ?? { type: "object", properties: {} }) as Record<string, unknown>,
      },
    }));

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    ...systemMessages,
    { role: "user", content: body.prompt },
  ];
  const toolCalls: ToolCallTraceEntry[] = [];
  let totalInputTokens = 0, totalOutputTokens = 0, finalText = "";

  for (let iter = 1; iter <= maxIter; iter++) {
    let response: OpenAI.Chat.ChatCompletion;
    try {
      response = await openaiClient.chat.completions.create({
        model, max_tokens: maxTokens, messages,
        ...(oaiTools.length > 0 ? { tools: oaiTools } : {}),
      });
    } catch (err) {
      return { resolved: false, shape: "llmCompletion", error: `openai (iter ${iter}): ${err instanceof Error ? err.message : String(err)}`, tool_calls: toolCalls };
    }
    totalInputTokens += response.usage?.prompt_tokens ?? 0;
    totalOutputTokens += response.usage?.completion_tokens ?? 0;

    const choice = response.choices[0];
    if (!choice) break;
    messages.push(choice.message);

    if (choice.finish_reason !== "tool_calls" || !choice.message.tool_calls?.length) {
      finalText = choice.message.content ?? "";
      break;
    }

    const toolResultMessages: OpenAI.Chat.ChatCompletionToolMessageParam[] = [];
    for (const tc of choice.message.tool_calls) {
      if (tc.type !== "function") continue;
      let toolInput: Record<string, unknown>;
      try { toolInput = JSON.parse(tc.function.arguments); } catch { toolInput = {}; }
      const start = Date.now();
      const r = await dispatchTool(dispatchEndpoint, dispatchApiKey, tc.function.name, toolInput);
      toolCalls.push({ iteration: iter, tool_name: tc.function.name, tool_input: toolInput, tool_output: r.ok ? r.result : { error: r.error }, duration_ms: Date.now() - start });
      toolResultMessages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: typeof r.result === "string" ? r.result : JSON.stringify(r.result ?? r.error ?? null),
      });
    }
    messages.push(...toolResultMessages);
  }

  return {
    resolved: true, shape: "llmCompletion", content: finalText,
    provider: "openai", model,
    tool_calls: toolCalls,
    iterations: toolCalls.length > 0 ? toolCalls[toolCalls.length - 1]!.iteration : 0,
    usage: { input_tokens: totalInputTokens, output_tokens: totalOutputTokens },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified resolver handler
// ─────────────────────────────────────────────────────────────────────────────

const llmCompletionHandler: ResolverHandler = async (ctx) => {
  const body = ctx.body as LlmCompletionRequest;

  if (!body.prompt || typeof body.prompt !== "string") {
    return { resolved: false, shape: "llmCompletion", error: "body must include non-empty 'prompt' string" };
  }

  const model = body.model ?? DEFAULT_MODEL;
  const provider = pickProvider(model, body.provider);

  if (!provider) {
    const configured: string[] = [];
    if (anthropic) configured.push("anthropic (ANTHROPIC_API_KEY)");
    if (openaiClient) configured.push("openai (OPENAI_API_KEY)");
    return {
      resolved: false, shape: "llmCompletion",
      error: configured.length === 0
        ? "No LLM provider configured. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY."
        : `No suitable provider for model '${model}'. Configured: ${configured.join(", ")}`,
    };
  }

  if (provider === "anthropic") return resolveWithAnthropic(body);
  return resolveWithOpenAI(body);
};

// ─────────────────────────────────────────────────────────────────────────────
// VesselDaemon
// ─────────────────────────────────────────────────────────────────────────────

const runtime = new ExecutionRuntime({
  attachedVessels: [
    { id: VESSEL_ID, kind: "custom" as never, resolverIds: ["llm_completion"] },
  ],
});

const executor = new ActivityExecutor(runtime);

const resolvers = new Map<string, ResolverHandler>([
  ["llm_completion", llmCompletionHandler],
]);

const daemon = new VesselDaemon({
  port: PORT,
  vesselId: VESSEL_ID,
  vesselName: "LLM Resolver Vessel",
  shapes: ["llm_completion", "llmCompletion"],
  executor,
  resolvers,
  discoveryEndpoint: DISCOVERY_ENDPOINT,
  apiKey: API_KEY,
  version: "0.2.0",
  enforceCompositionChain: false,
});

await daemon.start();

const providers: string[] = [];
if (anthropic) providers.push("anthropic");
if (openaiClient) {
  const base = OPENAI_BASE_URL ?? "https://api.openai.com/v1";
  const label = base.includes("11434") ? "ollama" : base.includes("groq") ? "groq" : base.includes("together") ? "together" : base.includes("1234") ? "lm-studio" : "openai-compat";
  providers.push(`${label} (${base})`);
}
console.log(
  `[llm-resolver-vessel] started port=${PORT} providers=[${providers.join(", ") || "NONE"}] default_model=${DEFAULT_MODEL} discovery=${DISCOVERY_ENDPOINT}`,
);

// ─────────────────────────────────────────────────────────────────────────────
// Periodic GC (Bun 1.3.14 RSS workaround — iter-9, substrate-wide)
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
