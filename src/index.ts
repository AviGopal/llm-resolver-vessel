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
import { selectArm, recordArmOutcome, llmModelPolicyHandler, llmModelPolicyWriteHandler } from "./model-policy.js";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.PORT ?? "8220", 10);
const VESSEL_ID = process.env.LLM_RESOLVER_VESSEL_ID ?? process.env.VESSEL_ID ?? "llm-resolver-vessel";
const DISCOVERY_ENDPOINT = process.env.DISCOVERY_VESSEL_ENDPOINT ?? "http://127.0.0.1:8100";
const API_KEY = process.env.LLM_RESOLVER_VESSEL_API_KEY ?? process.env.METABOB_API_KEY;

// Provider config
// Env values arrive quoted from some generators (VAR="") - a quote-only or empty
// value must read as ABSENT, or we construct clients with bogus credentials.
function cleanEnv(v: string | undefined): string | undefined {
  const t = (v ?? "").trim().replace(/^["']+|["']+$/g, "").trim();
  return t.length > 0 ? t : undefined;
}
const ANTHROPIC_API_KEY = cleanEnv(process.env.ANTHROPIC_API_KEY);
const OPENAI_API_KEY = cleanEnv(process.env.OPENAI_API_KEY);
const OPENAI_BASE_URL = cleanEnv(process.env.OPENAI_BASE_URL); // default: OpenAI; set for Ollama/Groq/etc.
const LLM_PROVIDER = (cleanEnv(process.env.LLM_PROVIDER) ?? "auto") as "anthropic" | "openai" | "auto";

const DEFAULT_MODEL = cleanEnv(process.env.LLM_DEFAULT_MODEL) ?? "claude-sonnet-5";
const DEFAULT_MAX_TOKENS = 4096;

// Retired Anthropic model ids that callers still hardcode and that 404 at the API.
const RETIRED_ANTHROPIC_MODEL_IDS = new Set(["claude-sonnet-4-20250514"]);

// Model prefixes that always route to the OpenAI-compatible path.
const OPENAI_MODEL_PREFIXES = [
  "gpt-", "o1-", "o3-", "o4-",
  "llama", "mistral", "mixtral", "gemma", "phi-",
  "qwen", "deepseek", "yi-", "command-", "nova-", "qwenvn",
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

// OpenAI-wire-compatible provider registry: add a service as data, not a new code path.
interface OpenAiWireProvider { id: string; baseURL: string; apiKeyEnv: string; models: string[]; }
const OPENAI_WIRE_PROVIDERS: OpenAiWireProvider[] = [
  { id: "chutes", baseURL: "https://llm.chutes.ai/v1", apiKeyEnv: "CHUTES_API_KEY",
    models: ["zai-org/GLM-5.1-TEE", "zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"] },
  // OpenRouter serves arbitrary vendor/model ids - it is the catch-all for any
  // slash-qualified model not explicitly mapped above (see resolve entry point).
  // Free-tier openrouter models (verified live 2026-07-14) — the last-resort
  // completion plane when both anthropic and chutes are credit-exhausted.
  // Free slugs drift; keep several and let the exhaustion fallback walk them.
  // PAID openrouter models first (verified live 2026-07-14, real balance) — the
  // sustained-capacity plane: reliable non-reasoning models that don't burn the
  // token budget on hidden reasoning the way the free reasoning models do, and
  // no account-wide free daily cap. The :free models stay LAST as true
  // last-resort (they 429 account-wide under load). Order within openrouter is a
  // cold-start prior, not a learned policy — see gap
  // tool-affordance-not-a-shaped-impulse / dispatch-time selection should be shaped.
  { id: "openrouter", baseURL: "https://openrouter.ai/api/v1", apiKeyEnv: "OPENROUTER_API_KEY",
    models: ["google/gemini-2.5-flash", "openai/gpt-4o-mini", "deepseek/deepseek-chat-v3-0324",
             "nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3-nano-30b-a3b:free", "tencent/hy3:free"] },
  { id: "google", baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/", apiKeyEnv: "GOOGLE_API_KEY",
    models: ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3-flash-preview"] },
  { id: "groq", baseURL: "https://api.groq.com/openai/v1", apiKeyEnv: "GROQ_API_KEY",
    models: ["llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b"] },
  { id: "mistral", baseURL: "https://api.mistral.ai/v1", apiKeyEnv: "MISTRAL_API_KEY",
    models: ["mistral-small-latest", "codestral-latest", "mistral-large-latest"] },
];
const modelClientMap = new Map<string, OpenAI>();

async function syncCompletionAdvertisement(): Promise<void> {
  // Quota-gated advertisement (law: a resolver must not advertise a shape it
  // cannot serve). When every keyed wire model AND the anthropic/openai lanes
  // are in exhaustion cooldown, DROP the completion-serving shapes so discovery
  // routes callers to a producer that still has quota (incl. a remote hub arm)
  // instead of into a dead local arm. The observability/policy shapes stay
  // advertised: llmQuotaState keeps exhaustion visible and llmModelPolicy stays
  // writable regardless of quota.
  //
  // Per-MODEL cooldown means a paid 402 never hides the working :free siblings
  // on the same baseURL — as long as ANY model (paid or :free) is uncooled,
  // hasCompletionQuota() is true and completion stays advertised.
  //
  // Resume is condition-driven, not traffic-driven: a dropped shape receives no
  // completion traffic, so nothing would arrive to clear a passively-expiring
  // cooldown (the advertisement-shrink deadlock). markExhausted /
  // markModelExhausted schedule a re-sync at cooldown expiry via
  // scheduleAdvertisementResume(), and a recovering provider re-syncs on the
  // success path (clearExhausted).
  const shapes = ["llmModelPolicy", "llmModelPolicy_write", "llmQuotaState"];
  if (hasCompletionQuota()) {
    shapes.unshift("llm_completion", "llmCompletion");
  } else {
    console.warn("[llm-resolver-vessel] all completion providers cooling — de-advertising llm_completion until quota returns");
  }
  try {
    await daemon.setShapes(shapes);
  } catch (err) {
    console.warn("[llm-resolver-vessel] syncCompletionAdvertisement: setShapes failed (non-fatal)", err);
  }
}

async function llmQuotaStateHandler(_ctx: { body: unknown }): Promise<{ resolved: boolean; shape: string; body: unknown }> {
  const providers: Record<string, { present: boolean; cooldown_until_ms: number | null }> = {};
  for (const p of OPENAI_WIRE_PROVIDERS) {
    const until = exhaustedUntil.get(p.baseURL) ?? null;
    providers[p.id] = {
      present: !!modelClientMap.get(p.models[0] ?? ""),
      cooldown_until_ms: until && until > Date.now() ? until : null,
    };
  }
  const anthropicUntil = exhaustedUntil.get("anthropic") ?? null;
  providers["anthropic"] = {
    present: anthropic !== null,
    cooldown_until_ms: anthropicUntil && anthropicUntil > Date.now() ? anthropicUntil : null,
  };
  return { resolved: true, shape: "llmQuotaState", body: { providers } };
}
let openrouterClient: OpenAI | null = null;
for (const p of OPENAI_WIRE_PROVIDERS) {
  const key = cleanEnv(process.env[p.apiKeyEnv]);
  if (!key) continue;
  const client = new OpenAI({ apiKey: key, baseURL: p.baseURL });
  for (const m of p.models) modelClientMap.set(m, client);
  if (p.id === "openrouter") openrouterClient = client;
  console.log(`[llm-resolver-vessel] OpenAI-wire provider '${p.id}': ${p.baseURL} (${p.models.length} models)`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider routing
// ─────────────────────────────────────────────────────────────────────────────

// Maps retired model IDs to current equivalents so callers using old hardcoded ids still work.
const RETIRED_MODEL_ALIASES: Record<string, string> = {
  "claude-sonnet-4-20250514": "claude-sonnet-5",
};

function remapModel(model: string): string {
  return RETIRED_MODEL_ALIASES[model] ?? model;
}

function pickProvider(model: string, explicitProvider?: string): "anthropic" | "openai" | null {
  model = remapModel(model);
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
const DEFAULT_MAX_TOOL_ITERATIONS = parseInt(process.env.LLM_MAX_TOOL_ITERATIONS ?? "20", 10);

const TOOL_ENDPOINT_CACHE = new Map<string, string>();

async function resolveToolEndpoint(toolName: string, fallback: string): Promise<string> {
  const cached = TOOL_ENDPOINT_CACHE.get(toolName);
  if (cached) return cached;
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (API_KEY) headers.Authorization = `ApiKey ${API_KEY}`;
    const res = await fetch(`${DISCOVERY_ENDPOINT}/resolve`, {
      method: "POST",
      headers,
      body: JSON.stringify({ pointer: { type: "vesselCapability", shape: toolName } }),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) {
      TOOL_ENDPOINT_CACHE.set(toolName, fallback);
      return fallback;
    }
    const body = (await res.json()) as { content?: { vessels?: Array<{ endpoint?: string; resolve_endpoint?: string }> } };
    const vessel = body?.content?.vessels?.[0];
    if (!vessel || !vessel.endpoint || !vessel.resolve_endpoint) {
      TOOL_ENDPOINT_CACHE.set(toolName, fallback);
      return fallback;
    }
    const re = vessel.resolve_endpoint;
    const url = re.startsWith("http://") || re.startsWith("https://")
      ? re
      : `${vessel.endpoint.replace(/\/$/, "")}${re.startsWith("/") ? re : "/" + re}`;
    TOOL_ENDPOINT_CACHE.set(toolName, url);
    return url;
  } catch {
    TOOL_ENDPOINT_CACHE.set(toolName, fallback);
    return fallback;
  }
}

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
  task_type?: string;
  type: "llm_completion";
  prompt: string;
  model?: string;
  provider?: "anthropic" | "openai" | "auto";
  max_tokens?: number;
  system?: string;
  images?: Array<{ media_type: string; data: string }>;
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
async function anthropicCreditFallback(
  err: unknown, body: LlmCompletionRequest): Promise<Record<string, unknown> | null> {
  if (Array.isArray(body.tools) && body.tools.length > 0) return null;
  const message = err instanceof Error ? err.message : String(err);
  const status = (err as { status?: number } | null)?.status;
  const errType = (err as { error?: { type?: string; message?: string } } | null)?.error?.type;
  const errMsg = (err as { error?: { type?: string; message?: string } } | null)?.error?.message ?? "";
  const combined = `${message} ${errMsg}`.toLowerCase();
  const isCreditDead = combined.includes("credit balance is too low") ||
    (status === 400 && errType === "invalid_request_error" && (combined.includes("billing") || combined.includes("credit")));
  if (!isCreditDead) return null;
  const fallbackModel = process.env.LLM_FALLBACK_MODEL;
  if (!fallbackModel) return null;
  const client = modelClientMap.get(fallbackModel);
  if (!client) return null;
  console.log(`[llm-resolver-vessel] anthropic credit-dead — falling back to ${fallbackModel}`);
  try {
    const messages: Array<{ role: "system" | "user"; content: string }> = [];
    if (body.system) messages.push({ role: "system", content: body.system });
    messages.push({ role: "user", content: body.prompt ?? "" });
    const completion = await client.chat.completions.create({
      model: fallbackModel,
      max_tokens: body.max_tokens ?? DEFAULT_MAX_TOKENS,
      messages,
    });
    const content = completion.choices?.[0]?.message?.content ?? "";
    return {
      resolved: true, shape: "llmCompletion", content,
      provider: "openai-wire", model: fallbackModel,
      fallback_from: "anthropic-credit",
    };
  } catch (fallbackErr) {
    console.error("[llm-resolver-vessel] fallback error:", fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr));
    return null;
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Prompt caching (Anthropic path)
//
// Caching is a byte-prefix match; render order is tools → system → messages.
// The tool-use loop below re-sends the full growing history every iteration
// (up to max_tool_iterations), so caching the shared prefix is the real win:
// system gets a stable breakpoint, and a single sliding breakpoint rides the
// last block of the last message each iteration (max 4 markers per request —
// we use at most 2). Prefixes below the model's minimum (1k–4k tokens)
// silently don't cache, which is harmless. Cache reads report in
// usage.cache_read_input_tokens.
// ─────────────────────────────────────────────────────────────────────────────

const CACHE_EPHEMERAL = { type: "ephemeral" as const };

function cachedSystem(system: string | undefined): Record<string, unknown> {
  if (!system) return {};
  return { system: [{ type: "text", text: system, cache_control: CACHE_EPHEMERAL }] };
}

// Move the message-side breakpoint to the last block of the last message,
// clearing any marker set on a previous iteration so we never exceed the
// 4-breakpoint request limit as the loop grows the history.
function slideCacheBreakpoint(messages: Array<{ role: string; content: unknown }>): void {
  for (const m of messages) {
    if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (b && typeof b === "object" && "cache_control" in (b as Record<string, unknown>)) {
          delete (b as Record<string, unknown>)["cache_control"];
        }
      }
    }
  }
  const last = messages[messages.length - 1];
  if (last && Array.isArray(last.content) && last.content.length > 0) {
    const b = last.content[last.content.length - 1];
    if (b && typeof b === "object") (b as Record<string, unknown>)["cache_control"] = CACHE_EPHEMERAL;
  }
}

async function resolveWithAnthropic(body: LlmCompletionRequest): Promise<Record<string, unknown>> {
  if (!anthropic) {
    return { resolved: false, shape: "llmCompletion", error: "ANTHROPIC_API_KEY not configured" };
  }
  const rawModel = body.model ?? DEFAULT_MODEL;
  const stripped = rawModel.startsWith("anthropic/") ? rawModel.slice("anthropic/".length) : rawModel;
  const model = RETIRED_ANTHROPIC_MODEL_IDS.has(stripped) ? DEFAULT_MODEL : stripped;
  const maxTokens = body.max_tokens ?? DEFAULT_MAX_TOKENS;
  const userContent: any = body.images && body.images.length > 0
    ? [
        ...body.images.map((img) => ({ type: "image", source: { type: "base64", media_type: img.media_type, data: img.data } })),
        { type: "text", text: body.prompt },
      ]
    : body.prompt;

  if (!body.tools || body.tools.length === 0) {
    try {
      const response = await anthropic.messages.create({
        model, max_tokens: maxTokens,
        ...cachedSystem(body.system),
        messages: [{ role: "user", content: userContent }],
      });
      const content = response.content
        .filter((b) => b.type === "text")
        .map((b) => (b as { type: "text"; text: string }).text)
        .join("");
      return {
        resolved: true, shape: "llmCompletion", content,
        provider: "anthropic", model,
        usage: {
          input_tokens: response.usage.input_tokens,
          output_tokens: response.usage.output_tokens,
          cache_creation_input_tokens: (response.usage as { cache_creation_input_tokens?: number }).cache_creation_input_tokens ?? 0,
          cache_read_input_tokens: (response.usage as { cache_read_input_tokens?: number }).cache_read_input_tokens ?? 0,
        },
        stop_reason: response.stop_reason,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const fallback = await anthropicCreditFallback(err, body);
      if (fallback) return fallback;
      console.error("[llm-resolver-vessel] anthropic error:", message);
      return { resolved: false, shape: "llmCompletion", error: message };
    }
  }

  // Tool-use loop (Anthropic)
  const dispatchEndpoint = body.tool_dispatch_endpoint ?? DEFAULT_TOOL_DISPATCH_ENDPOINT;
  const dispatchApiKey = body.tool_dispatch_api_key ?? process.env.METABOB_API_KEY ?? "";
  const maxIter = Math.max(1, Math.min(body.max_tool_iterations ?? DEFAULT_MAX_TOOL_ITERATIONS, 30));
  const hasClientSideTools = body.tools.some((t) => !t.type || t.type === "custom");
  if (hasClientSideTools && !dispatchApiKey) {
    return { resolved: false, shape: "llmCompletion", error: "tool-use with client-side tools requires METABOB_API_KEY" };
  }

  const messages: Array<{ role: "user" | "assistant"; content: unknown }> = [
    {
      role: "user",
      content: Array.isArray(userContent)
        ? userContent
        : [{ type: "text", text: userContent as string }],
    },
  ];
  const toolCalls: ToolCallTraceEntry[] = [];
  let totalInputTokens = 0, totalOutputTokens = 0, finalText = "";
  let totalCacheReadTokens = 0, totalCacheWriteTokens = 0;

  for (let iter = 1; iter <= maxIter; iter++) {
    let response;
    try {
      slideCacheBreakpoint(messages);
      response = await anthropic.messages.create({
        model, max_tokens: maxTokens,
        tools: body.tools as unknown as Anthropic.Messages.Tool[],
        ...cachedSystem(body.system),
        messages: messages as unknown as Anthropic.Messages.MessageParam[],
      });
    } catch (err) {
      return { resolved: false, shape: "llmCompletion", error: `anthropic (iter ${iter}): ${err instanceof Error ? err.message : String(err)}`, tool_calls: toolCalls };
    }
    totalInputTokens += response.usage.input_tokens;
    totalOutputTokens += response.usage.output_tokens;
    totalCacheReadTokens += (response.usage as { cache_read_input_tokens?: number }).cache_read_input_tokens ?? 0;
    totalCacheWriteTokens += (response.usage as { cache_creation_input_tokens?: number }).cache_creation_input_tokens ?? 0;
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
    usage: {
      input_tokens: totalInputTokens,
      output_tokens: totalOutputTokens,
      cache_creation_input_tokens: totalCacheWriteTokens,
      cache_read_input_tokens: totalCacheReadTokens,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAI-compatible resolution path (OpenAI, Ollama, Groq, Together, vLLM, …)
// ─────────────────────────────────────────────────────────────────────────────

async function resolveWithOpenAI(body: LlmCompletionRequest, client: OpenAI | null = openaiClient): Promise<Record<string, unknown>> {
  if (!client) {
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
      const response = await client.chat.completions.create({
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
        stop_reason: response.choices[0]?.finish_reason === "length" ? "max_tokens" : response.choices[0]?.finish_reason,
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
  const maxIter = Math.max(1, Math.min(body.max_tool_iterations ?? DEFAULT_MAX_TOOL_ITERATIONS, 30));
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
      response = await client.chat.completions.create({
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

// ─────────────────────────────────────────────────────────────────────────────
// Cross-request provider-exhaustion memory (auto-failover + auto-recovery).
// A provider that fails with a billing/quota error is skipped for a cooldown
// window instead of being retried on every request; after the window expires
// it is tried again automatically, and a success clears the mark — so when a
// key regains balance the substrate reverts to its preferred provider with no
// operator action. Cooldown is bootstrap-tunable via LLM_EXHAUSTION_COOLDOWN_MS.
const EXHAUSTION_COOLDOWN_MS =
  Number.parseInt(process.env.LLM_EXHAUSTION_COOLDOWN_MS ?? "", 10) > 0
    ? Number.parseInt(process.env.LLM_EXHAUSTION_COOLDOWN_MS ?? "", 10)
    : 10 * 60 * 1000;
const exhaustedUntil = new Map<string, number>();
const providerKeyOf = (client: OpenAI): string => String((client as { baseURL?: unknown }).baseURL ?? "openai-wire");
const inCooldown = (key: string): boolean => (exhaustedUntil.get(key) ?? 0) > Date.now();
const markExhausted = (key: string): void => {
  exhaustedUntil.set(key, Date.now() + EXHAUSTION_COOLDOWN_MS);
  console.warn(`[llm-resolver-vessel] provider '${key}' marked exhausted — cooling down ${Math.round(EXHAUSTION_COOLDOWN_MS / 1000)}s before retry`);
  void syncCompletionAdvertisement();
  scheduleAdvertisementResume();
};
const clearExhausted = (key: string): void => {
  if (exhaustedUntil.delete(key)) {
    console.warn(`[llm-resolver-vessel] provider '${key}' recovered — resuming preferred routing`);
    void syncCompletionAdvertisement();
  }
};
const isExhaustedProviderError = (e: unknown): boolean => {
  const m = String(e ?? "").toLowerCase();
  return (
    m.includes("credit balance") ||
    m.includes("insufficient_quota") ||
    m.includes("exceeded your current quota") ||
    m.includes("usage cap") ||
    // 402 as a standalone status code only — a bare substring match would
    // false-positive on token counts like "140250" and cool a healthy
    // provider down for the whole window.
    /(?:^|[^0-9])402(?:[^0-9]|$)/.test(m) ||
    m.includes("billing") ||
    m.includes("limit_rpd") ||
    m.includes("limit_rpm") ||
    m.includes("daily limit reached") ||
    m.includes("rate limit") ||
    m.includes("too many requests") ||
    m.includes("free-models-per-day") ||
    // 429 as a standalone status code only (regex-guarded like 402 above so a
    // token count like "14290" cannot false-positive a healthy provider).
    /(?:^|[^0-9])429(?:[^0-9]|$)/.test(m) ||
    // A "free" model that 404s with "unavailable for free / use the paid slug /
    // no endpoints" (openrouter free-tier drift) is dead to US — cool the MODEL
    // so the failover walk fires and the quota-gate de-advertises the arm,
    // instead of sinking the dispatch in ~30ms and keeping a dead arm falsely
    // advertised (which strands a spoke on its dead local arm instead of routing
    // through discovery to a quota-having producer). Narrowed to a standalone 404
    // AND a paid/unavailable phrase so a genuine wrong-model 404 on a healthy
    // provider does not cool the whole lane.
    (/(?:^|[^0-9])404(?:[^0-9]|$)/.test(m) &&
      (m.includes("unavailable") ||
        m.includes("paid") ||
        m.includes("use this slug") ||
        m.includes("no endpoints")))
  );
};

// Model-granular exhaustion. Provider-level cooldowns condemn every model on a
// baseURL when one 402s — which hid the working :free openrouter models behind
// a paid-model 402 (2026-07-18 completion-plane outage). Cool down the failing
// MODEL; siblings on the same client stay eligible.
const modelExhaustedUntil = new Map<string, number>();
const inModelCooldown = (m: string): boolean => (modelExhaustedUntil.get(m) ?? 0) > Date.now();

// Quota gate for advertisement: any uncooled keyed wire model, or the
// anthropic/openai lanes still outside cooldown, means completion is servable.
// Mirrors the availableModels expression used for policy arm selection
// (llmCompletionWithPolicyHandler) so the advertised set and the routable set
// never disagree — and it is model-granular, so :free siblings keep the shape
// advertised even when a paid model on the same baseURL is cooling.
const hasCompletionQuota = (): boolean =>
  [...modelClientMap.keys()].some((m) => !inModelCooldown(m)) ||
  (anthropic !== null && !inCooldown("anthropic")) ||
  (openaiClient !== null && !inCooldown("openai"));

// Condition-driven resume of a de-advertised completion plane. A dropped shape
// receives no traffic, so nothing would ever clear a passively-expiring model
// cooldown — re-run the advertisement sync once the exhaustion window lapses.
// One pending timer at a time; unref so it never holds the process open.
let advertisementResumeTimer: ReturnType<typeof setTimeout> | null = null;
const scheduleAdvertisementResume = (): void => {
  if (advertisementResumeTimer !== null) return;
  advertisementResumeTimer = setTimeout(() => {
    advertisementResumeTimer = null;
    void syncCompletionAdvertisement();
  }, EXHAUSTION_COOLDOWN_MS + 1000);
  (advertisementResumeTimer as { unref?: () => void }).unref?.();
};
const markModelExhausted = (m: string): void => {
  modelExhaustedUntil.set(m, Date.now() + EXHAUSTION_COOLDOWN_MS);
  console.warn(`[llm-resolver-vessel] model '${m}' marked exhausted — cooling down ${Math.round(EXHAUSTION_COOLDOWN_MS / 1000)}s`);
  void syncCompletionAdvertisement();
  scheduleAdvertisementResume();
};

// Shared billing-exhaustion fallback: walk every keyed wire model not in its
// own cooldown, in registry order (paid tiers first, :free last resort).
const walkFallbackModels = async (
  body: LlmCompletionRequest,
  fromModel: string,
): Promise<Record<string, unknown> | null> => {
  const tried = new Set<string>([fromModel]);
  for (const [fbModel, client] of modelClientMap) {
    if (tried.has(fbModel) || inModelCooldown(fbModel)) continue;
    tried.add(fbModel);
    console.warn(`[llm-resolver-vessel] provider exhausted for '${fromModel}' — falling back to '${fbModel}'`);
    const fb = await resolveWithOpenAI({ ...body, model: fbModel }, client);
    if (fb.resolved === true) {
      clearExhausted(providerKeyOf(client));
      return { ...fb, fallback_from: fromModel };
    }
    if (isExhaustedProviderError(fb.error)) markModelExhausted(fbModel);
  }
  return null;
};

const llmCompletionHandler: ResolverHandler = async (ctx) => {
  const body = ctx.body as LlmCompletionRequest;

  if (!body.prompt || typeof body.prompt !== "string") {
    return { resolved: false, shape: "llmCompletion", error: "body must include non-empty 'prompt' string" };
  }

  const model = body.model ?? DEFAULT_MODEL;
  const wireClient = modelClientMap.get(model);
  if (wireClient) {
    if (inModelCooldown(model)) {
      const walked = await walkFallbackModels(body, model);
      if (walked) return walked;
    }
    const r = await resolveWithOpenAI(body, wireClient);
    if (r.resolved === true || !isExhaustedProviderError(r.error)) return r;
    markModelExhausted(model);
    const walked = await walkFallbackModels(body, model);
    return walked ?? r;
  }
  // Any vendor/model id not explicitly mapped routes through OpenRouter when keyed.
  if (openrouterClient && model.includes("/") && !model.toLowerCase().startsWith("anthropic/")) {
    const r = await resolveWithOpenAI(body, openrouterClient);
    if (r.resolved === true || !isExhaustedProviderError(r.error)) return r;
    markModelExhausted(model);
    const walked = await walkFallbackModels(body, model);
    return walked ?? r;
  }
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

  // Billing/quota-exhaustion fallback (credit-outage law: use ALL keyed
  // providers; a dead primary must not take the whole substrate's LLM plane
  // down with it). Providers in exhaustion cooldown are skipped up front and
  // retried automatically after the window — auto-failover, auto-recovery.
  const primaryKey = provider;
  const primaryCoolingDown = inCooldown(primaryKey);
  let result: Record<string, unknown>;
  if (primaryCoolingDown) {
    result = { resolved: false, shape: "llmCompletion", error: `${primaryKey} cooling down after credit/quota exhaustion (auto-retries after cooldown)` };
  } else {
    result = provider === "anthropic" ? await resolveWithAnthropic(body) : await resolveWithOpenAI(body);
    if (result.resolved === true) {
      clearExhausted(primaryKey);
    } else if (isExhaustedProviderError(result.error)) {
      markExhausted(primaryKey);
    }
  }
  if (result.resolved !== true && (primaryCoolingDown || isExhaustedProviderError(result.error))) {
    const walked = await walkFallbackModels(body, model);
    if (walked) return walked;
    // Last resort: every fallback lane is dead or cooling down — try the
    // primary anyway even mid-cooldown (its balance may have just returned).
    if (primaryCoolingDown) {
      const retry = provider === "anthropic" ? await resolveWithAnthropic(body) : await resolveWithOpenAI(body);
      if (retry.resolved === true) {
        clearExhausted(primaryKey);
        return retry;
      }
      if (isExhaustedProviderError(retry.error)) markExhausted(primaryKey);
      return retry;
    }
  }
  return result;
};

// ─────────────────────────────────────────────────────────────────────────────
// VesselDaemon
// ─────────────────────────────────────────────────────────────────────────────

// Policy-selected completion: when the caller does not pin a model (absent or
// "auto"), select one by cost/reach/uncertainty (Thompson over the shaped
// llmModelPolicy) and record the choice + technical outcome. Pinned calls pass
// through ungraded (causal discipline).
const llmCompletionWithPolicyHandler: ResolverHandler = async (ctx) => {
  const body = ctx.body as LlmCompletionRequest;
  const pinned = typeof body.model === "string" && body.model.length > 0 && body.model !== "auto";
  if (pinned) return llmCompletionHandler(ctx);
  const availableModels = [...[...modelClientMap.keys()].filter(m => !inModelCooldown(m)), ...(anthropic && !inCooldown("anthropic") ? ["claude-sonnet-5","claude-haiku-4-5-20251001"] : [])];
const sel = await selectArm(body.task_type, availableModels);
  if (!sel) return llmCompletionHandler(ctx);
  const result = await llmCompletionHandler({ ...ctx, body: { ...body, model: sel.model } } as never);
  try {
    (result as Record<string, unknown>).model_selection = sel.meta;
    await recordArmOutcome(sel.model, (result as { resolved?: boolean }).resolved === true, body.task_type);
  } catch (err) {
    console.warn("[llm-resolver-vessel] policy outcome record failed (non-fatal):", err);
  }
  return result;
};

const runtime = new ExecutionRuntime({
  attachedVessels: [
    { id: VESSEL_ID, kind: "custom" as never, resolverIds: ["llm_completion"] },
  ],
});

const executor = new ActivityExecutor(runtime);

const resolvers = new Map<string, ResolverHandler>([
  ["llm_completion", llmCompletionWithPolicyHandler],
  ["llmModelPolicy", llmModelPolicyHandler as never],
  ["llmModelPolicy_write", llmModelPolicyWriteHandler as never],
  ["llmQuotaState", llmQuotaStateHandler as never],
]);

const daemon = new VesselDaemon({
  port: PORT,
  vesselId: VESSEL_ID,
  vesselName: "LLM Resolver Vessel",
  shapes: ["llm_completion", "llmCompletion", "llmModelPolicy", "llmModelPolicy_write", "llmQuotaState"],
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
