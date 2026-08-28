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
import { classifyPlane, type ProviderState } from "./plane-outage";
import { isScaleToZeroCold, isWarmFromHealth } from "./scale-to-zero";
import OpenAI from "openai";
import {
  ActivityExecutor,
  ExecutionRuntime,
  VesselDaemon,
} from "@avigopal/ias-executor-ts";
import type { ResolverHandler } from "@avigopal/ias-executor-ts";
import { isExhaustedProviderError, isUnreachableProviderError, isFailoverError, isUnauthenticatedProviderError } from "./provider-errors.js";
import { selectArm, recordArmOutcome, loadPolicy, llmModelPolicyHandler, llmModelPolicyWriteHandler, ensureArmsForModels, repriceSeededArm } from "./model-policy.js";
import { decideLastResort } from "./last-resort.js";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.PORT ?? "8220", 10);
const VESSEL_ID = process.env.LLM_RESOLVER_VESSEL_ID ?? process.env.VESSEL_ID ?? "llm-resolver-vessel";
const DISCOVERY_ENDPOINT = process.env.DISCOVERY_VESSEL_ENDPOINT ?? "http://127.0.0.1:8100";
const API_KEY = process.env.LLM_RESOLVER_VESSEL_API_KEY ?? process.env.METABOB_API_KEY;
// Bootstrap-tier fallback only: discovery owns the real address for substrateGap_write and is
// consulted first. This literal exists so a detector still has somewhere to report when the
// registry itself is the thing that is unreachable.
const DEV_VESSEL_FALLBACK = process.env.DEVELOPMENT_VESSEL_ENDPOINT ?? "http://127.0.0.1:8090";

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
const LLM_PINNED_PROVIDER = cleanEnv(process.env.LLM_PINNED_PROVIDER) ?? null;

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

// RunPod Serverless — a vendor wire host (api.runpod.ai) whose path carries a
// per-deployment endpoint id, so only the id, the served model ids and the key
// are operator input; the URL shape is fixed and belongs in the registry below
// like any other vendor. Distinct from the VLLM_* self-hosted forms further
// down, which point at a per-instance hostname behind a tunnel.
//
// This capacity SCALES TO ZERO. When no worker is up, the first request pays an
// image-pull + engine-load tail measured at 20-40 min on these baked images —
// so the arm is only offered while some worker is already warm (see
// isRunpodCold / isModelWilling). cost_per_mtok is therefore the honest WARM
// price; "cold" is expressed as not-routable, never as an inflated price,
// because the policy's cost term is bounded (cost_weight, 0.25) while a Beta
// draw spans [0,1] — a merely-expensive cold arm would still be explored, and
// each such pick would burn a 20-40 min timeout and charge the loss to the
// model's reach evidence rather than to the cold start.
const RUNPOD_ENDPOINT_ID = cleanEnv(process.env.RUNPOD_ENDPOINT_ID);
// Empty unless an endpoint is configured. The model-id default must NOT leak
// out when RUNPOD_ENDPOINT_ID is unset: these ids also name models a self-hosted
// VLLM_ENDPOINTS instance may serve, and a non-empty set here would gate that
// perfectly-warm arm on the readiness of an endpoint that does not exist.
const RUNPOD_MODELS = RUNPOD_ENDPOINT_ID
  ? (cleanEnv(process.env.RUNPOD_MODELS) ?? "Qwen/Qwen3-Coder-Next-FP8")
      .split(",").map((m) => m.trim()).filter(Boolean)
  : [];
// Operator-tunable prior, NOT a measured rate: RunPod publishes no per-token
// price for serverless (billing is GPU-seconds), so this seeds the arm in the
// same band as the cheap hosted arms until real cost/throughput evidence lands.
// Must fall back on a non-finite parse, not pass NaN through: NaN survives into
// the arm, JSON.stringify writes it as null, and `cost_per_mtok ?? 0` then reads
// back a ZERO-cost arm — the outbid-everything trap this design exists to avoid,
// reachable from a single typo'd env var.
const RUNPOD_COST_PER_MTOK = ((): number => {
  const n = Number(cleanEnv(process.env.RUNPOD_COST_PER_MTOK) ?? "0.35");
  return Number.isFinite(n) && n >= 0 ? n : 0.35;
})();

// OpenAI-wire-compatible provider registry: add a service as data, not a new code path.
// defaultKey lets a keyless self-hosted endpoint (vLLM with no --api-key) still
// construct an OpenAI client, which requires a non-empty apiKey string.
interface OpenAiWireProvider { id: string; baseURL: string; apiKeyEnv: string; models: string[]; defaultKey?: string; }
const OPENAI_WIRE_PROVIDERS: OpenAiWireProvider[] = [
  { id: "chutes", baseURL: "https://llm.chutes.ai/v1", apiKeyEnv: "CHUTES_API_KEY",
    models: ["zai-org/GLM-5.1-TEE", "zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"] },
  // FUNDED quota providers (groq + mistral) placed HIGH so the exhaustion
  // failover walk reaches them before the rate-limited openrouter-free / gemini
  // lanes — under load the fleet was burning all four gemini models (each 45s
  // cooling) before ever trying groq's llama-3.3, leaving the funded quota idle.
  { id: "groq", baseURL: "https://api.groq.com/openai/v1", apiKeyEnv: "GROQ_API_KEY",
    models: ["llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b"] },
  { id: "mistral", baseURL: "https://api.mistral.ai/v1", apiKeyEnv: "MISTRAL_API_KEY",
    models: ["mistral-small-latest", "codestral-latest", "mistral-large-latest"] },
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
             "nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3-nano-30b-a3b:free",
             // `tencent/hy3:free` was listed here and is NOT offered by OpenRouter
             // (verified against GET /api/v1/models, 400 models, 14 of them `:free`).
             // A configured id the provider does not serve is a PHANTOM ARM: it can
             // never succeed, yet it holds a policy posterior and burns a failover hop
             // every time it is drawn. Replaced with a verified code-oriented free slug.
             "cohere/north-mini-code:free"] },
  { id: "google", baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/", apiKeyEnv: "GOOGLE_API_KEY",
    models: ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3-flash-preview"] },
  // LAST in the registry on purpose: the exhaustion failover walk reads this
  // order, and scale-to-zero capacity is the one lane that must never be
  // reached ahead of an always-on provider.
  ...(RUNPOD_ENDPOINT_ID ? [{
    id: "runpod-serverless",
    baseURL: `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/openai/v1`,
    apiKeyEnv: "RUNPOD_API_KEY",
    models: RUNPOD_MODELS,
  }] : []),
];

// Self-hosted vLLM endpoints (Vast.ai / RunPod behind a Cloudflare Tunnel) are
// operator/deploy-specific: the baseURL is a per-instance stable hostname, not a
// fixed vendor URL, so they are injected via env rather than hardcoded here. Two
// forms (both may be set; VLLM_ENDPOINTS entries come first):
//   VLLM_ENDPOINTS  = JSON array [{id, baseURL, models, apiKeyEnv?}]   (multi-instance)
//   VLLM_BASE_URL + VLLM_MODELS (+ optional VLLM_API_KEY)              (single instance)
// Once appended, these behave as any other OpenAI-wire provider — modelClientMap,
// the exhaustion failover walk, quota-gated advertisement all apply unchanged.
// vLLM served-model ids come from the instance's MODEL / SERVED_MODEL_NAME; list
// exactly those here so callers can pin them and the policy can select them.
function loadSelfHostedVllmProviders(): OpenAiWireProvider[] {
  const out: OpenAiWireProvider[] = [];
  const splitModels = (s: string): string[] => s.split(",").map((m) => m.trim()).filter(Boolean);

  const endpointsRaw = cleanEnv(process.env.VLLM_ENDPOINTS);
  if (endpointsRaw) {
    try {
      const parsed = JSON.parse(endpointsRaw) as Array<{ id?: string; baseURL?: string; models?: string[] | string; apiKeyEnv?: string }>;
      for (const [i, e] of parsed.entries()) {
        const baseURL = cleanEnv(e.baseURL);
        const models = Array.isArray(e.models) ? e.models : splitModels(e.models ?? "");
        if (!baseURL || models.length === 0) {
          console.warn(`[llm-resolver-vessel] VLLM_ENDPOINTS[${i}] missing baseURL or models — skipping`);
          continue;
        }
        out.push({ id: cleanEnv(e.id) ?? `vllm-${i}`, baseURL, models, apiKeyEnv: e.apiKeyEnv ?? "VLLM_API_KEY", defaultKey: "EMPTY" });
      }
    } catch (err) {
      console.warn("[llm-resolver-vessel] VLLM_ENDPOINTS is not valid JSON — ignoring", err);
    }
  }

  const singleBase = cleanEnv(process.env.VLLM_BASE_URL);
  if (singleBase) {
    const models = splitModels(cleanEnv(process.env.VLLM_MODELS) ?? "");
    if (models.length === 0) {
      console.warn("[llm-resolver-vessel] VLLM_BASE_URL set but VLLM_MODELS empty — skipping single vLLM endpoint");
    } else {
      out.push({ id: cleanEnv(process.env.VLLM_ID) ?? "vllm", baseURL: singleBase, models, apiKeyEnv: "VLLM_API_KEY", defaultKey: "EMPTY" });
    }
  }
  return out;
}

const SELF_HOSTED_VLLM_PROVIDERS = loadSelfHostedVllmProviders();
for (const p of SELF_HOSTED_VLLM_PROVIDERS) {
  OPENAI_WIRE_PROVIDERS.push(p);
  console.log(`[llm-resolver-vessel] self-hosted vLLM provider '${p.id}': ${p.baseURL} (${p.models.length} models)`);
}
const SELF_HOSTED_VLLM_MODELS = [...new Set(SELF_HOSTED_VLLM_PROVIDERS.flatMap((p) => p.models))];

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
  // De-advertising is the correct REACTION and a poor DETECTION: the registry quietly loses a
  // shape and every caller degrades in silence. Observed 2026-08-07 — the whole plane was
  // credit-dead for the length of a session, reach grading fell back to deterministic oracles
  // only, and no gap was ever filed. File one (law 6).
  void reportPlaneState();
  try {
    await daemon.setShapes(shapes);
  } catch (err) {
    console.warn("[llm-resolver-vessel] syncCompletionAdvertisement: setShapes failed (non-fatal)", err);
  }
}

/**
 * File (or close) the plane-dark gap from the SAME provider table `llmQuotaState` serves, so
 * the gap and the observable state can never disagree.
 *
 * Fails open and silent by design: this vessel's job is serving completions, and a gap-store
 * hiccup must never interfere with that. It is a detector, not a dependency.
 */
let lastPlaneDark: boolean | null = null;
async function reportPlaneState(): Promise<void> {
  try {
    const state = (await llmQuotaStateHandler({ body: null })).body as { providers: Record<string, ProviderState> };
    const verdict = classifyPlane(state.providers ?? {}, Date.now());

    // Only write on a TRANSITION. A steady-state outage re-emitting on every advertisement
    // sync would bury the store under one row's worth of noise — the flood the stable gap ids
    // elsewhere in the fleet exist to avoid.
    if (lastPlaneDark === verdict.dark) return;
    lastPlaneDark = verdict.dark;

    const endpoint = await resolveToolEndpoint("substrateGap_write", `${DEV_VESSEL_FALLBACK}/v2/impulses/resolve`);
    const gap = verdict.dark
      ? { id: verdict.id, category: "infrastructure", source: "substrate_detected", status: "open", summary: verdict.summary, detected_at: new Date().toISOString() }
      : { id: "llm-completion-plane-dark", category: "infrastructure", source: "substrate_detected", status: "closed", summary: `[closed] The LLM completion plane recovered — ${verdict.reason}.`, detected_at: new Date().toISOString() };
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (API_KEY) headers.Authorization = `ApiKey ${API_KEY}`;
    await fetch(endpoint, {
      method: "POST", headers,
      body: JSON.stringify({ impulse: { type: "substrateGap_write", pointer: { type: "substrateGap_write", gap } } }),
      signal: AbortSignal.timeout(10_000),
    });
    console.log(`[llm-resolver-vessel] plane ${verdict.dark ? "DARK — filed" : "recovered — closed"} gap llm-completion-plane-dark`);
  } catch (err) {
    console.warn("[llm-resolver-vessel] reportPlaneState failed (non-fatal)", err instanceof Error ? err.message : String(err));
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
  // Self-hosted endpoints (defaultKey set) stay eligible even when their key env
  // is absent — a keyless vLLM ignores the token but the SDK needs a non-empty one.
  const key = cleanEnv(process.env[p.apiKeyEnv]) ?? p.defaultKey;
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
  // Filter by willingness, not merely by "a client exists": the credit-dead
  // path fires exactly when the fleet is already degraded, which is the worst
  // moment to fall back onto a lane that is cooling down or cold-started.
  const fallbackModel = (await selectArm("credit_dead_fallback", [...modelClientMap.keys()].filter(isModelWilling)))?.model;
  if (!fallbackModel) { markExhausted("anthropic", 30 * 60_000); return null; }
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
// Reachability failures (self-hosted spot instance down, gateway 5xx, DNS/TCP
// error) get a much SHORTER cooldown than billing exhaustion: a flapping Vast/
// RunPod endpoint should return to rotation in seconds once it's back, not sit
// out the full 10-minute credit-outage window.
const UNREACHABLE_COOLDOWN_MS =
  Number.parseInt(process.env.LLM_UNREACHABLE_COOLDOWN_MS ?? "", 10) > 0
    ? Number.parseInt(process.env.LLM_UNREACHABLE_COOLDOWN_MS ?? "", 10)
    : 30 * 1000;
// Transient rate-limits (429 / RPM / provider "overloaded") are NOT credit
// exhaustion: they clear in seconds. Under a thundering herd — a freshly-resumed
// arm taking the whole fleet's pent-up demand at once — every provider 429s, and
// giving those the 10-minute EXHAUSTION cooldown darkens the local arm for 10
// minutes per stampede (the observed flapping: it re-advertised, was swamped,
// re-exhausted its entire stack in ~4s, and went dark again). A short rate-limit
// cooldown lets it rejoin rotation in under a minute so the demand spreads across
// arms instead of collapsing onto the single funded hub producer.
const RATE_LIMIT_COOLDOWN_MS =
  Number.parseInt(process.env.LLM_RATE_LIMIT_COOLDOWN_MS ?? "", 10) > 0
    ? Number.parseInt(process.env.LLM_RATE_LIMIT_COOLDOWN_MS ?? "", 10)
    : 45 * 1000;
const exhaustedUntil = new Map<string, number>();
const providerKeyOf = (client: OpenAI): string => String((client as { baseURL?: unknown }).baseURL ?? "openai-wire");
const inCooldown = (key: string): boolean => (exhaustedUntil.get(key) ?? 0) > Date.now();
const markExhausted = (key: string, ms: number = EXHAUSTION_COOLDOWN_MS): void => {
  // MONOTONIC COOLDOWN. Never shorten a window already in force. A credit-dead
  // provider is deliberately cooled for 30 minutes by anthropicCreditFallback,
  // but the generic cooldownMsFor() call in the resolve path then re-marked the
  // same key with the 10-minute default and CLOBBERED it — so a permanently dead
  // lane was retried three times more often than intended, and every retry
  // re-flipped the llm_completion advertisement through discovery. Only extend.
  const until = Date.now() + ms;
  if ((exhaustedUntil.get(key) ?? 0) >= until) return;
  exhaustedUntil.set(key, until);
  console.warn(`[llm-resolver-vessel] provider '${key}' marked exhausted — cooling down ${Math.round(ms / 1000)}s before retry`);
  void syncCompletionAdvertisement();
  scheduleAdvertisementResume(ms);
};
const clearExhausted = (key: string): void => {
  if (exhaustedUntil.delete(key)) {
    console.warn(`[llm-resolver-vessel] provider '${key}' recovered — resuming preferred routing`);
    void syncCompletionAdvertisement();
  }
};
// HARD exhaustion — a genuinely dead lane (no credit, quota gone, daily cap hit).
// These deserve the full EXHAUSTION cooldown: retrying sooner just re-fails. This
// is the credit/quota subset of isExhaustedProviderError, WITHOUT the transient
// per-request rate-limits (which the same fn also matches so failover still fires).
const isHardExhaustedError = (e: unknown): boolean => {
  const m = String(e ?? "").toLowerCase();
  return (
    m.includes("credit balance") ||
    m.includes("insufficient_quota") ||
    m.includes("exceeded your current quota") ||
    m.includes("usage cap") ||
    /(?:^|[^0-9])402(?:[^0-9]|$)/.test(m) ||
    m.includes("billing") ||
    m.includes("limit_rpd") ||
    m.includes("daily limit reached") ||
    m.includes("free-models-per-day")
  );
};
// Transient rate-limit — recovers in seconds. Distinct from a daily/credit cap.
const isRateLimitError = (e: unknown): boolean => {
  if (isHardExhaustedError(e)) return false;
  const m = String(e ?? "").toLowerCase();
  return (
    m.includes("limit_rpm") ||
    m.includes("rate limit") ||
    m.includes("rate_limit") ||
    m.includes("too many requests") ||
    m.includes("overloaded") ||
    /(?:^|[^0-9])429(?:[^0-9]|$)/.test(m) ||
    /(?:^|[^0-9])529(?:[^0-9]|$)/.test(m)
  );
};
// Grade the cooldown by cause: hard exhaustion → full window; transient
// rate-limit → short; reachability blip → shortest. A cause that matches none of
// these (unexpected error) defaults conservative (full window).
// A bad credential does not heal on its own, but an operator may rotate it at
// any moment — so the lane must rejoin rotation by itself rather than being
// condemned for the process lifetime. Long enough that 566 futile calls a day
// become ~4, short enough that a rotation takes effect without a restart.
const UNAUTHENTICATED_COOLDOWN_MS =
  parseInt(process.env.LLM_UNAUTHENTICATED_COOLDOWN_MS ?? "900000", 10);

const cooldownMsFor = (e: unknown): number => {
  if (isHardExhaustedError(e)) return EXHAUSTION_COOLDOWN_MS;
  if (isRateLimitError(e)) return RATE_LIMIT_COOLDOWN_MS;
  if (isUnreachableProviderError(e)) return UNREACHABLE_COOLDOWN_MS;
  // Checked AFTER the others: a 402/429 carrying the word "unauthorized" is a
  // billing problem, and its cooldown should be the billing one.
  if (isUnauthenticatedProviderError(e)) return UNAUTHENTICATED_COOLDOWN_MS;
  return EXHAUSTION_COOLDOWN_MS;
};

// Model-granular exhaustion. Provider-level cooldowns condemn every model on a
// baseURL when one 402s — which hid the working :free openrouter models behind
// a paid-model 402 (2026-07-18 completion-plane outage). Cool down the failing
// MODEL; siblings on the same client stay eligible.
const modelExhaustedUntil = new Map<string, number>();
const inModelCooldown = (m: string): boolean => (modelExhaustedUntil.get(m) ?? 0) > Date.now();

// Warm state of the scale-to-zero lane. RunPod reports worker counts on /health
// WITHOUT dispatching a job, so polling never wakes — or bills — the endpoint.
// Polled in the background rather than probed on demand: willingness is checked
// on the resolve path, which must not take a network round trip, and a stale
// reading is harmless in both directions (a missed warm window costs one polling
// interval of eligibility; a missed cold transition costs one request that the
// endpoint queues anyway).
const RUNPOD_MODEL_SET: ReadonlySet<string> = new Set(RUNPOD_MODELS);
const RUNPOD_HEALTH_URL = RUNPOD_ENDPOINT_ID ? `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/health` : null;
const RUNPOD_POLL_MS = 30_000;
let runpodWarm = false;

async function refreshRunpodWarm(): Promise<void> {
  if (!RUNPOD_HEALTH_URL) return;
  const was = runpodWarm;
  try {
    const key = cleanEnv(process.env.RUNPOD_API_KEY);
    const res = await fetch(RUNPOD_HEALTH_URL, {
      headers: key ? { Authorization: `Bearer ${key}` } : {},
      signal: AbortSignal.timeout(5_000),
    });
    // A non-2xx (401, 5xx, rate limit) yields no reading — isWarmFromHealth is
    // fed only a body we actually parsed, and anything else falls to the catch.
    if (!res.ok) throw new Error(`health ${res.status}`);
    runpodWarm = isWarmFromHealth(await res.json());
  } catch {
    runpodWarm = false;  // fail closed — see scale-to-zero.ts
  }
  if (was !== runpodWarm) {
    console.log(`[llm-resolver-vessel] runpod-serverless is now ${runpodWarm ? "WARM — arm eligible" : "COLD — arm gated"}`);
  }
}

/** A scale-to-zero model with no worker up is not routable now, whatever its
 * quota says. Kept separate from cooldown state: this is capacity, not billing. */
function isRunpodCold(model: string): boolean {
  return isScaleToZeroCold(model, RUNPOD_MODEL_SET, runpodWarm);
}

if (RUNPOD_HEALTH_URL) {
  void refreshRunpodWarm();
  const t = setInterval(() => { void refreshRunpodWarm(); }, RUNPOD_POLL_MS);
  (t as unknown as { unref?: () => void }).unref?.();
}

// Willing = this resolver can route the model right now AND neither the model
// nor its provider is in exhaustion cooldown. Derived from actual routing
// capability (mirrors llmCompletionHandler's branches), never a hardcoded list,
// so every reachable policy arm is selectable and advertised-willing == routable.
// Models OpenRouter actually serves for us, as declared in the provider registry.
//
// This replaces a `model.includes("/")` heuristic that treated ANY slash-bearing
// id as an openrouter model. Slashes are just how most vendors namespace: chutes
// arms (`moonshotai/…`, `zai-org/…`) and runpod arms (`Qwen/…`) all matched, so a
// policy arm whose provider had no key — or had been removed entirely — still
// reported WILLING, got selected, failed, and burned a failover hop into the one
// model that did work. Measured: 6 unpinned probes selected 6 different arms and
// 5 of them were served by the same fallback. A selector that picks broadly and
// executes narrowly is not diversity; it is one model wearing five hats.
const OPENROUTER_MODEL_SET: ReadonlySet<string> = new Set(
  OPENAI_WIRE_PROVIDERS.find((p) => p.id === "openrouter")?.models ?? [],
);

function isModelWilling(model: string): boolean {
  // Gating here (rather than at the policy) covers every consumer of
  // willingness at once — arm selection, the exhaustion failover walk and
  // quota-gated advertisement — so a cold lane cannot be reached down any path.
  if (isRunpodCold(model)) return false;
  const client = modelClientMap.get(model);
  if (client) return !inModelCooldown(model) && !inCooldown(providerKeyOf(client));
  if (model.startsWith("claude") || model.startsWith("anthropic/")) {
    return anthropic !== null && !inCooldown("anthropic") && !inModelCooldown(model);
  }
  if (openrouterClient && OPENROUTER_MODEL_SET.has(model)) {
    return !inCooldown(providerKeyOf(openrouterClient)) && !inModelCooldown(model);
  }
  return false;
}

// Quota gate for advertisement: any uncooled keyed wire model, or the
// anthropic/openai lanes still outside cooldown, means completion is servable.
// Mirrors the availableModels expression used for policy arm selection
// (llmCompletionWithPolicyHandler) so the advertised set and the routable set
// never disagree — and it is model-granular, so :free siblings keep the shape
// advertised even when a paid model on the same baseURL is cooling.
// When LLM_PINNED_PROVIDER is set, check only that provider's quota.
const hasCompletionQuota = (): boolean => {
  if (LLM_PINNED_PROVIDER) {
    if (LLM_PINNED_PROVIDER === "anthropic") return anthropic !== null && !inCooldown("anthropic");
    if (LLM_PINNED_PROVIDER === "openai") return openaiClient !== null && !inCooldown("openai");
    const pinned = OPENAI_WIRE_PROVIDERS.find((p) => p.id === LLM_PINNED_PROVIDER);
    return pinned ? pinned.models.some((m) => modelClientMap.has(m) && !inModelCooldown(m)) : false;
  }
  return (
    [...modelClientMap.keys()].some((m) => !inModelCooldown(m)) ||
    (anthropic !== null && !inCooldown("anthropic")) ||
    (openaiClient !== null && !inCooldown("openai"))
  );
};

// Condition-driven resume of a de-advertised completion plane. A dropped shape
// receives no traffic, so nothing would ever clear a passively-expiring model
// cooldown — re-run the advertisement sync once the exhaustion window lapses.
// One pending timer at a time; unref so it never holds the process open.
let advertisementResumeTimer: ReturnType<typeof setTimeout> | null = null;
let advertisementResumeAt = 0;
const scheduleAdvertisementResume = (ms: number = EXHAUSTION_COOLDOWN_MS): void => {
  const fireAt = Date.now() + ms + 1000;
  // Keep the soonest pending resume: a short reachability cooldown must not be
  // shadowed by an earlier-scheduled long exhaustion window (else a recovered
  // spot endpoint stays de-advertised for the full 10 minutes).
  if (advertisementResumeTimer !== null && fireAt >= advertisementResumeAt) return;
  if (advertisementResumeTimer !== null) clearTimeout(advertisementResumeTimer);
  advertisementResumeAt = fireAt;
  advertisementResumeTimer = setTimeout(() => {
    advertisementResumeTimer = null;
    void syncCompletionAdvertisement();
  }, ms + 1000);
  (advertisementResumeTimer as { unref?: () => void }).unref?.();
};
const markModelExhausted = (m: string, ms: number = EXHAUSTION_COOLDOWN_MS): void => {
  modelExhaustedUntil.set(m, Date.now() + ms);
  console.warn(`[llm-resolver-vessel] model '${m}' marked exhausted — cooling down ${Math.round(ms / 1000)}s`);
  void syncCompletionAdvertisement();
  scheduleAdvertisementResume(ms);
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
    if (isFailoverError(fb.error)) markModelExhausted(fbModel, cooldownMsFor(fb.error));
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
    if (r.resolved === true || !isFailoverError(r.error)) return r;
    markModelExhausted(model, cooldownMsFor(r.error));
    const walked = await walkFallbackModels(body, model);
    return walked ?? r;
  }
  // Any vendor/model id not explicitly mapped routes through OpenRouter when keyed.
  if (openrouterClient && model.includes("/") && !model.toLowerCase().startsWith("anthropic/")) {
    const r = await resolveWithOpenAI(body, openrouterClient);
    if (r.resolved === true || !isFailoverError(r.error)) return r;
    markModelExhausted(model, cooldownMsFor(r.error));
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
    } else if (isFailoverError(result.error)) {
      markExhausted(primaryKey, cooldownMsFor(result.error));
    }
  }
  if (result.resolved !== true && (primaryCoolingDown || isFailoverError(result.error))) {
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
      if (isFailoverError(retry.error)) markExhausted(primaryKey, cooldownMsFor(retry.error));
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
  if (pinned) {
    // A model pin is a PREFERENCE, not a mandate. Model/tier is learned shaped selection
    // reading live quota — a hardcoded model must never STRAND the plane on a credit-dry
    // provider when another arm has quota. Honor the pin only when its arm is actually
    // servable; otherwise fall through to policy-selection over the arms that DO have quota
    // (and, when none do locally, to the de-advertise path so discovery routes to a peer
    // that can serve). This neutralises every hardcoded caller pin at the choke point.
    const pm = body.model as string;
    const wireLive = modelClientMap.has(pm) && !inModelCooldown(pm);
    const prov = pickProvider(pm, body.provider);
    const provLive = prov !== null && !inCooldown(prov);
    if (wireLive || provLive) return llmCompletionHandler(ctx);
    // pinned arm is dry → drop the pin and select a live arm by policy below.
  }
  // Willing set = every policy arm this resolver can route RIGHT NOW (law 1: the
  // selectable pool is derived+filtered, not a frozen subset). loadPolicy is cached.
  const policyForWilling = await loadPolicy();
  const availableModels = policyForWilling.arms.map((a) => a.model).filter((m) => isModelWilling(m));
  const sel = await selectArm(body.task_type, availableModels);
  if (!sel) {
    // NOTHING was selectable. The handler below would then fall through to
    // `body.model ?? DEFAULT_MODEL` and dial the default BLINDLY — which is
    // exactly how a credit-dead default kept being called in a tight loop
    // (`credit balance is too low ... cooling down 1800s`, hundreds of times)
    // while a perfectly warm arm sat unused because nothing had routed to it.
    //
    // A default is a LAST RESORT, not a route: it may only be dialled after the
    // same willingness check every other arm must pass. `isModelWilling` mirrors
    // all three routing branches (wire client, anthropic, openrouter), so this
    // cannot wrongly refuse a default that IS servable — the failure it prevents
    // is dialling one that demonstrably is not.
    const choice = decideLastResort({
      pinnedModel: typeof body.model === "string" ? body.model : undefined,
      defaultModel: DEFAULT_MODEL,
      armsChecked: availableModels.length,
      isWilling: isModelWilling,
      // Routable != policy arm. A provider's models can be routable without ever
      // having been seeded as arms, so "no willing ARM" must not be read as
      // "nothing can serve" — that refused while gemini-2.5-flash answered fine.
      routableModels: [...modelClientMap.keys()],
    });
    if ("refuse" in choice) {
      console.warn(`[llm-resolver-vessel] ${choice.refuse}`);
      // A REFUSAL IS A CONCLUSION, AND IT MUST REACH THE ADVERTISEMENT GATE.
      //
      // decideLastResort is a pure decision function: when it refuses it has just
      // established, against the same willingness predicate the router uses, that
      // NOTHING here can serve a completion. That conclusion used to die in a log
      // line. Nothing was marked exhausted, so hasCompletionQuota() never changed,
      // so llm_completion stayed advertised by a resolver that had just said it
      // could not serve it — violating the law stated in this repo and in
      // development-vessel/src/index.ts:52: A RESOLVER MUST NOT ADVERTISE A SHAPE
      // IT CANNOT SERVE.
      //
      // MEASURED ON A SPOKE, 2026-08-18. Local arms: ANTHROPIC_API_KEY set but
      // 401-invalid, OPENROUTER_API_KEY valid but 402 (no credits). The openrouter
      // MODELS cooled correctly, so clause 1 of hasCompletionQuota() went false —
      // but clause 2 is `anthropic !== null && !inCooldown("anthropic")`, and the
      // anthropic client is non-null whenever the KEY IS SET, validity unchecked.
      // A 401 is neither a quota error nor a reachability error, and the lane is
      // only cooled on the resolve path (line ~1134) which a refusal never reaches.
      // So the gate stayed permanently open on a lane that could never answer.
      //
      // The cost was not local. Both hub-egress fallbacks in development-vessel
      // (patch-with-tools.ts, llm-completion-dispatch.ts) are gated on "no llm arm
      // is discoverable locally", and their comments name the assumption they rest
      // on: "the local resolver de-advertises llm_completion on quota/credit
      // exhaustion". That assumption was false, so the fallbacks never engaged and
      // every spoke caller was routed into the dead local arm while a funded arm
      // sat on the hub. feature_compose failed with exactly this, which is how the
      // substrate was unable to author its own repair.
      //
      // Cooling the LANES (not the models) is the honest marker: the refusal is a
      // statement about lanes, and markExhausted is monotonic, re-syncs the
      // advertisement, and schedules its own resume — so recovery stays
      // condition-driven and a returning key re-advertises without traffic.
      markExhausted("anthropic");
      markExhausted("openai");
      return { resolved: false, shape: "llmCompletion", error: choice.refuse };
    }
    return llmCompletionHandler(ctx);
  }
  const result = await llmCompletionHandler({ ...ctx, body: { ...body, model: sel.model } } as never);
  try {
    (result as Record<string, unknown>).model_selection = sel.meta;
    // HONESTY (2026-07-31): drafting task-types grade on resolved===true = "the LLM answered",
    // which measures RESPONSIVENESS, not edit CAPABILITY — it false-inflates fast-but-weak arms
    // (haiku patch_with_tools 4020/36) and starves capable ones, so "auto" converges on a weak
    // drafter. Skip the dishonest grade for drafting tasks; true drafting capability must be
    // graded by the edit OUTCOME (reach/verified), which is wired separately (needs credit-
    // assignment: select-once-per-draft + outcome callback). Non-drafting tasks keep the grade.
    const DRAFTING_TASK_TYPES = new Set(["feature_compose", "patch_with_tools"]);
    // A PROVIDER FAILURE IS NOT A QUALITY SIGNAL.
    //
    // The outcome recorded here is `resolved === true`, so a call the provider
    // REFUSED — billing rejection, unreachable host — lands as `ok:false` and
    // increments the arm's beta exactly like a model that answered badly. The arm
    // is then penalised for its provider's account state.
    //
    // Measured 2026-08-11, after a credit outage on the paid provider:
    //
    //   claude-sonnet-5     alpha 1     beta 10.24
    //   claude-haiku-4-5    alpha 1.90  beta 40.05
    //   selected: nvidia/nemotron-3-nano-30b-a3b:free
    //
    // Selection had abandoned every capable arm, so drafting, symbol proposal and
    // reach judging fleet-wide ran on a tiny free model — and the outage taught it,
    // not the models. This is the mechanism behind a long run of "bad drafting".
    //
    // The vessel ALREADY classifies this error class: `isFailoverError` gates the
    // cooldown and the failover chain above. It simply was never consulted on the
    // LEARNING path, so a concept the code already owned did not reach the one
    // place that permanently records a judgement. Consulted here, where the error
    // is already in hand — no plumbing, no new predicate.
    //
    // Note this does NOT rescue the arm's reputation: it declines to record
    // anything, so the posterior is untouched rather than credited. An arm that is
    // genuinely bad still earns its beta from calls that actually ran.
    const armFailed = (result as { resolved?: boolean }).resolved !== true;
    const providerLevelFailure = armFailed && isFailoverError((result as { error?: unknown }).error);
    if (!DRAFTING_TASK_TYPES.has(body.task_type ?? "") && !providerLevelFailure) {
      await recordArmOutcome(sel.model, (result as { resolved?: boolean }).resolved === true, body.task_type);
    } else if (providerLevelFailure) {
      console.log(
        `[llm-resolver-vessel] NOT grading ${sel.model} — provider-level failure (billing/unreachable), not a quality outcome`,
      );
    }
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
    ["llmCompletion", llmCompletionWithPolicyHandler],
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

// Register self-hosted vLLM models as policy arms so auto-selection can pick
// them (idempotent — learned arm stats survive restarts). Best-effort: a policy
// write failure must not block the vessel from serving.
// RunPod models are excluded here and seeded below at their real price instead.
// A RunPod endpoint reachable via VLLM_ENDPOINTS looks like any other
// self-hosted vLLM to this path, and this path runs FIRST — so without the skip
// it wins the race and registers metered GPU-second capacity at cost 0, which
// ensureArmsForModels then refuses to correct because the arm already exists.
// The 0 default is right for what it was written for (an always-on box already
// paid for, e.g. the q3-30b tunnel instance) and wrong for scale-to-zero.
const SELF_HOSTED_ONLY_MODELS = SELF_HOSTED_VLLM_MODELS.filter((m) => !RUNPOD_MODEL_SET.has(m));
if (SELF_HOSTED_ONLY_MODELS.length > 0) {
  try {
    const added = await ensureArmsForModels(SELF_HOSTED_ONLY_MODELS);
    console.log(`[llm-resolver-vessel] self-hosted vLLM models registered as policy arms (${added} new): ${SELF_HOSTED_ONLY_MODELS.join(", ")}`);
  } catch (err) {
    console.warn("[llm-resolver-vessel] failed to seed vLLM policy arms (non-fatal):", err);
  }
}

// Register the RunPod Serverless models as policy arms at their WARM price.
// Seeding at the ensureArmsForModels default of 0 would be wrong here: a 0-cost
// arm wins the cost-discount term outright and would be picked ahead of every
// funded arm before any reach evidence exists. That default suits an always-on
// box already paid for; this capacity is metered per GPU-second. The arm is
// only ever SELECTABLE while warm — see isModelWilling.
if (RUNPOD_ENDPOINT_ID && cleanEnv(process.env.RUNPOD_API_KEY)) {
  try {
    const note = "runpod serverless (metered; gated while cold)";
    const added = await ensureArmsForModels(RUNPOD_MODELS, RUNPOD_COST_PER_MTOK, note);
    console.log(`[llm-resolver-vessel] runpod-serverless models registered as policy arms (${added} new) at ${RUNPOD_COST_PER_MTOK}/Mtok: ${RUNPOD_MODELS.join(", ")}`);
    // Repair arms the generic self-hosted path already created at 0 before the
    // skip above existed. Scoped to RUNPOD_MODELS so the always-on self-hosted
    // arms — for which 0 is correct — are never touched, and one-shot because
    // the repair rewrites the marker note it matches on.
    for (const m of RUNPOD_MODELS) {
      if (await repriceSeededArm(m, RUNPOD_COST_PER_MTOK, "self-hosted vLLM", note)) {
        console.log(`[llm-resolver-vessel] repriced '${m}' 0 -> ${RUNPOD_COST_PER_MTOK}/Mtok (was seeded cost-blind as generic self-hosted vLLM)`);
      }
    }
  } catch (err) {
    console.warn("[llm-resolver-vessel] failed to seed runpod policy arms (non-fatal):", err);
  }
}

// Register FUNDED wire-provider models (groq, mistral) as policy arms so the
// Thompson selector can actually route to their quota. They are in the provider
// registry (routable via modelClientMap) but were never seeded as ARMS, so
// selectArm never picked them as primary — under load the fleet burned the
// rate-limited anthropic/gemini/openrouter-free arms and never reached the
// funded capacity (observed: the hub arm flapping on gemini-2.5-* / cooling,
// while GROQ_API_KEY + MISTRAL_API_KEY sat unused). Seed each at its published
// per-Mtok cost so the cost-discount term is honest; only when the key is
// present. Idempotent — learned alpha/beta survive restarts.
const FUNDED_ARM_COST: Record<string, number> = {
  "llama-3.3-70b-versatile": 0.6, "moonshotai/kimi-k2-instruct": 1.0, "qwen/qwen3-32b": 0.5,
  "mistral-small-latest": 0.2, "codestral-latest": 0.3, "mistral-large-latest": 2.0,
  // openrouter, published per-Mtok input rates; `:free` slugs are genuinely 0 so
  // the cost-discount term prefers them and only escalates to paid when the free
  // arms are cooling (their daily quota is account-wide, so they DO run out).
  "google/gemini-2.5-flash": 0.3, "openai/gpt-4o-mini": 0.15,
  "deepseek/deepseek-chat-v3-0324": 0.3,
  "nvidia/nemotron-3-ultra-550b-a55b:free": 0, "nvidia/nemotron-3-nano-30b-a3b:free": 0,
  "cohere/north-mini-code:free": 0,
};
// DERIVE the providers to seed rather than hardcoding a list.
//
// This was `["groq", "mistral"]`, which excluded openrouter deliberately — at the
// time its arms were the rate-limited ones being burned while funded capacity sat
// idle. That reasoning inverted once openrouter became the only keyed provider:
// its six models were routable but never ARMS, so `selectArm` could not pick any
// of them, every unpinned call fell through to the same failover model, and the
// live policy held SIX arms of which all six were unusable (2 anthropic
// credit-dead, 2 chutes with no key, 1 slug the provider does not serve, 1 runpod
// that had been removed). A selectable pool that cannot intersect the routable set
// is not a selector — it is a single point of failure with extra steps.
//
// runpod-serverless is skipped here because it is seeded separately at its real
// GPU-second-derived price; seeding it at a wire default would misprice it.
const SEED_SKIP_PROVIDERS = new Set(["runpod-serverless"]);
for (const provId of OPENAI_WIRE_PROVIDERS.map((pr) => pr.id).filter((id) => !SEED_SKIP_PROVIDERS.has(id))) {
  const prov = OPENAI_WIRE_PROVIDERS.find((pr) => pr.id === provId);
  if (!prov || !cleanEnv(process.env[prov.apiKeyEnv])) continue;
  let seeded = 0;
  for (const m of prov.models) {
    try { seeded += await ensureArmsForModels([m], FUNDED_ARM_COST[m] ?? 0.6, `keyed ${provId}`); }
    catch (err) { console.warn(`[llm-resolver-vessel] failed to seed ${provId} arm '${m}' (non-fatal):`, err); }
  }
  console.log(`[llm-resolver-vessel] keyed ${provId} models registered as policy arms (${seeded} new): ${prov.models.join(", ")}`);
}

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
