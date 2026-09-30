/**
 * model-policy.ts - cost/reach/uncertainty-optimized model selection
 * (gap resolver-and-model-selection-not-cost-optimized).
 *
 * The policy is a SHAPED, live-editable document (llmModelPolicy /
 * llmModelPolicy_write impulses), stored as a JSON file owned by this vessel
 * and read at use time - never a frozen constant. Selection is Thompson
 * sampling: each arm carries alpha/beta reach evidence; a Beta draw per arm
 * (normal approximation) captures uncertainty, and a cost discount biases
 * toward cheap/free models until evidence justifies premium ones. Every
 * auto-selection is returned in the resolve body (model_selection) so traces
 * can observe and grade the choice; technical outcomes update alpha/beta
 * immediately, and graders may write task-level outcomes via
 * llmModelPolicy_write.
 */
import * as fs from "node:fs/promises";
import * as path from "node:path";

export interface PolicyArm {
  model: string;
  cost_per_mtok: number;
  alpha: number;
  beta: number;
  note?: string;
  task_alpha?: Record<string, number>;
  task_beta?: Record<string, number>;
  last_updated_at?: string;
}
export interface ModelPolicy {
  rev: number;
  updated_at: string;
  cost_weight: number;
  arms: PolicyArm[];
  /** Per-turn prompt ceiling (value-per-cost 1.6); read in index.ts. */
  max_input_tokens?: number;
  /** Input tokens one dispatch (dispatch_id, else execution_id) may send across ALL its turns and
   * calls before llm-resolver refuses further provider calls for it. See dispatch-budget.ts. */
  max_input_tokens_per_dispatch?: number;
  /** Longest a single tool result may enter a tool loop's message list. See tool-result-bound.ts. */
  tool_result_max_chars?: number;
  /** Total tool-result characters one request's tool loop may add across all its turns. */
  tool_results_total_max_chars?: number;
}

/**
 * Values a policy field takes when the stored policy does not set it. The stored policy file
 * predates these fields, so the defaults are applied at READ time, not seeded into the file.
 *
 * max_input_tokens_per_dispatch = 400k, from the spend summary of 2026-09-30 15:40-16:40 UTC across
 * both nodes: legitimate per-call input was feature_compose ~24k, patch_with_tools <= ~4.1k,
 * floor_tool_loop normally 5-10k (one 231k gpt-4o-mini outlier), everything else < 8k. 400k clears
 * the largest legitimate call ~16x and the 231k outlier, yet stops a runaway floor loop at about a
 * quarter of one observed 1.58M-token call: ~0.24 USD at the observed gpt-5 rate (1.906 USD /
 * 3.16M input tokens), ~12% of the fleet's 2 USD/h spend envelope instead of ~95%.
 */
/*
 * tool_result_max_chars = 16k chars (~4k tokens) and tool_results_total_max_chars = 64k chars
 * (~16k tokens) per request. In the same hour a normal floor_tool_loop call used 5-10k input
 * tokens IN TOTAL (prompt, every turn and every tool result), so 16k tokens of tool results clears
 * normal use while holding the re-sent list far under the 200k-token per-turn ceiling (~800k chars).
 * 16k chars is ~400 lines of source: enough to orient, after which the model is told to re-call
 * narrower. A single 525k-char result used to enter whole (~131k tokens, re-sent every turn).
 */
export const POLICY_DEFAULTS = {
  max_input_tokens_per_dispatch: 400_000,
  tool_result_max_chars: 16_000,
  tool_results_total_max_chars: 64_000,
} as const;

/** The numeric policy fields that fall back to POLICY_DEFAULTS and are writable via llmModelPolicy_write. */
export type DefaultedPolicyField = keyof typeof POLICY_DEFAULTS;
const DEFAULTED_FIELDS = Object.keys(POLICY_DEFAULTS) as DefaultedPolicyField[];

/** A defaulted field's value in force: the policy's when it is a finite number >= 1000 (smaller
 * values are ignored so refusal text never carries a 3-digit number that the provider-error
 * classifiers read as an HTTP status), else POLICY_DEFAULTS. */
export function effectivePolicyValue(policy: Partial<Pick<ModelPolicy, DefaultedPolicyField>>, field: DefaultedPolicyField): number {
  const v = policy[field];
  return typeof v === "number" && Number.isFinite(v) && v >= 1000 ? v : POLICY_DEFAULTS[field];
}

export function effectiveMaxInputTokensPerDispatch(policy: Pick<ModelPolicy, "max_input_tokens_per_dispatch">): number {
  return effectivePolicyValue(policy, "max_input_tokens_per_dispatch");
}

/** Tool-result allowances in force for one request (read once at the start of a tool loop). */
export async function readToolResultLimits(): Promise<{ per_result_max_chars: number; total_max_chars: number }> {
  const policy = await loadPolicy();
  return {
    per_result_max_chars: effectivePolicyValue(policy, "tool_result_max_chars"),
    total_max_chars: effectivePolicyValue(policy, "tool_results_total_max_chars"),
  };
}

/** Read at use time (loadPolicy's 10 s cache; a policy write refreshes it immediately). */
export async function readMaxInputTokensPerDispatch(): Promise<number> {
  return effectiveMaxInputTokensPerDispatch(await loadPolicy());
}

const POLICY_PATH = path.join(process.env.WORKSPACE_ROOT ?? "/workspace", "policies", "llm-model-policy.json");
const CACHE_TTL_MS = 10_000;

const DEFAULT_POLICY: ModelPolicy = {
  rev: 1,
  updated_at: "1970-01-01T00:00:00.000Z",
  cost_weight: 0.25,
  arms: [
    { model: "claude-sonnet-5", cost_per_mtok: 3.0, alpha: 1, beta: 1, note: "premium anchor (anthropic)" },
    { model: "claude-haiku-4-5-20251001", cost_per_mtok: 0.8, alpha: 1, beta: 1, note: "cheap anthropic" },
    { model: "moonshotai/Kimi-K2.6-TEE", cost_per_mtok: 0.3, alpha: 1, beta: 1, note: "chutes" },
    { model: "zai-org/GLM-5.2-TEE", cost_per_mtok: 0.25, alpha: 1, beta: 1, note: "chutes" },
    { model: "tencent/hy3:free", cost_per_mtok: 0.01, alpha: 1, beta: 1, note: "openrouter free tier" },
  ],
};

let cached: { at: number; policy: ModelPolicy } | null = null;

export async function loadPolicy(): Promise<ModelPolicy> {
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) return cached.policy;
  let policy: ModelPolicy;
  try {
    policy = JSON.parse(await fs.readFile(POLICY_PATH, "utf-8")) as ModelPolicy;
    if (!Array.isArray(policy.arms) || policy.arms.length === 0) policy = DEFAULT_POLICY;
  } catch {
    policy = DEFAULT_POLICY;
    try { await savePolicy(policy); } catch { /* first-write best effort */ }
  }
  cached = { at: Date.now(), policy };
  return policy;
}

// Saves are serialised and each writes its own temp file. Concurrent completions shared one
// ".tmp" path, so one rename moved another writer's file away and every loser failed with
// ENOENT: 157 arm outcomes lost in 2 minutes under load (2026-09-26), i.e. the model
// posterior stopped learning exactly when it was busiest.
let savePolicyChain: Promise<void> = Promise.resolve();

export async function savePolicy(policy: ModelPolicy): Promise<void> {
  const run = savePolicyChain.then(async () => {
    policy.updated_at = new Date().toISOString();
    await fs.mkdir(path.dirname(POLICY_PATH), { recursive: true });
    const tmp = `${POLICY_PATH}.tmp.${process.pid}.${Date.now()}.${Math.random().toString(36).slice(2)}`;
    await fs.writeFile(tmp, JSON.stringify(policy, null, 2), "utf-8");
    await fs.rename(tmp, POLICY_PATH);
    cached = { at: Date.now(), policy };
  });
  savePolicyChain = run.catch(() => {});
  return run;
}

/** Exponential decay pulling alpha and beta toward neutral prior (1,1) with 3-day half-life. */
function decayedCounts(alpha: number, beta: number, lastUpdatedAt: string | undefined, nowMs: number): { alpha: number; beta: number } {
  const ageMs = nowMs - (lastUpdatedAt ? new Date(lastUpdatedAt).getTime() : 0);
  const halfLifeMs = 3 * 24 * 60 * 60 * 1000;
  const d = Math.pow(0.5, ageMs / halfLifeMs);
  return {
    alpha: 1 + (alpha - 1) * d,
    beta: 1 + (beta - 1) * d,
  };
}

/** Beta(a,b) sample via normal approximation (adequate for bandit selection;
 * exact gamma sampling is overkill here and this stays dependency-free). */
function betaSample(a: number, b: number): number {
  const mean = a / (a + b);
  const variance = (a * b) / ((a + b) * (a + b) * (a + b + 1));
  const u1 = Math.random(), u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1 || 1e-12)) * Math.cos(2 * Math.PI * u2);
  const s = mean + z * Math.sqrt(variance);
  return Math.min(1, Math.max(0, s));
}

export interface ArmSelection {
  model: string;
  meta: Record<string, unknown>;
}

export async function selectArm(taskType?: string, availableModels?: string[]): Promise<ArmSelection | null> {
  const policy = await loadPolicy();
  if (policy.arms.length === 0) return null;
  const maxCost = Math.max(...policy.arms.map((a) => a.cost_per_mtok ?? 0), 0.01);
  let best: { arm: PolicyArm; score: number } | null = null;
  const considered: Array<Record<string, unknown>> = [];
  for (const arm of policy.arms) {
    if (availableModels && !availableModels.includes(arm.model)) continue;
    const hasTask = !!(taskType && arm.task_alpha && arm.task_alpha[taskType] !== undefined);
    const taskAlpha = hasTask ? arm.task_alpha![taskType]! : (taskType ? 1 : arm.alpha);
    const taskBeta = hasTask ? arm.task_beta![taskType]! : (taskType ? 1 : arm.beta);
    const decayed = decayedCounts(taskAlpha, taskBeta, arm.last_updated_at, Date.now());
    const draw = betaSample(decayed.alpha, decayed.beta);
    const score = draw - policy.cost_weight * ((arm.cost_per_mtok ?? 0) / maxCost);
    considered.push({ model: arm.model, draw: Number(draw.toFixed(4)), score: Number(score.toFixed(4)), alpha: decayed.alpha, beta: decayed.beta, cost_per_mtok: arm.cost_per_mtok });
    if (!best || score > best.score) best = { arm, score };
  }
  if (!best) return null;
  return {
    model: best.arm.model,
    meta: {
      policy_rev: policy.rev,
      selected: best.arm.model,
      sampled_score: Number(best.score.toFixed(4)),
      considered,
    },
  };
}

/** Technical-outcome update for an auto-selected arm. Only policy-chosen calls
 * are graded here (causal discipline: never grade pinned calls the policy did
 * not choose). Task-level verdicts arrive via llmModelPolicy_write. */
export async function recordArmOutcome(model: string, ok: boolean, taskType?: string): Promise<void> {
  const policy = await loadPolicy();
  const arm = policy.arms.find((a) => a.model === model);
  if (!arm) return;
  const now = Date.now();
  if (taskType) {
    arm.task_alpha = arm.task_alpha ?? {};
    arm.task_beta = arm.task_beta ?? {};
    const prevTaskAlpha = arm.task_alpha[taskType] ?? 1;
    const prevTaskBeta = arm.task_beta[taskType] ?? 1;
    const decayed = decayedCounts(prevTaskAlpha, prevTaskBeta, arm.last_updated_at, now);
    const a = decayed.alpha + (ok ? 1 : 0);
    const b = decayed.beta + (ok ? 0 : 1);
    arm.task_alpha[taskType] = a;
    arm.task_beta[taskType] = b;
  } else {
    const decayed = decayedCounts(arm.alpha, arm.beta, arm.last_updated_at, now);
    arm.alpha = decayed.alpha + (ok ? 1 : 0);
    arm.beta = decayed.beta + (ok ? 0 : 1);
  }
  arm.last_updated_at = new Date().toISOString();
  await savePolicy(policy);
}

/** Idempotently ensure a policy arm exists for each model (used to register
 * self-hosted vLLM models discovered from env at startup, so the Thompson
 * selector can pick them in auto mode). Existing arms are left untouched — their
 * learned alpha/beta and any operator-tuned cost survive restarts. Self-hosted
 * inference has no per-token vendor price, so new arms seed at cost 0, which
 * makes the cost-discount term favour them until reach evidence says otherwise. */
export async function ensureArmsForModels(models: string[], costPerMtok = 0, note = "self-hosted vLLM"): Promise<number> {
  if (models.length === 0) return 0;
  const policy = await loadPolicy();
  let added = 0;
  for (const model of models) {
    if (policy.arms.some((a) => a.model === model)) continue;
    policy.arms.push({ model, cost_per_mtok: costPerMtok, alpha: 1, beta: 1, note });
    added += 1;
  }
  if (added > 0) {
    policy.rev += 1;
    await savePolicy(policy);
  }
  return added;
}

/** One-shot repair for an arm that a COST-BLIND registration path created at the
 * `ensureArmsForModels` 0 default, when the model actually costs money.
 *
 * Needed because two paths can register the same model and the cheap one may
 * win the race: a RunPod Serverless model reachable via VLLM_ENDPOINTS is seeded
 * by the generic self-hosted path at 0 before the RunPod-aware path can seed it
 * at its real price, and `ensureArmsForModels` then leaves the existing arm
 * alone. A 0-cost arm takes the cost-discount term outright, so the mispricing
 * does not self-correct — nothing about later evidence touches `cost_per_mtok`.
 *
 * Deliberately narrow, because "repair" and "stomp the operator" are the same
 * edit seen from different sides. The CALLER scopes which models are eligible,
 * and the repair only fires while the arm still carries the exact seeding marker
 * it was created with AND is still at 0 — so a deliberate operator price
 * (including a deliberate 0) is never touched, and rewriting the note makes the
 * repair idempotent: it cannot fire twice. Learned alpha/beta and task stats are
 * preserved untouched — the price was wrong, the evidence was not. */
export async function repriceSeededArm(
  model: string, costPerMtok: number, fromNote: string, toNote: string,
): Promise<boolean> {
  const policy = await loadPolicy();
  const arm = policy.arms.find((a) => a.model === model);
  if (!arm) return false;
  if (arm.note !== fromNote || (arm.cost_per_mtok ?? 0) !== 0) return false;
  arm.cost_per_mtok = costPerMtok;
  arm.note = toNote;
  policy.rev += 1;
  await savePolicy(policy);
  return true;
}

export function providerFor(modelId: string): string {
  if (modelId.startsWith("claude")) return "anthropic";
  if (modelId.startsWith("gemini")) return "google";
  if (modelId.endsWith("-TEE")) return "chutes";
  if (modelId.includes("/")) return "openrouter";
  return "unknown";
}

import { readFile, writeFile, rename, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
const PENDING_ARM_OUTCOMES_PATH = path.join(path.dirname(POLICY_PATH), "pending-arm-outcomes.json");

export async function recordPendingArmOutcome(executionId: string, model: string): Promise<void> {
  let records: Array<{ executionId: string; model: string; at: string }> = [];
  try {
    records = JSON.parse(await readFile(PENDING_ARM_OUTCOMES_PATH, "utf-8"));
  } catch {
    records = [];
  }
  records.push({ executionId, model, at: new Date().toISOString() });
  await mkdir(dirname(PENDING_ARM_OUTCOMES_PATH), { recursive: true });
  const tmp = PENDING_ARM_OUTCOMES_PATH + ".tmp";
  await writeFile(tmp, JSON.stringify(records, null, 2), "utf-8");
  await rename(tmp, PENDING_ARM_OUTCOMES_PATH);
}

export async function gradeArmByExecution(executionId: string, reached: boolean, taskType?: string): Promise<void> {
  let records: Array<{ executionId: string; model: string; at: string }> = [];
  try {
    records = JSON.parse(await readFile(PENDING_ARM_OUTCOMES_PATH, "utf-8"));
  } catch {
    records = [];
  }
  const record = records.find((r) => r.executionId === executionId);
  if (!record) return;
  records = records.filter((r) => r.executionId !== executionId);
  await mkdir(dirname(PENDING_ARM_OUTCOMES_PATH), { recursive: true });
  const tmp = PENDING_ARM_OUTCOMES_PATH + ".tmp";
  await writeFile(tmp, JSON.stringify(records, null, 2), "utf-8");
  await rename(tmp, PENDING_ARM_OUTCOMES_PATH);
  await recordArmOutcome(record.model, reached, taskType);
}

export async function recordArmOutcomeFromPending(executionId: string, reached: boolean): Promise<void> {
  let records: Array<{ executionId: string; model: string; at: string }> = [];
  try {
    records = JSON.parse(await readFile(PENDING_ARM_OUTCOMES_PATH, "utf-8"));
  } catch {
    records = [];
  }
  const record = records.find((r) => r.executionId === executionId);
  if (!record) return;
  const policy = await loadPolicy();
  const arm = policy.arms.find((a) => a.model === record.model);
  if (!arm) return;
  if (reached) arm.alpha += 1; else arm.beta += 1;
  await savePolicy(policy);
  const updated = records.filter((r) => r.executionId !== executionId);
  await mkdir(dirname(PENDING_ARM_OUTCOMES_PATH), { recursive: true });
  const tmp = PENDING_ARM_OUTCOMES_PATH + ".tmp";
  await writeFile(tmp, JSON.stringify(updated, null, 2), "utf-8");
  await rename(tmp, PENDING_ARM_OUTCOMES_PATH);
}

function effectiveDefaultedValues(policy: ModelPolicy): Record<string, number> {
  return Object.fromEntries(DEFAULTED_FIELDS.map((f) => [f, effectivePolicyValue(policy, f)]));
}

export async function llmModelPolicyHandler(): Promise<{ resolved: boolean; shape: string; body: ModelPolicy & { effective: Record<string, number> } }> {
  const policy = await loadPolicy();
  // `effective` shows the value in force for fields the stored policy leaves unset (defaults).
  return { resolved: true, shape: "llmModelPolicy", body: { ...policy, effective: effectiveDefaultedValues(policy) } };
}

export async function llmModelPolicyWriteHandler(ctx: { body: unknown }): Promise<{ resolved: boolean; shape: string; body?: unknown; error?: string }> {
  const req = ctx.body as { arms?: PolicyArm[]; cost_weight?: number; merge?: boolean } & Partial<Record<DefaultedPolicyField, number | null>>;
  const fields = req != null && typeof req === "object" ? DEFAULTED_FIELDS.filter((f) => f in req) : [];
  if (!req || (!Array.isArray(req.arms) && typeof req.cost_weight !== "number" && fields.length === 0)) {
    return { resolved: false, shape: "llmModelPolicyWriteResult", error: `body must include arms[], cost_weight and/or one of ${DEFAULTED_FIELDS.join(", ")}` };
  }
  for (const f of fields) {
    const v = req[f];
    if (v !== null && !(typeof v === "number" && Number.isFinite(v) && v >= 1000)) {
      return { resolved: false, shape: "llmModelPolicyWriteResult", error: `${f} must be a number >= 1000, or null to use the default` };
    }
  }
  const policy = await loadPolicy();
  if (typeof req.cost_weight === "number") policy.cost_weight = req.cost_weight;
  for (const f of fields) {
    const v = req[f];
    if (v === null || v === undefined) delete policy[f];
    else policy[f] = v;
  }
  if (Array.isArray(req.arms)) {
    if (req.merge === false) {
      policy.arms = req.arms;
    } else {
      for (const incoming of req.arms) {
        const existing = policy.arms.find((a) => a.model === incoming.model);
        if (existing) Object.assign(existing, incoming);
        else policy.arms.push(incoming);
      }
    }
  }
  policy.rev += 1;
  await savePolicy(policy);
  return { resolved: true, shape: "llmModelPolicyWriteResult", body: { rev: policy.rev, arm_count: policy.arms.length, effective: effectiveDefaultedValues(policy) } };
}
