/**
 * An EMPTY COMPLETION is not an answer.
 *
 * A provider can return a turn with no text and no tool calls: a reasoning model that spent the
 * whole token budget on hidden reasoning (`finish_reason:"length"`, `content:null`, the text in
 * `message.reasoning`), a tool loop that ran out of iterations before a final turn, or a model that
 * answers a tool result with an empty `stop` turn. Each path used to return
 * `resolved:true, content:""`, so nothing downstream treated it as a failure until the caller's
 * very last step threw, no other arm was tried, and the learner credited the arm that answered empty.
 *
 * Every completion path now ends in `emptyCompletionRefusal` when its final text is blank. It is a
 * QUALITY failure of the model that served the turn, not a provider failure: it is not a failover
 * error (no cooldown, no exhaustion mark), but the resolver still walks on to another arm.
 *
 * Extracted from index.ts so it can be imported by tests (index.ts boots the daemon at import time).
 */

export const EMPTY_COMPLETION = "empty_completion";

/** Null, undefined, "" or whitespace only. Non-string content is not judged blank here. */
export function isBlankText(content: unknown): boolean {
  if (content === null || content === undefined) return true;
  return typeof content === "string" && content.trim().length === 0;
}

/** Whether a provider message carries reasoning. Only the boolean leaves this function: reasoning
 * text is never copied into a result or an error. */
export function reasoningPresent(message: unknown): boolean {
  if (!message || typeof message !== "object") return false;
  const m = message as Record<string, unknown>;
  for (const k of ["reasoning", "reasoning_content"]) {
    if (typeof m[k] === "string" && (m[k] as string).trim().length > 0) return true;
  }
  if (Array.isArray(m.reasoning_details) && m.reasoning_details.length > 0) return true;
  // Anthropic: thinking blocks in the content array.
  if (Array.isArray(m.content) && m.content.some((b) => b && typeof b === "object"
    && ((b as { type?: unknown }).type === "thinking" || (b as { type?: unknown }).type === "redacted_thinking"))) return true;
  return false;
}

export interface EmptyCompletionFacts {
  provider: string;
  /** The wire model id the provider was called with. */
  model: string;
  /** The arm id that returned the empty turn. */
  served_model: string;
  /** The arm id the call was asked to serve. */
  requested_model: string;
  /** The provider's own stop reason for the last turn (OpenAI `finish_reason`, Anthropic `stop_reason`). */
  stop_reason: string | null;
  completion_tokens: number;
  /** Provider turns taken. */
  iterations: number;
  reasoning_present: boolean;
  /** This resolver's own instance id, so a peer's empty can be told from a local one. */
  resolver: string;
  /** The request's own correlation ids, when it carried them. */
  execution_id?: unknown;
  dispatch_id?: unknown;
}

/** The refusal every completion path returns instead of `resolved:true` with blank text. The error
 * and the facts that tell its causes apart come first, so a reader that keeps only a prefix of the
 * serialized body still sees them. */
export function emptyCompletionRefusal(f: EmptyCompletionFacts, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    resolved: false,
    error: EMPTY_COMPLETION,
    stop_reason: f.stop_reason,
    served_model: f.served_model,
    requested_model: f.requested_model,
    iterations: f.iterations,
    completion_tokens: f.completion_tokens,
    reasoning_present: f.reasoning_present,
    resolver: f.resolver,
    ...(typeof f.execution_id === "string" && f.execution_id.length > 0 ? { execution_id: f.execution_id } : {}),
    ...(typeof f.dispatch_id === "string" && f.dispatch_id.length > 0 ? { dispatch_id: f.dispatch_id } : {}),
    shape: "llmCompletion",
    failure_mode: EMPTY_COMPLETION,
    provider: f.provider,
    model: f.model,
    ...extra,
  };
}

export function isEmptyCompletion(result: unknown): boolean {
  return !!result && typeof result === "object" && (result as { error?: unknown }).error === EMPTY_COMPLETION;
}

/** The arms that answered empty, as `result` records them (`empty_arms`, else its own served_model). */
export function emptyArmsOf(result: Record<string, unknown>): string[] {
  if (Array.isArray(result.empty_arms)) return result.empty_arms.filter((m): m is string => typeof m === "string" && m.length > 0);
  if (isEmptyCompletion(result) && typeof result.served_model === "string" && result.served_model.length > 0) return [result.served_model];
  return [];
}

/** `result` with the dispatch's empty arms attached; the same object when there were none. */
export function withEmptyArms(result: Record<string, unknown>, empties: readonly string[]): Record<string, unknown> {
  if (empties.length === 0) return result;
  return { ...result, empty_arms: [...new Set(empties)] };
}

/** The learner's success: resolved AND non-blank text. A blank answer is never a success, whichever
 * path produced it. */
export function completionAnswered(result: Record<string, unknown>): boolean {
  return result.resolved === true && !isBlankText(result.content);
}

/** The arms to charge one ok:false for answering empty in this dispatch: every arm in the result's
 * empty list, once, except `graded` — the arm the main grade already recorded for this result. */
export function emptyArmsToCharge(result: Record<string, unknown>, graded: string | null): string[] {
  return [...new Set(emptyArmsOf(result))].filter((m) => m !== graded);
}
