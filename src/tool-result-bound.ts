/**
 * tool-result-bound.ts - bound what one tool result, and all tool results of one request, add to
 * the message list a tool-use loop re-sends every turn.
 *
 * WHY: the tool loops appended every tool result VERBATIM and re-sent the whole message list on
 * every turn. One append can move a turn from under the 200k-token prompt ceiling to 1.6M: the
 * ceiling refused 20+ goal-host:floor_tool_loop turns estimated at 202k to 1,673,516 input tokens,
 * but only after the earlier turns had been paid for. The ceiling detects the growth; this stops it.
 *
 * WHAT: a result within the per-result allowance passes UNCHANGED. A larger one is replaced by a
 * record: a truncation marker, the record's own shape, the producing tool, the full size, a head
 * excerpt, and a pointer (the tool and its arguments) so the model can call it again, narrower.
 * A per-request total allowance bounds the sum as well: once it is spent, results arrive as the
 * record with no excerpt. Both allowances are llmModelPolicy fields read at use time (law 1).
 */

/** Longest a tool call's arguments are echoed in a record's pointer. */
export const POINTER_ARGS_MAX_CHARS = 400;

export interface ToolResultLimits { per_result_max_chars: number; total_max_chars: number }

export interface ToolCallOutcome { ok: boolean; result: unknown; error?: string }

/** The text a tool result has always been sent as. */
export function toolResultText(r: ToolCallOutcome): string {
  return typeof r.result === "string" ? r.result : JSON.stringify(r.result ?? r.error ?? null) ?? "null";
}

function resultKind(v: unknown): string {
  if (typeof v === "string") return "text";
  if (Array.isArray(v)) return "array";
  if (v === null || v === undefined) return "empty";
  return typeof v === "object" ? "object" : typeof v;
}

function compactArgs(toolInput: unknown): unknown {
  let text: string;
  try { text = JSON.stringify(toolInput ?? {}) ?? "{}"; } catch { return "(unserialisable arguments)"; }
  if (text.length <= POINTER_ARGS_MAX_CHARS) return toolInput ?? {};
  return `${text.slice(0, POINTER_ARGS_MAX_CHARS)}... (${text.length} chars)`;
}

/** Per-request bounder: create one per tool-use loop, before its first turn. */
export class ToolResultBudget {
  private remaining: number;
  constructor(private readonly limits: ToolResultLimits) {
    this.remaining = Math.max(0, limits.total_max_chars);
  }

  /** Characters of excerpt still allowed for this request. */
  get remainingChars(): number { return this.remaining; }

  /** The message content for one tool result: unchanged when it fits, otherwise a bounded record. */
  bound(toolName: string, toolInput: unknown, r: ToolCallOutcome): string {
    const full = toolResultText(r);
    const allowed = Math.min(this.limits.per_result_max_chars, this.remaining);
    if (full.length <= allowed) {
      this.remaining -= full.length;
      return full;
    }
    const shown = Math.max(0, allowed);
    this.remaining -= shown;
    const totalSpent = shown < this.limits.per_result_max_chars;
    const record = {
      shape: "toolResultExcerpt",
      truncated: true,
      producer: toolName,
      result_kind: resultKind(r.result ?? r.error),
      ok: r.ok,
      size_chars: full.length,
      shown_chars: shown,
      reason: totalSpent
        ? "this request's total tool-result allowance (llmModelPolicy.tool_results_total_max_chars) is spent or nearly spent"
        : "the result is longer than the per-result allowance (llmModelPolicy.tool_result_max_chars)",
      pointer: { tool: toolName, args: compactArgs(toolInput) },
      hint: totalSpent
        ? "Answer from what you have already observed, or make one narrow call."
        : `Only the head is shown. Call ${toolName} again with narrower arguments (a path, a line range, a filter or a limit) to read a specific part.`,
    };
    return `[TOOL RESULT TRUNCATED] ${JSON.stringify(record)}${shown > 0 ? `\n--- head (${shown} of ${full.length} chars) ---\n${full.slice(0, shown)}` : ""}`;
  }
}
