// A pure module for request validation.

type CompletionRequest = { type: "llm_completion", prompt: string };

type Validation = null | { resolved: false; shape: string; error: string; failure_mode: string; field: string };

export function validateCompletionRequest(body: unknown): Validation {
  // Only the prompt is classified as a caller fault. A body without `type:
  // "llm_completion"` is NOT malformed: llmCompletionHandler's old inline check
  // only guarded the prompt, and openrouter-model-set pins that requests sent
  // without the type field still reach the model logic.
  const completionRequest = (body ?? {}) as CompletionRequest;

  if (typeof completionRequest.prompt !== "string" || completionRequest.prompt.trim() === "") {
    return {
      resolved: false,
      shape: "llmCompletion",
      error: "body must include non-empty 'prompt' string",
      failure_mode: "malformed_request",
      field: "prompt",
    };
  }

  return null;
}

const malformedRequestCounts: Record<string, { caller: string; task_type: string; count: number; last_execution_id?: string }> = {};

export function recordMalformedRequest(body: { caller?: string; task_type?: string; execution_id?: string }): void {
  const caller = body.caller ?? "unknown";
  const task_type = body.task_type ?? "unknown";
  const key = `${caller}:${task_type}`;
  if (!malformedRequestCounts[key]) {
    malformedRequestCounts[key] = { caller, task_type, count: 0 };
  }
  malformedRequestCounts[key].count++;
  malformedRequestCounts[key].last_execution_id = body.execution_id;
}

export function malformedRequestSummary(): Array<{ caller: string; task_type: string; count: number; last_execution_id?: string }> {
  return Object.values(malformedRequestCounts);
}

