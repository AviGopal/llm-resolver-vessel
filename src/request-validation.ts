// A pure module for request validation.

type CompletionRequest = { type: "llm_completion", prompt: string };

type Validation = null | { resolved: false; shape: string; error: string; failure_mode: string; field: string };

export function validateCompletionRequest(body: unknown): Validation {
  if (typeof body !== "object" || body === null || !("type" in body) || body.type !== "llm_completion") {
    return {
      resolved: false,
      shape: "llmCompletion",
      error: "body must be an object with type 'llm_completion'",
      failure_mode: "malformed_request",
      field: "body",
    };
  }

  const completionRequest = body as CompletionRequest;

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

const malformedRequestCounts: Record<string, { count: number; last_execution_id?: string }> = {};

export function recordMalformedRequest(body: { caller: string; task_type: string; execution_id?: string }): void {
  const key = `${body.caller}:${body.task_type}`;
  if (!malformedRequestCounts[key]) {
    malformedRequestCounts[key] = { count: 0 };
  }
  malformedRequestCounts[key].count++;
  malformedRequestCounts[key].last_execution_id = body.execution_id;
}

export function malformedRequestSummary(): Array<{ caller: string; task_type: string; count: number; last_execution_id?: string }> {
  return Object.entries(malformedRequestCounts).map(([key, value]) => {
    const [caller, task_type] = key.split(":");
    return { caller, task_type, ...value };
  });
}
