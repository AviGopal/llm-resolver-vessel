/**
 * Normalize a resolve request body to the flat fields the handlers read.
 *
 * The handlers read their arguments from the top level of the body
 * (`body.prompt`, `body.model`, ...). A request that arrives through discovery's
 * `/resolve` — local or forwarded across federation — carries them inside
 * `{ pointer: {...} }` (the format this vessel's registry row declares) or
 * `{ impulse: { pointer: {...} } }`. Read that way, every routed llm_completion
 * failed with "body must include non-empty 'prompt' string", so a node with no
 * local arm could never use a peer's.
 *
 * Top-level fields win over pointer fields, so a flat request is unchanged.
 */
export function unwrapPointerBody(body: unknown): unknown {
  if (typeof body !== "object" || body === null || Array.isArray(body)) return body;
  const b = body as Record<string, unknown>;
  const impulse = b.impulse as Record<string, unknown> | undefined;
  const pointer =
    (typeof b.pointer === "object" && b.pointer !== null ? b.pointer : undefined) ??
    (typeof impulse === "object" && impulse !== null && typeof impulse.pointer === "object" && impulse.pointer !== null
      ? impulse.pointer
      : undefined);
  if (pointer === undefined || Array.isArray(pointer)) return body;
  const { pointer: _p, impulse: _i, ...rest } = b;
  return { ...(pointer as Record<string, unknown>), ...rest };
}
