// EVERY COMPLETION CALL'S OUTCOME, COUNTED BY OUTCOME AND PRODUCER.
//
// The spend windows counted only calls that carried usage, and nothing logged a success, so neither node could show
// that a completion answered 200 after an outage (research). Each call now adds one to `<outcome>|<producer>` in the
// spend windows: outcome is resolved, empty (an empty completion), unserved (a strict pin that could not serve) or
// failed; producer is the model that served, else the model asked for.
import { isEmptyCompletion } from "./empty-completion";

export type Outcome = "resolved" | "empty" | "unserved" | "failed";

export const outcomeOf = (result: unknown): Outcome => {
  const r = (result ?? {}) as Record<string, unknown>;
  if (r.resolved === true) return "resolved";
  if (isEmptyCompletion(r)) return "empty";
  if (typeof r.unserved === "string") return "unserved";
  return "failed";
};

export const producerOf = (result: unknown, asked: unknown): string => {
  const r = (result ?? {}) as Record<string, unknown>;
  for (const v of [r.served_model, r.model, r.requested_model, asked]) if (typeof v === "string" && v.length > 0) return v;
  return "unknown";
};

/** Adds one call's outcome to a count map; returns its key. */
export const countOutcome = (counts: Record<string, number>, result: unknown, asked: unknown): string => {
  const k = `${outcomeOf(result)}|${producerOf(result, asked)}`;
  counts[k] = (counts[k] ?? 0) + 1;
  return k;
};
