// Why a provider or model was marked exhausted, in the words the provider used: its HTTP status, its error code, and
// its message, redacted by the same function the daemon uses for a thrown resolver error (ias-executor-ts). A cooled
// lane that says only "billing/unreachable" cannot be told apart from a key's spending limit or a model the account
// has no access to; the provider's own answer can.
import { redactedMessage } from "@avigopal/ias-executor-ts";

export type ProviderReason = { status?: number; code?: string; message: string };

const codeOf = (e: Record<string, unknown> | undefined): string | undefined => {
  if (e === undefined) return undefined;
  const inner = (typeof e.error === "object" && e.error !== null ? e.error : undefined) as Record<string, unknown> | undefined;
  const deeper = (typeof inner?.error === "object" && inner.error !== null ? inner.error : undefined) as Record<string, unknown> | undefined;
  const c = [e.code, inner?.code, inner?.type, deeper?.code, deeper?.type].find((x) => typeof x === "string" || typeof x === "number");
  return c === undefined ? undefined : String(c);
};

/** What a failed provider call said: the SDK error's status and code, and its message, redacted. */
export const reasonOf = (err: unknown): ProviderReason => {
  const e = typeof err === "object" && err !== null ? (err as Record<string, unknown>) : undefined;
  const status = typeof e?.status === "number" ? e.status : undefined;
  const code = codeOf(e);
  return { ...(status !== undefined ? { status } : {}), ...(code !== undefined ? { code: redactedMessage(code) } : {}), message: redactedMessage(err) };
};

/** A reason as one line; a result with none (a refusal, a cooling lane) says so. */
export const reasonText = (r: unknown): string => {
  const x = typeof r === "object" && r !== null ? (r as Partial<ProviderReason>) : undefined;
  if (typeof x?.message !== "string") return "no provider answer recorded";
  return `provider answered ${typeof x.status === "number" ? x.status : "no status"}${typeof x.code === "string" ? ` ${x.code}` : ""}: ${x.message}`;
};
