// Why an OpenAI-wire completion stopped, said from what the resolver holds, not only from what the provider said.
//
// A provider can answer finish_reason "stop" for an output its token cap cut: the text then reads as whole
// downstream and is parsed as if it were. The resolver holds the cap it sent and the output tokens it was
// billed, so an output that reached its cap is said cut, whatever the provider said. This fails closed: an
// output that ends exactly at the cap of its own accord is said cut too, which costs a retry, never a cut
// output parsed as whole.

/** stop_reason for a wire completion: "max_tokens" when the provider said "length" or the output reached `cap`; the provider's own word beside it. */
export function wireStopReason(finish: string | null | undefined, completionTokens: number | undefined, cap: number): { stop_reason: string | null | undefined; provider_stop_reason: string | null | undefined } {
  const cut = finish === "length" || (typeof completionTokens === "number" && completionTokens >= cap);
  return { stop_reason: cut ? "max_tokens" : finish, provider_stop_reason: finish };
}
