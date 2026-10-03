// THE NODE KEY NEVER FOLLOWS A CALLER-SUPPLIED TOOL-DISPATCH ENDPOINT.
//
// An llm completion with tools takes `tool_dispatch_endpoint` and `tool_dispatch_api_key` from its
// request body. The endpoint is the fallback when discovery has no producer for a tool, and the key used to
// default to this vessel's METABOB_API_KEY. So a caller who named its own endpoint, and no key, received
// `Authorization: ApiKey <node key>` on every tool call.
//
// The rule:
//   - a caller-supplied endpoint gets NO node key: only the caller's own tool_dispatch_api_key, or nothing;
//   - tool_dispatch_api_key is honoured only together with that override (the caller's key to the caller's
//     endpoint). It is never sent to a configured or discovered endpoint;
//   - no override: the configured or discovered endpoint gets the node key, as before;
//   - an endpoint a request supplied is never remembered as the tool's endpoint for later requests.
//
// Every case uses a RECORDING fetch stub. Discovery answers 404 ("no producer") unless a case gives it a
// vessel.
import { describe, expect, test, afterEach } from "bun:test";
import { makeToolEndpointResolver, toolDispatchRoute, dispatchToolCall } from "./tool-dispatch";

const NODE_KEY = "node-key-under-test";
const CALLER_KEY = "caller-own-key";
const DISCOVERY = "http://discovery.test";
const CONFIGURED = "http://127.0.0.1:8090/v2/impulses/resolve";
const ATTACKER = "http://attacker.invalid/v2/impulses/resolve";
const DISCOVERED = "http://tool-vessel.test/v2/impulses/resolve";

type Seen = { url: string; authorization: string | null };
const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function recordingFetch(discoveryAnswers: boolean): Seen[] {
  const seen: Seen[] = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : (input as Request).url;
    seen.push({ url, authorization: new Headers(init?.headers ?? {}).get("authorization") });
    if (url.startsWith(DISCOVERY)) {
      return discoveryAnswers
        ? new Response(JSON.stringify({ content: { vessels: [{ endpoint: "http://tool-vessel.test", resolve_endpoint: "/v2/impulses/resolve" }] } }), { status: 200 })
        : new Response("{}", { status: 404 });
    }
    return new Response(JSON.stringify({ shape: "x", body: { ok: true } }), { status: 200 });
  }) as unknown as typeof fetch;
  return seen;
}

/** One request's tool call, through the same composition both loops in index.ts use. */
async function send(
  resolve: ReturnType<typeof makeToolEndpointResolver>,
  body: { tool_dispatch_endpoint?: string; tool_dispatch_api_key?: string },
): Promise<void> {
  const route = toolDispatchRoute(body, CONFIGURED, NODE_KEY);
  await dispatchToolCall(resolve, route, "web_search", { query: "q" }, new Set(["web_search"]));
}

const toolCalls = (seen: Seen[]) => seen.filter((s) => !s.url.startsWith(DISCOVERY));

describe("tool dispatch: a caller-supplied endpoint gets no node key", () => {
  test("MUST-FAIL (i): an override with no caller key carries no Authorization", async () => {
    const seen = recordingFetch(false);
    await send(makeToolEndpointResolver(DISCOVERY, NODE_KEY), { tool_dispatch_endpoint: ATTACKER });
    expect(toolCalls(seen).map((s) => s.url)).toEqual([ATTACKER]);
    expect(toolCalls(seen)[0]!.authorization).toBeNull();
  });

  // (ii) already holds at base (the caller's key won over the node key); it pins that the fix keeps it.
  test("(ii): an override with the caller's key carries only the caller's key", async () => {
    const seen = recordingFetch(false);
    await send(makeToolEndpointResolver(DISCOVERY, NODE_KEY), { tool_dispatch_endpoint: ATTACKER, tool_dispatch_api_key: CALLER_KEY });
    expect(toolCalls(seen)).toEqual([{ url: ATTACKER, authorization: `ApiKey ${CALLER_KEY}` }]);
  });

  test("MUST-FAIL (iii): an endpoint one request supplied is never used, with the node key, for the next", async () => {
    const resolve = makeToolEndpointResolver(DISCOVERY, NODE_KEY);
    const seen = recordingFetch(false);
    await send(resolve, { tool_dispatch_endpoint: ATTACKER });
    await send(resolve, {});
    const second = toolCalls(seen)[1]!;
    expect(second).toEqual({ url: CONFIGURED, authorization: `ApiKey ${NODE_KEY}` });
  });

  test("MUST-FAIL (iv): a caller key is never sent to the configured endpoint", async () => {
    const seen = recordingFetch(false);
    await send(makeToolEndpointResolver(DISCOVERY, NODE_KEY), { tool_dispatch_api_key: CALLER_KEY });
    expect(toolCalls(seen)).toEqual([{ url: CONFIGURED, authorization: `ApiKey ${NODE_KEY}` }]);
  });

  test("a blank override means unset: the configured endpoint, with the node key", async () => {
    const seen = recordingFetch(false);
    await send(makeToolEndpointResolver(DISCOVERY, NODE_KEY), { tool_dispatch_endpoint: "  " });
    expect(toolCalls(seen)).toEqual([{ url: CONFIGURED, authorization: `ApiKey ${NODE_KEY}` }]);
  });

  test("control: no override, discovery has no producer: the configured endpoint gets the node key", async () => {
    const seen = recordingFetch(false);
    await send(makeToolEndpointResolver(DISCOVERY, NODE_KEY), {});
    expect(toolCalls(seen)).toEqual([{ url: CONFIGURED, authorization: `ApiKey ${NODE_KEY}` }]);
  });

  test("control: discovery names a producer: it gets the node key", async () => {
    const seen = recordingFetch(true);
    await send(makeToolEndpointResolver(DISCOVERY, NODE_KEY), {});
    expect(toolCalls(seen)).toEqual([{ url: DISCOVERED, authorization: `ApiKey ${NODE_KEY}` }]);
  });

  test("MUST-FAIL (v): with an override, a discovered producer still gets the node key, not the caller's key", async () => {
    const seen = recordingFetch(true);
    await send(makeToolEndpointResolver(DISCOVERY, NODE_KEY), { tool_dispatch_endpoint: ATTACKER, tool_dispatch_api_key: CALLER_KEY });
    expect(toolCalls(seen)).toEqual([{ url: DISCOVERED, authorization: `ApiKey ${NODE_KEY}` }]);
  });
});
