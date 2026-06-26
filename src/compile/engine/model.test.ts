import { test } from "node:test";
import assert from "node:assert/strict";
import {
  deadline,
  resolveProvider,
  buildClaudeCodeArgs,
  parseClaudeCodeOutput,
  estimateCostUsd,
  newUsageMeter,
  recordUsage,
  withRetry,
  isRetryableError,
  type ModelClient,
} from "./model.js";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// A stub that fails the first `failTimes` calls with `err`, then returns `ok`. Counts calls.
function flakyClient(
  failTimes: number,
  err: Error,
  ok = "OK",
): { client: ModelClient; calls: () => number } {
  let calls = 0;
  return {
    calls: () => calls,
    client: {
      async complete(): Promise<string> {
        calls++;
        if (calls <= failTimes) throw err;
        return ok;
      },
    },
  };
}

// A sleep that records the delays it was asked to wait (so backoff is asserted without real time).
function recordingSleep(): {
  fn: (ms: number) => Promise<void>;
  delays: number[];
} {
  const delays: number[] = [];
  return { delays, fn: async (ms: number) => void delays.push(ms) };
}

// ── Provider selection (borrow-the-harness: run on the Claude Code subscription, no API key) ──

test("resolveProvider: an explicit choice wins over everything (flag/env override)", () => {
  assert.equal(
    resolveProvider({ explicit: "api", env: {}, claudeAvailable: true }),
    "api",
  );
  assert.equal(
    resolveProvider({
      explicit: "claude-code",
      env: { ANTHROPIC_API_KEY: "sk-x" },
      claudeAvailable: false,
    }),
    "claude-code",
    "explicit claude-code is honored even with a key set",
  );
});

test("resolveProvider: ADA_MODEL_PROVIDER env is the next authority after an explicit flag", () => {
  assert.equal(
    resolveProvider({
      env: { ADA_MODEL_PROVIDER: "claude-code", ANTHROPIC_API_KEY: "sk-x" },
      claudeAvailable: false,
    }),
    "claude-code",
    "the env override beats a present API key (the user's case: key present but unusable)",
  );
});

test("resolveProvider: with no override, PREFER the Claude Code subscription when the binary is present (zero-key default)", () => {
  assert.equal(
    resolveProvider({ env: {}, claudeAvailable: true }),
    "claude-code",
    "the non-expert default: use the subscription that's already authenticated, no key needed",
  );
});

test("resolveProvider: falls back to the API key when claude is absent; to 'api' (clear no-key error) when neither", () => {
  assert.equal(
    resolveProvider({
      env: { ANTHROPIC_API_KEY: "sk-x" },
      claudeAvailable: false,
    }),
    "api",
    "no claude binary but a key → API",
  );
  assert.equal(
    resolveProvider({ env: {}, claudeAvailable: false }),
    "api",
    "neither available → API path, which surfaces the actionable no-key error",
  );
});

test("buildClaudeCodeArgs: headless, JSON out (for usage capture), MCP/skills off — never --bare (it breaks subscription auth)", () => {
  const args = buildClaudeCodeArgs({});
  assert.ok(args.includes("-p"), "headless print mode");
  assert.equal(
    args[args.indexOf("--output-format") + 1],
    "json",
    "JSON output so we can read per-call token usage + cost (the spend report)",
  );
  assert.ok(args.includes("--strict-mcp-config"), "no MCP servers loaded");
  assert.ok(args.includes("--disable-slash-commands"), "no skills loaded");
  assert.ok(
    !args.includes("--bare"),
    "MUST NOT use --bare — it forces API-key auth and breaks the subscription",
  );
});

// ── Spend reporting (the agent needs to see token + $ cost; the user needs to save money) ──

test("parseClaudeCodeOutput pulls the text AND the usage/cost from claude -p's JSON event array", () => {
  // The real shape (verified): an array of stream events; the `result` event carries the answer,
  // a usage block, and total_cost_usd. The big number is cache_read — Claude Code's own ~31K-token
  // system prompt rides every call (the cost of borrowing the harness).
  const stdout = JSON.stringify([
    { type: "system", subtype: "init", session_id: "x" },
    { type: "assistant", message: {} },
    {
      type: "result",
      result: '{"id":"X.1","label":"a node"}',
      total_cost_usd: 0.0307745,
      usage: {
        input_tokens: 3039,
        output_tokens: 412,
        cache_read_input_tokens: 30959,
        cache_creation_input_tokens: 0,
      },
    },
  ]);
  const { text, usage } = parseClaudeCodeOutput(stdout);
  assert.equal(
    text,
    '{"id":"X.1","label":"a node"}',
    "the answer text is extracted",
  );
  assert.equal(usage?.inputTokens, 3039);
  assert.equal(usage?.outputTokens, 412);
  assert.equal(
    usage?.cacheReadTokens,
    30959,
    "the harness-overhead cache read is captured",
  );
  assert.equal(
    usage?.costUsd,
    0.0307745,
    "provider-reported $ is exact, not estimated",
  );
});

test("parseClaudeCodeOutput falls back to raw text when output isn't the JSON envelope (no usage)", () => {
  const { text, usage } = parseClaudeCodeOutput("just plain text\n");
  assert.equal(text, "just plain text");
  assert.equal(usage, null, "no usage when there's no JSON envelope");
});

test("estimateCostUsd prices by model family (API path, where claude doesn't hand us a $ figure)", () => {
  // 1M in + 1M out on Opus 4.8 = $5 + $25 = $30; on Sonnet 4.6 = $3 + $15 = $18. Unknown model → null (honest).
  // Pricing per platform.claude.com/docs/en/about-claude/models/overview (verified 2026-06-26).
  assert.equal(estimateCostUsd("claude-opus-4-8", 1_000_000, 1_000_000), 30);
  assert.equal(estimateCostUsd("claude-sonnet-4-6", 1_000_000, 1_000_000), 18);
  assert.equal(estimateCostUsd("some-unknown-model", 1000, 1000), null);
});

test("recordUsage accumulates a per-compile meter — calls, tokens, cache, and summed cost", () => {
  const m = newUsageMeter();
  recordUsage(
    m,
    {
      inputTokens: 3000,
      outputTokens: 400,
      cacheReadTokens: 31000,
      cacheCreationTokens: 0,
      costUsd: 0.03,
    },
    false,
  );
  recordUsage(
    m,
    {
      inputTokens: 3500,
      outputTokens: 900,
      cacheReadTokens: 31000,
      cacheCreationTokens: 0,
      costUsd: 0.04,
    },
    false,
  );
  assert.equal(m.calls, 2);
  assert.equal(m.inputTokens, 6500);
  assert.equal(m.outputTokens, 1300);
  assert.equal(
    m.cacheReadTokens,
    62000,
    "the harness overhead totals across calls",
  );
  assert.equal(Math.round(m.costUsd * 100) / 100, 0.07);
  assert.equal(
    m.costEstimated,
    false,
    "provider-exact costs are not flagged as estimates",
  );
});

test("buildClaudeCodeArgs: a model override maps to claude --model; absent → no flag (claude's default)", () => {
  const withModel = buildClaudeCodeArgs({ model: "sonnet" });
  assert.equal(
    withModel[withModel.indexOf("--model") + 1],
    "sonnet",
    "the model is threaded to claude --model",
  );
  assert.ok(
    !buildClaudeCodeArgs({}).includes("--model"),
    "no model → let claude use its configured default",
  );
});

test("deadline aborts its signal after the budget elapses (MODELCALL.003: a hung call can't hang the compile)", async () => {
  const d = deadline(20);
  assert.equal(d.signal.aborted, false, "not aborted before the budget");
  await sleep(45);
  assert.equal(d.signal.aborted, true, "aborted once the budget elapsed");
  d.clear();
});

test("deadline.clear() cancels the abort — a call that returns in time is never aborted", async () => {
  const d = deadline(20);
  d.clear(); // the call returned: stop the timer
  await sleep(45);
  assert.equal(d.signal.aborted, false, "cleared timer never fires");
});

test("the abort signal is a real AbortSignal — wired to fetch's signal option", () => {
  const d = deadline(1000);
  assert.ok(
    typeof d.signal === "object" && "aborted" in d.signal,
    "deadline yields an AbortSignal fetch accepts",
  );
  d.clear();
});

// ── Per-call retry (unattended robustness: a transient hiccup must not abort a whole compile) ──

test("isRetryableError: TRANSIENT failures retry — timeout, usage limit, network, 5xx, empty", () => {
  for (const m of [
    "claude -p timed out after 300000ms",
    "claude -p failed (usage limit reached)",
    "Anthropic request failed (network): ECONNRESET",
    "Anthropic API error 529 Overloaded",
    "Anthropic API error 503 Service Unavailable",
    "Anthropic API returned no text content",
  ]) {
    assert.equal(isRetryableError(new Error(m)), true, `should retry: ${m}`);
  }
});

test("isRetryableError: PERMANENT failures do NOT retry — auth / no-key / missing binary", () => {
  for (const m of [
    'claude -p failed (Not logged in). If it says "Not logged in"...',
    "ANTHROPIC_API_KEY is not set. The compile-time model call...",
    "Could not run the 'claude' CLI (ENOENT). Install Claude Code...",
    "Anthropic API error 401 Unauthorized",
    "Anthropic API error 403 authentication_error",
  ]) {
    assert.equal(
      isRetryableError(new Error(m)),
      false,
      `should NOT retry: ${m}`,
    );
  }
});

test("withRetry: a clean first call passes straight through (one call, no sleep)", async () => {
  const { client, calls } = flakyClient(0, new Error("unused"));
  const sl = recordingSleep();
  const r = await withRetry(client, { sleep: sl.fn }).complete("x");
  assert.equal(r, "OK");
  assert.equal(calls(), 1, "no retry when the first call succeeds");
  assert.equal(sl.delays.length, 0, "never slept");
});

test("withRetry: a TRANSIENT failure then success — retries and returns the good result", async () => {
  const { client, calls } = flakyClient(
    1,
    new Error("claude -p timed out after 300000ms"),
  );
  const sl = recordingSleep();
  const r = await withRetry(client, {
    sleep: sl.fn,
    baseDelayMs: 1000,
  }).complete("x");
  assert.equal(r, "OK");
  assert.equal(calls(), 2, "one retry after the transient failure");
  assert.deepEqual(sl.delays, [1000], "backed off once before the retry");
});

test("withRetry: exhausting maxAttempts throws the last error (called exactly maxAttempts times)", async () => {
  const { client, calls } = flakyClient(
    99,
    new Error("claude -p failed (usage limit reached)"),
  );
  const sl = recordingSleep();
  await assert.rejects(
    withRetry(client, {
      sleep: sl.fn,
      maxAttempts: 3,
      baseDelayMs: 1000,
    }).complete("x"),
    /usage limit/,
  );
  assert.equal(calls(), 3, "tried exactly maxAttempts times");
  assert.equal(
    sl.delays.length,
    2,
    "slept between the 3 attempts (after #1 and #2)",
  );
});

test("withRetry: a PERMANENT error throws immediately — no retry, no sleep", async () => {
  const { client, calls } = flakyClient(
    99,
    new Error("claude -p failed (Not logged in)"),
  );
  const sl = recordingSleep();
  await assert.rejects(
    withRetry(client, { sleep: sl.fn }).complete("x"),
    /Not logged in/,
  );
  assert.equal(calls(), 1, "permanent error is not retried");
  assert.equal(sl.delays.length, 0, "never slept");
});

test("withRetry: backoff is exponential and capped at maxDelayMs", async () => {
  const { client } = flakyClient(99, new Error("network blip"));
  const sl = recordingSleep();
  await assert.rejects(
    withRetry(client, {
      sleep: sl.fn,
      maxAttempts: 5,
      baseDelayMs: 1000,
      maxDelayMs: 4000,
    }).complete("x"),
  );
  // 1000, 2000, 4000 (cap), 4000 — exponential then clamped.
  assert.deepEqual(sl.delays, [1000, 2000, 4000, 4000]);
});
