/**
 * The SINGLE model boundary. This is the only module in the engine permitted to touch a
 * network: AXIOM A1 (the model is invoked at COMPILE TIME only) and AXIOM A9 (that single
 * compile-time call is the only outbound call Ada is allowed). Everything downstream —
 * excavate.ts, orchestrate.ts, the rubric gate, assembly — is pure and model-free (AXIOM
 * A3). The grep-guard tests assert `fetch`/`anthropic`/`openai` appear HERE and nowhere
 * else in the engine.
 *
 * Two providers sit behind the one `ModelClient` seam (A9 — still a single compile-time call):
 *   • `anthropicClient` — the direct Messages API (needs ANTHROPIC_API_KEY, pay-per-token).
 *   • `claudeCodeClient` — BORROWS the local `claude` CLI in headless mode, so the call runs on
 *     the user's Claude Code SUBSCRIPTION with NO API key. This is the "borrow the harness, don't
 *     build the runtime" axiom applied to billing: sovereignty (A9) is stronger, not weaker —
 *     Ada stores no key; auth lives in Claude Code's own credential store.
 * `resolveProvider` picks between them; `defaultModelClient` builds the chosen one.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";

export interface ModelClient {
  /** One compile-time completion. The only network the engine is permitted (A1/A9). */
  complete(prompt: string): Promise<string>;
}

/** The default boundary: no client configured yet. Throws rather than silently faking. */
export function notConfigured(): ModelClient {
  return {
    async complete(): Promise<string> {
      throw new Error(
        "No model configured. Inject a ModelClient — the real compile-time client lives " +
          "in engine/model.ts behind this interface (A1/A9: the only permitted outbound call).",
      );
    },
  };
}

// ── Spend metering (the agent reads this; the user uses it to cut cost) ──────────────────────

/** One call's token usage + cost. `costUsd` is provider-exact when known, else null (estimate it). */
export interface CallUsage {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  costUsd: number | null;
}

/** A per-compile accumulator: how many calls, how many tokens, and how much it cost. */
export interface UsageMeter {
  calls: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  costUsd: number;
  /** True if ANY call's cost was estimated from a price table (API path) vs provider-reported. */
  costEstimated: boolean;
}

export function newUsageMeter(): UsageMeter {
  return {
    calls: 0,
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheCreationTokens: 0,
    costUsd: 0,
    costEstimated: false,
  };
}

/** Fold one call's usage into the meter. `estimated` flags a table-derived (not exact) cost. */
export function recordUsage(
  meter: UsageMeter,
  u: CallUsage,
  estimated: boolean,
): void {
  meter.calls += 1;
  meter.inputTokens += u.inputTokens;
  meter.outputTokens += u.outputTokens;
  meter.cacheReadTokens += u.cacheReadTokens;
  meter.cacheCreationTokens += u.cacheCreationTokens;
  if (typeof u.costUsd === "number") meter.costUsd += u.costUsd;
  if (estimated) meter.costEstimated = true;
}

/**
 * Published per-million-token prices, by model family — for estimating the API path's cost.
 * Source: https://platform.claude.com/docs/en/about-claude/models/overview (verified 2026-06-26).
 * Opus 4.8 $5/$25 · Sonnet 4.6 $3/$15 · Haiku 4.5 $1/$5. (The old $15/$75 was Opus 4.1, now deprecated.)
 */
const PRICE_PER_MTOK: Record<string, { in: number; out: number }> = {
  opus: { in: 5, out: 25 },
  sonnet: { in: 3, out: 15 },
  haiku: { in: 1, out: 5 },
};

/** Estimate USD for an API call from the model family. Unknown family → null (don't guess). */
export function estimateCostUsd(
  model: string,
  inTokens: number,
  outTokens: number,
): number | null {
  const m = model.toLowerCase();
  const fam = m.includes("opus")
    ? "opus"
    : m.includes("sonnet")
      ? "sonnet"
      : m.includes("haiku")
        ? "haiku"
        : null;
  if (!fam) return null;
  const p = PRICE_PER_MTOK[fam]!;
  return (inTokens / 1_000_000) * p.in + (outTokens / 1_000_000) * p.out;
}

/**
 * Pull the answer text AND the usage/cost out of `claude -p --output-format json`. That format is
 * an ARRAY of stream events; the `result` event carries `.result` (the text), a `.usage` block, and
 * `.total_cost_usd` (provider-exact). The big `cache_read_input_tokens` is Claude Code's own system
 * prompt riding every call — the overhead of borrowing the harness. Falls back to raw text (no usage)
 * if the output isn't the JSON envelope (e.g. an error string), so callers degrade cleanly.
 */
export function parseClaudeCodeOutput(stdout: string): {
  text: string;
  usage: CallUsage | null;
} {
  const raw = stdout.trim();
  try {
    const parsed = JSON.parse(raw) as unknown;
    const ev = (
      Array.isArray(parsed)
        ? parsed.find(
            (e) =>
              e &&
              typeof e === "object" &&
              (e as { type?: string }).type === "result",
          )
        : parsed
    ) as Record<string, unknown> | undefined;
    if (ev && typeof ev["result"] === "string") {
      const u = (ev["usage"] ?? {}) as Record<string, unknown>;
      const num = (v: unknown): number => (typeof v === "number" ? v : 0);
      return {
        text: ev["result"] as string,
        usage: {
          inputTokens: num(u["input_tokens"]),
          outputTokens: num(u["output_tokens"]),
          cacheReadTokens: num(u["cache_read_input_tokens"]),
          cacheCreationTokens: num(u["cache_creation_input_tokens"]),
          costUsd:
            typeof ev["total_cost_usd"] === "number"
              ? (ev["total_cost_usd"] as number)
              : null,
        },
      };
    }
  } catch {
    // not the JSON envelope — fall through to raw text
  }
  return { text: raw, usage: null };
}

/** Tuning for the live client. Defaults are conservative; never carries a key. */
export interface AnthropicOptions {
  /** Model id. Override via `ADA_MODEL` env or this arg; defaults to a current Claude. */
  model?: string;
  /** Output token ceiling for one excavation completion. */
  maxTokens?: number;
  /** Wall-clock budget for one call before it aborts. Env `ADA_MODEL_TIMEOUT_MS` else 120s. */
  timeoutMs?: number;
  /** Optional spend meter — each call folds its token usage + cost in, for the compile's report. */
  meter?: UsageMeter;
}

/** Default per-call wall-clock ceiling — a hung connection must not hang the whole compile. */
const DEFAULT_TIMEOUT_MS = 120_000;
/**
 * The Claude Code provider is slower per call than the raw API: each completion is a fresh `claude`
 * process (cold-start) plus generation, measured at ~45–75s for a real excavation/normalize prompt.
 * So it gets a much higher default ceiling than the API path — a single slow call must not abort
 * the whole compile. Still overridable via ADA_MODEL_TIMEOUT_MS / `timeoutMs`.
 */
const CLAUDE_CODE_DEFAULT_TIMEOUT_MS = 300_000;

/**
 * An abort budget: aborts the returned signal after `ms`, with a `clear()` to cancel the timer
 * on a normal return. Pure timer logic (the timer is unref'd so it never holds the process open),
 * so the timeout contract is unit-testable without a network. MODELCALL.003 (production-ready).
 */
export function deadline(ms: number): {
  signal: AbortSignal;
  clear: () => void;
} {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), ms);
  (t as { unref?: () => void }).unref?.();
  return { signal: ac.signal, clear: () => clearTimeout(t) };
}

const ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages";
// The dated Messages API version header (stable contract, not a secret).
const ANTHROPIC_VERSION = "2023-06-01";
// Default to the most capable current Claude — the "first node must impress" bet (A8)
// rides on excavation quality. Override with ADA_MODEL (e.g. claude-sonnet-4-6) for
// faster/cheaper runs.
const DEFAULT_MODEL = "claude-opus-4-8";
const DEFAULT_MAX_TOKENS = 4096;

/** Narrow shape of the bits of the Messages response we read. */
interface MessagesResponse {
  content?: Array<{ type?: string; text?: string }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
  };
}

/**
 * The REAL compile-time client (AXIOM A1/A9). Uses Node 22 global `fetch` — NO SDK, NO new
 * dependency. The API key is read from `process.env.ANTHROPIC_API_KEY` ONLY: never
 * hardcoded, never defaulted, never logged. If the env var is absent we throw a clear,
 * actionable error rather than fall back to a fake response. The prompt is sent to the
 * Messages API but is NEVER logged, and the key is NEVER logged or echoed.
 *
 * Request/response shape verified against the Anthropic Messages API:
 *   POST /v1/messages  · headers: x-api-key, anthropic-version, content-type
 *   body: { model, max_tokens, messages:[{role:"user",content}] }
 *   response: { content: [{ type:"text", text }] }  (text blocks concatenated)
 */
export function anthropicClient(options: AnthropicOptions = {}): ModelClient {
  const model = options.model ?? process.env["ADA_MODEL"] ?? DEFAULT_MODEL;
  const maxTokens = options.maxTokens ?? DEFAULT_MAX_TOKENS;
  const envTimeout = Number(process.env["ADA_MODEL_TIMEOUT_MS"]);
  const timeoutMs =
    options.timeoutMs ??
    (Number.isFinite(envTimeout) && envTimeout > 0
      ? envTimeout
      : DEFAULT_TIMEOUT_MS);
  return {
    async complete(prompt: string): Promise<string> {
      // Key from env at call time (the CLI loads it from ./.env or ~/.ada/.env into
      // process.env at startup; env always wins). Never defaulted, never logged (A1/A9).
      const apiKey = process.env["ANTHROPIC_API_KEY"];
      if (!apiKey) {
        throw new Error(
          "ANTHROPIC_API_KEY is not set. The compile-time model call (AXIOM A1/A9) reads it " +
            "from the environment only — never hardcoded, never logged. Set it once:\n" +
            "  mkdir -p ~/.ada && printf 'ANTHROPIC_API_KEY=sk-ant-...\\n' >> ~/.ada/.env\n" +
            "  # or for this session: export ANTHROPIC_API_KEY=sk-ant-...\n" +
            '  then retry: ada compile --engine "<your intent>"   (check with: ada key)',
        );
      }

      // Abort budget (MODELCALL.003): a hung connection must not hang the whole compile.
      const budget = deadline(timeoutMs);
      let res: Response;
      try {
        res = await fetch(ANTHROPIC_MESSAGES_URL, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-api-key": apiKey,
            "anthropic-version": ANTHROPIC_VERSION,
          },
          body: JSON.stringify({
            model,
            max_tokens: maxTokens,
            messages: [{ role: "user", content: prompt }],
          }),
          signal: budget.signal,
        });
      } catch (cause) {
        // A timeout aborts the request — surface it as a timeout, not a generic network error.
        if (budget.signal.aborted) {
          throw new Error(
            `Anthropic request timed out after ${timeoutMs}ms with no response. ` +
              "Retry, or raise the budget via ADA_MODEL_TIMEOUT_MS.",
          );
        }
        // Network-level failure. Surface a clean message WITHOUT the key or the prompt.
        const detail = cause instanceof Error ? cause.message : String(cause);
        throw new Error(`Anthropic request failed (network): ${detail}`);
      } finally {
        // Response headers are in hand (or it threw) — stop the abort timer either way.
        budget.clear();
      }

      if (!res.ok) {
        // Read the body for the API's own error text, but NEVER log the key or prompt.
        const body = await res.text().catch(() => "");
        const trimmed = body.slice(0, 500);
        throw new Error(
          `Anthropic API error ${res.status} ${res.statusText}` +
            (trimmed ? `: ${trimmed}` : ""),
        );
      }

      const data = (await res.json()) as MessagesResponse;
      const text = (data.content ?? [])
        .filter((b) => b.type === "text" && typeof b.text === "string")
        .map((b) => b.text as string)
        .join("");
      if (!text) {
        throw new Error(
          "Anthropic API returned no text content. Expected a content block of " +
            "type 'text'; got an empty or non-text response.",
        );
      }
      // The API doesn't hand us a $ figure → estimate from the model family's price table.
      if (options.meter) {
        const inTok = data.usage?.input_tokens ?? 0;
        const outTok = data.usage?.output_tokens ?? 0;
        recordUsage(
          options.meter,
          {
            inputTokens: inTok,
            outputTokens: outTok,
            cacheReadTokens: data.usage?.cache_read_input_tokens ?? 0,
            cacheCreationTokens: data.usage?.cache_creation_input_tokens ?? 0,
            costUsd: estimateCostUsd(model, inTok, outTok),
          },
          true,
        );
      }
      return text;
    },
  };
}

// ── Provider B: borrow the Claude Code subscription (no API key) ─────────────────────────────

/** Which provider satisfies the one compile-time call. */
export type ModelProvider = "api" | "claude-code";

function isProvider(v: unknown): v is ModelProvider {
  return v === "api" || v === "claude-code";
}

/**
 * Decide which provider serves the compile-time call. Pure (env + availability in, choice out),
 * so the precedence is unit-testable with no spawn. Precedence, highest first:
 *   1. an explicit choice (a `--provider` flag),
 *   2. the `ADA_MODEL_PROVIDER` env (the user's case: a key is present but unusable → force CC),
 *   3. PREFER the Claude Code subscription when the `claude` binary is present — the zero-key
 *      default that makes a non-expert's first run work with no setup,
 *   4. else the API key if set, else 'api' (so the clear, actionable no-key error surfaces).
 */
export function resolveProvider(args: {
  explicit?: string | undefined;
  env: Record<string, string | undefined>;
  claudeAvailable: boolean;
}): ModelProvider {
  if (isProvider(args.explicit)) return args.explicit;
  if (isProvider(args.env["ADA_MODEL_PROVIDER"]))
    return args.env["ADA_MODEL_PROVIDER"] as ModelProvider;
  if (args.claudeAvailable) return "claude-code";
  if ((args.env["ANTHROPIC_API_KEY"] ?? "").trim()) return "api";
  return "api";
}

/**
 * Is the `claude` CLI on PATH? Synchronous PATH scan (no spawn) so it's cheap and side-effect
 * free. Used by `resolveProvider` to default to the subscription when it's actually available.
 */
export function claudeCodeAvailable(
  env: Record<string, string | undefined> = process.env,
): boolean {
  const path = env["PATH"];
  if (!path) return false;
  return path
    .split(delimiter)
    .some((dir) => dir && existsSync(join(dir, "claude")));
}

/**
 * The headless `claude` argv for a one-shot text completion. NEVER `--bare` — bare mode forces
 * ANTHROPIC_API_KEY / apiKeyHelper auth and DROPS the subscription OAuth, defeating the whole
 * point (verified: `--bare` → "Not logged in"). `--strict-mcp-config`/`--disable-slash-commands`
 * keep OAuth while skipping the user's MCP servers + skills (cleaner, no tool use). Pure → testable.
 */
export function buildClaudeCodeArgs(opts: { model?: string } = {}): string[] {
  return [
    "-p",
    "--output-format",
    "json", // JSON envelope so we can read per-call token usage + total_cost_usd (the spend report)
    "--strict-mcp-config",
    "--disable-slash-commands",
    ...(opts.model ? ["--model", opts.model] : []),
  ];
}

/**
 * BORROW the harness: run the compile-time call through the local `claude` CLI on the user's
 * subscription — no API key. The prompt goes in on STDIN (no ARG_MAX limit); the answer + the token
 * usage + the cost are read from stdout (`--output-format json`). Run in a NEUTRAL cwd (a temp dir)
 * so the target repo's CLAUDE.md/hooks never load (which would be slow, polluting, and — inside Ada's
 * own repo — recursive). ANTHROPIC_API_KEY is STRIPPED from the child env so a present-but-dead key
 * can't route this to the paid API. Timeout via the `deadline` budget, enforced by killing the child.
 * When a `meter` is provided, each call's provider-exact usage + cost is folded in (the spend report).
 */
export function claudeCodeClient(options: AnthropicOptions = {}): ModelClient {
  const model = options.model ?? process.env["ADA_MODEL"];
  const envTimeout = Number(process.env["ADA_MODEL_TIMEOUT_MS"]);
  const timeoutMs =
    options.timeoutMs ??
    (Number.isFinite(envTimeout) && envTimeout > 0
      ? envTimeout
      : CLAUDE_CODE_DEFAULT_TIMEOUT_MS);
  return {
    async complete(prompt: string): Promise<string> {
      const childEnv: Record<string, string | undefined> = { ...process.env };
      delete childEnv["ANTHROPIC_API_KEY"]; // force subscription auth (A9: no key in play)
      const args = buildClaudeCodeArgs(model ? { model } : {});
      return await new Promise<string>((resolve, reject) => {
        const child = spawn("claude", args, {
          cwd: tmpdir(),
          env: childEnv,
          stdio: ["pipe", "pipe", "pipe"],
        });
        let out = "";
        let err = "";
        const timer = setTimeout(() => {
          child.kill("SIGKILL");
          reject(
            new Error(
              `claude -p timed out after ${timeoutMs}ms. Raise ADA_MODEL_TIMEOUT_MS, or use ` +
                "--provider=api with ANTHROPIC_API_KEY set.",
            ),
          );
        }, timeoutMs);
        (timer as { unref?: () => void }).unref?.();
        child.stdout.on("data", (d) => (out += String(d)));
        child.stderr.on("data", (d) => (err += String(d)));
        child.on("error", (e) => {
          clearTimeout(timer);
          reject(
            new Error(
              `Could not run the 'claude' CLI (${e.message}). Install Claude Code and run ` +
                "`claude login`, or use --provider=api with ANTHROPIC_API_KEY set.",
            ),
          );
        });
        child.on("close", (code) => {
          clearTimeout(timer);
          const { text, usage } = parseClaudeCodeOutput(out);
          if (code !== 0 || !text) {
            const detail = (err.trim() || out.trim() || `exit ${code}`).slice(
              0,
              500,
            );
            reject(
              new Error(
                `claude -p failed (${detail}). If it says "Not logged in", run \`claude login\`; ` +
                  "if it's a usage limit, wait or use --provider=api.",
              ),
            );
            return;
          }
          // Provider-EXACT spend (claude reports total_cost_usd) — fold it into the meter.
          if (options.meter && usage) recordUsage(options.meter, usage, false);
          resolve(text);
        });
        child.stdin.write(prompt);
        child.stdin.end();
      });
    },
  };
}

// ── Per-call retry (unattended robustness) ───────────────────────────────────────────────────

/**
 * Should a failed compile-time call be retried? TRANSIENT failures (a timeout, a momentary usage /
 * rate limit, a network blip, a 5xx/overload, an empty response) recover on a backed-off retry — the
 * exact class that killed two unattended compiles when run concurrently. PERMANENT failures (not
 * logged in, no API key, a missing `claude` binary, an auth error) will NEVER recover by retrying, so
 * they fail fast. Pure (message in, verdict out) → unit-testable. The default is transient-biased: an
 * unknown error retries, because aborting a long compile is costlier than one extra call.
 */
export function isRetryableError(err: unknown): boolean {
  const msg = (err instanceof Error ? err.message : String(err)).toLowerCase();
  // Permanent — a retry cannot fix these, so don't waste the backoff.
  if (msg.includes("not logged in")) return false;
  if (msg.includes("anthropic_api_key is not set")) return false;
  if (msg.includes("could not run the 'claude' cli")) return false;
  if (/\b(401|403)\b/.test(msg) || msg.includes("authentication")) return false;
  // Everything else is treated as transient (timeout / usage limit / network / 5xx / empty / unknown).
  return true;
}

/** Tuning for `withRetry`. All optional; the defaults suit an unattended scheduled compile. */
export interface RetryOptions {
  /** Total attempts INCLUDING the first. Default 3 (env `ADA_MODEL_RETRIES` overrides the default). */
  maxAttempts?: number;
  /** First backoff in ms; each retry doubles it. Default 2000. */
  baseDelayMs?: number;
  /** Backoff ceiling in ms (the doubling is clamped here). Default 30000. */
  maxDelayMs?: number;
  /** Injected sleep (tests pass a no-op recorder); default is a real unref'd timer. */
  sleep?: (ms: number) => Promise<void>;
  /** Override the retryable classifier (tests). Default `isRetryableError`. */
  isRetryable?: (err: unknown) => boolean;
  /** Observe each retry (attempt #, the delay about to be waited, the error) — e.g. progress logging. */
  onRetry?: (attempt: number, delayMs: number, err: unknown) => void;
}

const realSleep = (ms: number): Promise<void> =>
  new Promise((r) => {
    const t = setTimeout(r, ms);
    (t as { unref?: () => void }).unref?.();
  });

/**
 * Decorate any `ModelClient` so a TRANSIENT failure backs off and retries instead of aborting the
 * whole compile. Exponential backoff (baseDelayMs · 2^n, clamped to maxDelayMs); permanent errors
 * (auth/no-key/missing-binary) throw immediately. This is the load-bearing fix for unattended Ada:
 * one hung/limit-hit call no longer kills a multi-call compile. Wrapping is transparent — still ONE
 * logical compile-time call per `complete()` from the caller's view (A1/A9), just resilient.
 */
export function withRetry(
  client: ModelClient,
  options: RetryOptions = {},
): ModelClient {
  const maxAttempts = options.maxAttempts ?? 3;
  const baseDelayMs = options.baseDelayMs ?? 2000;
  const maxDelayMs = options.maxDelayMs ?? 30_000;
  const sleep = options.sleep ?? realSleep;
  const retryable = options.isRetryable ?? isRetryableError;
  return {
    async complete(prompt: string): Promise<string> {
      let lastErr: unknown;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          return await client.complete(prompt);
        } catch (err) {
          lastErr = err;
          if (attempt >= maxAttempts || !retryable(err)) throw err;
          const delayMs = Math.min(
            maxDelayMs,
            baseDelayMs * 2 ** (attempt - 1),
          );
          options.onRetry?.(attempt, delayMs, err);
          await sleep(delayMs);
        }
      }
      throw lastErr; // unreachable: the loop returns or throws on the last attempt
    },
  };
}

/** Options for the provider-selecting default client: tuning + an explicit provider override. */
export interface DefaultClientOptions extends AnthropicOptions {
  /** Force a provider (a `--provider` flag). Undefined → `resolveProvider` decides. */
  provider?: string | undefined;
}

/**
 * The one door every caller goes through (engineCompile injects nothing → this picks the provider).
 * Borrows the Claude Code subscription by default when available (no key); honors an explicit
 * `--provider`/`ADA_MODEL_PROVIDER`; falls back to the API key. Still ONE compile-time call (A9).
 */
export function defaultModelClient(
  options: DefaultClientOptions = {},
): ModelClient {
  const provider = resolveProvider({
    explicit: options.provider,
    env: process.env,
    claudeAvailable: claudeCodeAvailable(),
  });
  const tuning: AnthropicOptions = {
    ...(options.model ? { model: options.model } : {}),
    ...(options.maxTokens ? { maxTokens: options.maxTokens } : {}),
    ...(options.timeoutMs ? { timeoutMs: options.timeoutMs } : {}),
    ...(options.meter ? { meter: options.meter } : {}),
  };
  const base =
    provider === "claude-code"
      ? claudeCodeClient(tuning)
      : anthropicClient(tuning);
  // Resilience for unattended runs: a transient timeout / rate-limit / network hiccup retries with
  // backoff instead of aborting the whole compile. Permanent errors (auth/no-key) still fail fast.
  // `ADA_MODEL_RETRIES` overrides the attempt count (0/1 → effectively no retry); default 3.
  const envRetries = Number(process.env["ADA_MODEL_RETRIES"]);
  const maxAttempts =
    Number.isFinite(envRetries) && envRetries >= 1 ? Math.floor(envRetries) : 3;
  return withRetry(base, { maxAttempts });
}
