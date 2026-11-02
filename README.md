# Agent runtime limits, measured: a reproducible case pack

**Snapshot of 2026-11-01, CrewAI added 2026-11-02.** Versions move fast; re-run `lab/reproduce.sh` before citing a number.

Small, offline experiments that show what agent-framework limits actually bound.
Every model is a scripted test double, so nothing calls an API and every count is deterministic.
Tested on macOS arm64, Python 3.12 (3.12.11, CrewAI on 3.12.13), with the exact versions in `requirements-*.txt`.

This is **not** a new budget library, a security product or a vulnerability report.
Most behaviours below are documented or intended. The value is one table, one shape,
real numbers, and links to the people already working on each gap.

## 1. A turn/step limit is not a tree budget

Same shape everywhere: the parent may make 2 model calls, each child agent makes 3.

| Framework (version) | Limit tried | Tree total, 1 child | Tree total, 2 parallel children | Child inherits the parent's limit value? | Shared remaining budget by default? |
|---|---|---|---|---|---|
| openai-agents 0.24.0 | `Runner.run(max_turns=2)`, `as_tool(max_turns=3)` | **5** | **8** | No. Omitted `as_tool(max_turns)` means 10, whatever the parent has | No |
| pydantic-ai-slim 2.64.0 | `UsageLimits(request_limit=2)` on the parent | **5** | **8** | No. A delegate run starts its own counter (default 50) | No (opt-in, see below) |
| langgraph 1.2.19 | `recursion_limit=6` | **5** | **8** | Yes, the value is passed down | No. Each graph counts its own supersteps |
| google-adk 2.13.0 | `RunConfig(max_llm_calls=3)` with `AgentTool` | **5** | **8** | Yes, the value is passed down | No. Counted per invocation |
| smolagents 1.26.0 | parent `max_steps=2`, child `max_steps=3` | **5** | **8** | No | No |
| crewai 1.15.34 (added 2026-11-02) | parent `max_iter=2`, coworker `max_iter=3`, `allow_delegation=True` | **5** | **9** (two children, one after the other) | No. A coworker without `max_iter` uses its own default (25) | No |

Source-level notes from the same versions (not run here): Microsoft Agent Framework 1.24.0 and
AutoGen AgentChat 0.7.5 also count per agent/run. Claude Agent SDK's `max_budget_usd` does include
subagent spend (dollars, checked after responses).

### The delegation style matters more than the framework

| Framework | Style | Parent limit | Calls | Result |
|---|---|---|---|---|
| openai-agents 0.24.0 | handoff (same run loop) | `max_turns=2` | 2 | stopped: `MaxTurnsExceeded` |
| openai-agents 0.24.0 | `agent.as_tool` (nested run) | `max_turns=2` | 5 | completed |
| google-adk 2.13.0 | `transfer_to_agent` (same invocation) | `max_llm_calls=3` | 3 | stopped |
| google-adk 2.13.0 | `AgentTool` (nested runner) | `max_llm_calls=3` | 5 | completed |

### Details that surprised us

- **openai-agents:** parent `RunHooks.on_llm_start` fired for the 2 parent calls only, not for the
  3 child calls, while the final `usage.requests` was 5. Passing the **same hooks object** to
  `as_tool(hooks=...)` made it see all 5 calls live, so a hook-based shared counter is the cheapest
  in-process guard in this SDK (related discussion: [#2036](https://github.com/openai/openai-agents-python/issues/2036)).
- **pydantic-ai:** without `usage=ctx.usage` the parent's result reports **2 requests** for 5 real calls.
  Passing only `usage=ctx.usage` lets the child run past the parent's `request_limit=2` (4 calls before
  the parent stops). Passing `usage` **and** `usage_limits` enforces it, except under parallel
  delegation: `request_limit=4` allowed **5** calls. This is already tracked in
  [pydantic-ai#10142](https://github.com/pydantic/pydantic-ai/issues/10142) and
  [#9255](https://github.com/pydantic/pydantic-ai/issues/9255); the design question of inherited
  budgets is [#11918](https://github.com/pydantic/pydantic-ai/issues/11918).
- **smolagents:** `max_steps=3` on a looping child made **4** model calls (one extra call to write a
  final answer when the limit is hit).
- **crewai:** in ReAct mode each model call makes one delegation, so two coworkers need a third parent
  call. `max_iter=2` still allowed it: when the limit is hit CrewAI makes one more call for a forced
  final answer, so `max_iter=N` means up to N+1 model calls (`max_iter=3` on a looping coworker: 4).
  A delegation requested in that forced call is not executed (a third coworker never ran). Separately,
  calling the same tool with the same arguments and result 5 times in a row stops the loop
  (`MAX_REPEATED_TOOL_CALLS = 5` in `crewai/utilities/agent_utils.py`); our looping coworker stopped
  after 7 calls that way. Native function-calling mode and `Process.hierarchical` were not run.
- **langgraph:** `recursion_limit` counts supersteps, not model calls. With the limit at 5 the tree
  stopped inside the child; at 6 it completed with 5 (single) or 8 (parallel) model calls.

Worst case for nesting without a shared budget is roughly
`parent_limit + fan_out × child_limit` per level, multiplied by depth.

## 2. A tool timeout does not undo a side effect

A `send_email` tool records a line after 0.5 s; the framework timeout is 0.2 s; the scripted model
retries once when told the call failed.

| Variant | openai-agents 0.24.0 | pydantic-ai-slim 2.64.0 |
|---|---|---|
| Sync tool (worker thread) | timeout **refused at definition** ("only supported for async handlers") | model told "Timed out", run ends with an error, **2 emails sent** |
| Async tool, effect after the wait | 0 sent (cancelled in time) | 0 sent |
| Async tool, effect before a slow ack | **2 sent** | **2 sent** |
| Same, with an idempotency key | 1 sent | 1 sent |

This is documented behavior, not a bug: pydantic-ai's [Timeouts](https://ai.pydantic.dev/timeouts/) page
says a `def` tool "is not actually stopped", its cancellation docs say side effects remain, and its
production guide says to make side-effecting tools safe to repeat (idempotency key). The repro shows
what those three facts add up to when the timeout's retry kicks in. openai-agents avoids the sync case
entirely.

## 3. Restart and refresh tokens (MCP Python SDK 2.3.0)

`mcp_refresh_restart.py` restores a stored, server-expired access token plus a valid refresh token into
a new `OAuthClientProvider`. The client did **not** use the refresh token and went straight to an
interactive login. This is already reported as
[python-sdk#3250](https://github.com/modelcontextprotocol/python-sdk/issues/3250) (P1) with an open fix,
[PR #3263](https://github.com/modelcontextprotocol/python-sdk/pull/3263). We include it only as an
independent reproduction. The MCP spec requires the RFC 8707 `resource` parameter in token requests;
the conformance suite currently checks refresh only for DPoP/EMA (open PRs #524, #484).

## Reproduce

One command, with [uv](https://docs.astral.sh/uv/) installed (no API keys; only package installs touch the network):

```sh
cd lab && ./reproduce.sh
```

It creates one throwaway virtual environment per framework (their dependencies conflict), installs the
pinned `requirements-*.txt`, rewrites every file in `evidence/`, and runs `check_matrix.py`.
`check_matrix.py` asserts every number in the tables above from the JSON evidence. Two deliberately
wrong expectations (`--broken-control`, `--broken-side-effect-control`) must fail; the script stops if
either passes.

## Limits of this evidence

Scripted models, one event loop, one machine. No real provider retries, streaming, billing,
multiple processes or production collectors. Versions move fast: ADK changed `AgentTool` config
inheritance in 2026-08, pydantic-ai and Microsoft Agent Framework have budget work open. Re-run before
citing. Prior art we build on rather than replace: [openai-agents#5453](https://github.com/openai/openai-agents-python/issues/5453)
(inherited delegation depth, 2026-10-29), [#3353](https://github.com/openai/openai-agents-python/issues/3353),
LiteLLM and Claude Agent SDK dollar budgets.

## License

MIT, see [LICENSE](LICENSE).
