"""Tool timeout vs. real side effects. Scripted models, no API.

A `send_email` tool writes one line to a ledger (the side effect) after 0.5 s.
The framework's per-tool timeout is 0.2 s. The scripted model behaves like a
reasonable model: when told the call failed or timed out, it calls the tool once more.

What we measure: how many emails were actually sent, versus what the model was told.
Variants: sync tool (runs in a worker thread), async tool (cancellable), async tool
that commits before a slow acknowledgement, and the same plus an idempotency key.

Run:  python side_effect_timeout.py openai   |   python side_effect_timeout.py pydantic
"""
import asyncio
import json
import sys
import threading
import time
from importlib.metadata import version

LEDGER: list[str] = []
LOCK = threading.Lock()
SEEN_KEYS: set[str] = set()


def record(email_id, dedupe):
    with LOCK:
        if dedupe and email_id in SEEN_KEYS:
            return "duplicate ignored"
        SEEN_KEYS.add(email_id)
        LEDGER.append(email_id)
        return "sent"


def sync_send(email_id: str) -> str:
    time.sleep(0.5)                      # slow provider, then the effect happens
    return record(email_id, dedupe=False)


async def async_send_after(email_id: str) -> str:
    await asyncio.sleep(0.5)             # cancellation lands here: no effect
    return record(email_id, dedupe=False)


async def async_commit_then_ack(email_id: str) -> str:
    record(email_id, dedupe=False)       # effect first ...
    await asyncio.sleep(0.5)             # ... slow acknowledgement gets cancelled
    return "sent"


async def async_commit_then_ack_dedupe(email_id: str) -> str:
    result = record(email_id, dedupe=True)
    await asyncio.sleep(0.5)
    return result


VARIANTS = {
    "sync_thread": sync_send,
    "async_effect_after_wait": async_send_after,
    "async_effect_before_ack": async_commit_then_ack,
    "async_effect_before_ack_with_idempotency_key": async_commit_then_ack_dedupe,
}


async def settle():
    await asyncio.sleep(0.8)  # let any orphaned worker thread finish


# ---------------- OpenAI Agents SDK ----------------
async def openai_run(variant):
    from agents import Agent, Model, ModelResponse, RunConfig, Runner, Usage, function_tool, set_tracing_disabled
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
    set_tracing_disabled(True)
    told = []

    class Scripted(Model):
        def __init__(self):
            self.n = 0

        async def get_response(self, system_instructions, input, *a, **k):
            self.n += 1
            outputs = [i for i in input if isinstance(i, dict) and i.get("type") == "function_call_output"]
            if outputs:
                told.append(str(outputs[-1].get("output"))[:120])
            if self.n <= 2:   # first call, then one retry after a failure report
                out = [ResponseFunctionToolCall(type="function_call", name="send_email",
                                                arguments=json.dumps({"email_id": "invoice-42"}),
                                                call_id=f"c{self.n}", id=f"fc{self.n}")]
            else:
                out = [ResponseOutputMessage(type="message", role="assistant", id="m", status="completed",
                                             content=[ResponseOutputText(type="output_text", text="done", annotations=[])])]
            return ModelResponse(output=out, usage=Usage(), response_id=f"r{self.n}")

        async def stream_response(self, *a, **k):
            raise NotImplementedError
            yield

    fn = VARIANTS[variant]
    try:
        tool = function_tool(fn, name_override="send_email", timeout=0.2)
    except ValueError as error:   # the SDK refuses timeouts on sync handlers
        return [f"refused at definition: {error}"]
    agent = Agent(name="mailer", model=Scripted(), tools=[tool])
    await Runner.run(agent, "send invoice 42", run_config=RunConfig(tracing_disabled=True))
    told_before_settle = list(told)
    await settle()
    return told_before_settle


# ---------------- PydanticAI ----------------
async def pydantic_run(variant):
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    told = []
    state = {"n": 0}

    async def fn_model(messages, info: AgentInfo):
        state["n"] += 1
        last = messages[-1]
        for part in getattr(last, "parts", []):
            if isinstance(part, (RetryPromptPart, ToolReturnPart)):
                told.append(f"{type(part).__name__}: {str(part.content)[:100]}")
        if state["n"] <= 2:
            return ModelResponse(parts=[ToolCallPart("send_email", {"email_id": "invoice-42"})])
        return ModelResponse(parts=[TextPart("done")])

    from pydantic_ai.exceptions import UnexpectedModelBehavior
    agent = Agent(FunctionModel(fn_model))
    agent.tool_plain(VARIANTS[variant], name="send_email", timeout=0.2)
    try:
        await agent.run("send invoice 42")
    except UnexpectedModelBehavior as error:   # default: 1 retry, the 2nd timeout ends the run
        told.append(f"run failed: {str(error)[:90]}")
    told_before_settle = list(told)
    await settle()
    return told_before_settle


async def main(which):
    runner = openai_run if which == "openai" else pydantic_run
    pkg = "openai-agents" if which == "openai" else "pydantic-ai-slim"
    cases = {}
    for variant in VARIANTS:
        LEDGER.clear()
        SEEN_KEYS.clear()
        t0 = time.monotonic()
        told = await runner(variant)
        cases[variant] = {"model_was_told": told, "emails_actually_sent": len(LEDGER),
                          "seconds": round(time.monotonic() - t0, 2)}
    print(json.dumps({"framework": pkg, "version": version(pkg), "tool_timeout_s": 0.2, "provider_latency_s": 0.5,
                      "model_retries_after_failure": 1, "cases": cases, "network_or_model_api_calls": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
