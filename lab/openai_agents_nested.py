"""Offline SDK contract experiment. Scripted models never call an API."""
import asyncio
import json
from dataclasses import dataclass
from importlib.metadata import version

from agents import Agent, Model, ModelResponse, Runner, RunConfig, Usage, function_tool, set_tracing_disabled
from agents.exceptions import UserError
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

set_tracing_disabled(True)


class BudgetExceeded(Exception):
    pass


@dataclass
class SharedAdmission:
    limit: int
    used: int = 0

    def reserve(self):
        # No await between check and reserve: one event loop, this fixture only.
        if self.used >= self.limit:
            raise BudgetExceeded("shared request limit")
        self.used += 1


class ScriptedModel(Model):
    def __init__(self, name, steps, observed, budget=None):
        self.name, self.steps, self.observed, self.budget = name, iter(steps), observed, budget

    async def get_response(self, *args, **kwargs):
        if self.budget is not None:
            self.budget.reserve()
        self.observed.append(self.name)
        step = next(self.steps)
        i = len(self.observed)
        await asyncio.sleep(0)
        if isinstance(step, (tuple, list)):
            calls = [step] if isinstance(step, tuple) else step
            output = [ResponseFunctionToolCall(type="function_call", name=call[0], arguments=json.dumps(call[1]), call_id=f"call_{i}_{n}", id=f"fc_{i}_{n}") for n, call in enumerate(calls)]
        else:
            output = [ResponseOutputMessage(type="message", role="assistant", id=f"msg_{i}", status="completed", content=[ResponseOutputText(type="output_text", text=step, annotations=[])])]
        return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=10, output_tokens=2, total_tokens=12), response_id=f"response_{i}")

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("streaming is outside this experiment")
        yield


@function_tool
def tick() -> str:
    """A deterministic no-op tool."""
    return "ok"


async def scenario(limit=None, parallel=False):
    calls = []
    budget = SharedAdmission(limit) if limit is not None else None
    config = RunConfig(tracing_disabled=True)
    names = ["child_a", "child_b"] if parallel else ["child"]
    child_tools = []
    for name in names:
        child = Agent(name=name, model=ScriptedModel(name, [("tick", {}), ("tick", {}), "done"], calls, budget), tools=[tick])
        child_tools.append(child.as_tool(tool_name=name, tool_description="Run child", max_turns=3, run_config=config, failure_error_function=None))
    parent_steps = [[(name, {"input": "go"}) for name in names], "done"]
    parent = Agent(name="parent", model=ScriptedModel("parent", parent_steps, calls, budget), tools=child_tools)
    status, usage = "completed", None
    try:
        result = await Runner.run(parent, "go", max_turns=2, run_config=config)
        usage = result.context_wrapper.usage.requests
    except BudgetExceeded:
        status = "budget_exceeded"
    except UserError as error:
        # Tool errors are wrapped by this SDK; require the typed cause.
        cause = error.__cause__
        while cause is not None and not isinstance(cause, BudgetExceeded):
            cause = cause.__cause__
        if cause is None:
            raise
        status = "budget_exceeded"
    return {"status": status, "actual_calls": calls, "reported_requests": usage, "reserved": budget.used if budget else None}


async def main():
    unchanged = await scenario()
    limited = await scenario(3)
    generous = await scenario(5)
    parallel = await scenario(parallel=True)
    parallel_limited = await scenario(4, parallel=True)
    assert unchanged["actual_calls"] == ["parent", "child", "child", "child", "parent"]
    assert unchanged["reported_requests"] == 5
    assert limited["status"] == "budget_exceeded" and len(limited["actual_calls"]) == 3
    assert generous["status"] == "completed" and len(generous["actual_calls"]) == 5
    assert parallel["status"] == "completed" and len(parallel["actual_calls"]) == 8
    assert parallel["reported_requests"] == 8
    assert parallel_limited["status"] == "budget_exceeded" and len(parallel_limited["actual_calls"]) == 4
    print(json.dumps({"sdk": version("openai-agents"), "outer_max_turns": 2, "inner_max_turns": 3,
        "turn_limit_only": unchanged, "shared_limit_3": limited, "shared_limit_5": generous,
        "parallel_turn_limit_only": parallel, "parallel_shared_limit_4": parallel_limited,
        "cases": 5, "result": "TURN_LIMIT_IS_NOT_GLOBAL_REQUEST_BUDGET",
        "limitation": "Single event loop scripted model admission; not dollar/token/provider retries/distributed/streaming enforcement",
        "network_or_model_api_calls": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
