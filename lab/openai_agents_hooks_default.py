"""Offline OpenAI Agents SDK follow-up: what a parent can and cannot see or bound.

1. Do parent RunHooks (on_llm_start) fire for model calls made inside an agent.as_tool child?
2. If as_tool(max_turns=...) is omitted, does the child inherit the parent's max_turns?
3. Does the parent's usage include child requests (aggregation)?
A scripted Model never calls an API.
"""
import asyncio
import json
from importlib.metadata import version

from agents import Agent, Model, ModelResponse, RunConfig, RunHooks, Runner, Usage, function_tool, set_tracing_disabled
from agents.exceptions import MaxTurnsExceeded, UserError
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

set_tracing_disabled(True)
CALLS: list[str] = []


class ScriptedModel(Model):
    def __init__(self, name, steps, loop_tool=None):
        self.name, self.steps, self.loop_tool = name, list(steps), loop_tool

    async def get_response(self, *args, **kwargs):
        CALLS.append(self.name)
        i = len(CALLS)
        step = self.steps.pop(0) if self.steps else ((self.loop_tool, {}) if self.loop_tool else "done")
        await asyncio.sleep(0)
        if isinstance(step, (tuple, list)):
            calls = [step] if isinstance(step, tuple) else step
            output = [ResponseFunctionToolCall(type="function_call", name=c[0], arguments=json.dumps(c[1]),
                                               call_id=f"call_{i}_{n}", id=f"fc_{i}_{n}") for n, c in enumerate(calls)]
        else:
            output = [ResponseOutputMessage(type="message", role="assistant", id=f"msg_{i}", status="completed",
                                            content=[ResponseOutputText(type="output_text", text=step, annotations=[])])]
        return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=10, output_tokens=2, total_tokens=12),
                             response_id=f"r_{i}")

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError
        yield


@function_tool
def tick() -> str:
    """A deterministic no-op tool."""
    return "ok"


class CountingHooks(RunHooks):
    def __init__(self):
        self.seen = []

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self.seen.append(agent.name)


async def hooks_case(share_hooks=False):
    CALLS.clear()
    hooks = CountingHooks()
    config = RunConfig(tracing_disabled=True)
    child = Agent(name="child", model=ScriptedModel("child", [("tick", {}), ("tick", {}), "done"]), tools=[tick])
    parent = Agent(name="parent", model=ScriptedModel("parent", [("child", {"input": "go"}), "done"]),
                   tools=[child.as_tool(tool_name="child", tool_description="Run child", max_turns=3, run_config=config,
                                         hooks=hooks if share_hooks else None)])
    result = await Runner.run(parent, "go", max_turns=2, run_config=config, hooks=hooks)
    return {"actual_calls": list(CALLS), "parent_hooks_saw": hooks.seen,
            "parent_usage_requests": result.context_wrapper.usage.requests}


async def default_case(parent_max_turns):
    CALLS.clear()
    config = RunConfig(tracing_disabled=True)
    # Child loops on a tool forever; as_tool max_turns is omitted.
    child = Agent(name="child", model=ScriptedModel("child", [], loop_tool="tick"), tools=[tick])
    parent = Agent(name="parent", model=ScriptedModel("parent", [("child", {"input": "go"}), "done"]),
                   tools=[child.as_tool(tool_name="child", tool_description="Run child", run_config=config,
                                        failure_error_function=None)])
    status = "completed"
    try:
        await Runner.run(parent, "go", max_turns=parent_max_turns, run_config=config)
    except (MaxTurnsExceeded, UserError) as error:
        cause = error
        while cause is not None and not isinstance(cause, MaxTurnsExceeded):
            cause = cause.__cause__
        status = f"max_turns_exceeded: {cause}" if cause else f"other: {error!r}"
    return {"parent_max_turns": parent_max_turns, "as_tool_max_turns": "omitted", "status": status,
            "child_calls_before_stop": CALLS.count("child"), "total_calls": len(CALLS)}


async def main():
    hooks = await hooks_case()
    hooks_shared = await hooks_case(share_hooks=True)
    d2 = await default_case(2)
    d50 = await default_case(50)
    print(json.dumps({"sdk": version("openai-agents"), "hooks_visibility": hooks, "hooks_visibility_when_passed_to_as_tool": hooks_shared,
                      "omitted_as_tool_max_turns_parent2": d2, "omitted_as_tool_max_turns_parent50": d50,
                      "network_or_model_api_calls": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
