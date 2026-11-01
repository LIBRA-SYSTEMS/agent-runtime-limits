"""Offline PydanticAI nested request-limit experiment. FunctionModel never calls an API.

Same shape as the OpenAI Agents experiment: the parent may make 2 requests,
each delegated child makes 3 requests (tick, tick, done).
"""
import asyncio
import json
from importlib.metadata import version

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits


def responses_so_far(messages):
    return sum(isinstance(m, ModelResponse) for m in messages)


def build(calls, parallel, mode, parent_limit):
    """mode: 'separate' (no usage passed), 'shared_usage' (usage=ctx.usage),
    'shared_usage_and_limits' (usage=ctx.usage, usage_limits=parent limits)."""

    def child_model(name):
        async def fn(messages, info: AgentInfo):
            calls.append(name)
            await asyncio.sleep(0)
            if responses_so_far(messages) < 2:
                return ModelResponse(parts=[ToolCallPart("tick", {})])
            return ModelResponse(parts=[TextPart("done")])
        return FunctionModel(fn)

    names = ["child_a", "child_b"] if parallel else ["child"]
    children = {}
    for name in names:
        child = Agent(child_model(name), name=name)

        @child.tool_plain
        def tick() -> str:
            return "ok"

        children[name] = child

    async def parent_fn(messages, info: AgentInfo):
        calls.append("parent")
        await asyncio.sleep(0)
        if responses_so_far(messages) == 0:
            return ModelResponse(parts=[ToolCallPart(f"run_{n}", {}) for n in names])
        return ModelResponse(parts=[TextPart("done")])

    parent = Agent(FunctionModel(parent_fn), name="parent")
    limits = UsageLimits(request_limit=parent_limit)

    def make_runner(child):
        async def run_child(ctx: RunContext[None]) -> str:
            kwargs = {}
            if mode in ("shared_usage", "shared_usage_and_limits"):
                kwargs["usage"] = ctx.usage
            if mode == "shared_usage_and_limits":
                kwargs["usage_limits"] = limits
            result = await child.run("go", **kwargs)
            return result.output
        return run_child

    for name in names:
        parent.tool(make_runner(children[name]), name=f"run_{name}")

    return parent, limits


async def scenario(mode, parent_limit, parallel=False):
    calls = []
    parent, limits = build(calls, parallel, mode, parent_limit)
    status, reported = "completed", None
    try:
        result = await parent.run("go", usage_limits=limits)
        reported = result.usage.requests
    except UsageLimitExceeded as error:
        status = f"limit_exceeded: {error}"
    return {"mode": mode, "parent_request_limit": parent_limit, "parallel": parallel,
            "status": status, "actual_calls": calls, "total_calls": len(calls),
            "parent_reported_requests": reported}


async def main():
    cases = {
        "separate_limit2": await scenario("separate", 2),
        "separate_limit2_parallel": await scenario("separate", 2, parallel=True),
        "shared_usage_limit2": await scenario("shared_usage", 2),
        "shared_usage_limit5": await scenario("shared_usage", 5),
        "shared_usage_and_limits_limit3": await scenario("shared_usage_and_limits", 3),
        "shared_usage_and_limits_limit4_parallel": await scenario("shared_usage_and_limits", 4, parallel=True),
    }
    print(json.dumps({"framework": "pydantic-ai-slim", "version": version("pydantic-ai-slim"),
                      "model": "FunctionModel (scripted, no API)", "cases": cases,
                      "network_or_model_api_calls": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
