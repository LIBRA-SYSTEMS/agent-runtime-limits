"""Offline Google ADK nested max_llm_calls experiment. A scripted BaseLlm never calls an API."""
import asyncio
import json
from importlib.metadata import version
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

CALLS: list[str] = []


def function_responses(req: LlmRequest) -> int:
    return sum(1 for c in req.contents or [] for p in c.parts or [] if p.function_response)


class ScriptedLlm(BaseLlm):
    role: str = "child"
    children: list[str] = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        CALLS.append(self.model)
        await asyncio.sleep(0)
        done = function_responses(llm_request)
        if self.role == "child" and done < 2:
            part = types.Part(function_call=types.FunctionCall(name="tick", args={}))
            yield LlmResponse(content=types.Content(role="model", parts=[part]))
        elif self.role == "parent" and done == 0:
            parts = [types.Part(function_call=types.FunctionCall(name=n, args={"request": "go"})) for n in self.children]
            yield LlmResponse(content=types.Content(role="model", parts=parts))
        else:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="done")]))


def tick() -> str:
    """A deterministic no-op tool."""
    return "ok"


async def scenario(max_llm_calls, parallel=False):
    CALLS.clear()
    names = ["child_a", "child_b"] if parallel else ["child"]
    children = [LlmAgent(name=n, model=ScriptedLlm(model=n), tools=[tick], instruction="x") for n in names]
    parent = LlmAgent(name="parent", model=ScriptedLlm(model="parent", role="parent", children=names),
                      tools=[AgentTool(agent=c) for c in children], instruction="x")
    runner = InMemoryRunner(agent=parent, app_name="budget")
    session = await runner.session_service.create_session(app_name="budget", user_id="u")
    status = "completed"
    try:
        async for _ in runner.run_async(user_id="u", session_id=session.id,
                                        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
                                        run_config=RunConfig(max_llm_calls=max_llm_calls)):
            pass
    except LlmCallsLimitExceededError as error:
        status = f"limit_exceeded: {error}"
    except Exception as error:  # a wrapped limit error from a nested tool
        cause = error
        while cause is not None and not isinstance(cause, LlmCallsLimitExceededError):
            cause = cause.__cause__ or cause.__context__
        if cause is None:
            raise
        status = f"limit_exceeded(wrapped): {cause}"
    return {"max_llm_calls": max_llm_calls, "parallel": parallel, "status": status,
            "actual_calls": list(CALLS), "total_calls": len(CALLS)}


async def main():
    cases = {
        "limit2_single": await scenario(2),
        "limit3_single": await scenario(3),
        "limit2_parallel": await scenario(2, parallel=True),
        "limit3_parallel": await scenario(3, parallel=True),
        "limit1_single": await scenario(1),
    }
    print(json.dumps({"framework": "google-adk", "version": version("google-adk"),
                      "model": "scripted BaseLlm (no API)", "cases": cases,
                      "network_or_model_api_calls": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
