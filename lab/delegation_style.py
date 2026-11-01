"""Same framework, two delegation styles: does the parent's limit cover the child?

OpenAI Agents 0.24.0: handoff (same Runner loop) vs agent.as_tool (nested Runner).
Run with the matching venv:  python delegation_style.py openai   |   python delegation_style.py adk
A scripted model never calls an API.
"""
import asyncio
import json
import sys
from importlib.metadata import version

CALLS: list[str] = []


async def openai_case():
    from agents import Agent, Model, ModelResponse, RunConfig, Runner, Usage, function_tool, handoff, set_tracing_disabled
    from agents.exceptions import MaxTurnsExceeded
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
    set_tracing_disabled(True)

    class ScriptedModel(Model):
        def __init__(self, name, steps):
            self.name, self.steps = name, list(steps)

        async def get_response(self, *args, **kwargs):
            CALLS.append(self.name)
            i = len(CALLS)
            step = self.steps.pop(0) if self.steps else "done"
            if isinstance(step, tuple):
                out = [ResponseFunctionToolCall(type="function_call", name=step[0], arguments=json.dumps(step[1]),
                                                call_id=f"c{i}", id=f"fc{i}")]
            else:
                out = [ResponseOutputMessage(type="message", role="assistant", id=f"m{i}", status="completed",
                                             content=[ResponseOutputText(type="output_text", text=step, annotations=[])])]
            return ModelResponse(output=out, usage=Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2),
                                 response_id=f"r{i}")

        async def stream_response(self, *a, **k):
            raise NotImplementedError
            yield

    @function_tool
    def tick() -> str:
        """No-op."""
        return "ok"

    async def run(style, parent_max_turns):
        CALLS.clear()
        cfg = RunConfig(tracing_disabled=True)
        child = Agent(name="child", model=ScriptedModel("child", [("tick", {}), ("tick", {}), "done"]), tools=[tick])
        if style == "handoff":
            parent = Agent(name="parent", model=ScriptedModel("parent", [("transfer_to_child", {})]), handoffs=[handoff(child)])
        else:
            parent = Agent(name="parent", model=ScriptedModel("parent", [("child", {"input": "go"}), "done"]),
                           tools=[child.as_tool(tool_name="child", tool_description="c", max_turns=3, run_config=cfg)])
        status = "completed"
        try:
            await Runner.run(parent, "go", max_turns=parent_max_turns, run_config=cfg)
        except MaxTurnsExceeded as e:
            status = f"max_turns_exceeded: {e}"
        return {"style": style, "parent_max_turns": parent_max_turns, "status": status,
                "actual_calls": list(CALLS), "total_calls": len(CALLS)}

    return {"framework": "openai-agents", "version": version("openai-agents"), "cases": {
        "handoff_limit2": await run("handoff", 2),
        "handoff_limit4": await run("handoff", 4),
        "as_tool_limit2": await run("as_tool", 2),
    }}


async def adk_case():
    from google.adk.agents import LlmAgent
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError
    from google.adk.agents.run_config import RunConfig
    from google.adk.models.base_llm import BaseLlm
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.agent_tool import AgentTool
    from google.genai import types

    class ScriptedLlm(BaseLlm):
        script: list = []

        async def generate_content_async(self, llm_request, stream=False):
            CALLS.append(self.model)
            step = self.script.pop(0) if self.script else "done"
            if isinstance(step, tuple):
                part = types.Part(function_call=types.FunctionCall(name=step[0], args=step[1]))
            else:
                part = types.Part(text=step)
            yield __import__("google.adk.models.llm_response", fromlist=["LlmResponse"]).LlmResponse(
                content=types.Content(role="model", parts=[part]))

    def tick() -> str:
        """No-op."""
        return "ok"

    async def run(style, limit):
        CALLS.clear()
        child = LlmAgent(name="child", model=ScriptedLlm(model="child", script=[("tick", {}), ("tick", {}), "done"]),
                         tools=[tick], instruction="x")
        if style == "transfer":
            parent = LlmAgent(name="parent", model=ScriptedLlm(model="parent", script=[("transfer_to_agent", {"agent_name": "child"})]),
                              sub_agents=[child], instruction="x")
        else:
            parent = LlmAgent(name="parent", model=ScriptedLlm(model="parent", script=[("child", {"request": "go"}), "done"]),
                              tools=[AgentTool(agent=child)], instruction="x")
        runner = InMemoryRunner(agent=parent, app_name="b")
        s = await runner.session_service.create_session(app_name="b", user_id="u")
        status = "completed"
        try:
            async for _ in runner.run_async(user_id="u", session_id=s.id,
                                            new_message=types.Content(role="user", parts=[types.Part(text="go")]),
                                            run_config=RunConfig(max_llm_calls=limit)):
                pass
        except Exception as e:
            c = e
            while c is not None and not isinstance(c, LlmCallsLimitExceededError):
                c = c.__cause__ or c.__context__
            if c is None:
                raise
            status = f"limit_exceeded: {c}"
        return {"style": style, "max_llm_calls": limit, "status": status, "actual_calls": list(CALLS), "total_calls": len(CALLS)}

    return {"framework": "google-adk", "version": version("google-adk"), "cases": {
        "transfer_limit3": await run("transfer", 3),
        "transfer_limit4": await run("transfer", 4),
        "agent_tool_limit3": await run("agent_tool", 3),
    }}


if __name__ == "__main__":
    which = sys.argv[1]
    result = asyncio.run(openai_case() if which == "openai" else adk_case())
    result["network_or_model_api_calls"] = 0
    print(json.dumps(result, indent=2))
