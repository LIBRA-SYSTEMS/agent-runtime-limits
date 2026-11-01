"""Offline smolagents managed-agent max_steps experiment. A scripted Model never calls an API."""
import json
from importlib.metadata import version

from smolagents import Tool, ToolCallingAgent
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model

CALLS: list[str] = []


class ScriptedModel(Model):
    def __init__(self, name, steps):
        super().__init__(model_id=name)
        self.name, self.steps = name, list(steps)

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        CALLS.append(self.name)
        step = self.steps.pop(0) if self.steps else [("final_answer", {"answer": "forced"})]
        calls = [ChatMessageToolCall(id=f"{self.name}_{len(CALLS)}_{i}", type="function",
                                     function=ChatMessageToolCallFunction(name=n, arguments=a))
                 for i, (n, a) in enumerate(step)]
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=calls)


class Tick(Tool):
    name = "tick"
    description = "A deterministic no-op tool."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "ok"


def scenario(parent_max_steps, child_max_steps, parallel=False, child_loops=False):
    CALLS.clear()
    names = ["child_a", "child_b"] if parallel else ["child"]
    child_script = ([[("tick", {})]] * 10) if child_loops else [[("tick", {})], [("tick", {})], [("final_answer", {"answer": "done"})]]
    children = [ToolCallingAgent(tools=[Tick()], model=ScriptedModel(n, child_script), name=n,
                                 description="Run child", max_steps=child_max_steps, verbosity_level=-1)
                for n in names]
    parent_script = [[(n, {"task": "go"}) for n in names], [("final_answer", {"answer": "done"})]]
    parent = ToolCallingAgent(tools=[], model=ScriptedModel("parent", parent_script), managed_agents=children,
                              max_steps=parent_max_steps, verbosity_level=-1)
    result = parent.run("go", return_full_result=True)
    return {"parent_max_steps": parent_max_steps, "child_max_steps": child_max_steps, "parallel": parallel,
            "child_loops": child_loops, "state": result.state, "actual_calls": list(CALLS), "total_calls": len(CALLS)}


def main():
    cases = {
        "parent2_child3_single": scenario(2, 3),
        "parent2_child3_parallel": scenario(2, 3, parallel=True),
        "parent2_child3_child_loops": scenario(2, 3, child_loops=True),
    }
    print(json.dumps({"framework": "smolagents", "version": version("smolagents"),
                      "model": "scripted Model (no API)", "cases": cases,
                      "network_or_model_api_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
