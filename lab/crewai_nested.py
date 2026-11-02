"""Offline CrewAI delegation experiment. A scripted BaseLLM never calls an API.

Same shape as the other frameworks: the parent (manager) agent is limited to 2 iterations,
each coworker agent makes 3 model calls. Delegation uses CrewAI's own `allow_delegation=True`
("Delegate work to coworker" tool), in ReAct text mode.
"""
import json
import os
from importlib.metadata import version

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402
from crewai.tools import BaseTool  # noqa: E402

CALLS: list[str] = []


def act(tool, payload):
    return f"Thought: use a tool\nAction: {tool}\nAction Input: {json.dumps(payload)}"


FINAL = "Thought: I now know the final answer\nFinal Answer: done"


class ScriptedLLM(BaseLLM):
    name: str = "scripted"
    script: list = []

    def __init__(self, name, script):
        super().__init__(model=f"scripted-{name}")
        self.name, self.script = name, list(script)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        CALLS.append(self.name)
        return self.script.pop(0) if self.script else FINAL

    def supports_function_calling(self):
        return False

    def supports_stop_words(self):
        return False

    def get_context_window_size(self):
        return 8192


class Tick(BaseTool):
    name: str = "tick"
    description: str = "A deterministic tool that echoes its step number."

    def _run(self, n: int = 0) -> str:
        return f"ok {n}"


def child(name, loops=False, child_max_iter=3, same_args=False):
    # Distinct arguments by default, so CrewAI's repeated-call guard (same tool, same args,
    # same result 5 times in a row) does not stop the loop before max_iter does.
    if loops:
        steps = [act("tick", {"n": 0 if same_args else i}) for i in range(10)]
    else:
        steps = [act("tick", {"n": 1}), act("tick", {"n": 2}), FINAL]
    extra = {} if child_max_iter is None else {"max_iter": child_max_iter}
    return Agent(role=name, goal="help", backstory="coworker", llm=ScriptedLLM(name, steps), tools=[Tick()],
                 allow_delegation=False, verbose=False, **extra)


def scenario(parent_max_iter, children=("child",), child_loops=False, child_max_iter=3, same_args=False):
    CALLS.clear()
    parent_steps = [act("Delegate work to coworker", {"task": "go", "context": "none", "coworker": n})
                    for n in children] + [FINAL]
    parent = Agent(role="parent", goal="finish", backstory="manager", llm=ScriptedLLM("parent", parent_steps),
                   max_iter=parent_max_iter, allow_delegation=True, verbose=False)
    workers = [child(n, child_loops, child_max_iter, same_args) for n in children]
    task = Task(description="go", expected_output="done", agent=parent)
    crew = Crew(agents=[parent, *workers], tasks=[task], process=Process.sequential, verbose=False)
    status = "completed"
    try:
        crew.kickoff()
    except Exception as error:  # a limit may surface as an exception
        status = f"{type(error).__name__}: {str(error)[:120]}"
    return {"parent_max_iter": parent_max_iter, "children": list(children), "child_loops": child_loops,
            "status": status, "actual_calls": list(CALLS), "total_calls": len(CALLS),
            "parent_calls": CALLS.count("parent")}


def main():
    cases = {
        "parent2_child3_single": scenario(2),
        "parent2_two_children_sequential": scenario(2, ("child_a", "child_b")),
        "parent3_two_children_sequential": scenario(3, ("child_a", "child_b")),
        "parent2_three_children_sequential": scenario(2, ("child_a", "child_b", "child_c")),
        "parent2_child3_child_loops": scenario(2, child_loops=True),
        "parent2_child_default_limit_loops": scenario(2, child_loops=True, child_max_iter=None),
        "parent2_child_default_limit_same_args_loops": scenario(2, child_loops=True, child_max_iter=None,
                                                                same_args=True),
    }
    print(json.dumps({"framework": "crewai", "version": version("crewai"),
                      "model": "scripted BaseLLM (no API), ReAct text mode", "cases": cases,
                      "network_or_model_api_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
