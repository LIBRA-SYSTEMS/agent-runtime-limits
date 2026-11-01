"""Offline LangGraph nested recursion_limit experiment. No model or API is used.

A "model" node is a counting function. The parent loop is agent -> delegate -> agent.
Each child graph is model -> tool -> model -> tool -> model (3 model calls, 5 steps).
Question: is `recursion_limit` one budget for the whole tree, or per graph invocation?
"""
import json
import operator
from importlib.metadata import version
from typing import Annotated, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


class ChildState(TypedDict):
    turns: int


class ParentState(TypedDict):
    turns: int


def build_child(name, calls):
    def model(state: ChildState):
        calls.append(name)
        return {"turns": state["turns"] + 1}

    def tool(state: ChildState):
        return {}

    g = StateGraph(ChildState)
    g.add_node("model", model)
    g.add_node("tool", tool)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", lambda s: "tool" if s["turns"] < 3 else END)
    g.add_edge("tool", "model")
    return g.compile()


def build_parent(calls, parallel, pass_config):
    names = ["child_a", "child_b"] if parallel else ["child"]
    children = [build_child(n, calls) for n in names]

    def agent(state: ParentState):
        calls.append("parent")
        return {"turns": state["turns"] + 1}

    def make_delegate(child):
        def delegate(state: ParentState, config):
            # A tool that runs a sub-agent graph, as many apps do.
            if pass_config:
                child.invoke({"turns": 0}, config)
            else:
                child.invoke({"turns": 0})
            return {}
        return delegate

    g = StateGraph(ParentState)
    g.add_node("agent", agent)
    for n, child in zip(names, children):
        g.add_node(f"delegate_{n}", make_delegate(child))
    g.add_edge(START, "agent")
    route = [f"delegate_{n}" for n in names]
    g.add_conditional_edges("agent", lambda s: route if s["turns"] < 2 else END, route + [END])
    # Parallel delegates join back into the agent.
    g.add_edge(route, "agent") if len(route) > 1 else g.add_edge(route[0], "agent")
    return g.compile()


def scenario(limit, parallel=False, pass_config=True):
    calls = []
    graph = build_parent(calls, parallel, pass_config)
    status = "completed"
    try:
        graph.invoke({"turns": 0}, {"recursion_limit": limit})
    except GraphRecursionError as error:
        status = "recursion_limit: " + str(error).splitlines()[0]
    return {"recursion_limit": limit, "parallel": parallel, "child_gets_config": pass_config,
            "status": status, "actual_model_calls": calls, "total_model_calls": len(calls)}


def main():
    cases = {
        # Parent alone needs 3 steps (agent, delegate, agent). Child needs 5 steps.
        "limit5_single": scenario(5),
        "limit4_single": scenario(4),
        "limit5_parallel": scenario(5, parallel=True),
        "limit5_single_no_config": scenario(5, pass_config=False),
        "limit3_single": scenario(3),
        # Smallest limit at which the child (5 steps) fits: the whole tree then completes.
        "limit6_single": scenario(6),
        "limit6_parallel": scenario(6, parallel=True),
    }
    print(json.dumps({"framework": "langgraph", "version": version("langgraph"),
                      "model": "counting function nodes (no LLM)", "cases": cases,
                      "network_or_model_api_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
