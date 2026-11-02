"""Check the cross-framework nested-budget evidence and build one summary table.

Shape of every experiment: parent allows 2 model calls (or a limit just above that),
each child makes 3 model calls. "Tree total" is what actually reached the model.
Run:  python3 check_matrix.py [--broken-control | --broken-side-effect-control]
"""
import json
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent.parent / "evidence"


def load(name):
    return json.loads((EV / name).read_text())


def rows():
    oa = load("openai_agents_nested.json")
    oh = load("openai_agents_hooks_default.json")
    pa = load("pydanticai_nested.json")
    lg = load("langgraph_nested.json")
    ad = load("adk_nested.json")
    sm = load("smolagents_nested.json")
    cr = load("crewai_nested.json")["cases"]
    return [
        {"framework": "openai-agents", "version": oa["sdk"], "limit": "max_turns (per run)",
         "single": len(oa["turn_limit_only"]["actual_calls"]), "parallel": len(oa["parallel_turn_limit_only"]["actual_calls"]),
         "child_inherits_limit": oh["omitted_as_tool_max_turns_parent2"]["child_calls_before_stop"] != 10,
         "parent_sees_child_live": oh["hooks_visibility"]["parent_hooks_saw"].count("child") > 0,
         "reported_total": oh["hooks_visibility"]["parent_usage_requests"]},
        {"framework": "pydantic-ai", "version": pa["version"], "limit": "UsageLimits.request_limit (per run unless usage+limits passed)",
         "single": pa["cases"]["separate_limit2"]["total_calls"], "parallel": pa["cases"]["separate_limit2_parallel"]["total_calls"],
         "child_inherits_limit": False,
         "parent_sees_child_live": False,
         "reported_total": pa["cases"]["separate_limit2"]["parent_reported_requests"],
         "opt_in_shared_overshoot": pa["cases"]["shared_usage_and_limits_limit4_parallel"]["total_calls"]},
        {"framework": "langgraph", "version": lg["version"], "limit": "recursion_limit (value inherited, counted per graph)",
         "single": lg["cases"]["limit6_single"]["total_model_calls"], "parallel": lg["cases"]["limit6_parallel"]["total_model_calls"],
         "child_inherits_limit": True, "parent_sees_child_live": None, "reported_total": None},
        {"framework": "google-adk", "version": ad["version"], "limit": "RunConfig.max_llm_calls (value inherited, counted per invocation)",
         "single": ad["cases"]["limit3_single"]["total_calls"], "parallel": ad["cases"]["limit3_parallel"]["total_calls"],
         "child_inherits_limit": True, "parent_sees_child_live": None, "reported_total": None},
        {"framework": "smolagents", "version": sm["version"], "limit": "max_steps (per agent)",
         "single": sm["cases"]["parent2_child3_single"]["total_calls"], "parallel": sm["cases"]["parent2_child3_parallel"]["total_calls"],
         "child_inherits_limit": False, "parent_sees_child_live": None, "reported_total": None,
         "max_steps3_loop_calls": sm["cases"]["parent2_child3_child_loops"]["actual_calls"].count("child")},
        {"framework": "crewai", "version": load("crewai_nested.json")["version"],
         "limit": "Agent.max_iter (per agent; one delegation per model call)",
         "single": cr["parent2_child3_single"]["total_calls"],
         "parallel": cr["parent2_two_children_sequential"]["total_calls"],  # two children, delegated one after the other
         "child_inherits_limit": cr["parent2_child_default_limit_loops"]["actual_calls"].count("child") <= 3,
         "parent_sees_child_live": None, "reported_total": None,
         "max_iter3_loop_calls": cr["parent2_child3_child_loops"]["actual_calls"].count("child"),
         "max_iter2_parent_calls_three_delegations": cr["parent2_three_children_sequential"]["parent_calls"],
         "third_child_ran": "child_c" in cr["parent2_three_children_sequential"]["actual_calls"],
         "same_args_loop_child_calls": cr["parent2_child_default_limit_same_args_loops"]["actual_calls"].count("child")},
    ]


def check(table, broken=False, broken_side_effect=False):
    by = {r["framework"]: r for r in table}
    expect_single = 2 if broken else 5   # broken control: pretend the parent limit bounds the whole tree
    for name in ("openai-agents", "pydantic-ai", "langgraph", "google-adk", "smolagents"):
        assert by[name]["single"] == expect_single, (name, by[name]["single"])
        assert by[name]["parallel"] == (2 if broken else 8), (name, by[name]["parallel"])
    # LangGraph limit 5 stops the tree mid-child (limit is inherited), limit 6 completes 5 calls.
    lg = load("langgraph_nested.json")["cases"]
    assert lg["limit5_single"]["status"].startswith("recursion_limit")
    assert lg["limit5_parallel"]["total_model_calls"] == 7
    assert by["openai-agents"]["child_inherits_limit"] is False
    assert by["openai-agents"]["parent_sees_child_live"] is False
    assert by["openai-agents"]["reported_total"] == 5
    assert by["pydantic-ai"]["reported_total"] == 2          # parent result under-reports 5 real calls
    assert by["pydantic-ai"]["opt_in_shared_overshoot"] == 5  # request_limit=4, known pydantic-ai#10142
    assert by["smolagents"]["max_steps3_loop_calls"] == 4     # max_steps=3 costs one extra summary call
    # CrewAI: same 5 for one child; two children cost 9 because each delegation takes its own parent call.
    cr = by["crewai"]
    assert cr["single"] == expect_single, ("crewai", cr["single"])
    assert cr["parallel"] == (2 if broken else 9), ("crewai", cr["parallel"])
    assert cr["child_inherits_limit"] is False                 # child without max_iter ran 11 calls (default 25)
    assert cr["max_iter3_loop_calls"] == 4                     # max_iter=3 plus one forced final-answer call
    assert cr["max_iter2_parent_calls_three_delegations"] == 3 # max_iter=2 -> 3rd call is the forced answer
    assert cr["third_child_ran"] is False                      # the forced call's delegation is not executed
    assert cr["same_args_loop_child_calls"] == 7               # repeated-call guard (5 identical calls) stops it
    hv = load("openai_agents_hooks_default.json")["hooks_visibility_when_passed_to_as_tool"]["parent_hooks_saw"]
    assert hv.count("child") == 3   # same RunHooks passed to as_tool(hooks=...) sees child calls live
    # Same framework, two delegation styles.
    do, da = load("delegation_openai.json")["cases"], load("delegation_adk.json")["cases"]
    assert do["handoff_limit2"]["total_calls"] == 2 and do["handoff_limit2"]["status"].startswith("max_turns")
    assert do["as_tool_limit2"]["total_calls"] == 5 and do["as_tool_limit2"]["status"] == "completed"
    assert da["transfer_limit3"]["total_calls"] == 3 and da["transfer_limit3"]["status"].startswith("limit")
    assert da["agent_tool_limit3"]["total_calls"] == 5 and da["agent_tool_limit3"]["status"] == "completed"
    # Tool timeout vs side effects (a model that retries once after a failure report).
    so, sp = load("side_effect_openai.json")["cases"], load("side_effect_pydantic.json")["cases"]
    assert so["sync_thread"]["model_was_told"][0].startswith("refused at definition")
    assert sp["sync_thread"]["emails_actually_sent"] == 2
    for cases in (so, sp):
        assert cases["async_effect_after_wait"]["emails_actually_sent"] == 0
        assert cases["async_effect_before_ack"]["emails_actually_sent"] == 2
        assert cases["async_effect_before_ack_with_idempotency_key"]["emails_actually_sent"] == (2 if broken_side_effect else 1)


if __name__ == "__main__":
    table = rows()
    broken = "--broken-control" in sys.argv
    broken_side_effect = "--broken-side-effect-control" in sys.argv
    try:
        check(table, broken=broken, broken_side_effect=broken_side_effect)
        verdict = "PASS"
    except AssertionError as error:
        verdict = f"FAIL {error}"
    print(json.dumps({"broken_control": broken, "broken_side_effect_control": broken_side_effect, "verdict": verdict, "table": table}, indent=2))
    sys.exit(0 if verdict == "PASS" else 1)
