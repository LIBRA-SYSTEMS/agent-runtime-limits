#!/usr/bin/env bash
# Rebuild every evidence file from scratch, then check the tables.
# Needs: uv (https://docs.astral.sh/uv/). No API keys, no network calls besides package installs.
# Usage: ./reproduce.sh            (writes ../evidence/*.json, then runs check_matrix.py)
set -euo pipefail
LAB="$(cd "$(dirname "$0")" && pwd)"
EV="$LAB/../evidence"
ENVS="$(mktemp -d)"
trap 'rm -rf "$ENVS"' EXIT
mkdir -p "$EV"

env_for() {  # $1 = requirements suffix; prints the python path of a fresh venv
  local dir="$ENVS/$1"
  if [ ! -x "$dir/bin/python" ]; then
    uv venv -q --python 3.12 "$dir"
    uv pip install -q --python "$dir/bin/python" -r "$LAB/requirements-$1.txt"
  fi
  echo "$dir/bin/python"
}

run() {  # $1 = requirements suffix, $2 = output json, rest = script and args
  local py; py="$(env_for "$1")"; local out="$2"; shift 2
  echo "  $out"
  # Some cases hit a limit on purpose and the framework logs a traceback; keep it out of sight unless the script fails.
  if ! (cd "$LAB" && "$py" -I "$@" > "$EV/$out" 2> "$ENVS/$out.log"); then
    cat "$ENVS/$out.log"; echo "FAILED: $*"; exit 1
  fi
}

echo "Running experiments (scripted models, 0 API calls):"
run openai-agents openai_agents_nested.json        openai_agents_nested.py
run openai-agents openai_agents_hooks_default.json openai_agents_hooks_default.py
run openai-agents delegation_openai.json           delegation_style.py openai
run openai-agents side_effect_openai.json          side_effect_timeout.py openai
run openai-agents mcp_refresh_restart.json         mcp_refresh_restart.py
run pydantic-ai   pydanticai_nested.json           pydanticai_nested.py
run pydantic-ai   side_effect_pydantic.json        side_effect_timeout.py pydantic
run langgraph     langgraph_nested.json            langgraph_nested.py
run google-adk    adk_nested.json                  adk_nested.py
run google-adk    delegation_adk.json              delegation_style.py adk
run smolagents    smolagents_nested.json           smolagents_nested.py

echo "Checking tables:"
python3 "$LAB/check_matrix.py" > "$EV/matrix-check.json" && echo "  matrix-check: PASS"
# The two deliberately wrong expectations must fail; if either passes, the checker is not testing anything.
if python3 "$LAB/check_matrix.py" --broken-control > "$EV/matrix-broken-control.json"; then
  echo "  broken-control unexpectedly PASSED"; exit 1; fi
if python3 "$LAB/check_matrix.py" --broken-side-effect-control > "$EV/matrix-broken-side-effect-control.json"; then
  echo "  broken-side-effect-control unexpectedly PASSED"; exit 1; fi
echo "  both broken controls: FAIL (as expected)"
