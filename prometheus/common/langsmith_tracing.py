"""LangSmith tracing of a run (entry 114). LangChain and LangGraph trace themselves once the LANGSMITH_* variables are in the PROCESS environment; this module puts them there.

`start()` (`configure()`, and a line on stderr when it is on) is called first by each command's `main` (the desk, the session, the recommendations runner, the Trader's replay, the Account Manager);
the web server calls `configure()` and prints its line:

  - it is ON when LANGSMITH_API_KEY is set (the environment, then `.env` through `rag.config.setting`, never by opening a file here) and LANGSMITH_TRACING is not "false";
  - the traces go to the AGENT ADDRESS, not to a project: LANGSMITH_AGENT_ID (default `prometheus`) and LANGSMITH_AGENT_ENVIRONMENT (default `local`) at LANGSMITH_ENDPOINT
    (default https://api.smith.langchain.com). The SDK traces NOTHING when an address and LANGSMITH_PROJECT are both set, so a project in the process environment is removed and the
    user is told. Agent addressing is a beta of the SDK, enabled per workspace: a workspace without it rejects the runs (they are lost, the run itself is not affected);
  - the tests turn it off (`tests/conftest.py` sets LANGSMITH_TRACING=false), so no fake run is ever sent.

WITH TRACING ON, the prompts, the model replies and the account's figures of a run are sent to LangSmith.
"""
import atexit
import os
import sys

from prometheus.rag.config import setting

DEFAULTS = {
    "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
    "LANGSMITH_AGENT_ID": "prometheus",
    "LANGSMITH_AGENT_ENVIRONMENT": "local",
}
PROJECT_NAMES = ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")

_flush_registered = False


def _clear_sdk_cache() -> None:
    """The SDK caches what it read from the environment; a value set here must be seen."""
    try:
        from langsmith import utils
        utils.get_env_var.cache_clear()
    except (ImportError, AttributeError):
        pass


def _flush() -> None:
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:   # never let the end of a run fail for its traces
        pass


def start() -> None:
    """`configure()` for a command: says so on stderr only when tracing is ON (a command's own output and errors are unchanged when it is off)."""
    line = configure()
    if " ON" in line:
        print(line, file=sys.stderr)


def configure() -> str:
    """Turn LangSmith tracing on for this process, or leave it off. Returns the line to print (it names no key)."""
    global _flush_registered
    key = setting("LANGSMITH_API_KEY", "")
    if setting("LANGSMITH_TRACING", "").strip().lower() == "false":
        os.environ["LANGSMITH_TRACING"] = "false"
        _clear_sdk_cache()
        return "tracing: LangSmith is off (LANGSMITH_TRACING=false)"
    if not key:
        return "tracing: LangSmith is off (set LANGSMITH_API_KEY to turn it on)"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_TRACING"] = "true"
    values = {name: setting(name, default) for name, default in DEFAULTS.items()}
    os.environ.update(values)
    removed = [name for name in PROJECT_NAMES if os.environ.pop(name, None)]
    _clear_sdk_cache()
    if not _flush_registered:
        atexit.register(_flush)
        _flush_registered = True
    line = (f"tracing: LangSmith is ON, agent {values['LANGSMITH_AGENT_ID']!r} ({values['LANGSMITH_AGENT_ENVIRONMENT']}) at {values['LANGSMITH_ENDPOINT']} "
            "(prompts, replies and the account's figures of a run are sent to LangSmith)")
    if removed:
        line += f"; {' and '.join(removed)} ignored: the SDK traces nothing when a project and an agent address are both set"
    return line
