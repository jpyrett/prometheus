"""How fast a Time Machine run goes, and what it costs in fidelity (design/proving_ground_spec.md P14; Jacob, 2026-11-20: *"this backtesting does not need to be so precise ... it should take a few minutes"*).

    speed = request.speed or config.time_machine.mode        # "fast" (the default) or "faithful"
    models = build_models(speed, config)                      # {role: a chat model or None}: None is "the agent builds its own, exactly as the live desk does"
    with limit_of(models.call_limit):                         # the wall limit of one model call of the day (the live desk's is 300 seconds)
        ...

TWO MODES, ONE SET OF RULES. Nothing a program computes changes between them: every score, gate, threshold, verdict rule, size and fill is the same code and reads the same config. What a mode changes is only HOW THE MODELS ARE ASKED:

  faithful   the agents build their own models, as the live desk does: the Advisor's and the Critic's reasoning effort and output length are the live ones, the routing is OpenRouter's own. A day takes as long as it does live.
  fast       the Advisor's, the Critic's and the Data Clerk's models are built here with the settings of `configs/proving_v1.json` section `time_machine.fast`: a LOWER reasoning effort, a capped output length, a shorter wait for one call and fewer
             retries (so one slow call cannot hold a day for minutes), and the providers sorted by THROUGHPUT (OpenRouter's `provider.sort`: the same model, served from the host that produces the most tokens a second). Every run that is fast says so
             (the caveat `fast_models`), and the research cache keeps a fast day apart from a faithful one (`engine.research_key`).

Two things are the same in both modes because they change no answer, only when it arrives: the days of a run are researched at once (`parallel_days`), and the records of one review by the Critic are reviewed at once (`Critic(parallel=...)`).

A model that is built here goes to `Advisor(model=...)`, `Critic(model=...)` and `DataClerk(model=...)`, the constructors the agents already have: the agents are not changed for it. No model, no network, nothing is read from `.env` here except through `get_chat_model`.
"""
import contextlib
from dataclasses import dataclass, field

from prometheus.proving.config import ModeSettings, ProvingConfig, RoleSettings

SPEEDS = ("fast", "faithful")


def speed_of(requested: str | None, config: ProvingConfig) -> str:
    """The mode of a run: the request's own, else the config's default. Raises ValueError for anything that is not a mode."""
    speed = requested or config.time_machine.mode
    if speed not in SPEEDS:
        raise ValueError(f"speed must be one of {', '.join(SPEEDS)}, not {speed!r}")
    return speed


def request_body(role: RoleSettings, provider_sort: str | None) -> dict:
    """The `extra_body` OpenRouter is sent for a role: the reasoning effort (only when the config names one) and the provider routing (only when it sorts). `require_parameters` goes with the effort so a provider that cannot honour it is
    never picked: a lower effort that was silently ignored would be the same slow run under a false caveat."""
    body: dict = {}
    if role.effort:
        body["reasoning"] = {"effort": role.effort}
    provider: dict = {}
    if provider_sort:
        provider["sort"] = provider_sort
    if role.effort or role.max_output_tokens:
        provider["require_parameters"] = True
    if provider:
        body["provider"] = provider
    return body


def build_model(role: str, settings: RoleSettings, provider_sort: str | None):
    """One chat model of the fast mode (a LangChain `ChatOpenAI` on OpenRouter, through `models.chat.get_chat_model`)."""
    from prometheus.models.chat import get_chat_model
    options: dict = {"timeout": settings.timeout_seconds, "max_retries": settings.max_retries}
    if settings.max_output_tokens:
        options["max_tokens"] = settings.max_output_tokens
    body = request_body(settings, provider_sort)
    if body:
        options["extra_body"] = body
    return get_chat_model(role, **options)


@dataclass
class Models:
    """The models of a run by role (None: the agent builds its own, as the live desk does) and the limits of its calls."""
    speed: str
    advisor: object = None
    critic: object = None
    data_clerk: object = None
    call_limit: float | None = None          # seconds one model call of the day may take (None: the live desk's own, `advisor.calls.MODEL_TIMEOUT`)
    parallel_records: int = 1                # the records of one review the Critic reviews at once
    settings: dict = field(default_factory=dict)


def build_models(speed: str, config: ProvingConfig) -> Models:
    """The models of a run in this `speed`. `faithful` builds none (the agents do, exactly as live) and `fast` builds the three of its section."""
    mode: ModeSettings = config.time_machine.fast if speed == "fast" else config.time_machine.faithful
    models = Models(speed=speed, call_limit=mode.call_limit_seconds, parallel_records=mode.parallel_records, settings=mode_summary(speed, config))
    if speed == "fast":
        for role in ("advisor", "critic", "data_clerk"):
            settings = getattr(mode, role)
            if settings is not None:
                setattr(models, role, build_model(role, settings, mode.provider_sort))
    return models


def mode_summary(speed: str, config: ProvingConfig) -> dict:
    """What a mode asks of the models that can change what they SAY, as plain JSON: the provider sort and, for each role, the reasoning effort and the cap on the reply. It is part of the research cache's key (a changed effort or cap is a
    different research); what only changes how long a call is waited for or how many records are checked at once (timeouts, retries, the call limit, `parallel_records`) and the measured note of a run's duration are NOT in it: the same
    answers come out of them."""
    mode: ModeSettings = config.time_machine.fast if speed == "fast" else config.time_machine.faithful
    summary: dict = {"provider_sort": mode.provider_sort}
    for role in ("advisor", "critic", "data_clerk"):
        settings = getattr(mode, role)
        summary[role] = None if settings is None else {"effort": settings.effort, "max_output_tokens": settings.max_output_tokens}
    return summary


@contextlib.contextmanager
def limit_of(seconds: float | None):
    """Within the block, one model call of the Advisor's calls (`advisor.calls.invoke_with_limit`) waits at most `seconds` (None: the live limit). A ContextVar, so it follows `contextvars.copy_context()` into the day's threads and
    never reaches another run's."""
    from prometheus.agents.advisor import calls
    token = calls.CALL_LIMIT.set(seconds)
    try:
        yield
    finally:
        calls.CALL_LIMIT.reset(token)
