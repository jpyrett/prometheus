"""The Critic: an independent reviewer that approves or denies each recommendation of the Advisor (Critic v1 spec, design choices entries 56 to 59).

    critic = Critic()                                                       # Z.ai GLM 5.3 through OpenRouter (the `critic` role: CRITIC_MODEL), the real Data Clerk
    order = critic.review_recommendations(order, view, now)                 # status "recommended" -> "reviewed"; one review per recommendation on `order.reviews`

    python -m prometheus.application.recommendations MCD --no-fetch --no-ledger          # the whole desk, Critic included (paid: Finnhub and the critic's model)

The pattern is the Advisor's (prometheus/agents/advisor/advisor.py): a class that takes a chat model (a scripted one in tests) or builds the role's model with `get_chat_model`, one public method
that does the job (`review.py` holds the nine steps, as `advisor/recommend.py` holds the Advisor's twelve), a prompt module (`prompts/critic.py`), thresholds in a config file
(`configs/critic_v1.json`), and CODE that checks everything the model says.

WHAT MAKES IT A CRITIC AND NOT A SECOND ADVISOR. It does not recalculate the Advisor's work with the Advisor's code. It asks the DATA CLERK, directly, to calculate the key numbers again, to give a
current price, and to read the cited sources again from the store; it compares the old values with the new ones and denies what does not match. It makes its own judgements of risk, timing and
value from the numbers IT asked for. It uses a model of a different producer than the Advisor's, for the one thing code cannot do: say whether a quote really is good news for the company, with a
control claim planted in every call to prove the model read the text. The model never decides anything: the verdict is arithmetic over findings, and every label the model gives can only lower a result.

THE CRITIC ONLY DEALS WITH RECOMMENDATIONS. It approves or denies a company's recommendation (accuracy, evidence, the company's risk against the user's tolerance, timing, value, chart). It does not know the user's money and it
sizes nothing: how much to buy or sell is the Account Manager's (agents/account_manager/sizing.py), made after the review, from the user's money and limits.

THE CRITIC NEVER TRADES AND NEVER FETCHES (spec P8): its Data Clerk requests read stored data and Finnhub's quote, metric and symbol-search endpoints; it never sets `allow_fetch`.

VERSION 2 (config_version "2", design/critic_v2_evidence_spec.md, design choices entry 133): A DENIAL REASON MUST EARN ITS PLACE. Each finding code is scored against the outcomes by a program (`proving/report_card.py`), and the reasons that denied winners
were changed: `fresh_negative_news` and `blend_not_supported` inform and no longer deny (C2, C3: `evidence.py`), `company_mismatch` compares the names in `names.py` (C4), the new blocker `trend_not_confirmed` denies a trend recommendation whose
stock fell over the last five sessions (C5: `chart.py`), and the model's labels are kept in a store so that the same quote gets the same label (C6: `labels.py`). It still sizes nothing, sees no portfolio and has no liquidity, stop-distance or market-cap rule (C7).
"""
from prometheus.agents.advisor.config import load_config as load_advisor_config
from prometheus.agents.critic.config import CriticConfig, load_config
from prometheus.agents.critic.labels import LabelStore, open_store
from prometheus.agents.critic.review import review_order
from prometheus.common.trace import traced
from prometheus.common.user_profile import RiskView
from prometheus.common.work_order import WorkOrder

EFFORTS = ("low", "high", "max")   # the reasoning efforts OpenRouter lists for GLM 5.3; `max` is the model's own default


class Critic:
    def __init__(self, model=None, model_id: str | None = None, clerk=None, config_path=None, ledger=None, history=None, effort: str | None = None, advisor_config=None, parallel: int = 1,
                 label_store: LabelStore | None | bool = True):
        """model: a chat model (anything with `invoke`; a scripted one in tests); otherwise OpenRouter through `get_chat_model` with the model of the `critic` role (CRITIC_MODEL, Z.ai GLM 5.3 by
        default), or `model_id`, built when first needed: a model that cannot start (a missing key) is a note on the review, never an exception. clerk: the Data Clerk that is asked for the numbers,
        the quote and the sources (anything with `request_metrics`, `request_quotes` and `request_verification`; a fake in tests; the real one, with no model of its own to call, when left out).
        config_path: the Critic's settings (configs/critic_v1.json). ledger: where the reviews are appended (data/ledger/reviews.jsonl; a test passes one in a temporary folder). history: the ADVISOR'S
        recommendation ledger, read only, for the two warnings about earlier recommendations of the same symbol (None skips them). advisor_config: the Advisor's settings, whose formulas the Critic
        works again (configs/advisor_v1.json when left out).

        effort: None (the default) sends no reasoning parameter, so GLM 5.3 runs at its own default, `max`. Do not lower it to save time: on OpenRouter's listed benchmarks the model at `low`
        avoids a made-up answer far less often than at `max`. "low", "high" or "max" sends OpenRouter's `reasoning` effort, for an experiment.

        parallel: how many records of one review are checked at once (1, the default, is the sequential loop every live run uses; the Proving Ground's Time Machine passes more, `review._check_all`: the same review comes out, in the same order).

        label_store: where the model's labels are kept so that the same quote gets the same label (Critic v2, `labels.py`): True (the default) opens the store the config names (`evidence.label_store`, None there turns it off); a `LabelStore` is
        used as it is (a test passes one in a temporary folder); None or False keeps and reads no label."""
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)} (or None for the model's own default)")
        if not isinstance(parallel, int) or isinstance(parallel, bool) or parallel < 1:
            raise ValueError("parallel must be a whole number of at least 1")
        self.parallel = parallel
        self._model, self._model_id, self._clerk, self._config_path, self._ledger = model, model_id, clerk, config_path, ledger
        self._advisor_config, self._config = advisor_config, None
        self._label_store = label_store
        self.history, self.effort = history, effort
        self.errors: list[str] = []            # why the critic's model could not start, when it could not: copied onto the run of every review
        self._model_tried = model is not None

    @property
    def config(self) -> CriticConfig:
        """The Critic's thresholds (configs/critic_v1.json), read and checked once."""
        if self._config is None:
            self._config = load_config(self._config_path)
        return self._config

    @property
    def label_store(self) -> LabelStore | None:
        """The label store of this Critic (see `__init__`): the config's own folder when left as True, else what was given (None: none)."""
        if self._label_store is True:
            self._label_store = open_store(self.config.evidence.label_store)
        return self._label_store or None

    @property
    def advisor_config(self):
        """The Advisor's thresholds: the constants of the formulas the Critic works again."""
        if self._advisor_config is None:
            self._advisor_config = load_advisor_config()
        return self._advisor_config

    @property
    def model(self):
        """The critic's chat model, built on first use. None when it could not be built (the reason is in `errors`)."""
        if not self._model_tried:
            self._model_tried = True
            try:
                from prometheus.models.chat import get_chat_model
                extra = {"extra_body": {"reasoning": {"effort": self.effort}}} if self.effort else {}
                self._model = get_chat_model("critic", model=self._model_id, timeout=self.config.limits.model_timeout_seconds, **extra)
            except (RuntimeError, ValueError) as error:  # for example a missing key: the message names the setting, never a value
                self.errors.append(f"the critic's model could not start: {error}")
        return self._model

    @property
    def model_name(self) -> str | None:
        """The OpenRouter id of the critic's model: what the model says it is, else the id asked for, else the `critic` role's setting."""
        if self._model is not None and getattr(self._model, "model_name", None):
            return self._model.model_name
        if self._model_id:
            return self._model_id
        if self._model is None and self._model_tried:
            return None
        from prometheus.models.chat import model_name
        return model_name("critic")

    @property
    def clerk(self):
        if self._clerk is None:
            from prometheus.agents.data_clerk.data_clerk_agent import DataClerk
            self._clerk = DataClerk(model=object())  # the numbers, the quote and the sources are code: this Data Clerk's own model is never called here
        return self._clerk

    @property
    def ledger(self):
        if self._ledger is None:
            from prometheus.memory.ledger import Ledger
            self._ledger = Ledger(self.config.ledger.path)
        return self._ledger

    @traced("critic.review_recommendations")
    def review_recommendations(self, order: WorkOrder, risk: RiskView, now=None) -> WorkOrder:
        """Approve or deny each recommendation on the order (see `review.py` for the nine steps) and return the SAME order with `status == "reviewed"`, `reviews` (one per recommendation, in the same
        order, joined by `record_id`) and `critic_runs`. `risk` is what the user accepts (`UserProfile.risk_view()`: the tolerance, age, horizon and exclusions, never the whole profile and no size limit), `now` an aware
        datetime (the real clock when left out). The Critic is given no portfolio and no money: how much to buy is the Account Manager's. Never raises for a data, API or model failure: what could not be checked is a finding, and is not approved."""
        return review_order(self, order, risk, now)
