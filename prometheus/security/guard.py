"""The doors of the security layer (S1 to S9; design choices entry 116): where a text is screened, and what happens to it.

    guard = Guard.code_only()                 # rules + router + audit, no model: the default of `build_desk` and of every test
    guard = Guard.with_scorer()               # rules + the LLM Risk Scorer + audit: the real parts of a session and of the web page

    verdict = guard.screen_request(text, account_id=12)        # the user's request: BLOCK and REDACT_OR_REVIEW refuse the run (`verdict.refused`, `verdict.message`)
    verdict = guard.screen_document(text, doc_id=..., corpus="news")      # a retrieved document: code rules only; a refused one is dropped from the evidence
    verdict = guard.screen_message(text, account_id=12)        # a message to the user: `verdict.text` is the text with secrets and another account's number redacted
    guard.check_rate(account_id)                               # the rate limit: a refusing Verdict, or None when the request may run
    guard.trust(order)                                         # the Trust Score of each recommendation: recorded, shown nowhere else, changes no decision

THE RISK SCORE (S2) of a REQUEST (entry 121). A rule of the request door is EXACT (a program's own check: control tokens, hidden characters, another account's number, SQL as a statement) or a HINT (a word pattern). With a scorer, the risk is the
MAXIMUM of the exact rules' scores and the model's; a hint adds NOTHING to it (the model reads the text alone, it is never told about the hints), it is kept on the verdict as `hint:<rule>` and in the audit log, and when it fired while the model
and the exact rules left the action at ALLOW, the action is raised to ALLOW_AND_FLAG (allowed, logged for a person to look at). The model is called on every request unless an EXACT rule already gives the BLOCK threshold. With NO scorer
(`Guard.code_only()`: a desk that has no model to decide) a hint keeps its YAML score, so that desk is no weaker than before. A document and a message have only exact rules. `router.act` turns the risk into an action (S3).
FAIL CLOSED (S5): a bug in a door is a refusal for that input (rule `guard_error`), never a pass. Every screen is a `traced("security.<door>")` span and one line of the audit log (S8); what a span or a line holds of the text is its hash and a short
redacted preview, never the text (the screened text goes to the span inside a holder that a trace shows as its type only).

Nothing here refuses, cuts or resizes a trade: a request that screens ALLOW or ALLOW_AND_FLAG runs exactly as it always did.
"""
import copy
from dataclasses import dataclass, field

from prometheus.common.trace import traced
from prometheus.security import rules, router
from prometheus.security.audit import AuditLog, sha256_of
from prometheus.security.policy import Policy, load_policy
from prometheus.security.rate import RateLimiter
from prometheus.security.router import ALLOW, ALLOW_AND_FLAG, BLOCK, REDACT_OR_REVIEW
from prometheus.security.scorer import RiskScorer

REFUSED = (BLOCK, REDACT_OR_REVIEW)
RATE_RULE, GUARD_ERROR_RULE, SCORER_FAILURE_RULE = "rate_limit", "guard_error", "scorer_failure"


class _Held:
    """The text being screened, in a holder that is not data: a trace shows its type and nothing else (a request may hold a key the user pasted; a document is not copied into a trace)."""

    def __init__(self, text):
        self.text = text


@dataclass(frozen=True)
class Verdict:
    """What a door decided: the action, the risk, which rules fired, why, the message for the user when it refuses, and (a message) the text to show."""
    door: str
    action: str
    risk: float
    rules: tuple = ()
    reason: str = ""
    message: str = ""            # a refused request: what the user is shown
    text: str = ""               # a message: the text with secrets and another account's number redacted
    scored_by: str = "code"      # "code", "code+model" or "rate"
    scores: dict = field(default_factory=dict)      # the scorer's four labels, when it was used
    sha256: str = ""

    @property
    def refused(self) -> bool:
        """BLOCK, and REDACT_OR_REVIEW (which fails closed: there is no operator console yet, the audit log marks it for a person to look later)."""
        return self.action in REFUSED

    @property
    def review(self) -> bool:
        return self.action == REDACT_OR_REVIEW

    def as_dict(self) -> dict:
        return {"door": self.door, "action": self.action, "risk": self.risk, "rules": list(self.rules), "reason": self.reason, "message": self.message, "scored_by": self.scored_by,
                "scores": dict(self.scores), "review": self.review, "sha256": self.sha256}


def account_number_of(account_id) -> int | None:
    """The requester's account NUMBER when `account_id` is one (a whole number, or text of digits); a sample key such as `jacob_sample` is a name, not a number."""
    if isinstance(account_id, bool):
        return None
    if isinstance(account_id, int):
        return account_id if account_id >= 1 else None
    if isinstance(account_id, str) and account_id.strip().isdigit():
        return int(account_id.strip()) or None
    return None


class Guard:
    def __init__(self, policy: Policy | None = None, scorer: RiskScorer | None = None, audit: AuditLog | None = None, limiter: RateLimiter | None = None, ledger=None):
        self.policy = policy or load_policy()
        self.scorer, self.limiter, self.ledger = scorer, limiter, ledger
        self.audit = audit if audit is not None else AuditLog(policy=self.policy)

    # ---- the two ways to build one

    @classmethod
    def code_only(cls, policy: Policy | None = None, audit: AuditLog | None = None, clock=None, ledger=None) -> "Guard":
        """Rules, router, rate limit and audit; NO model. `clock` (seconds, monotonic) is for a test of the rate limit."""
        policy = policy or load_policy()
        limiter = RateLimiter(policy.rate.max_requests_per_minute, policy.rate.max_requests_per_hour, **({"clock": clock} if clock else {}))
        return cls(policy, None, audit, limiter, ledger)

    @classmethod
    def with_scorer(cls, model=None, policy: Policy | None = None, audit: AuditLog | None = None, clock=None, ledger=None) -> "Guard":
        """Rules, the LLM Risk Scorer (the `guard` role's model, built when first needed; or `model`), router, rate limit and audit."""
        policy = policy or load_policy()
        guard = cls.code_only(policy, audit, clock, ledger)
        guard.scorer = RiskScorer(model, policy)
        return guard

    def without_rate_limit(self) -> "Guard":
        """The same guard (policy, scorer, audit, ledger) that does not count requests: for a desk whose requests were already counted at the door of the page (`POST /api/runs`)."""
        other = copy.copy(self)
        other.limiter = None
        return other

    # ---- the rate limit (S9)

    @traced("security.rate")
    def check_rate(self, account_id=None) -> Verdict | None:
        """None when this request may run (it is counted); a BLOCK verdict with the policy's message when the account is over its limit, logged."""
        if self.limiter is None or self.limiter.allow(account_id):
            return None
        verdict = Verdict("request", BLOCK, 1.0, (RATE_RULE,), "over the rate limit", self.policy.messages["rate"], scored_by="rate")
        self.audit.write(self.audit.line(door="request", action=BLOCK, risk=1.0, rule_ids=verdict.rules, reason=verdict.reason, text="", account_id=account_id))
        return verdict

    # ---- the user's request (S4, S10)

    def screen_request(self, text, *, account_id=None, rate: bool = True) -> Verdict:
        """Screen what the user typed. `rate` counts it against the account's limit first (a desk behind the page's own count passes False)."""
        return self._request(_Held(text), account_id, rate)

    @traced("security.request")
    def _request(self, held: _Held, account_id=None, rate: bool = True) -> Verdict:
        text = held.text if isinstance(held.text, str) else ""
        try:
            if rate and (refused := self.check_rate(account_id)) is not None:
                return refused
            findings = rules.screen("request", text, self.policy, account_number_of(account_id))
            exact, hints = rules.exact_of(findings), rules.hints_of(findings)
            ids = [f.name for f in findings]
            details = [f"{f.name}: {f.detail}" for f in findings]
            risk, scored_by, scores = rules.risk_of(findings), "code", {}                        # no scorer: a hint keeps its score (Q5)
            if self.scorer is not None:
                risk = rules.risk_of(exact)
                if risk < self.policy.router.block:                                              # S2: the model is not called when an EXACT rule already blocks
                    result = self.scorer.score(text)                                             # it is given the text alone: a hint cannot talk it into a refusal
                    scored_by, scores = "code+model", dict(result.parts)
                    risk = max(risk, result.risk)
                    details.append(f"model: {rules.redact_secrets(result.reason, self.policy)}" if result.reason else "model: no reason")      # the model may quote a key it was shown: the verdict, the trace and the log never carry it
                    if result.failed:
                        ids.append(SCORER_FAILURE_RULE)
            action = router.act(risk, self.policy.router)
            if self.scorer is not None and hints and action == ALLOW:                              # a hint fired and neither the exact rules nor the model saw a risk: allowed, and logged for a person to look at
                action = ALLOW_AND_FLAG
                details.append("flagged: a word pattern fired that the model did not rate")
            verdict = Verdict("request", action, risk, tuple(dict.fromkeys(ids)), "; ".join(details), self._refusal(action), scored_by=scored_by, scores=scores, sha256=sha256_of(text))
        except Exception as error:      # a bug in the guard is a refusal for this input, never a pass (S5)
            verdict = Verdict("request", BLOCK, 1.0, (GUARD_ERROR_RULE,), f"the guard failed ({type(error).__name__})", self.policy.messages["block"], sha256=sha256_of(text))
        self.audit.write(self.audit.line(door="request", action=verdict.action, risk=verdict.risk, rule_ids=verdict.rules, reason=self._clean(verdict.reason), text=text, account_id=account_id, scored_by=verdict.scored_by))
        return verdict

    def _refusal(self, action: str) -> str:
        return self.policy.messages["block"] if action == BLOCK else self.policy.messages["review"] if action == REDACT_OR_REVIEW else ""

    def _clean(self, reason: str) -> str:
        """What a log line keeps of a reason: the model's words with every secret pattern redacted."""
        return rules.redact_secrets(reason, self.policy)[: self.policy.audit.preview_chars]

    # ---- a document (S4, S10): code rules only, no model call per document

    def screen_document(self, text, *, doc_id: str = "", corpus: str = "") -> Verdict:
        """Screen one retrieved document (news, press release, filing text). BLOCK and REDACT_OR_REVIEW drop it from the evidence (`verdict.refused`); a lower action keeps it. Logged with its id and the rules."""
        return self._document(_Held(text), doc_id, corpus)

    @traced("security.document")
    def _document(self, held: _Held, doc_id: str = "", corpus: str = "") -> Verdict:
        text = held.text if isinstance(held.text, str) else ""
        try:
            findings = rules.screen("document", text, self.policy)
            risk = rules.risk_of(findings)
            action = router.act(risk, self.policy.router)
            verdict = Verdict("document", action, risk, tuple(f.rule for f in findings), "; ".join(f"{f.rule}: {f.detail}" for f in findings), sha256=sha256_of(text))
        except Exception as error:
            verdict = Verdict("document", BLOCK, 1.0, (GUARD_ERROR_RULE,), f"the guard failed ({type(error).__name__})", sha256=sha256_of(text))
        self.audit.write(self.audit.line(door="document", action=verdict.action, risk=verdict.risk, rule_ids=verdict.rules, reason=verdict.reason, text=text, doc_id=str(doc_id), corpus=str(corpus)))
        return verdict

    # ---- a message to the user (S4, S10): redacted in place

    def screen_message(self, text, *, account_id=None) -> Verdict:
        """Screen a message the user is about to read. `verdict.text` is the message with each secret and each account number that is not the user's replaced by the policy's placeholder; the rest is shown as it is."""
        return self._message(_Held(text), account_id)

    @traced("security.message")
    def _message(self, held: _Held, account_id=None) -> Verdict:
        text = held.text if isinstance(held.text, str) else str(held.text)
        try:
            shown, findings = rules.redact(text, self.policy, account_number_of(account_id))
            if any(f.score >= 1.0 for f in findings):      # a rule of the message door failed: nothing of the message is shown
                shown = self.policy.messages["redacted"]
            risk = rules.risk_of(findings)
            action = router.act(risk, self.policy.router)
            verdict = Verdict("message", action, risk, tuple(f.rule for f in findings), "; ".join(f"{f.rule}: {f.detail}" for f in findings), text=shown, sha256=sha256_of(text))
        except Exception as error:
            verdict = Verdict("message", BLOCK, 1.0, (GUARD_ERROR_RULE,), f"the guard failed ({type(error).__name__})", text=self.policy.messages["redacted"], sha256=sha256_of(text))
        self.audit.write(self.audit.line(door="message", action=verdict.action, risk=verdict.risk, rule_ids=verdict.rules, reason=verdict.reason, text=text, preview_text=verdict.text, account_id=account_id))
        return verdict

    # ---- the Trust Score (S7): reliability, never blocks anything

    def trust(self, order) -> list[dict]:
        """The Trust Score of each recommendation on the work order: [{symbol, record_id, trust, grounded, source, confidence, history}]. Recorded on the audit log and in the trace and changes nothing else. Never raises."""
        scored = []
        try:
            for record in list(getattr(order, "recommendations", None) or []):
                if isinstance(record, dict):
                    scored.append(self._trust_one(str(record.get("symbol") or ""), _Held(record)))
        except Exception:      # reliability is a measure, never a gate: whatever went wrong, the run goes on
            return scored
        return scored

    @traced("security.trust_score")
    def _trust_one(self, symbol: str, held: _Held) -> dict:
        record = held.text
        signals = router.signals_of(record, self._history(symbol), self.policy)
        value = router.trust_score(signals, self.policy.trust)
        entry = {"symbol": symbol, "record_id": str(((record.get("identity") or {}).get("record_id")) or ""), "trust": round(value, 4), **{k: round(v, 4) for k, v in signals.items()}}
        self.audit.write(self.audit.trust_line(entry))
        return entry

    def _history(self, symbol: str) -> float | None:
        """The ledger's past hit rate for the symbol: the share of its earlier records whose `outcome.hit` is True. None without a ledger or without any record that has an outcome (the ledger holds none yet: the outcomes are added later)."""
        if self.ledger is None or not symbol:
            return None
        try:
            hits = [r["outcome"]["hit"] for r in self.ledger.recent(symbol) if isinstance(r.get("outcome"), dict) and isinstance(r["outcome"].get("hit"), bool)]
        except Exception:
            return None
        return sum(hits) / len(hits) if hits else None
