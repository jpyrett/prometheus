"""The code rules (S10): plain regular expressions from the policy, run first, free, and the same answer for the same text.

    findings = screen("request", text, policy, account_number=7)       # [Finding(rule, score, detail, kind)]
    risk_of(findings)                                                   # the highest score, 0.0 for none (S2)
    exact_of(findings), hints_of(findings)                              # the two kinds of the request door (entry 121)
    redacted, findings = redact(text, policy, account_number=7)         # a MESSAGE: secrets and another account's number become "[redacted]"

A rule is a score given to a text when one of its patterns matches. A rule of the REQUEST door is EXACT (a program's own check: control tokens, hidden characters, another account's number, SQL as a statement; it blocks alone) or a HINT (a word pattern: it points
at the text, and the guard model decides; `Guard` joins them, entry 121). THE RULES NEVER FAIL OPEN (S5): a rule that raises is a finding of score 1.0 for that text. The text is NORMALIZED before the rules read it (Unicode NFKC, zero-width,
bidi and tag characters removed, runs of spaces made one) so that "ig<zero-width>nore previous instructions" is still seen; a rule marked `raw` (`hidden_text`) reads the text as it came. A finding says the rule and WHICH pattern, never the
text that matched (a message's match can be a password).

Nothing here knows a stock, a trade or a size: the rules are about injection, other people's data, leaving the application, secrets, abuse and hidden text.
"""
import re
import unicodedata
from dataclasses import dataclass

from prometheus.security.policy import EXACT, HINT, Policy, Rule

SPACES = re.compile(r"[ \t\f\v]+")


HINT_MARK = "hint:"


@dataclass(frozen=True)
class Finding:
    """One rule that fired: its id, its score, and which pattern (a number), never the matched text. `kind` is the rule's (entry 121): an `exact` finding blocks alone, a `hint` only points at the text (the request door's guard model decides)."""
    rule: str
    score: float
    detail: str
    kind: str = EXACT

    @property
    def name(self) -> str:
        """What a verdict and the audit log call it: the rule's id, and for a hint `hint:<id>`."""
        return HINT_MARK + self.rule if self.kind == HINT else self.rule


def normalize(text: str) -> str:
    """The text as the rules read it: NFKC, with every format character (zero-width, bidi, tag characters, the soft hyphen, the byte order mark) removed and runs of spaces and tabs made one space. Newlines stay (a role marker is at a line start)."""
    kept = "".join(ch for ch in unicodedata.normalize("NFKC", text) if unicodedata.category(ch) != "Cf")
    return SPACES.sub(" ", kept)


def _account_number(rule: Rule, text: str, account_number: int | None) -> Finding | None:
    """An account number in the text that is not the requester's. With no known requester there is nothing to compare with: no finding."""
    if account_number is None:
        return None
    for index, pattern in enumerate(rule.patterns, 1):
        for match in pattern.finditer(text):
            if int(match.group(1)) != account_number:
                return Finding(rule.id, rule.score, f"pattern {index}: a number that is not the requester's", rule.kind)
    return None


def _near(rule: Rule, text: str) -> Finding | None:
    """A word of `a` within `window` characters of a word of `b`: an imperative aimed at an AI inside a document."""
    for pattern in rule.a:
        for match in pattern.finditer(text):
            around = text[max(0, match.start() - rule.window): match.end() + rule.window]
            for index, other in enumerate(rule.b, 1):
                if other.search(around):
                    return Finding(rule.id, rule.score, f"an AI word near imperative {index}", rule.kind)
    return None


def _apply(rule: Rule, raw: str, normalized: str, account_number: int | None) -> Finding | None:
    text = raw if rule.raw else normalized
    if rule.match == "account_number":
        return _account_number(rule, text, account_number)
    if rule.match == "near":
        return _near(rule, text)
    for index, pattern in enumerate(rule.patterns, 1):
        if pattern.search(text):
            return Finding(rule.id, rule.score, f"pattern {index}", rule.kind)
    return None


def screen(door: str, text: str, policy: Policy, account_number: int | None = None) -> list[Finding]:
    """Every rule of this door that fires for `text`, in the policy's order. A rule that raises is a finding of score 1.0 (S5)."""
    normalized = normalize(text)
    findings = []
    for rule in policy.rules[door]:
        try:
            found = _apply(rule, text, normalized, account_number)
        except Exception as error:      # code rules never fail open: this text is as risky as can be
            found = Finding(rule.id, 1.0, f"the rule failed ({type(error).__name__})")      # always exact, even for a hint: a rule that cannot run is a block, never a pass
        if found is not None:
            findings.append(found)
    return findings


def risk_of(findings: list[Finding]) -> float:
    return max((f.score for f in findings), default=0.0)


def exact_of(findings: list[Finding]) -> list[Finding]:
    """The findings of rules a program checks exactly (entry 121): the only ones that block on their own."""
    return [f for f in findings if f.kind == EXACT]


def hints_of(findings: list[Finding]) -> list[Finding]:
    """The findings of word-pattern hints (entry 121): kept on the verdict and in the audit log; with the guard model they add nothing to the risk."""
    return [f for f in findings if f.kind == HINT]


def redact(text: str, policy: Policy, account_number: int | None = None) -> tuple[str, list[Finding]]:
    """A message to the user with each secret and each account number that is not the user's replaced by the policy's placeholder, and the findings (the rule and the number of places). A rule that raises changes nothing here and
    is a finding of score 1.0: the caller decides (the Guard withholds the whole message then)."""
    placeholder = policy.messages["redacted"]
    findings = []
    for rule in policy.rules["message"]:
        count = 0
        try:
            for pattern in rule.patterns:
                if rule.match == "account_number":
                    if account_number is None:
                        continue

                    def swap(match, number=account_number):
                        nonlocal count
                        if int(match.group(1)) == number:
                            return match.group(0)
                        count += 1
                        return match.group(0)[: match.start(1) - match.start(0)] + placeholder

                    text = pattern.sub(swap, text)
                else:
                    text, n = pattern.subn(placeholder, text)
                    count += n
        except Exception as error:
            findings.append(Finding(rule.id, 1.0, f"the rule failed ({type(error).__name__})"))
            continue
        if count:
            findings.append(Finding(rule.id, rule.score, f"{count} place{'s' if count != 1 else ''} redacted"))
    return text, findings


def redact_secrets(text: str, policy: Policy) -> str:
    """The text with only the message door's `secrets` rule applied: what the audit log keeps of a request (an account number there is the user's own words)."""
    placeholder = policy.messages["redacted"]
    for rule in policy.rules["message"]:
        if rule.id == "secrets":
            for pattern in rule.patterns:
                text = pattern.sub(placeholder, text)
    return text
