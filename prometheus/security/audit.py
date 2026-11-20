"""The audit log (S8): append-only JSON lines, one per screened input, at `logs/security_audit.jsonl` (gitignored; the folder is created when missing).

    {"time": "2026-11-20T09:12:03Z", "door": "request", "account": "12", "action": "BLOCK", "risk": 0.9, "rules": ["prompt_injection"], "reason": "pattern 1",
     "sha256": "...", "preview": "the first 200 characters, every secret pattern redacted"}

A line holds NEVER a key, a password or a whole document: the text is a SHA-256 and at most its first `preview_chars` characters with the message door's secret patterns replaced. A write that fails is printed to stderr ONCE and never stops
a run. `PROMETHEUS_AUDIT=false` (the environment, then `.env`) turns the log off: the tests do (`tests/conftest.py`), and a test of the log itself gives the log `enabled=True` and a path of its own.
"""
import hashlib
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from prometheus.rag.config import setting
from prometheus.security import rules
from prometheus.security.policy import Policy, load_policy

OFF = ("false", "no", "off", "0")


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path | None = None, enabled: bool | None = None, policy: Policy | None = None, clock=None):
        """`path` default is the policy's `audit.path`. `enabled` None reads PROMETHEUS_AUDIT (default true) at each write; a bool is final."""
        self.policy = policy or load_policy()
        self.path = Path(path) if path else self.policy.audit.path
        self._enabled = enabled
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._failed = False

    @property
    def enabled(self) -> bool:
        return self._enabled if self._enabled is not None else setting("PROMETHEUS_AUDIT", "true").strip().lower() not in OFF

    def line(self, *, door: str, action: str, risk: float, rule_ids, reason: str, text: str, account_id=None, preview_text: str | None = None, **extra) -> dict:
        """The record of one screened input (what `write` appends). The SHA-256 is of `text`; the preview is of `preview_text` when given (a message: the redacted one) and else of `text`, tidied, cut and with every secret pattern redacted."""
        shown = text if preview_text is None else preview_text
        preview = rules.redact_secrets(" ".join(shown.split()) if isinstance(shown, str) else "", self.policy)[: self.policy.audit.preview_chars]
        return {"time": self._now(), "door": door, "account": None if account_id is None else str(account_id), "action": action, "risk": round(risk, 4),
                "rules": list(rule_ids), "reason": reason, **extra, "sha256": sha256_of(text if isinstance(text, str) else ""), "preview": preview}

    def trust_line(self, entry: dict) -> dict:
        """The record of one Trust Score: the recommendation (symbol and record id), the score and its four signals. No text of any kind."""
        return {"time": self._now(), "door": "trust", **entry}

    def _now(self) -> str:
        return self._clock().strftime("%Y-%m-%dT%H:%M:%SZ")

    def write(self, record: dict) -> None:
        """Append one line. Never raises: a write that fails is printed to stderr once."""
        if not self.enabled:
            return
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as error:
            if not self._failed:
                self._failed = True
                print(f"security audit log: a line could not be written ({type(error).__name__}); the run goes on", file=sys.stderr)
