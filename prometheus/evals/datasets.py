"""The two datasets of the offline evaluation (design choices entry 117; design/offline_evaluation_spec.md E3, E4), built from files that are already in the repo. Nothing is invented: every example is a line of the red-team set or a real session capture.

    redteam_examples()    one example per line of `configs/security/red_team_v1.jsonl` (entry 116): inputs {text, door[, account_id]}, reference {expected_action, category, is_attack}
    session_examples()    one example per real capture of `tests/fixtures/session_real` that holds a work order: inputs {request, mode, example_key}, reference {recorded: <the summary>}
    summary_of(capture)   what a session did, as a small record: what was understood, the recommendations, the Critic's reviews, the decisions, the trades and fills, the cash before and after. NEVER article text:
                          a decision's reason and a review's text can quote a headline, so only codes, numbers and symbols are copied

EVERY example has a stable `example_key` (in its metadata) and a stable id (`example_id`: a UUID made from the dataset and the key), so an upload is IDEMPOTENT: `plan_upload` adds the keys a dataset lacks, updates the ones whose content changed and never
makes a second copy; an example in the dataset that the files no longer hold is only reported, never deleted. The red-team key is a hash of the door and the text (an edited attack is another attack); the session key is the capture's file name.
"""
import hashlib
import json
import uuid
from datetime import date

from prometheus.rag.config import REPO_ROOT

REDTEAM = "prometheus-security-redteam"
SESSIONS = "prometheus-sessions"
DATASETS = (REDTEAM, SESSIONS)
RED_TEAM_FILE = REPO_ROOT / "configs" / "security" / "red_team_v1.jsonl"
CAPTURES = REPO_ROOT / "tests" / "fixtures" / "session_real"
GUARD_REPLIES = REPO_ROOT / "tests" / "fixtures" / "security" / "guard_replies.json"
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "prometheus/offline-evals")
MAX_TEXT = 300          # a string in a summary is cut here: it is a code or a short system message, never a document


def description(name: str) -> str:
    """What a dataset says about itself on the LangSmith page: where its examples come from and the date it was created."""
    today = date.today().isoformat()
    if name == REDTEAM:
        return (f"Prometheus security red team (design choices entries 116 and 117): every line of configs/security/red_team_v1.jsonl, the attacks and the benign requests with the action a person expects "
                f"(BLOCK, REDACT_OR_REVIEW or ALLOW). Created {today} by scripts/run_offline_evals.py; re-run with --upload to add or update examples.")
    if name == SESSIONS:
        return (f"Prometheus real sessions (design choices entry 117): one example per real session capture of tests/fixtures/session_real that holds a work order, with what the desk did (understood, recommendations, "
                f"the Critic's verdicts, decisions, trades, cash before and after). Created {today} by scripts/run_offline_evals.py; no article text is stored.")
    raise ValueError(f"unknown dataset {name!r}: choose one of {', '.join(DATASETS)}")


def example_id(dataset: str, key: str) -> str:
    """The id of an example: the same for the same dataset and key, on every machine and every run."""
    return str(uuid.uuid5(NAMESPACE, f"{dataset}/{key}"))


# ---------------------------------------------------------------- the red team

def redteam_examples() -> list[dict]:
    """One example per line of the red-team file, in file order. Raises ValueError for a line that has no text or label, or two lines with the same door and text (they would share a key)."""
    examples, seen = [], set()
    for line, raw in enumerate(RED_TEAM_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        text, expected, category = row.get("text"), row.get("expected_action"), row.get("category")
        if not isinstance(text, str) or not text or not isinstance(expected, str) or not isinstance(category, str):
            raise ValueError(f"{RED_TEAM_FILE.name} line {line}: a line needs text, expected_action and category")
        door = row.get("door", "request")
        key = "redteam:" + hashlib.sha256(f"{door}\n{text}".encode("utf-8")).hexdigest()[:12]
        if key in seen:
            raise ValueError(f"{RED_TEAM_FILE.name} line {line}: the same {door} text twice")
        seen.add(key)
        inputs = {"text": text, "door": door}
        if "account_id" in row:
            inputs["account_id"] = row["account_id"]
        examples.append({
            "key": key,
            "inputs": inputs,
            "outputs": {"expected_action": expected, "category": category, "is_attack": category != "benign"},
            "metadata": {"example_key": key, "source": "configs/security/red_team_v1.jsonl", "line": line, "category": category, "door": door, "needs_llm": bool(row.get("needs_llm"))},
        })
    return examples


# ---------------------------------------------------------------- the real sessions

def capture_paths() -> dict:
    """{example_key: path} of every real capture that holds a work order (a dict `order`), by file name."""
    found = {}
    for path in sorted(CAPTURES.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("order"), dict) and isinstance(data.get("request"), str):
            found[f"session:{path.stem}"] = path
    return found


def _short(value) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "..."


def _kept(record: dict) -> dict:
    return {k: v for k, v in record.items() if v is not None and v != [] and v != {}}


def _action_of(record: dict):
    action = record.get("action")
    return action.get("action") if isinstance(action, dict) else action      # schema 3 has a record, schema 2 a word


def summary_of(capture: dict) -> dict:
    """What the session did, from the capture of a real run (`{request, intent, path, order}`); numbers stay as the capture holds them. Only codes, numbers, symbols and short system words: no article text, no reason sentence, no thesis."""
    order, intent = capture["order"], capture.get("intent") or {}
    decisions = [d for d in order.get("trade_decisions") or [] if isinstance(d, dict)]
    by_record = {d.get("record_id"): d for d in decisions}
    runs = [r for r in order.get("trader_runs") or [] if isinstance(r, dict)]
    understood = _kept({k: intent.get(k) for k in ("action", "topic", "symbols", "companies", "count", "allocation", "budget")})
    understood["orders"] = [_kept({k: o.get(k) for k in ("side", "symbol", "company", "shares")}) for o in intent.get("orders") or [] if isinstance(o, dict)]
    trades = []
    for trade in order.get("trades") or []:
        if not isinstance(trade, dict):
            continue
        fill = trade.get("fill") if isinstance(trade.get("fill"), dict) else {}
        decision = by_record.get(trade.get("record_id")) or {}
        origin = "raise_funds" if trade.get("kind") == "funding" else decision.get("origin") or "recommendation"
        trades.append(_kept({"trade_id": trade.get("trade_id"), "kind": trade.get("kind"), "record_id": trade.get("record_id"), "symbol": trade.get("symbol"), "side": trade.get("side"), "status": trade.get("status"),
                             "reject_code": trade.get("reject_code"), "origin": origin, "shares": fill.get("shares"), "price": fill.get("price"), "amount": fill.get("amount"), "commission": fill.get("commission")}))
    return {
        "request": capture["request"],
        "mode": next((d.get("mode") for d in decisions if d.get("mode")), None),
        "understood": understood,
        "path": [str(p) for p in capture.get("path") or []],
        "status": order.get("status"),
        "errors": [_short(e) for e in order.get("errors") or []],
        "recommendations": [_kept({"record_id": (r.get("identity") or {}).get("record_id"), "symbol": r.get("symbol"), "stance": r.get("stance"), "score": r.get("score"), "action": _action_of(r)})
                            for r in order.get("recommendations") or [] if isinstance(r, dict)],
        "reviews": [_kept({"record_id": r.get("record_id"), "symbol": r.get("symbol"), "approved": r.get("approved"), "score": r.get("score"), "requires_human": r.get("requires_human"),
                           "denial_codes": [str(c) for c in r.get("denial_reasons") or []]}) for r in order.get("reviews") or [] if isinstance(r, dict)],
        "decisions": [_kept({"record_id": d.get("record_id"), "symbol": d.get("symbol"), "action": d.get("action"), "will_trade": d.get("will_trade"), "origin": d.get("origin"), "shares": d.get("shares"),
                             "dollars": d.get("dollars"), "est_cost": d.get("est_cost"), "pot": (d.get("size") or {}).get("pot")}) for d in decisions],
        "trades": trades,
        "budget": intent.get("budget"),
        "cash_before": ((runs[0].get("portfolio_before") or {}).get("cash") if runs else None),
        "cash_after": ((runs[-1].get("portfolio_after") or {}).get("cash") if runs else None),
    }


def recorded_summary(key: str) -> dict:
    """The summary of the capture an example key names. Raises KeyError for a key no capture has."""
    paths = capture_paths()
    if key not in paths:
        raise KeyError(f"no real capture has the example key {key!r}")
    return summary_of(json.loads(paths[key].read_text(encoding="utf-8")))


def session_examples() -> list[dict]:
    """One example per real capture that holds a work order, in file-name order."""
    examples = []
    for key, path in capture_paths().items():
        summary = summary_of(json.loads(path.read_text(encoding="utf-8")))
        examples.append({
            "key": key,
            "inputs": {"request": summary["request"], "mode": summary["mode"] or "unknown", "example_key": key},
            "outputs": {"recorded": summary},
            "metadata": {"example_key": key, "source": str(path.relative_to(REPO_ROOT)), "status": summary["status"], "path": "/".join(summary["path"])},
        })
    return examples


def build(name: str | None = None) -> dict:
    """{dataset name: examples}, or just the one named."""
    both = {REDTEAM: redteam_examples(), SESSIONS: session_examples()}
    if name is None:
        return both
    if name not in both:
        raise ValueError(f"unknown dataset {name!r}: choose one of {', '.join(DATASETS)}")
    return {name: both[name]}


# ---------------------------------------------------------------- the upload, as a plan (pure: the command makes the calls)

def plan_upload(dataset: str, wanted: list[dict], existing: list[dict]) -> dict:
    """What an upload must do to make `dataset` hold `wanted`. `existing` are the dataset's examples as dicts {id, inputs, outputs, metadata}.

    {"create": the wanted examples whose key the dataset lacks, "update": [(id, example)] the ones whose inputs, reference outputs or metadata differ, "same": how many are already as wanted,
     "orphans": the keys the dataset has that the files no longer hold (reported, never deleted)}. An example is found by its `example_key`, or by its id when it has none.
    """
    have = {}
    for old in existing:
        key = (old.get("metadata") or {}).get("example_key")
        have[key or str(old.get("id"))] = old
    create, update, same = [], [], 0
    for new in wanted:
        old = have.get(new["key"]) or have.get(example_id(dataset, new["key"]))
        if old is None:
            create.append(new)
        elif (old.get("inputs") != new["inputs"] or old.get("outputs") != new["outputs"]
              or any((old.get("metadata") or {}).get(k) != v for k, v in new["metadata"].items())):
            update.append((str(old["id"]), new))
        else:
            same += 1
    wanted_keys = {n["key"] for n in wanted} | {example_id(dataset, n["key"]) for n in wanted}
    return {"create": create, "update": update, "same": same, "orphans": sorted(k for k in have if k not in wanted_keys)}
