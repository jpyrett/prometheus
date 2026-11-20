"""Capture the REAL replies of the security layer's guard model (the LLM Risk Scorer, role `guard`, GUARD_MODEL) to the red-team requests and to the real user requests, so that a test can replay them offline
(design choices entry 116; design/agentic_security_spec.md section 3).

    python scripts/capture_guard_replies.py                        # says what it would do and stops
    python scripts/capture_guard_replies.py --yes                  # does it
    python scripts/capture_guard_replies.py --yes --limit 3        # the first three texts only (to see what a call costs)

WHAT IT CALLS (so it asks for --yes): the guard model on OpenRouter, once per text (a few cents in all; a call that fails or is not valid JSON is tried once more, as the scorer does). It fetches NOTHING, reads no stored data and calls no other agent.
The texts are the attack requests and the benign requests of `configs/security/red_team_v1.jsonl` (the documents are code-only and never go to a model) and every real user request of `tests/fixtures/session_real/*.json`.
It writes ONLY `tests/fixtures/security/guard_replies.json`: for each text the category, the RAW reply of the model (every try), the labels and the risk the scorer's code took from them, the tokens, and the model's name.
The key is read by `get_chat_model` through `rag.config.setting`; it is never printed or saved.
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

from prometheus.models.chat import model_name  # noqa: E402
from prometheus.rag.config import REPO_ROOT  # noqa: E402
from prometheus.security.policy import load_policy  # noqa: E402
from prometheus.security.scorer import RiskScorer  # noqa: E402

RED_TEAM = REPO_ROOT / "configs" / "security" / "red_team_v1.jsonl"
SESSIONS = REPO_ROOT / "tests" / "fixtures" / "session_real"
OUT = REPO_ROOT / "tests" / "fixtures" / "security" / "guard_replies.json"
WORKERS = 4


def real_requests() -> list[str]:
    """Every real user request of the captures: the `request` of a session capture, the requests of a forms capture, in file order, without repeats."""
    found: list[str] = []
    for path in sorted(SESSIONS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("request"), str):
            found.append(data["request"])
        if isinstance(data.get("requests"), dict):
            found.extend(data["requests"])
    return list(dict.fromkeys(found))


def texts() -> list[dict]:
    items = []
    for line in RED_TEAM.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("door", "request") != "request":
            continue
        items.append({"text": row["text"], "category": row["category"], "kind": "benign" if row["category"] == "benign" else "attack", "needs_llm": bool(row.get("needs_llm"))})
    seen = {item["text"] for item in items}
    items += [{"text": t, "category": "real", "kind": "real", "needs_llm": False} for t in real_requests() if t not in seen]
    return items


class Recording:
    """The real chat model, remembering the raw text of every reply and the tokens it cost (one instance per text: a thread of its own)."""

    def __init__(self, model):
        self.model, self.model_name = model, getattr(model, "model_name", None)
        self.replies: list[str] = []
        self.tokens_in = self.tokens_out = 0
        self.errors: list[str] = []

    def invoke(self, messages):
        try:
            reply = self.model.invoke(messages)
        except Exception as error:
            self.errors.append(type(error).__name__)
            raise
        content = reply.content if isinstance(reply.content, str) else ""
        self.replies.append(content)
        usage = getattr(reply, "usage_metadata", None) or {}
        self.tokens_in += usage.get("input_tokens", 0)
        self.tokens_out += usage.get("output_tokens", 0)
        return AIMessage(content=content)


def capture_one(item: dict) -> dict:
    scorer = RiskScorer()
    recording = Recording(scorer.model)
    scorer._model = recording
    score = scorer.score(item["text"])
    return {**item, "replies": recording.replies, "errors": recording.errors, "risk": score.risk, "parts": score.parts, "reason": score.reason, "failed": score.failed, "attempts": score.attempts,
            "tokens_in": recording.tokens_in, "tokens_out": recording.tokens_out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="really call the guard model (OpenRouter, paid, a few cents)")
    parser.add_argument("--limit", type=int, help="only the first N texts")
    args = parser.parse_args(argv)
    items = texts()[: args.limit] if args.limit else texts()
    policy = load_policy()
    kinds = {k: sum(1 for i in items if i["kind"] == k) for k in ("attack", "benign", "real")}
    print(f"{len(items)} texts ({kinds['attack']} attacks, {kinds['benign']} benign, {kinds['real']} real user requests) -> the guard model, role `guard`, reasoning effort {policy.scorer.reasoning_effort}, max {policy.scorer.max_tokens} tokens a reply")
    if not args.yes:
        print("nothing was called: add --yes to call the model (OpenRouter, paid). It writes only " + str(OUT.relative_to(REPO_ROOT)))
        return 0
    lock, done = threading.Lock(), [0]

    def work(item):
        out = capture_one(item)
        with lock:
            done[0] += 1
            print(f"  [{done[0]}/{len(items)}] {item['kind']:<6} risk {out['risk']:.2f}{' FAILED' if out['failed'] else ''}  {item['text'][:70]!r}", flush=True)
        return out

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(work, items))
    except RuntimeError as error:      # a missing key: the message names the setting, never a value
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    model = model_name("guard")
    tokens_in, tokens_out = sum(r["tokens_in"] for r in results), sum(r["tokens_out"] for r in results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "_note": "REAL replies of the security layer's guard model (the LLM Risk Scorer, role `guard`) to the red-team requests of configs/security/red_team_v1.jsonl and to every real user request of tests/fixtures/session_real, captured with "
                 "scripts/capture_guard_replies.py. `replies` is the raw text of every try (a second try only after a failed call or a reply that was not valid JSON in range); `risk`, `parts`, `reason` and `failed` are what the scorer's code took from them.",
        "model": model, "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reasoning_effort": policy.scorer.reasoning_effort, "max_tokens": policy.scorer.max_tokens,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "items": results}, indent=1, ensure_ascii=True), encoding="utf-8")
    failed = sum(r["failed"] for r in results)
    print(f"saved {OUT.relative_to(REPO_ROOT)}: {len(results)} texts, {failed} failed, {tokens_in} tokens in, {tokens_out} tokens out ({model})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
