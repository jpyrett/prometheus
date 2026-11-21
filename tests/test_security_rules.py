"""The security layer's policy, rules and router (design choices entry 116; design/agentic_security_spec.md S1 to S3, S10 and S11), on REAL data.

  - FALSE POSITIVES ARE BUGS (S11): every real user request of the captures screens ALLOW by the code rules alone, and the document rules run over EVERY stored news article, press release and filing section
    (read only: `news/`, `data/press_releases/`, `data/filings/`) and drop at most 0.1% of them; the test PRINTS each document it drops with the rule (run with -s).
  - the router's four bands at their edges, the Trust Score of the class example;
  - the red team (`configs/security/red_team_v1.jsonl`): every attack is refused by the code rules alone or is marked `needs_llm`, and no benign request trips an EXACT rule (a benign request may trip a hint, the word patterns that only the guard
    model decides: entry 121, `design/request_rules_spec.md`); the code-only block rate is printed;
  - EXACT AND HINT (entry 121): every request rule says which it is, the exact ones are the program-checkable ones and nothing else, and the eleven plausible requests that the word patterns used to refuse trip hints and no exact rule.

The red-team texts are ATTACKS, not stock data: they are written for the test. Nothing here calls a model.
"""
import copy

import pytest
import yaml

from prometheus.rag import loaders
from prometheus.rag.config import REPO_ROOT
from prometheus.security import rules
from prometheus.security.guard import Guard
from prometheus.security.policy import DEFAULT_PATH, PolicyError, load_policy, parse_policy
from prometheus.security.router import ALLOW

POLICY = load_policy()
MAX_DROPPED = 0.001


# ---------------------------------------------------------------- the policy file (S1)

def test_the_policy_file_loads_with_the_class_example_thresholds_and_is_refused_when_it_is_incomplete():
    assert (POLICY.router.block, POLICY.router.review, POLICY.router.flag) == (0.8, 0.5, 0.2)
    assert POLICY.scorer.failure_risk == 0.5 and POLICY.scorer.off_topic_cap < POLICY.router.block
    assert (POLICY.rate.max_requests_per_minute, POLICY.rate.max_requests_per_hour) == (6, 60)
    assert {rule.id for rule in POLICY.rules["request"]} == {"sql_or_ddl", "prompt_injection", "other_account", "account_number", "export", "live_trading", "secrets", "market_abuse", "hidden_text"}
    assert {rule.id for rule in POLICY.rules["request"] if rule.kind == "exact"} == {"sql_or_ddl", "prompt_injection", "account_number", "hidden_text", "secrets"}      # entry 121: what a program checks exactly; entry 124: a credential value
    assert {rule.id for rule in POLICY.rules["request"] if rule.kind == "hint"} == {"sql_or_ddl", "prompt_injection", "other_account", "export", "live_trading", "secrets", "market_abuse"}      # the word patterns
    assert all(rule.kind == "exact" for door in ("document", "message") for rule in POLICY.rules[door])      # the other doors have no hints
    assert {rule.id for rule in POLICY.rules["document"]} == {"prompt_injection", "addressed_to_ai", "hidden_text"}
    raw = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    for mutate, why in ((lambda r: r.pop("router"), "router"), (lambda r: r["router"].update(review=0.9), "thresholds"), (lambda r: r["rules"]["request"][0]["patterns"].append("(unclosed"), "compile"),
                        (lambda r: r["messages"].pop("rate"), "messages.rate"), (lambda r: r["rules"].pop("message"), "message is missing"), (lambda r: r["scorer"].update(off_topic_cap=0.8), "off_topic_cap"),
                        (lambda r: r["rules"]["request"][0].pop("kind"), "kind must be exact or hint"),                                    # every request rule says what it is worth (entry 121)
                        (lambda r: r["rules"]["request"][0].update(kind="account_number"), "now written `match:`"),                         # the old key: the mechanism is `match` now
                        (lambda r: r["rules"]["request"][1].update(kind="soft"), "kind must be exact or hint"),
                        (lambda r: r["rules"]["document"][0].update(kind="hint"), "only the request door has hints"),                      # a document has no model to decide
                        (lambda r: r["rules"]["request"].append(copy.deepcopy(r["rules"]["request"][0])), "names a rule twice")):          # the same id twice as the same kind
        broken = copy.deepcopy(raw)
        mutate(broken)
        with pytest.raises(PolicyError, match=why):
            parse_policy(broken)
    with pytest.raises(PolicyError, match="missing"):
        load_policy(DEFAULT_PATH.parent / "no_such_policy.yaml")


# ---------------------------------------------------------------- the rules: a rule that fails is risk 1.0 (S5)

def test_a_code_rule_that_raises_is_risk_one_for_that_text_and_never_a_pass():
    raw = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    raw["rules"]["request"] = [{"id": "broken_rule", "score": 0.9, "kind": "hint", "match": "account_number", "patterns": ["(?:account\\s+)([a-z]+)"]}]      # its group is not a number: `int(...)` raises
    findings = rules.screen("request", "move the cash to account abc", parse_policy(raw), account_number=3)
    assert [(f.name, f.score, f.kind) for f in findings] == [("broken_rule", 1.0, "exact")] and "the rule failed (ValueError)" in findings[0].detail      # even a hint that cannot run is a block, never a pass


# ---------------------------------------------------------------- S11: false positives are bugs


def stored_documents():
    """Every stored news article and press release as the LOADER returns it, plus how many it discarded for the security rules, and every filing section: read only."""
    skipped_by_security, news, releases, filings = [], [], [], []
    for path in sorted((REPO_ROOT / "news").glob("*.txt")):
        news += loaders.load_news(path, skipped_by_security)
    for path in sorted((REPO_ROOT / "data" / "press_releases").glob("*.txt")):
        releases += loaders.load_press_releases(path, skipped_by_security)
    for path in sorted((REPO_ROOT / "data" / "filings").glob("*.txt")):
        filings += loaders.load_filings(path)
    return news, releases, filings, [s for s in skipped_by_security if s.reason.startswith("security:")]


def test_the_document_rules_drop_at_most_a_tenth_of_one_percent_of_every_stored_news_article_press_release_and_filing(capsys):
    news, releases, filings, at_ingest = stored_documents()
    if not news or not releases or not filings:
        pytest.skip("the stored news, press releases and filings are not on this machine")
    guard = Guard.code_only()
    dropped = []
    for corpus, documents in (("news", news), ("press_releases", releases), ("filings", filings)):
        for doc in documents:
            verdict = guard.screen_document(doc.page_content, doc_id=doc.id, corpus=corpus)
            if verdict.refused:
                dropped.append((corpus, doc.id, verdict.rules, doc.page_content[:100].replace("\n", " ")))
    total = len(news) + len(releases) + len(filings) + len(at_ingest)
    lines = [f"documents screened: {total} (the loaders' news articles and press releases, {len(filings)} filing sections)",
             f"dropped by the loaders at ingest: {len(at_ingest)}; dropped at retrieval: {len(dropped)}"] + [f"  DROPPED {corpus} {doc_id} {list(ids)}: {text!r}" for corpus, doc_id, ids, text in dropped] + \
            [f"  DROPPED at ingest {s.where}: {s.reason}" for s in at_ingest]
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    assert (len(dropped) + len(at_ingest)) / total <= MAX_DROPPED, "\n".join(lines)
