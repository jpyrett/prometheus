"""Offline evaluation of Prometheus (design choices entry 117; design/offline_evaluation_spec.md): two LangSmith datasets, code evaluators and the Opus risk judge, for the week 6 metrics.

    python scripts/run_offline_evals.py                       # the same as --dry-run
    python scripts/run_offline_evals.py --dry-run             # FREE: builds the examples, runs the code evaluators on the recorded outputs, prints the metrics, sends nothing (no model, no network)
    python scripts/run_offline_evals.py --upload              # creates or UPDATES the two datasets in LangSmith (idempotent: it never makes a second copy of an example)
    python scripts/run_offline_evals.py --upload --run all    # and runs the two experiments (PAID: the Opus judge on every request and session, the Qwen guard on the red-team requests; about a dollar or two)
    python scripts/run_offline_evals.py --run sessions --no-judge --limit 3

THE DATASETS: `prometheus-security-redteam` (a line of configs/security/red_team_v1.jsonl each: the attacks and the benign requests with the action a person expects) and `prometheus-sessions` (one real session capture of
tests/fixtures/session_real each). NO AGENT IS RERUN: the red-team target runs the guard (code rules, then the Qwen model of entry 116; documents code only), the sessions target REPLAYS the recorded output, so an experiment costs the judge
and the guard and nothing else. The ACTION of the judge's score is code's (`security.router.act`).
WHAT IT CALLS (so it is not run without a yes from Jacob): `--upload` and `--run` call LangSmith (LANGSMITH_API_KEY); `--run` calls OpenRouter (OPENROUTER_API_KEY): the `judge` role's model (JUDGE_MODEL, default anthropic/claude-opus-5.5)
and the `guard` role's (GUARD_MODEL). It fetches no data, runs no desk and no trade. Neither key is ever printed. Exit 0 when it ran: a low score is a result, not a failure.
"""
import argparse
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prometheus.common import langsmith_tracing  # noqa: E402
from prometheus.evals import datasets  # noqa: E402
from prometheus.evals import evaluators as ev  # noqa: E402
from prometheus.evals import targets  # noqa: E402
from prometheus.models.chat import model_name  # noqa: E402

CONCURRENCY = 4
RUNS = {"redteam": datasets.REDTEAM, "sessions": datasets.SESSIONS, "outcomes": datasets.OUTCOMES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="build the examples and run the code evaluators locally on the recorded outputs; sends nothing (free)")
    parser.add_argument("--upload", action="store_true", help="create or update the two datasets in LangSmith")
    parser.add_argument("--run", choices=("redteam", "sessions", "outcomes", "all"), help="run the experiment(s) with langsmith.evaluate (paid)")
    parser.add_argument("--no-judge", action="store_true", help="leave the Opus judge out (the code evaluators and the guard still run)")
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N examples of each dataset")
    return parser


def examples_of(name: str, limit: int | None) -> list[dict]:
    examples = datasets.build(name)[name]
    return examples[:limit] if limit else examples


def dry_run(names: list[str], limit: int | None) -> None:
    """Free: the code evaluators on the recorded outputs. The red team uses the code rules and the guard model's REAL recorded replies (entry 116); the sessions use the recorded captures."""
    print(f"DRY RUN (nothing is sent; no model is called): {date.today().isoformat()}")
    for name in names:
        examples = examples_of(name, limit)
        if name == datasets.OUTCOMES:
            rows = ev.evaluate_locally(examples, lambda inputs: next(e["outputs"] for e in examples if e["key"] == inputs["example_key"]), ev.OUTCOME_EVALUATORS)
        elif name == datasets.REDTEAM:
            rows = ev.evaluate_locally(examples, targets.recorded_redteam_target(), ev.REDTEAM_EVALUATORS)
            print("(the guard here is the real Guard.with_scorer: its exact rules and hints over the replies of the guard model recorded on 2026-11-20, tests/fixtures/security/guard_replies.json; the judge is not run)")
        else:
            rows = ev.evaluate_locally(examples, targets.sessions_target, ev.SESSION_EVALUATORS)
        print("\n".join(ev.report_lines(name, rows)))
        print()


def client_or_exit():
    from langsmith import Client
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set (put it in .env): nothing was sent", file=sys.stderr)
        raise SystemExit(2)
    return Client()


def upload(client, name: str, limit: int | None) -> None:
    """Create the dataset when it is missing and bring its examples to what the files hold: add the missing keys, update the changed ones, touch nothing else."""
    wanted = examples_of(name, limit)
    created = False
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description=datasets.description(name))
        created = True
    existing = [{"id": str(e.id), "inputs": e.inputs, "outputs": e.outputs, "metadata": e.metadata} for e in client.list_examples(dataset_name=name)]
    plan = datasets.plan_upload(name, wanted, existing)
    if plan["create"]:
        client.create_examples(dataset_name=name, examples=[{"id": datasets.example_id(name, e["key"]), "inputs": e["inputs"], "reference_outputs": e["outputs"], "metadata": e["metadata"]} for e in plan["create"]])
    if plan["update"]:
        client.update_examples(dataset_name=name, updates=[{"id": old_id, "inputs": e["inputs"], "outputs": e["outputs"], "metadata": e["metadata"]} for old_id, e in plan["update"]])
    total = client.read_dataset(dataset_name=name).example_count
    print(f"{name}: {'created' if created else 'exists'}; {len(plan['create'])} example(s) added, {len(plan['update'])} updated, {plan['same']} already as wanted; the dataset now holds {total}"
          + (f"; {len(plan['orphans'])} example(s) are not in the files any more (left in place)" if plan["orphans"] else ""))


def rows_of(results) -> list[dict]:
    """The rows of an experiment in the shape the metrics read: the example, the run's outputs and the flat results of every evaluator."""
    rows = []
    for row in results:
        example, run = row["example"], row["run"]
        flat = [{"key": r.key, "score": r.score, "value": r.value, "comment": r.comment} for r in row["evaluation_results"]["results"]]
        rows.append({"example": {"inputs": example.inputs or {}, "outputs": example.outputs or {}}, "outputs": run.outputs or {}, "results": flat, "error": run.error})
    return rows


def run_experiment(client, name: str, limit: int | None, no_judge: bool, usage) -> list[dict]:
    from langsmith import evaluate
    from prometheus.evals import judge
    if not client.has_dataset(dataset_name=name):
        print(f"{name} does not exist in LangSmith: run with --upload first", file=sys.stderr)
        raise SystemExit(2)
    if name == datasets.OUTCOMES:
        target, evaluators = targets.outcomes_target, list(ev.OUTCOME_EVALUATORS)
    elif name == datasets.REDTEAM:
        target, evaluators = targets.redteam_target(), list(ev.REDTEAM_EVALUATORS)
        if not no_judge:
            evaluators.append(judge.redteam_judge(judge.build_judge(callbacks=[usage])))
    else:
        target, evaluators = targets.sessions_target, list(ev.SESSION_EVALUATORS)
        if not no_judge:
            evaluators.append(judge.sessions_judge(judge.build_judge(callbacks=[usage])))
    data = list(client.list_examples(dataset_name=name, limit=limit)) if limit else name
    prefix = f"{name}-{date.today():%Y%m%d}"
    results = evaluate(target, data=data, evaluators=evaluators, experiment_prefix=prefix, max_concurrency=CONCURRENCY, client=client,
                       description=f"Offline evaluation of {name} (design choices entry 117)" + ("; no judge" if no_judge else f"; judge {model_name('judge')}"),
                       metadata={"dataset": name, "judge": not no_judge, "limit": limit})
    rows = rows_of(results)
    errors = [r for r in rows if r["error"]]
    print(f"experiment {results.experiment_name}: {len(rows)} examples, {len(errors)} target error(s)")
    try:
        print(f"  {results.url}")
    except Exception as error:      # a missing URL never hides the metrics
        print(f"  (no URL: {type(error).__name__})")
    print("\n".join(ev.report_lines(name, rows)))
    if not no_judge:
        expected = sum(1 for r in rows if r["example"]["inputs"].get("door", "request") == "request")
        print(f"  judge results: {len(ev.scores(rows, 'risk_score'))} scored, {len(ev.refused_rows(rows))} refused by the judge's provider, of {expected} examples")
    print()
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = list(RUNS.values()) if args.run in (None, "all") else [RUNS[args.run]]
    if args.dry_run or not (args.upload or args.run):
        dry_run(list(RUNS.values()), args.limit)
        if not args.dry_run:
            print("(no --upload and no --run: this was the dry run)")
        return 0
    langsmith_tracing.start()
    client = client_or_exit()
    if args.upload:
        for name in names:
            upload(client, name, args.limit)
    if args.run:
        from langchain_core.callbacks import UsageMetadataCallbackHandler
        usage = UsageMetadataCallbackHandler()
        for name in names:
            run_experiment(client, name, args.limit, args.no_judge, usage)
        print(f"judge tokens (the models the judge used): {usage.usage_metadata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
