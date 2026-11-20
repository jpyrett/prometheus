"""Offline evaluation (design choices entry 117; design/offline_evaluation_spec.md): LangSmith datasets, code evaluators and the Opus risk judge, for the week 6 metrics.

    datasets.py     builds the examples from files that are already in the repo: the red-team set (`configs/security/red_team_v1.jsonl`) and the real session captures (`tests/fixtures/session_real`); syncs them to LangSmith
    evaluators.py   the code evaluators (pure functions over `outputs` / `reference_outputs`), and the local run and the metrics that print them
    judge.py        the Opus risk judge (`openevals.create_llm_as_judge` with the rubric of `prometheus/prompts/risk_judge.py`); the ACTION is code's (`security.router.act`)
    targets.py      the functions `langsmith.evaluate` runs: the guard on a request, and the replay of a recorded session

    python scripts/run_offline_evals.py --dry-run          # free: builds the examples, runs the code evaluators on the recorded outputs, prints the metrics, sends nothing

The examples are real: the attacks and benign requests Jacob's red team wrote with their labels, and what the real sessions did. Nothing here reruns the desk (no model of an agent, no data fetch, no trade).
"""
