"""python -m prometheus.proving COMMAND  (./test_proving.sh COMMAND): the Proving Ground's command line (design/proving_ground_spec.md section 7.5).

    time-machine --prompt "..." --cash 10000 --from 2026-06-01 --to 2026-11-13 --cadence weekly [--risk moderate] [--neutral] [--strict] [--faithful] [--no-cache] [--save FILE] [--yes]     PAID (models): prints the plan, asks first
    track [--save FILE]                         the Track Record: every recommendation of the saved runs scored against the stored bars                                          free
    live --account N [--save FILE]              the paper account against live quotes and the plan of each recommendation that bought it (never sells)                           Finnhub quotes
    coverage                                    what is stored per month (bars, news, press releases, filings, market news) and the usable range                                  free
    index                                       build the bar index of every symbol with stored news (derived data, safe to delete)                                              free
    show RUN_ID [--json]                        a saved run: its headline and statistics                                                                                         free
    critic-report [--run RUN_ID ...] [--save FILE]   the report card of the Critic's rules: every finding code scored against the outcomes (Critic v2, rule C1)                         free

`langsmith_tracing.configure()` is the first call (entry 114). Nothing here fetches stored data, writes the account book, the real ledgers, the news files or the vector store (P6): results go to `data/proving/` (and `data/index/`).
"""
import argparse
import sys
from pathlib import Path

from prometheus.common import langsmith_tracing
from prometheus.common.money import dumps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m prometheus.proving", description="The Proving Ground: did the desk's recommendations work? (design/proving_ground_spec.md)")
    commands = parser.add_subparsers(dest="command", required=True)
    tm = commands.add_parser("time-machine", help="replay the real desk at past decision days (PAID: the models; no fetch)")
    tm.add_argument("--prompt", required=True, help="the request, in the user's own words")
    tm.add_argument("--cash", required=True, help="the test's starting cash in dollars, for example 10000")
    tm.add_argument("--from", dest="start", required=True, help="the first day, YYYY-MM-DD (a trading day)")
    tm.add_argument("--to", dest="end", required=True, help="the last decision day, YYYY-MM-DD")
    tm.add_argument("--cadence", default=None, help="daily, twice_weekly, weekly or every_two_weeks (default: the config's)")
    tm.add_argument("--risk", choices=("conservative", "moderate", "aggressive"), default="moderate")
    tm.add_argument("--neutral", action="store_true", help="a neutral recommendation counts as approved (ALLOW_NEUTRAL_RECOMMENDATIONS for this run)")
    tm.add_argument("--strict", action="store_true", help="only what the Critic APPROVES may trade")
    tm.add_argument("--faithful", action="store_true", help="ask the models exactly what the live desk asks (the default is the config's fast mode: lower reasoning effort, capped replies, the fastest provider; it takes much longer)")
    tm.add_argument("--no-cache", action="store_true", help="research every decision day again (the cache is neither read nor written)")
    tm.add_argument("--save", metavar="FILE", help="also write the run to this JSON file")
    tm.add_argument("--yes", action="store_true", help="do not ask before the paid run")
    track = commands.add_parser("track", help="the Track Record (free)")
    track.add_argument("--save", metavar="FILE", help="also write the result to this JSON file")
    live = commands.add_parser("live", help="the live monitor (Finnhub quotes; sells nothing)")
    live.add_argument("--account", type=int, required=True, help="the account number")
    live.add_argument("--save", metavar="FILE", help="also write the snapshot to this JSON file")
    commands.add_parser("coverage", help="what is stored per month (free)")
    commands.add_parser("index", help="build the bar index (free)")
    report = commands.add_parser("critic-report", help="the report card of the Critic's rules: each finding code against the outcomes (free)")
    report.add_argument("--run", action="append", metavar="RUN_ID", help="a saved run to score (repeat it for several); the default is every saved Time Machine run and the Track Record")
    report.add_argument("--save", metavar="FILE", help="also write the card to this JSON file")
    show = commands.add_parser("show", help="a saved run (free)")
    show.add_argument("run_id")
    show.add_argument("--json", action="store_true", help="the whole run as JSON")
    return parser


def _write(path: str | None, text: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        print(f"(saved in {path})")


def run_live(args) -> int:
    from prometheus.agents.account_manager.account_manager import AccountManager
    from prometheus.persistence.accounts import open_account_store
    from prometheus.proving import live
    try:
        manager = AccountManager()
        manager.connect(open_account_store(), args.account)
        shot = live.snapshot(manager, args.account)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(live.report(shot))
    _write(args.save, dumps(shot.model_dump(mode="json"), indent=1))
    return 0


def run_track(args) -> int:
    from prometheus.proving import track
    run = track.build()
    print(track.report(run))
    _write(args.save, dumps(run.model_dump(mode="json"), indent=1))
    return 0


def run_show(args) -> int:
    from prometheus.proving.store import ProvingStore
    from prometheus.proving import track
    try:
        run = ProvingStore().load_run(args.run_id)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(dumps(run.model_dump(mode="json"), indent=1) if args.json else track.report(run))
    return 0


def run_critic_report(args) -> int:
    from prometheus.proving import report_card
    from prometheus.proving.store import ProvingStore
    store = ProvingStore()
    try:
        runs = [store.load_run(run_id) for run_id in args.run] if args.run else None
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    card = report_card.build(runs, store=store)
    print(report_card.report(card))
    _write(args.save, dumps(card, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    langsmith_tracing.start()          # entry 114: the first call of every entry point
    if args.command == "live":
        return run_live(args)
    if args.command == "track":
        return run_track(args)
    if args.command == "show":
        return run_show(args)
    if args.command == "critic-report":
        return run_critic_report(args)
    if args.command == "coverage":
        from prometheus.proving import coverage
        print(coverage.report(coverage.compute()))
        return 0
    if args.command == "index":
        from prometheus.agents.data_clerk import bar_index
        return bar_index.main()
    from prometheus.proving import engine
    return engine.main(args)


if __name__ == "__main__":
    sys.exit(main())
