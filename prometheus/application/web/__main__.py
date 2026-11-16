"""python -m prometheus.application.web [--host 127.0.0.1] [--port 8000] [--replay FILE] [--replay-speed 1.0] [--question-timeout 600] [--dev]

Starts the web bridge of the desk and, when `ui/dist` is built, the page at `/`. It is bound to 127.0.0.1 by default: one user, one run at a time. Starting the server costs nothing; pressing Go! in live mode is a PAID run
(OpenRouter and Finnhub) that writes the UI's own account in Postgres. `--replay FILE` plays a recorded session instead: no key, no network, no database.
"""
import argparse
import sys
from pathlib import Path

DEFAULT_REPLAY = "tests/fixtures/session_real/bob_jones_oil.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m prometheus.application.web", description="The web bridge of the Prometheus desk (design/ui_v1_spec.md).")
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen on (default 127.0.0.1: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replay", nargs="?", const=DEFAULT_REPLAY, metavar="FILE", help=f"play a recorded session (a capture made with `session --save`) instead of the live desk: free, no key, no database (default {DEFAULT_REPLAY})")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="how long each stage takes in the replay, 1.0 is about 10 seconds for a whole run, 0 is no pause")
    parser.add_argument("--question-timeout", type=float, default=600.0, help="seconds an open question waits for an answer before it counts as no answer (default 600)")
    parser.add_argument("--dev", action="store_true", help="allow the page served by Vite (localhost:5173) to call the API")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import uvicorn

    from prometheus.application.web.app import create_app
    from prometheus.application.web.replay import Replay, load_capture

    replay = None
    if args.replay:
        try:
            replay = Replay(load_capture(Path(args.replay)), speed=args.replay_speed, question_timeout=args.question_timeout)
        except (ValueError, OSError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
    app = create_app(replay=replay, dev=args.dev, question_timeout=args.question_timeout)
    note = f"REPLAY of {Path(args.replay).name}: no model, no quote, no database" if replay else "LIVE: pressing Go! is a PAID run (OpenRouter, Finnhub) that writes the UI's own account in Postgres"
    from prometheus.application.web import tracing
    print(tracing.startup_line())
    print(f"Prometheus desk on http://{args.host}:{args.port}   [{note}]")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
