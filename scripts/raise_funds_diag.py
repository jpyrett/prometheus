"""LIVE diagnosis of ONE purchase of `claude` (9 COST): the latency of every model call, and what the beam width and the reasoning effort do (design/week4_tot_guide.md section 9m).

    python scripts/raise_funds_diag.py low 3          # reasoning effort low, beam 3 (the defaults)
    python scripts/raise_funds_diag.py low 1          # beam 1: a depth-first dive, nothing kept in reserve
    python scripts/raise_funds_diag.py medium 3 60    # effort medium, a limit of 60 seconds ("0": none)

PAID (a few cents, 30 to 90 seconds; effort high can take minutes: a single call took 24 seconds and 3,454 tokens). Needs Jacob's yes. It prints one line per model call (when it started, how long it took,
tokens in and out) and a summary line: time, depth, levels, calls, the time of the first completed plan, nodes, the plan and its score.
"""
import sys, time, threading
sys.path.insert(0, ".")
from prometheus.agents.account_manager.account_manager import AccountManager
from prometheus.common import portfolio as pf
effort = sys.argv[1]; beam = int(sys.argv[2]); deadline = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "0" else None
m = AccountManager(model_id="openai/gpt-oss-120b", effort=effort, providers=(None if len(sys.argv) > 4 and sys.argv[4] == "default" else "auto"))
real, t0, lock = m.model, time.perf_counter(), threading.Lock()
class Wrap:
    model_name = real.model_name
    def invoke(self, messages):
        t = time.perf_counter(); r = real.invoke(messages); u = r.usage_metadata or {}
        with lock: print(f"  call at {t - t0:5.1f}s took {time.perf_counter() - t:5.1f}s  in {u.get('input_tokens')} out {u.get('output_tokens')}", flush=True)
        return r
m.model = Wrap()
r = m.raise_funds(pf.load_portfolio("configs/portfolios/claude.json"), "COST", 9, lean=True, explore=True, green_first=True, up_to=True, parallel=True, branching=3, beam_width=beam, **({"deadline": deadline} if deadline else {}))
st = r["stats"]; plan = r.get("plan")
print(f"effort {effort} beam {beam} deadline {deadline}: {time.perf_counter() - t0:.0f}s depth {r['limits']['max_depth']} levels {st['levels']} calls {st['llm_calls']} first plan {st.get('first_plan_seconds')} nodes {st['nodes']} "
      + (f"score {plan['score']} {[(x['symbol'], x['shares']) for x in plan['sells']]}" if plan else r['status']) + f" | {r['stop'][:70]}")
