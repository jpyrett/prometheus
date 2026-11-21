"""The Proving Ground: did the desk's recommendations work? (design/proving_ground_spec.md; design choices entry 126.)

One package, three views of one question (the Time Machine's backtest, the Track Record, the Live monitor), and in it the PURE core, which is what this file lists. Plain Python: NO model anywhere in these modules, nothing is
fetched, nothing is written outside `data/proving/`. Only `engine.py` and `live.py` (later steps) reach an agent.

    config.py     `load_config()`: `configs/proving_v1.json`, every threshold of the Proving Ground, a missing or unknown key is an error
    record.py     the pydantic models of the result (`RecOutcome`, `EquityPoint`, `Stats`, `ProvingRun` ...), the closed list of caveats with their one sentence, and the readers of a real recommendation (`plan_of`, `verdict_of`)
    splits.py     the stored split table of a symbol and the adjustment of a position across a split (P10)
    outcomes.py   `score(...)`: one recommendation against the stored bars (the entry by `simulator.fill_backtest`, the exit by `simulator.follow`: reused, never rewritten) -> a `RecOutcome`
    portfolio.py  `SimPortfolio`: the Time Machine's OWN money, in memory (the fills of the Trader, the exits of the plan, the equity of each day; every identity checked on every step)
    stats.py      `compute(...)`: the statistics of a result, each with its `n` and a small-sample mark, and the headline and Critic sentences from a closed set of templates
    store.py      the result as files under `data/proving/` (runs, the research cache, the sessions): atomic writes, never under `data/historical/` or `data/processed/`

MONEY, PRICES AND SHARES ARE `decimal.Decimal` through the doors of `common/money.py` and `common/shares.py` (entry 112, P7); a RETURN, a drawdown, a rate, a mean are MEASUREMENTS and stay `float`, worked out from Decimal values at the
very end and to six places (a `_pct` here is a fraction, 0.0077 is 0.77%, as everywhere in the desk). No size limit, risk rule, liquidity rule or minimum order is added anywhere: the plan decides the exit (P5) and nothing else does.
"""
