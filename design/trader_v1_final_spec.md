# Trader v1: final spec (paper trading and backtesting)

Written 2026-11-13 by Fable 5.2 for Jacob Pyrett, to be coded by Sonnet 5.5. Sources: `design/week5_submission_jpyrett.pdf` (the week 5 architecture), `design/jpyrett_capstone_tradingbot.md` (the week 2 Trader loop), and the code as it stands on `dev` (the Critic, the trade decisions, the Postgres account book).

The build notes go in `design/how_to_code_trader.md` (the same kind of file as `design/how_to_code_critic.md`). Where the build has to differ from this spec, say so there under "Deviations from the spec", with the reason.

---

## 1. What the Trader is

The Trader is the fifth agent of the desk. The Account Manager hands it a work order whose recommendations the Critic has reviewed and the Account Manager has decided on. The Trader executes the trades that are permitted, against a simulated broker, and reports what was filled back to the Account Manager, which updates the user's portfolio.

There is no live trading in this version. There are two settings:

| mode | prices come from | what it answers |
|---|---|---|
| `paper` (default) | the current quote, asked through the Account Manager | "If I placed these trades now, what would my portfolio look like?" |
| `backtest` | the stored daily bars, asked through the Account Manager | "If I had placed these trades when they were approved, how would they have done?" |

`live` is refused in code. No broker SDK is imported and the Trader makes no network call of its own.

What the week 2 design asked for, and where it is here:

| week 2 design | this spec |
|---|---|
| `WORK_ORDER.Trades[Trade_ID]` is Active, Filled or Rejected | `order.trades`, each with `trade_id` and `status` (`active`, `filled`, `rejected`) |
| ReAct: `verify_request` → `check_trade_status` → `AccountManager.notify` | the model's four tools: `verify_request`, `place_order`, `check_trade_status`, `notify_account_manager` |
| "The Trader does not call the Data Clerk" | every price and bar is asked through the Account Manager |
| "reports to Account Manager" so the portfolio and funds can be updated | `AccountManager.notify_trades` |
| guardrails so "the Trader cannot execute any trade without permission" | `verify_trade` inside the tools, section 2 |

### The model

The Trader's model is `openai/gpt-oss-120b` through OpenRouter (a new role `trader`, setting `TRADER_MODEL`), at **temperature 0**.

Temperature 0 makes the model repeatable: the same order gives the same tool calls. It does not make the model truthful. What stops a made-up trade is that the model has no way to express one: its tools take a `trade_id` that code issued, and code owns the symbol, the side, the shares and the price limits. That is the design. Do not describe temperature 0 as the safeguard in any docstring or doc.

The model's job is the part a person on a trading desk does: work through the tickets in order, choose the order type inside the limits it is given, check what happened, report to the Account Manager, and say in two sentences what was done. Code checks all of it.

---

## 2. The rules (each one gets a test that fails when the rule is removed)

- **T1. Permission is code.** No trade without `work_order.verify_trade(order, record_id, now) == []`. It is checked when the ticket is built and again inside `place_order`.
- **T2. The model cannot size, price or pick.** A tool takes a `trade_id` and, for `place_order`, an order type and a limit price that must be inside the limits code computed. No tool argument names a symbol, a side or a number of shares. No tool creates a trade.
- **T3. Once.** A `trade_id` is placed at most once. Running the Trader again on the same order does not trade again: a `filled` or `rejected` trade is final.
- **T4. Paper only.** The mode is `paper` or `backtest`. Any other mode is refused. `PAPER_TRADING` set to anything but true refuses the whole run.
- **T5. The Account Manager is the only door.** The portfolio, the cash, the quotes, the bars and the booking of fills all go through `AccountManager` methods. The Trader's modules import nothing from `prometheus.agents.data_clerk`, `prometheus.agents.critic`, `prometheus.tools.data_clerk`, or `prometheus.agents.advisor` (one exception: `prometheus.agents.advisor.calls`, the shared JSON helper, if you need it). From `prometheus.rag` only `config` and `loaders.check_symbol`.
- **T6. Code finishes the job.** After the model's loop, code places the permitted trades the model left, and notifies the Account Manager if the model did not. Each trade says who placed it (`placed_by`: `model` or `code`). A model that fails, times out or answers nonsense never stops a permitted trade and never causes one that is not permitted.
- **T7. Never raises** for a data, model or account failure. A trade that cannot be executed is `rejected` with a code from a closed list and a sentence. A naive `now` is a programming mistake and raises `ValueError`, as everywhere else.
- **T8. The model's words reach two places only:** a trade's `note` and the run's `summary`. The summary is checked by code and replaced by code's own when it is wrong.
- **T9. No text from the web reaches the Trader's model.** Its prompt and its tool results hold symbols, numbers and words from closed lists, all made by code. No thesis, no news quote, no company name (the Advisor's `company` is a model's field). A hidden instruction in an article cannot reach this agent.
- **T10. Other agents' work is not edited.** `order.recommendations`, `order.reviews` and `order.trade_decisions` are byte for byte the same after the Trader ran.

---

## 3. Where it sits in the graph

```
START -> account_manager -> data_clerk -> advisor -> critic -> account_manager_decision -> trader -> END
                                                                        |                    ^
                                                                        +-> account_manager_funds (with_funds) -+
```

`build_desk(with_trader=True)` needs `with_critic=True` and raises `ValueError` without it. The `trader` node runs only when at least one decision says `yes_auto` or `yes_user`. With `with_funds`, the funds node runs first and its answer (`state["funds"]`) is passed to the Trader as the funding plan.

The node, in `desk.py`:

```python
def trader(state):
    order = state["order"]
    try:
        manager, portfolio = account(state)
        agent = trader_factory(order)
    except (RuntimeError, ValueError, OSError) as error:
        order.errors = [*order.errors, f"the Trader could not start: {error}"]
        return {"order": order}
    spec = state["request"].get("trade") or {}
    return {"order": manager.perform_trades(order, portfolio, agent, clock(), funding=state.get("funds"), approver=approver, book=bool(spec.get("book")))}
```

---

## 4. The steps of `Trader.execute_trades`

```python
class Trader:
    def __init__(self, model=None, model_id=None, config_path=None, mode=None, effort="auto", providers="auto", use_model=True, sweep=True, ledger=None): ...
    def execute_trades(self, order, account, portfolio=None, now=None, funding=None, approver=None, book=False) -> WorkOrder: ...
```

`account` is the `AccountManager`. `portfolio` is what the desk loaded (the JSON file); the Trader asks `account.trading_portfolio(portfolio)` for the one to use. `funding` is a RaiseFunds result or None. `approver` is the same callable the decisions use. `book` asks for the fills to be saved in the Postgres account book.

The steps live in `prometheus/agents/trader/execute.py`, as the Critic's live in `review.py`:

1. **Preconditions.** `PAPER_TRADING` is true; the mode is `paper` or `backtest`; `book` is false in backtest mode (a replay never writes the account book); the order's status is `decided`, `reviewed` or `traded`; it has trade decisions. A failed precondition is one message on `order.errors` and a `trader_runs` entry, and nothing else changes.
2. **Account.** `account.trading_portfolio(portfolio)`. The portfolio's user must be the profile's user (the same rule as `decide_trades`). A portfolio that cannot be had (none given and no account, a database error) is a failed precondition. When the order already carries trades from an earlier run that were filled and not booked, they are applied to this portfolio first, in order, with `apply_fill`: those fills live only on the order, and without this a second run would start from a portfolio that never had them. Booked fills are already in the account.
3. **Tickets** (section 5). One per decision that says `yes_auto` or `yes_user` and has no trade on the order yet. Trades already on the order are loaded: `filled` and `rejected` are final, `active` is looked at again. Tickets beyond `limits.max_trades` are rejected `not_attempted`, and the reason gives the limit.
4. **Prices** (section 6). One request for every symbol the tickets need. Paper: `account.request_prices`. Backtest: `account.request_bars`. An `active` trade from an earlier run is judged again here, by code, with the order it already has: `verify_trade` first (a permission that ran out makes it `rejected`), then the fill rules against the new prices. The model cannot place it again.
5. **Funding consent** (section 7). A purchase the cash does not cover, with a valid funding plan, needs the user's yes for the sells.
6. **The model's loop** (section 8), when `use_model` is true, the model could start, and at least one ticket is `new`.
7. **The sweep** (section 9), when `sweep` is true: code places what is still `new` and notifies the Account Manager if that has not happened.
8. **The output guard** `work_order.verify_trades(order)` (section 11), then the Account Manager is notified. With a finding, nothing is booked.
9. **Outcomes** (section 10), backtest mode only: each filled purchase is followed through the later bars.
10. **Records.** `order.trades`, one more entry on `order.trader_runs`, the trades ledger (paper mode only), the status.

**Status.** The order becomes `traded` when it was `decided` and every permitted decision has a trade that is `filled` or `rejected`. With a decision still `pending_user` or a trade still `active`, the status stays what it was, and a later call carries on from there. An order that carries trades is `schema_version` 4.

---

## 5. Tickets (`tickets.py`, code only)

A ticket is a `Trade` with status `new`. Built from the decision, the review and the Advisor's trade plan (`recommendation["trade"]`), read as plain dicts.

| field | from |
|---|---|
| `trade_id` | `T1`, `T2`, ... in the order of the decisions, continuing after any trades already on the order. A funding sell is `T1-F1`, `T1-F2` |
| `kind` | `recommendation` or `funding` |
| `side` | `sell` when the decision's action is `reduce`, else `buy`. There is no short sale |
| `shares_approved` | the decision's `shares` |
| `shares` | a sell: `min(shares_approved, portfolio.sellable_shares(symbol, today))`, the T+1 rule. A buy: `shares_approved`, cut for cash when it is placed |
| `ref_price` | the review's `timing.deal_price`, else the plan's `entry_mid` |
| limits, a buy | `high = min(entry_zone[1], ref_price * (1 + max_chase_pct))`, `low = max(entry_zone[0], ref_price * (1 - limit_band_pct))`. Without an entry zone only the percentages apply |
| limits, a sell | `low = ref_price * (1 - max_chase_pct)`, `high = ref_price * (1 + limit_band_pct)` |
| default order | the Advisor's own plan: `limit` at the plan's `limit_price` moved inside the limits; `market` when the plan has no limit price |

`today` is `eastern_date(now)`.

A ticket that cannot be traded is built as `rejected` at once, and the model never sees it as placeable:

| code | when |
|---|---|
| `not_permitted` | `verify_trade` gave reasons (they go in `reason`) |
| `approval_expired` | the only reason is that the approval ran out |
| `not_held`, `not_sellable_today` | a sell of a stock the portfolio does not hold, or whose shares were all bought today |
| `price_unavailable` | no reference price, or step 4 gave no quote or bar for the symbol |

The closed list of codes (in `record.py`; a trade cannot be built `rejected` with another): `not_permitted`, `approval_expired`, `price_unavailable`, `price_outside_limits`, `limit_not_reached`, `insufficient_cash`, `not_held`, `not_sellable_today`, `funding_failed`, `not_attempted`, `account_refused`, `internal_error`.

**What "market" means here.** A market order fills at the market price only when that price is inside the limits. Above `high` (a buy) or below `low` (a sell) it is rejected `price_outside_limits`: the price has left what the Critic judged, and a new review is needed. So no order of any type ever fills outside the limits.

---

## 6. Prices, the paper broker and the simulator

`simulator.py` holds pure functions: no I/O, no clock, no model. `broker.py` holds `PaperBroker`, which keeps the running portfolio of one run and applies the rules below to one price snapshot.

**One snapshot per run.** An order is judged once, against the prices fetched in step 4. `check_trade_status` reads the status; it does not fetch again. A later run takes a new snapshot.

A bar is `{"date": "YYYY-MM-DD", "open", "high", "low", "close", "volume"}`.

### Fill rules for a buy (a sell is the mirror image)

`slip = costs.slippage_bps / 10000`. Fill prices are rounded to 4 places. Every amount (a fill's `amount`, the cash a buy needs, the proceeds of a sale) is `portfolio.trade_amount(shares, price)`, which rounds half up as the account book does; Python's own `round` goes to the even cent and would put the broker a cent away from the book.

**One trade date per run.** Paper: `eastern_date(now)`. Backtest: the run's entry date, defined below. It is the `today` of `apply_fill`, the date of every fill, and the `traded_on` given to `notify_trades`.

**Paper**, quote price `q`:

| order | result |
|---|---|
| market | `p = q * (1 + slip)`. `p > high`: rejected `price_outside_limits`. Else filled at `p` |
| limit `L` | `q <= L`: filled at `min(L, q * (1 + slip))`. Else `active` (a resting order: the price is above the limit) |

**Backtest**, entry bar `b`:

The run has one entry date, found in two steps:

1. from the calendar alone (`market_calendar`): the first trading day whose regular open (09:30 Eastern) is strictly after `now`. A decision at 10:30 on Thursday gives Friday; a decision at 08:00 on Thursday gives Thursday. This is the rule that keeps the future out of the decision;
2. from the bars: the earliest bar date on or after that day, among the bars of every symbol that was asked for (the benchmark included). So a day whose file is missing moves the whole run to the next stored day.

No such bar: the future is not stored yet, and every ticket is `active` (waiting for the next session's bar). A ticket whose symbol has no bar on the entry date, while other symbols do, is rejected `price_unavailable`. The entry bar `b` of a ticket is its symbol's bar on the entry date.

| order | result |
|---|---|
| market | `p = b.open * (1 + slip)`. `p > high`: rejected `price_outside_limits`. Else filled at `p` on `b.date` |
| limit `L` | `b.open <= L`: filled at `min(L, b.open * (1 + slip))`. Else `b.low <= L`: filled at `L`. Else rejected `limit_not_reached` (a day order, and the day is over) |
| no entry bar stored | `active`: waiting for the next session's bar |

**Cash, when a buy fills.** `shares = min(shares, floor((cash - commission) / fill_price))`. Zero shares: rejected `insufficient_cash`. Fewer than approved: filled, and `reason` says the size was cut for cash. This is `orders.cash_policy: "reduce"`. Buying fewer shares than approved is inside the permission; buying more never is.

**Order of placing.** Every `reduce` sell is placed before any buy. The `place_order` tool refuses a buy while a sell ticket is still `new` and names it.

### What the snapshot asks for

Paper: `account.request_prices([symbols of the tickets and of the funding sells])`. A symbol with no quote: its ticket is rejected `price_unavailable`.

Backtest: `account.request_bars([those symbols, plus backtest.benchmark], start=eastern_date(now), end=eastern_date(now) + backtest.max_follow_days)`. The real clock is never read: the store returns the bars it has.

---

## 7. Funding (a purchase the cash does not cover)

Jacob's sample portfolio holds $51.23 of cash, so most approved purchases carry the decision's note "needs RaiseFunds". With `--raise-funds` the desk's funds node has already run the Tree of Thought for the first permitted purchase, and its answer arrives as `funding`.

The funding plan is used when all of this holds (`tickets.funding_legs`, code):

- `funding["status"] == "plan"`, with `funding["plan"]["sells"]` not empty;
- `funding["target"]` is the symbol of a `new` buy ticket;
- every sell names a holding, is for at least 1 share, at most `portfolio.sellable_shares(symbol, today)`, and is not the target itself.

`funding["status"] == "funded"` means the cash already covers it: no sells. Anything else: no sells, and the run notes why.

**Consent.** A sale is never made automatically (the existing rule in `decisions.py`: "a sale is never traded automatically"). So while `funding.require_user_approval` is true, in Ask mode and in Auto mode, the Trader calls `approver(card)` once per funded purchase, with:

```python
{"text": "<the sells, one per line: symbol, shares, about $ proceeds, gain>", "symbol": "PG", "shares": 44, "est_cost": 7352.40,
 "question": "Sell these to pay for 44 shares of PG (about $7,352)? [y/N] ", "kind": "funding"}
```

`terminal_approver` gains one line: it asks `card.get("question")` when the card has one, else its own question as today. True attaches the sells. False, None or no approver: no sells are attached and the run says "the funding sells were not approved".

**Execution.** The funding sells belong to the purchase: the model never places them. Placing a funded buy does this, in code, in one go:

1. check the buy itself first: permission, and that its price is inside the limits and the order would fill now. If the buy would not fill (a resting limit, a rejected price), nothing is sold;
2. work out what the sells would bring (each at its snapshot price less slippage; a sell with no price counts as nothing). If the cash plus those proceeds does not pay for even one share of the buy, nothing is sold: the buy is rejected `insufficient_cash` and each funding sell `funding_failed`;
3. fill each funding sell (a sell that cannot fill is `rejected` `funding_failed` and the others go on);
4. fill the buy with the cash now there (the cash rule of section 6 applies).

Step 2 is what stops the worst case, holdings sold and nothing bought. The output guard backs it up: a filled funding sell whose purchase is not filled is a finding (section 11).

The sells are trades of their own on the order (`kind: "funding"`, `funds_for: "T1"`). A funding sell carries the `record_id` and the `review_id` of the purchase it pays for.

Without a funding plan and without the cash, the buy is cut or rejected `insufficient_cash`, and the reason says that `--raise-funds` would plan the sells.

Only the first permitted purchase can be funded, because the funds node plans one purchase. That is the desk as it is today; say so in the notes.

---

## 8. The model's loop (`trader.py`, `prometheus/tools/trader.py`, `prometheus/prompts/trader.py`)

A LangGraph tool loop built like the Data Clerk's (`data_clerk_agent.py`, `_graph`): an `agent` node, a `ToolNode` whose errors become short messages, a budget around every tool, and a last turn without tools.

```
agent --tool calls--> tools --> agent ... --no tool calls, or the last turn--> end
```

**Write the `tools` node yourself; do not use the prebuilt `ToolNode`.** The prebuilt one runs the calls of one turn at the same time, and two buys placed in one turn would then race for the same cash, so the same order could give two different results. This node runs the calls one after the other, in the order the model wrote them, and turns each result or error into a `ToolMessage`. That keeps a run repeatable, which is the point of temperature 0.

### The tools (`TRADER_TOOLS`, the allow-list of four)

Docstrings are written for the model, as in `prometheus/tools/data_clerk.py`.

**`verify_request()`** returns the tickets of this work order, and nothing a model wrote:

```json
{"mode": "paper", "cash": 51.23,
 "trades": [{"trade_id": "T1", "symbol": "PG", "side": "buy", "shares": 44, "status": "new",
             "market_price": 167.10, "default_order": {"order_type": "limit", "limit_price": 167.10},
             "limit_range": [165.46, 168.74], "funded_by": [{"symbol": "MSFT", "shares": 5}]},
            {"trade_id": "T2", "symbol": "XYZ", "side": "buy", "shares": 10, "status": "rejected", "reject_code": "approval_expired"}]}
```

**`place_order(trade_id, order_type="", limit_price=0.0, note="")`** places one ticket. Empty `order_type` means the default order. Refusals the model can fix (message "Invalid arguments: ..."):

- an unknown `trade_id`; a trade that is not `new` ("T1 was already placed": rule T3);
- an `order_type` that is not `market` or `limit`; `market` when `orders.allow_market` is false;
- a `limit_price` outside the limit range (the message gives the range);
- a buy while a sell ticket is still `new`;
- any call after the Account Manager was notified.

Before placing, it runs `verify_trade` again with the run's `now` (rule T1). It returns `{trade_id, status, shares, fill_price, amount, reject_code}`. `note` is kept on the trade, cut to `limits.max_note_words`.

**No tool result carries a trade's `reason`** (decided after phase 2; rule T9). The sentence of a rejected trade is built from `verify_trade`, which repeats two values of the order (`will_trade`, `valid_until_utc`); on an order read from a file those could be any text. The model reads the `reject_code`, a word from the closed list. The sentence stays on the record and in the report. A refusal message (a call the model can fix) is still a sentence: it is written by code from the trade id, the limits and the model's own argument cut short.

**`check_trade_status(trade_id="")`** returns the status of one trade, or of all when empty.

**`notify_account_manager()`** reports the filled trades to the Account Manager (step 8). Refused while a ticket is still `new` (the message names it). A second call returns the first call's answer. It returns `{cash_before, cash_after, filled, rejected, active, booked}`.

### The budget

`place_order`: once per `trade_id`. `verify_request`: 2. `check_trade_status`: the number of tickets plus 2. `notify_account_manager`: 2. Turns: `limits.base_turns` plus the number of `new` tickets, at most `limits.max_turns`. A call over a limit gets "Limit reached: ..." and is counted in the run.

### The prompt

The system prompt says: you are the Trader of a trading desk; the Account Manager gave you a work order whose trades were approved; you act only through your tools; you cannot change a symbol, a side or a number of shares; call `verify_request` first, place each `new` trade once (sells before buys), check the status, call `notify_account_manager`, then answer. The user message is one line made by code: the work order's id, the mode, and how many trades are to be placed.

The model's last message must be JSON only:

```json
{"summary": "at most 60 words", "trades": [{"trade_id": "T1", "status": "filled"}]}
```

### The model

Built on first use, like the Critic's: a model that cannot start (a missing key) is a note on the run and the sweep still trades.

```python
get_chat_model("trader", model=model_id, temperature=0.0, timeout=config.limits.model_timeout_seconds, extra_body=extra)
```

`effort` and `providers` follow the Account Manager's `"auto"` rule: for gpt-oss, reasoning effort `low` and providers Cerebras then Groq. Add `"require_parameters": True` to the provider block, so OpenRouter only uses a host that supports tool calls. `TEMPERATURE = 0.0` is a module constant and a test asserts it reaches `get_chat_model`.

### Checking the summary (rule T8)

The model's `summary` is kept only when: the JSON parses; every `trade_id` it lists exists and its status is the real one; the text is at most `limits.max_summary_words`; and every number in the text is a number of the trades (a share count, a price, an amount, a cash balance, a count of trades), compared at the number of decimals written. Otherwise the summary is code's own sentence, `summary_by` is `code`, and the run's `dropped` says what was wrong.

---

## 9. The sweep

After the loop, or in place of it:

1. each `reduce` sell still `new`, then each buy still `new`, is placed with its default order, `placed_by: "code"`;
2. the Account Manager is notified if it was not.

`Trader(sweep=False)` turns it off: what the model left is rejected `not_attempted`. `Trader(use_model=False)` runs no model at all, so the sweep does everything. That is the free, repeatable way to run a backtest.

---

## 10. Outcomes: following a trade (backtest mode)

`simulator.follow(...)`, a pure function. It follows each filled purchase of a recommendation through the bars after its entry, using the Advisor's own plan: `stop_price`, `target_price` (None for a capital-return plan) and `time_stop_date`.

On the entry bar only the stop can fire: the order of prices inside one day is not known, so the bad case is assumed. On each later bar, the first rule that holds:

| check | exit |
|---|---|
| `open <= stop` | at `open * (1 - slip)`, reason `stop` (it gapped through) |
| `low <= stop` | at `stop * (1 - slip)`, reason `stop` |
| a target, and `open >= target` | at `open`, reason `target` |
| a target, and `high >= target` | at `target`, reason `target` |
| `date >= time_stop_date` | at `close * (1 - slip)`, reason `time_stop` |

The stop is checked before the target, so a day that touches both counts as a stop. With no exit by the last stored bar the trade is `open` and marked at the last close.

```
Outcome: status (open | closed), exit_reason, exit_date, exit_price, sessions_held, realized_gain, return_pct,
         last_date, last_close, unrealized_gain, benchmark_return_pct, excess_return_pct
```

The benchmark is `backtest.benchmark` (SPY) over the same dates: entry bar open to exit date close, or to the last close. `None` when its bars are not stored.

Outcomes are a simulation. They are never booked in the account book, and paper mode has none (the position is open and the future is not known).

---

## 11. The work order (`prometheus/common/work_order.py`)

- Status `traded` after `decided`, in `STATUSES`, in the `Literal`, and in `HAS_DATA`.
- `trades: list[dict]` and `trader_runs: list[dict]`, both empty by default. `new_work_order` sets them. `schema_version` 4 once trades are written.
- `TRADE_STATUSES = ("new", "active", "filled", "rejected")`. On a finished order a trade is never `new`.
- A new action `fetch_bars` (in `ACTION_TYPES` and the `Literal`), a request field `bars: dict | None` (`{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`), and `DataResult.bars: dict[str, list[dict]]` (per symbol, oldest first). `new_work_order(..., bars=None)`.
- `verify_bar_request(order)`: the action is `fetch_bars`; the subject is 1 to 30 symbols; `bars` has exactly `start` and `end`, both ISO dates, `start <= end`, at most 400 days apart; the order has no data yet. `verify_request` (the search check) still refuses a `fetch_bars` order.
- `trade_of(order, trade_id)`, and `trades_of(order, record_id)`.
- **`verify_trades(order) -> list[str]`**, the output guard. Empty when the trades agree with the decisions. A reason for each of: a `filled` recommendation trade whose `record_id` has no decision that says `yes_auto` or `yes_user`; filled shares above the decision's shares; two filled trades for one `record_id`; a `filled` trade with no fill, or a fill with a price or share count that is not positive; a funding trade whose `funds_for` names no buy on the order; a funding trade that is not a sell; a `filled` funding trade whose purchase is not `filled` (added after phase 1: phase 2 adds this check and its test); a status outside `TRADE_STATUSES`; a `rejected` trade with no code. Phase 1 also checks a missing or repeated `trade_id` and, for a filled recommendation trade, that the symbol and the side are the decision's.
- Update the docstrings that say "The Trader is not built" (`verify_trade`, the module text, `AccountManager.record_trade`).
- Two functions that read the status do not know `traded` yet (phase 3): `decisions.decide_trades` must return a `traded` order unchanged, as it does a `decided` one, and the Critic's `review_order` must return it without a message, as it does a `reviewed` or `decided` one.

---

## 12. The Data Clerk's part: `request_bars`

The Trader does not call it. The Account Manager does.

- `DataClerkTools.daily_bars(symbols, start, end)` in `prometheus/tools/data_clerk.py`: reads `BarStore` (the same root rule as `metrics_for`), returns `{"bars": {symbol: [bar, ...]}, "gaps": [...]}`, oldest first, dates from `start` to `end` inclusive. Load only as many day files as the range needs. It is **not** in `DATA_CLERK_TOOLS` (the 13-tool pin holds): the model has no use for it.
- `DataClerk.request_bars(order)`: no model call (a test hands it a model that raises if invoked). It checks `verify_bar_request`, calls `daily_bars`, and puts the answer on `order.data.bars`. A symbol with no bar in the range is an empty list and a note: a result, not a failure (the future may not be stored yet). The order is `data_failed` only when it is refused or the store could not be read.

Nothing here fetches. It reads `data/historical` and never writes there.

---

## 13. The Account Manager's part (`account_manager.py`)

```python
def connect(self, store, account_id) -> None
    """Attach the Postgres account book after construction (the same checks as __init__), and forget a profile read from a file."""

def trading_portfolio(self, portfolio: Portfolio | None = None) -> Portfolio
    """The portfolio the Trader works from: the saved account when connected (list_portfolio), else the one given. ValueError when there is neither."""

def request_bars(self, symbols: list[str], start: date, end: date) -> WorkOrder
    """Ask the Data Clerk for stored daily bars: a `fetch_bars` order, answered by DataClerk.request_bars. Beside request_prices."""

def perform_trades(self, order, portfolio, trader, now=None, funding=None, approver=None, book=False) -> WorkOrder
    """The design's Perform Trade: hand the decided order to the Trader and return it with its trades."""
    return trader.execute_trades(order, self, portfolio, now, funding=funding, approver=approver, book=book)

def notify_trades(self, portfolio: Portfolio, trades: list[dict], traded_on: date, book: bool = False) -> tuple[Portfolio, list[str]]
    """The design's AccountManager.notify. Apply each filled trade to the portfolio, in order (`apply_fill`), and return the NEW portfolio and the problems.
    With `book` true and a connected account, each fill is also booked with `record_trade` (source 'bot', the recommendation's record_id). A fill that cannot be
    applied or booked is named in the problems and the others still go through. `book` true without a connected account is one problem and nothing is booked."""
```

`prometheus/common/portfolio.py` gains the pure function behind it:

```python
def apply_fill(portfolio, symbol, side, shares, price, today, name=None) -> Portfolio
```

A buy: the cost is `round(shares * price, 2)`, the cash must cover it, and the position follows the rules `after_purchase` already has (average cost basis, `acquired`, `recent_shares` for the T+1 lock). Reuse that code; do not write the lock rules twice. A sell: the shares must be held, the proceeds are added to the cash, `recent_shares` is kept inside the position, and a position sold out is removed. It returns a new portfolio and raises `ValueError` with the reason.

The broker keeps its running portfolio with the same `apply_fill`. At step 8 the portfolio the Account Manager returns must equal the broker's; a difference is an error on the order.

---

## 14. Records (`prometheus/agents/trader/record.py`)

Pydantic, `extra="forbid"`, as the Advisor's and the Critic's.

```
Trade:  trade_id, kind, record_id, review_id, funds_for, symbol, side, shares_approved, shares, order_type, limit_price,
        ref_price, limit_low, limit_high, status, reject_code, reason, placed_by, note, placed_utc, mode,
        fill: {price, shares, amount, commission, date, price_source (quote | bar_open | bar_limit), slippage_bps} | None,
        booked: bool, outcome: Outcome | None

Run (an entry of order.trader_runs):
        model, temperature, mode, now_utc, use_model, turns, tokens_in, tokens_out, tool_calls [{name, args, outcome}],
        placed_by_model, placed_by_code, summary, summary_by, portfolio_before, portfolio_after {cash, positions},
        price_order (the quote or bars order's request_id), funding {status, approved, note}, booked, config_version, dropped [...], errors [...]

MODEL_FIELDS = ("trades[*].note", "trader_runs[*].summary")
```

A validator refuses a `rejected` trade whose code is not in the closed list. A second one (decided after phase 2) refuses a `symbol` that is not a ticker (`check_symbol`): a decision whose symbol is not one gets no ticket, and the run says so in its errors. So the one piece of the order the model is shown as text is always a ticker.

---

## 15. The settings (`configs/trader_v1.json`, read by `prometheus/agents/trader/config.py`)

Loaded and checked once, `extra="forbid"`, frozen, like `critic/config.py`. No threshold lives in the code.

```json
{
  "config_version": "1",
  "_note": "Every threshold of the Trader (design/trader_v1_final_spec.md). Read once; a missing or unknown key is an error. The model never chooses one.",
  "mode": "paper",
  "limits": {"max_trades": 20, "base_turns": 5, "max_turns": 12, "model_timeout_seconds": 60, "max_note_words": 20, "max_summary_words": 60},
  "orders": {"max_chase_pct": 0.01, "limit_band_pct": 0.03, "allow_market": true, "cash_policy": "reduce"},
  "costs": {"slippage_bps": 5, "commission_per_order": 0.0},
  "funding": {"require_user_approval": true},
  "backtest": {"fill_at": "next_open", "max_follow_days": 90, "benchmark": "SPY"},
  "ledger": {"path": "data/ledger/trades.jsonl"}
}
```

`mode` must be `paper` or `backtest`. `cash_policy` must be `reduce`. `fill_at` must be `next_open`. `commission_per_order` must be 0 (decided after phase 2: `apply_fill` and the account book charge no commission, so another value would overstate the cash). The keys exist so that a later version can add a value; this version refuses any other.

The price request of step 4 is made in groups of at most 30 symbols, the limit of a quote order and of a bars order, and the answers are joined (decided after phase 2).

`PAPER_TRADING` is read with `prometheus.rag.config.setting("PAPER_TRADING", "true")` when a run starts; true, yes, on or 1 pass. `tests/conftest.py` gets an autouse fixture that sets it to `true`, so no test inherits Jacob's `.env` (the same reason as `market_hours_count`).

`prometheus/models/chat.py`: `ROLES["trader"] = ("TRADER_MODEL", "openai/gpt-oss-120b")`. Add `TRADER_MODEL=openai/gpt-oss-120b` with a comment to `.env.example` (never `.env`). `tests/test_chat_model.py` pins the roles and uses "trader" as its example of an unknown role: update both.

**The trades ledger.** `data/ledger/trades.jsonl`, append-only, through the existing `Ledger` class: one line per trade that became `filled` or `rejected`, paper mode only. A backtest is a replay and writes none. The autouse guard in `tests/conftest.py` already stops any test from writing under `data/ledger/`; tests pass their own ledger.

---

## 16. The report and the commands

`prometheus/agents/trader/report.py`: `trade_section(order) -> list[str]`, plain text from the records, no model. It shows: a header (the model, temperature 0, the mode, where the prices came from); one line per trade (id, symbol, side, shares of approved, the order, FILLED at a price and amount, or ACTIVE, or REJECTED with the code and the sentence; who placed it); the funding sells under their purchase; the account before and after (cash, and each position that changed), and whether it was kept in memory or booked; in backtest mode one line per outcome with the gain, the return, the sessions held and the benchmark; the summary and whose it is.

**The desk** (`python -m prometheus.langgraph.desk`):

| flag | meaning |
|---|---|
| `--trade [paper\|backtest]` | add the Trader (default `paper`). Needs `--critic`; the parser says so otherwise |
| `--no-trader-model` | no model: the sweep places the default orders |
| `--account NAME_OR_NUMBER` | use a saved account from Postgres for the profile and the portfolio, instead of the JSON files |
| `--book` | save the fills in that account. Needs `--account` and paper mode |

`build_desk(..., with_trader=False, trader_factory=None, store_factory=None)`. The request carries `trade: {"mode", "book"}` and `account: {"account": name or number}`. In `account(state)`, when `account` is given: open the store (`store_factory`, default `open_account_store`), find the id (`account_id_for_seed`, or the number), `manager.connect(...)` first and only then set a profile that was given (`connect` forgets a profile read from the default file, so the account's own row is used when none was given), and the portfolio is `manager.list_portfolio()`. A database error becomes a `RuntimeError` that names the kind of failure and no connection detail, so the node reports it.

`report()` prints the TRADER section after the Critic's, and `trader` joins the timing line. An order that ended `traded` is a finished path. When `--trade` is given, the exit code is 0 for an order that is `traded`, `decided` or `reviewed`.

**Replay a saved order** (`python -m prometheus.agents.trader.replay`). The command is wiring, so it lives in its own module `replay.py`: it builds the `AccountManager` with `clerk=DataClerk(model=object())` (the bars and the quotes are code: that Data Clerk's own model is never called, and no key is needed for it) and hands it to the Trader. `replay.py` is the one Trader module that may import the Data Clerk; rule T5 and its test cover all the others.

```
--order FILE          a saved work order: the desk's saved JSON (it has an "order" key) or a bare order
--mode paper|backtest (default backtest)
--portfolio FILE | --cash N      --profile FILE
--now "YYYY-MM-DD HH:MM"         US Eastern; default: the latest decided_utc of the permitted decisions (the moment the permission was given)
--no-model            the sweep only
--yes-funding         answer yes to the funding question (a replay has nobody to ask)
--json                --save FILE
```

`--mode backtest --no-model` calls no model and no paid API: it reads the stored bars and prints. It is the one Trader command that is free. Everything else here costs: a model run calls OpenRouter, and paper mode calls Finnhub for quotes.

**The runner** (`prometheus/application/recommendations.py` and `./test_recommendations.sh`): `run(..., trader=None, funding=None, book=False)` calls `manager.perform_trades` after the decisions, the report gains the TRADER section, and the flags `--trade [paper|backtest]` and `--no-trader-model` are passed through by the script. `--no-ledger` gives the Trader a throwaway ledger too. This is the last piece: if something has to be left out, leave this out and say so in the notes.

---

## 17. Files

```
prometheus/agents/trader/trader.py      class Trader, the model, the loop
prometheus/agents/trader/execute.py     the ten steps
prometheus/agents/trader/replay.py      the replay command (wiring: the only module here that imports the Data Clerk)
prometheus/agents/trader/tickets.py     decisions -> tickets, the limits, the funding sells
prometheus/agents/trader/broker.py      PaperBroker: the running portfolio, placing against the snapshot
prometheus/agents/trader/simulator.py   pure fill rules and `follow`
prometheus/agents/trader/record.py      Trade, Fill, Outcome, Run, the closed lists
prometheus/agents/trader/config.py      configs/trader_v1.json
prometheus/agents/trader/report.py      the TRADER section
prometheus/tools/trader.py              the four tools, the budget, the tool errors
prometheus/prompts/trader.py            the system prompt, the last-turn text
configs/trader_v1.json
```

No `__init__.py` in `prometheus/agents/trader/`: the agent folders are namespace packages.

Changed: `common/work_order.py`, `common/portfolio.py`, `agents/account_manager/account_manager.py`, `agents/account_manager/decisions.py` (the `question` line), `agents/data_clerk/data_clerk_agent.py`, `tools/data_clerk.py`, `models/chat.py`, `langgraph/desk.py`, `application/recommendations.py`, `test_recommendations.sh`, `.env.example`, `tests/conftest.py`.

---

## 18. Tests (fakes only; nothing here may call a paid API or a real model)

`tests/trader_helpers.py`:

- `decided(name="PG_open", answer=True, mode=None)`: the captured order of `critic_helpers.captured`, reviewed by `make_critic` and decided by `decisions.decide_trades`. With `PG_open` and a yes that is one permitted purchase: 44 shares of PG, reference price 167.10, entry zone 165.4559 to 168.7441, plan limit 167.10, stop 157.2351, target 168.3783, time stop 2026-12-31, approval valid until 2026-11-05T16:30:00Z, `NOW` 10:30 Eastern that day, cash 51.23.
- `MarketClerk(quotes=..., bars=...)`: answers `request_quotes` and `request_bars`, and records what it was asked.
- `manager(...)`: a real `AccountManager(model=object(), clerk=MarketClerk(...), profile=...)`.
- `ScriptedModel`, `call(name, **args)`, `say(text)`: copy the pattern of `tests/test_data_clerk_agent.py`.

Test files, and what each proves:

| file | proves |
|---|---|
| `test_trader_simulator.py` | every fill rule and every `follow` rule, each at its boundary (a price equal to the limit, a stop touched exactly, stop and target on one day, a gap through the stop, the entry bar, no bars) |
| `test_trader_tickets.py` | sides and sizes, the T+1 cut, the limits with and without an entry zone, each rejection code, the funding plan accepted and each way it is refused |
| `test_trader_broker.py` | paper and backtest placing, the entry-bar rule on both sides of 09:30, slippage, the cash cut, a funded buy that would not fill sells nothing |
| `test_trader_tools.py` | each refusal of each tool, the budget, sells before buys, nothing after the notify |
| `test_trader.py` | the whole step. The model places everything; the model places nothing and the sweep does; the model raises; the model cannot start; `use_model=False`; `sweep=False`; a summary with a wrong status or an invented number is replaced; a decision still pending; a second run trades nothing new; an `active` trade filled by a later run; an approval that ran out; `PAPER_TRADING=false`; a mode of `live`; `book` in backtest; the status and `schema_version` 4; temperature 0 reaches `get_chat_model`; the imports rule T5 (read the source of every Trader module); the prompt and every tool result hold no thesis and no quote of the record (T9); the recommendations, reviews and decisions are unchanged (T10) |
| `test_trader_config.py` | the file loads; an unknown key, a missing key and each refused value fail |
| `test_work_order_trader.py` | the status, the new fields, `verify_bar_request`, `verify_trades` (each reason, by changing one thing in a good order) |
| `test_data_clerk_bars.py` (or an extension of an existing file) | `daily_bars` and `request_bars` over day files written with `bar_helpers.write_day_files` in a temporary folder; no model is called |
| `test_portfolio.py` (extended) | `apply_fill`: a new position, an average cost, the T+1 fields, a sale, a sold-out position, each refusal |
| `test_account_manager_trades.py` | `connect`, `trading_portfolio`, `request_bars`, `notify_trades` in memory and with a fake store, `perform_trades` |
| `test_account_book.py` (extended) | one test on a throwaway account in the real local Postgres: fills booked through `notify_trades`, skipped when the password file is absent, as that file's other tests are |
| `test_desk_trader.py` | the graph's nodes and edges with and without funds; the path of one order; the Trader not run when nothing is permitted; `with_trader` without the Critic refused; the report; the flags; `--account` with a fake store |
| `test_recommendations_trader.py` | the runner with a Trader |

For each of T1 to T10, break the guard on purpose once, see its test fail, and put the guard back. List those in the build notes, as the Critic's notes do.

Run the suite with caches off, so nothing has to be cleaned up afterwards:

```
PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/cmu/bin/python -m pytest -q -p no:cacheprovider --deselect tests/fastmcp_tools.py::test_forced_failure_known_demo
```

---

## 19. What must not happen during the build (from `AGENTS.md`)

- No paid call: no desk run with a symbol, no `test_recommendations.sh`, no `Critic()` or `AccountManager()` with a real model, no quote, no fetch, no `PROMETHEUS_LIVE_TESTS`. Tests use fakes. The replay with `--mode backtest --no-model` is free and may be run.
- Never read, print or edit `.env`. Never write under `data/historical/` or `data/processed/`. Never read `secrets/`.
- No file is deleted. No version in `requirements.txt` changes; this build needs no new dependency.
- No commit.

---

## 20. Docs to update in the same change

- `docs/design_choices.md`: new numbered entries at the end, starting at 73 (problem, decision, alternatives, consequences). Suggested: 73, the Trader trades on paper and the model has four guarded tools; 74, the Data Clerk's `fetch_bars` and the Account Manager as the Trader's only door; 75, funding sells need a yes, and the backtest's fill and exit rules. Do not edit an older entry.
- `HowToRun.txt`: a new part for the Trader: the commands of section 16, and which are free.
- `README.md`: the layout.
- `AGENTS.md`: a "Where things are" line for the Trader, and in "Never without asking Jacob" the paid Trader commands (`desk --trade`, the replay with a model or in paper mode).
- `docs/SESSION_NOTES.md`: a dated line that the Trader is built and where its notes are. Do not rewrite the older sections.
- `design/how_to_code_trader.md`: the build notes, phase by phase.

---

## 21. Decisions made here that Jacob may want to change

1. **A tool loop with a code sweep**, not one JSON plan and not a free ReAct loop. The loop is the week 2 design; the sweep is the Data Clerk's own pattern (entry 40) and is what makes a backtest free and repeatable.
2. **Funding sells always need a yes**, in Auto mode too. `funding.require_user_approval` turns that off.
3. **Fills stay in memory** unless `--book` is given, and a backtest never books.
4. **A backtest enters at the next session's open** and exits by the Advisor's own stop, target and time stop, with the stop winning a day that touches both.
5. **A market order still has a price limit** (1% past the price the Critic judged, and never outside the entry zone).
6. **A purchase the cash does not fully cover is cut to the shares it can pay for**, not refused.

## 22. Not in this version

- Live trading (Alpaca). `broker.py` is where a second broker would go.
- A backtest over a range of dates. It needs the Advisor and the Critic to run as of a past date with a quote from that date; today the Critic's quote is live. The simulator's functions are pure so that harness can call them.
- Booking a simulated exit in the account book, and following a paper position after the run.
- Funding more than one purchase in a run.
- Extended-hours order rules (a real broker takes only limit orders before 09:30 and after 16:00).

---

## Amendment of 2026-11-14: the per-trade handshake with the Account Manager (design choices entry 81; `design/trader_handshake_spec.md`)

This amendment SUPERSEDES the parts of the text above named here, and nothing else. The text above is left as it was written.

- **Rule T5 changed.** "The portfolio, the cash, the quotes, the bars and the booking of fills all go through `AccountManager` methods" now means two more doors: `AccountManager.approve_trade` and `AccountManager.record_fill`. The Trader holds NO cash and NO portfolio of its own (not a copy, not a running total): the Account Manager and the Postgres book own the funds and the positions. The Trader's calls to the Account Manager are `request_prices`, `request_bars`, `trading_portfolio(portfolio, book)`, `apply_earlier_fills`, `current_portfolio`, `approve_trade`, `preview_trade` and `record_fill`; a test lists them and fails on any other. The rest of T5 (the imports) is as it was.
- **Section 6, "the paper broker keeps the running portfolio of one run", is superseded.** `PaperBroker` has no `portfolio`; it is given the Account Manager and the work order. Every trade, whoever places it (the model's `place_order` or the sweep, a funding sale included), goes through one funnel: the price rules give the fill price, `approve_trade` (the Account Manager reads the funds as they are now and approves a purchase for the whole shares the cash pays for, a sale for the shares held and sellable under T+1, never more than the decision's), the fill for at most the shares approved, the single-trade output check, and `record_fill` (the Account Manager applies the fill and, with a saved account and `book`, books it in ONE Postgres transaction), which returns before the next trade starts. "Cash, when a buy fills" is therefore the Account Manager's arithmetic (`portfolio.affordable_shares`, which moved there from `simulator.py`); the fill rules, the limits and `cash_policy: "reduce"` are unchanged. A refused approval is a `rejected` trade with the Account Manager's code (`insufficient_cash` stays). A fill the book refuses, or could not confirm, is `rejected` `account_refused` with no `Fill`; after one that could not be confirmed (a database failure) nothing more is placed and the order is not marked `traded`.
- **Step 2 (account): the earlier-fills sentence moved.** "When the order already carries trades from an earlier run that were filled and not booked, they are applied to this portfolio first, in order, with `apply_fill`" is now done by the ACCOUNT MANAGER (`apply_earlier_fills`, on its working portfolio, sales first, nothing booked) before the first approval; the Trader only reads the portfolio it then reports, to report it (`portfolio_before`) and to build the tickets. With a run that books, the earlier fills are not applied (the book has what it was told) and an unbooked one is named as a problem.
- **Step 8 (the output guard, then the Account Manager is notified) is superseded.** Each fill is checked by `work_order.verify_fill` (the single-trade part of `verify_trades`) BEFORE it is reported; a finding means the fill is neither reported nor booked, the trade is `rejected` `internal_error`, the finding is on the run, and nothing more is placed. The end-of-run `Notifier` tells the Account Manager NOTHING: it closes the run, REPORTS the whole-run `verify_trades` (a finding is a message and the order is not `traded`; what was booked stays booked), reads the account back from the Account Manager for `portfolio_after` and the cash figures, and counts what was booked (`Trade.booked` is set when its fill is booked). "With a finding, nothing is booked" no longer holds.
- **Section 7 (funding) keeps its steps**, each sale a trade through the same funnel, but the purchase is approved only after the sales are recorded (against the cash the book then shows), and the two cash computations of the Trader became questions to the Account Manager (`preview_trade`): whether the cash covers the purchase (before the user is asked), and whether the estimated proceeds would pay for one share (before anything is sold).
- **Section 8, `notify_account_manager`**, keeps its name, its budget and its refusal while a ticket is `new`, but no longer reports anything: it closes the run, and returns `{cash_before, cash_after, filled, rejected, active, booked}` from the Account Manager's own figures. `place_order`'s result gains `account` (`booked`, `applied`, `refused` or `unconfirmed`, a word of a closed list) and `cash_left` (a number the Account Manager gave). A refusal reaches the model as a reject code, never as the book's sentence (T8, T9).
- **Section 14:** the list of reject codes is unchanged (twelve; `account_refused` was already in it). A trade can now be `rejected` `account_refused` at the moment of its fill, not only at the end of the run.
- **No waiting (Jacob, 2026-11-14: "The handshaking should not wait if there are no further trades to execute").** An approval is asked only for a trade that is about to be placed (the price rules fill it now) and a `record_fill` only for a fill that was approved. A trade that is rejected before it fills, refused at its approval or resting never causes a `record_fill`. After the last trade's `record_fill` is answered the run goes straight to step 8 and on; nothing asks for a "next" trade, polls, sleeps or times out. With no ticket to place there is no handshake at all, and step 2 reads the account only as `portfolio_before` and step 8 only as `portfolio_after`.
- **No halting (Jacob, 2026-11-15, design choices entry 82; amendment 7 of `design/trader_handshake_spec.md`).** The sentences above and in step 8 that stop the run after an output-check finding or an unconfirmed booking are superseded: that trade is failed (`internal_error` or `account_refused`), shown to the user by the Account Manager, never retried, and the next trade is placed. A booking is now three reads around one locked commit (SELECT, COMMIT under row-level locks on the portfolio, SELECT again).


## TODO (2026-11-15): stop-loss orders and a monitoring mode (future version)

Jacob: stop-loss orders in the Trader and a monitoring mode, in a future version: today the plan's stop is only a price on the record (only a backtest follows a filled purchase through the stored bars, `simulator.follow`); nothing rests at the broker and nothing watches an open position. Text and open questions: `design/week6_updated_design.md`, "The Trader, the PaperBroker and the handshake" and Open and next 9. This spec's step list, order types (market, limit) and the exits of `simulator.follow` are unchanged until then.

## Amendment of 2026-11-15 (design choices entry 110): a trade may be for a fraction of a share

Jacob: *"It is possible to buy fractional shares ... $30000 split evenly is $10000 for stock x, $10000 for stock b, $10000 for stock c"* (`design/fractional_shares_spec.md`). Everywhere this spec says a trade is for "a whole number of shares" or "at least one share", read: **a number of shares above zero, with at most six decimals** (`prometheus/common/shares.py`); a whole number is still a Python `int` in every record, and only a quantity with a fraction is a `float`. What changes in the Trader: `Trade.shares_approved`, `Trade.shares` and `Fill.shares` are `Shares` (`PositiveShares` for a fill); a ticket approves "a quantity above zero, else 0" (`tickets._quantity`, which was `_whole` and read a fraction as 0 shares approved); `Fill.amount` is still `trade_amount(shares, price)`, the one rounding to the cent; a purchase cut for the cash is cut to a FRACTIONAL quantity when the order was fractional (`portfolio.affordable_shares`: for a whole order exactly as before), and a sale is all of the approved quantity or none; every printed quantity goes through `shares.text`, and the account line compares a holding's change as quantities, not floats. `work_order.verify_fill` and `verify_trades` check that a fill is a quantity above zero and not above the decision's, for a fraction too. The model's tools still cannot name a number of shares, and they return the quantity as a JSON number. The funding sells of RaiseFunds (section 7) stay WHOLE shares; a purchase of a fraction is raised for the next whole share. Nothing about the order types, the limits, the handshake or the exits of `simulator.follow` changes. The Trader fills at the quote of that moment (paper: five basis points above the close here), so the dollars FILLED can differ from the dollars the Account Manager sized at (the real session `tests/fixtures/session_real/auto_even_split_fractions.json`: $10,000.00 sized, $10,005.00 filled for each of three stocks).

## Amendment of 2026-11-16 (design choices entry 111): no slippage, and an order may be in dollars

Jacob, on a real session that sized $30,000.00 and filled $30,015.00: *"Slippage is not allowed. This is real money. It is impossible to spend more than the amount in the portfolio and budget. ... It is OK to have unused money left over but never is allowed to spend more than is available."* (`design/dollar_orders_spec.md`.) What changes in the Trader: **`costs.slippage_bps` is 0** in `configs/trader_v1.json`: a paper fill is at the quote itself and a backtest fill at the open; the simulator keeps its argument (its own tests pass a number of their own), `Fill.slippage_bps` stays on the record, and the real captures made with 5 basis points are history. Everywhere this spec says a fill is "the quote plus slippage", read "the quote". **A trade may be an order in dollars**: `Trade.dollars` is copied by `tickets.build_tickets` from the decision (`work_order.decision_dollars`); for such a trade `shares_approved` and `shares` are the ESTIMATE at the price the Account Manager sized at, the Trader still decides no number (it asks, the Account Manager answers with the shares its dollars pay for at the fill price, `handshake.decide_approval(..., dollars=)`), and `PaperBroker._fill` fills the shares APPROVED, not the smaller of the estimate and the approval: fewer than the estimate when the price is higher, more when it is lower, never costing more than the dollars. The reason of the trade says so ("an order of $3,750.00: 23.847376 shares at 157.2500 (the estimate was 23.686205)"). A purchase of a request is also held by its POT: the approval is cut to what is left of the money of the request, and the reason says "the money of this request" when that, and not the cash, was the limit. `work_order.verify_fill` checks the fill's AMOUNT against the dollars for an order in dollars (and the old check, more shares than the decision gave, for an order in shares: one of the two always runs), and `verify_trades` fails a work order whose purchases cost more than their pot. The model's tools are unchanged and still cannot name a symbol, a side, a number of shares or a number of dollars.

## Amendment of 2026-11-16 (design choices entry 112): money, prices and share counts are `decimal.Decimal`

Jacob: *"Need to always use decimal.Decimal type when dealing with money and share counts."* The Trader's money is a `Decimal` now (`design/decimal_money_spec.md`). `Fill.price` (four places), `Fill.amount` and `Fill.commission` (the cent), `Fill.shares`, `Trade.shares`, `Trade.shares_approved`, `Trade.dollars`, `Trade.limit_price`, `ref_price`, `limit_low` and `limit_high`, `Snapshot.cash` and an `Outcome`'s exit price, close and gains are Decimals (`money.Money`, `Price`, `shares.Shares`: the type refuses a bool and a string and writes a JSON NUMBER); an `Outcome`'s RETURNS (`return_pct`, the benchmark's and the excess) stay floats, worked out from the Decimal prices at the very end (a measurement). The settings `orders.max_chase_pct`, `orders.limit_band_pct`, `costs.slippage_bps` and `costs.commission_per_order` are read as Decimals from the digits the file writes (`money.Rate`). The fill rules of section 6 are the same rules in Decimal arithmetic: `p = q (1 + slip)` is exact and rounded once to four places, half up (`price_scale`; `simulator.r4` is gone); the limit range of `tickets.limit_range` is exact arithmetic on the reference price and the ratios, rounded to four places; the paper broker's `amount` is `trade_amount(shares, price)`. A quote, a stored bar's open, high, low and close, the Critic's deal price and the Advisor's entry zone, limit, stop and target are floats outside the account: each crosses into the Trader through the PRICE DOOR once (`execute._quoted`, `simulator.bar_price`, `tickets._price`, `execute._plan_price`: four places, half up), and a float given to a fill rule is a `TypeError`. The model's `limit_price` is a number it wrote: `TraderTools.place_order` turns it into a Decimal at its first line. The trade dicts the broker hands to the Account Manager and to `verify_fill` are `model_dump()` (Python mode: Decimals), and the ledger line is `model_dump(mode="json")` (numbers). What the tools show the model goes through the one encoder (`money.dumps`): a Decimal is a number, never a string. Nothing else about the Trader changes: no limit, no minimum order, no slippage.


## Amendment of 2026-11-20 (design choices entry 128): the venue, and what a live venue must still bring

Jacob: *"We will be adding ability to trade live using a stock broker in a later version of this app, so make sure the new simulation paper testing code is clean and can hook up to live (real) trading."* The Trader's funnel (`PaperBroker`: permission, the handshake approve -> fill -> verify -> record, one trade at a time, never retried) no longer knows HOW an order is filled. It asks a `Venue` (`agents/trader/venue.py`): `prices(account, symbols, now, errors)`, `price_of`, `missing`, `execute(trade, shares, prices) -> Execution` (filled, active or rejected by the venue's own rules) and `follow` (the backtest's exits; None elsewhere). `PaperVenue` is the quote, `BacktestVenue` the stored bars, and `venue_for(mode, config)` is the one place that maps a mode to a venue; any other mode is refused there, and a venue that is not simulated is refused unless `PAPER_TRADING` is off AND the venue is registered. None is registered: **no live trading is built**.

A live venue is a third class with `simulated = False` and `books = True`: `prices` from the broker's quote, `execute` = submit the order and wait for the fill report. The handshake already orders the calls correctly for a real fill (the Account Manager approves at the funds as they are, the venue fills, the fill is verified, the Account Manager books it, and only then the next trade). What a live venue must still bring, and what this spec does not have yet:

| Missing | Why it matters |
|---|---|
| A broker order id on the trade, and idempotency | A retry after a lost reply must not place the order twice; today a trade is never retried, and a live venue needs the id to ask the broker what happened instead. |
| Partial fills | `Execution` is filled, active or rejected. A real fill may be for fewer shares than approved, in several parts; `record_fill` must be called for what filled, and the rest decided. |
| An order that stays open past the run | `active` (a resting limit order) ends with the run today. A live order lives at the broker: it needs a run that can be resumed, or a cancel at the end. |
| Cancel | There is no cancel in the Trader. |
| Reconciliation | The broker's cash and positions are a second source of truth beside the Postgres book. A difference must stop trading and be shown to the user, never be settled silently. |
| Stop orders | The plan's stop is a price on the record. A live venue could rest it at the broker; that is the stop-loss TODO, and it is a new origin of a sale (entry 105). |
| Market hours and halts | The simulated venues fill whenever they are asked. |


## Amendment of 2026-11-21 (design choices entry 143, B1): a sale that fills lower, the sales stay

The venue has one more door, `report_fill(trade, shares, estimate, prices)`: what was really FILLED, asked AFTER the Account Manager approved `shares` at the price of `estimate` (what `execute` said it would fill at). A simulated venue returns the estimate (a paper fill is the quote, a backtest fill the bar's price). A live venue SUBMITS the approved shares there, never before the approval, and may report a sale at a LOWER price; a purchase must never be reported above the price it was approved at (a live venue sends a limit order at it: nothing may cost more than the account's money, entry 111). `PaperBroker._fill` books what filled; a `rejected` or `active` report after the approval is a rejected trade.

When funding sales were booked and brought less than they were estimated at, the Account Manager's approval of the funded purchase, which reads the cash the book shows then, refuses it (`insufficient_cash`; the cash is the only check). `PaperBroker._sales_fell_short` turns that into the thirteenth reject code, `sales_fell_short`: the sales were approved, filled and booked and STAY (nothing here undoes a booked sale), the purchase is `rejected`, its reason says what the sales brought against what was planned, a note goes on the work order, and the Trader's summary names the sales that stay. The whole-run output check `verify_trades` does not call that an error (a filled funding sale whose purchase was rejected for ANY OTHER reason is still the worst case of section 7 and is still reported). A purchase that still fills (cut to the cash, or paid in full) gets a note when the sales brought less. Section 14's list of reject codes is now thirteen.

What a live venue still must bring is the table of the amendment of entry 128 (an order id, partial fills, an order past the run, cancel, reconciliation, stop orders, market hours): this one row, "a sale that fills lower", is no longer in it.
