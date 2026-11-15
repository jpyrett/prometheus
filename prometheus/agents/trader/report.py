"""What the user reads: the TRADER section of a report (Trader v1 spec section 16). Plain text from the records of `order.trades` and `order.trader_runs`: no model writes a line.

    lines = trade_section(order)         # the whole TRADER section; empty when the Trader never ran

    TRADER  (model gpt-oss-120b at temperature 0; PAPER mode, prices: the current quotes (order 5d9fb2368171); run 2026-11-05 10:30 ET)
      T1  buy   44 of 44 PG   limit 167.10   FILLED 44 at 167.0835 = $7,351.67   placed by model
          T1-F1 sell 5 MSFT   market         FILLED 5 at 419.79 = $2,098.95      placed by code
      T2  buy   10 of 10 KO   REJECTED not_permitted: the Critic did not approve it
      account  cash $10,000.00 -> $2,648.33; PG 0 -> 44 shares; kept in memory only: the account book was not changed
      outcome T1  closed by target on 2026-11-09 after 1 session: gain $101.00 (+1.38%), SPY +1.00% (excess +0.38%)
      summary (by model): Bought 44 shares of PG at 167.0835.

A trade shows what was approved and what was done: the shares as "filled of approved", the order, FILLED at a price and amount, ACTIVE (a resting order) or REJECTED with its code and the sentence, and who placed it (the
model or the program's sweep). The funding sells are listed under the purchase they pay for. The account lines come from the run's before and after (`Run.portfolio_before`, `portfolio_after`). In backtest mode each
filled purchase has an OUTCOME line (the exit, the gain, the return, the sessions held, the benchmark). The summary says whose it is: the model's when code kept it, code's own otherwise.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")      # only for the words of a message
PRICE_WORDS = {"paper": "the current quotes", "backtest": "the stored daily bars"}


def _money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0 else f"-${-x:,.2f}"


def _signed(x: float) -> str:
    return f"+{_money(x)}" if x >= 0 else _money(x)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _et(text: str | None) -> str:
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).astimezone(EASTERN).strftime("%Y-%m-%d %H:%M ET")
    except ValueError:
        return "?"


def _order(trade: dict) -> str:
    kind, price = trade.get("order_type"), trade.get("limit_price")
    return "market" if kind == "market" else f"limit {price:,.2f}" if kind == "limit" and isinstance(price, (int, float)) else "no order"


def header(order) -> str:
    """The first line: the model and its temperature, the mode, where the prices came from and when the run was."""
    run = order.trader_runs[-1]
    model = f"model {run.get('model')} at temperature {run.get('temperature', 0):g}" if run.get("model") else "no model: the program placed the orders" if not run.get("use_model", True) else "no model could run: the program placed the orders"
    prices = PRICE_WORDS.get(run.get("mode"), "no prices") + (f" (order {run['price_order']})" if run.get("price_order") else "")
    runs = len(order.trader_runs)
    return f"TRADER  ({model}; {str(run.get('mode')).upper()} mode, prices: {prices}; run {_et(run.get('now_utc'))}" + (f"; run {runs} of this order" if runs > 1 else "") + ")"


def trade_line(trade: dict) -> str:
    """One line for one trade (see the module text)."""
    shares = trade.get("shares", 0)
    approved = trade.get("shares_approved", shares)
    left = f"{trade['trade_id']:<6} {trade['side']:<4} {shares} of {approved} {trade['symbol']:<5}"
    status, fill = trade.get("status"), trade.get("fill") or {}
    if status == "filled" and fill:
        right = f"{_order(trade):<13} FILLED {fill['shares']} at {fill['price']:,.4f} = {_money(fill['amount'])}"
    elif status == "active":
        right = f"{_order(trade):<13} ACTIVE: {trade.get('reason') or 'a resting order'}"
    elif status == "rejected":
        right = f"REJECTED {trade.get('reject_code')}: {trade.get('reason')}"
    else:
        right = f"{_order(trade):<13} NOT PLACED (status {status})"
    by = f"   placed by {trade['placed_by']}" if trade.get("placed_by") else ""
    booked = "   booked" if trade.get("booked") else ""
    return f"  {left}  {right}{by}{booked}"


def _positions(snapshot: dict | None) -> dict[str, int]:
    return {p["symbol"]: p["shares"] for p in (snapshot or {}).get("positions", [])}


def account_line(run: dict) -> str | None:
    """The account before and after: the cash, and each position that changed; and whether the fills were saved in the account book or only kept in memory."""
    before, after = run.get("portfolio_before"), run.get("portfolio_after")
    if not before or not after:
        return None
    was, now = _positions(before), _positions(after)
    moved = [f"{s} {was.get(s, 0)} -> {now.get(s, 0)} shares" for s in dict.fromkeys([*was, *now]) if was.get(s, 0) != now.get(s, 0)]
    kept = "booked in the account book" if run.get("booked") else "kept in memory only: the account book was not changed"
    return f"  account  cash {_money(before['cash'])} -> {_money(after['cash'])}" + "".join(f"; {m}" for m in moved) + f"; {kept}"


def outcome_line(trade: dict) -> str | None:
    """The simulated outcome of a filled purchase (backtest mode): the exit and the gain, or the open position and the unrealised gain, and the benchmark."""
    o = trade.get("outcome")
    if not o:
        return None
    bench = "" if o.get("benchmark_return_pct") is None else f", benchmark {_pct(o['benchmark_return_pct'])} (excess {_pct(o.get('excess_return_pct'))})"
    sessions = o.get("sessions_held")
    held = "" if sessions is None else f" after {sessions} session{'' if sessions == 1 else 's'}"
    if o["status"] == "closed":
        return f"  outcome  {trade['trade_id']}  closed by {str(o['exit_reason']).replace('_', ' ')} on {o['exit_date']} at {o['exit_price']:,.4f}{held}: gain {_signed(o['realized_gain'])} ({_pct(o['return_pct'])}){bench}"
    if o.get("last_date") is None:
        return f"  outcome  {trade['trade_id']}  open: no bar after the entry is stored"
    return f"  outcome  {trade['trade_id']}  still open on {o['last_date']} at {o['last_close']:,.4f}{held}: unrealised {_signed(o['unrealized_gain'])} ({_pct(o['return_pct'])}){bench}"


def trade_section(order) -> list[str]:
    """The TRADER section of a report: the header, one line per trade (the funding sells under their purchase, a trade's note under it), the funding answer, the account, the outcomes (backtest) and the summary with
    whose it is, then what went wrong. Empty when the Trader never ran. A run that was refused shows its header and its reason."""
    if not order.trader_runs:
        return []
    run = order.trader_runs[-1]
    lines = [header(order)]
    trades = [t for t in order.trades if isinstance(t, dict)]
    by_purchase: dict[str, list[dict]] = {}
    for t in trades:
        if t.get("kind") == "funding":
            by_purchase.setdefault(t.get("funds_for"), []).append(t)
    for t in (x for x in trades if x.get("kind") != "funding"):
        lines.append(trade_line(t))
        if t.get("note"):
            lines.append(f"      note: {t['note']}")
        lines += ["      " + trade_line(leg).strip() + "   (funds " + t["trade_id"] + ")" for leg in by_purchase.pop(t["trade_id"], [])]
    lines += ["      " + trade_line(leg).strip() + f"   (funds {leg.get('funds_for')}, which is not on this order)" for legs in by_purchase.values() for leg in legs]
    if run.get("portfolio_before") is None:
        lines.append("  the Trader did not run: nothing was traded (the reason is below)")
    elif not trades:
        lines.append("  no trades: no decision of this order permits one")
    funding = run.get("funding")
    if funding:
        lines.append(f"  funding  RaiseFunds said {funding.get('status')}: {funding.get('note') or 'nothing to say'}" + ("" if funding.get("approved") is None else f" (you said {'yes' if funding['approved'] else 'no'})"))
    account = account_line(run)
    if account:
        lines.append(account)
    lines += [line for line in (outcome_line(t) for t in trades) if line]
    if run.get("summary"):
        lines.append(f"  summary (by {run.get('summary_by')}): {run['summary']}")
    lines += [f"  trader note: {e}" for e in [*run.get("errors", []), *run.get("dropped", [])][:8]]
    return lines
