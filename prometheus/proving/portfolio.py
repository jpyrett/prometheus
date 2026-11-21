"""The Time Machine's OWN money: a simulated account in memory (design/proving_ground_spec.md section 5, P5, P6, P7 and P10).

    account = SimPortfolio(Decimal("10000.00"), user="Bob Jones")
    account.apply_fills(order.trades, {record_id: LotInfo(plan=record.plan_of(rec), key=outcome_key)})      # the Trader's filled purchases, as they are: the fill's shares, price, amount and date, never recomputed
    exits = account.step(day, bars)                      # P5: the plan's stop, target and time stop on that day's bar, for every open position: the sale proceeds go to the cash
    point = account.mark(day, bars)                      # an `EquityPoint`: cash + the positions at the day's close
    account.as_desk_portfolio()                          # the `common.portfolio.Portfolio` the desk reads (what the Account Manager decides and sizes with)

WHAT IT IS NOT. It is not the account book: nothing here touches Postgres (P6) and nothing is booked. It does not decide anything and it adds no rule: the Account Manager sizes and the Trader fills (P7: entry 111 applies unchanged, the
pot, the handshake), and a position leaves only by its plan (P5). It sells nothing else: `apply_fills` refuses a sale.

THE EXIT IS `simulator.follow`, NOT A SECOND COPY OF IT. `step(day, bars)` calls `follow` for each open position on the bars from its entry to `day` (so the stop before the target, only the stop on the entry bar, the time stop at the first
bar on or after its date, are THE rules) and sells at the price `follow` gives, on the date `follow` gives, which is `day` when the engine steps every day; a position `follow` says closed earlier than `day` (a day was skipped) is sold on
that earlier date. The proceeds are `trade_amount(shares, price)` to the cent, the gain is the proceeds less the lot's cost (the fill's own `amount`), and it must equal `follow`'s `realized_gain`. Call `apply_fills` for a day's fills BEFORE
`step(day)`: a position bought at the open can be stopped on that same bar, by the stop only.

THE IDENTITIES, CHECKED ON EVERY STEP (`check`; a break raises `IdentityError`, an AssertionError, which an engine lets stop the run): the cash is never negative; cash = starting cash - what was paid + what came in; the sum of the
realized gains = the proceeds - the cost of the closed lots; what was paid = the cost of the open lots + the cost of the closed lots; a fill's amount is `trade_amount(shares, price)`; and a day's equity = cash + the positions at the
close (each `trade_amount(shares, close)`).

MARKS. A position is marked at the close of the day's bar; a held symbol with NO bar that day keeps its previous close and the point says so (`stale`): a price is never invented. The drawdown of a point is its equity against the highest
equity so far, the starting cash counting as the first (a fraction at or below zero, a float: a measurement).

SPLITS (P10). With a split table for a symbol (`splits={"NFLX": SplitTable}`) a lot is put through each split on its date (`SplitTable.adjust`: the shares multiplied, the cost basis and the plan prices divided), and `follow` reads the bars in the
share basis of the ENTRY (`splits.to_basis`), so the plan made before the split means what it meant (a plan made the day before a split and bought on the split day is brought into the entry's basis when `LotInfo.decision_day` says when it was
made). A lot that crossed a split is worked in the share basis of its entry for the money (see `record.Exit`). A symbol with NO table is never followed across a suspected split (a close-to-open move past `split_suspect_pct`): the bars from that
day on are left out, so the position stays open at its last good close, marked stale, and `warnings` says why: a split must never read as a loss and hit a stop.

A TRAILING STOP (design/advisor_exit_plan_spec.md E4: the TREND family's plan, no target): a lot whose plan trails (`PlanView.trailing`) is followed with `simulator.follow(trail_distance=)`, the SAME call, so the stop rises with the highest close and the exit
is `trailing_stop` when the stop that fired was above the plan's own. `positions()` shows the stop IN FORCE (`simulator.trailing_stop`: the plan's stop or the highest close since the entry less the distance, whichever is higher, at the last mark). The distance is a
plan price, divided by a split's factor like the stop.

MONEY, PRICES AND SHARES ARE `Decimal` through the doors (entry 112): the cash is to the cent, a price has four places, a quantity six; a bar's floats cross the price door once inside the simulator. The only floats are the drawdown of a point
(a fraction, a measurement). No size limit, risk rule, liquidity rule or minimum order is added.
"""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from prometheus.agents.trader import simulator
from prometheus.agents.trader.config import load_config as load_trader_config
from prometheus.agents.trader.record import Trade
from prometheus.common import shares as sh
from prometheus.common.money import EXACT, non_negative_money, price_scale
from prometheus.common.portfolio import Portfolio, apply_fill, trade_amount
from prometheus.proving import splits as splits_mod
from prometheus.proving.config import load_config
from prometheus.proving.record import EquityPoint, Exit, PlanView, Position

ZERO = Decimal("0.00")


class IdentityError(AssertionError):
    """An identity of the account does not hold (the cash went negative, the gains do not add up ...): a programming mistake or a fill the Account Manager should never have approved. It is an AssertionError on purpose: a run stops on it."""


@dataclass
class LotInfo:
    """What the engine knows about a purchase that the Trader's trade does not say: the Advisor's plan (the stop, the target and the time stop that will close it), the key of its outcome and the company's name."""
    plan: PlanView | None = None
    key: str | None = None
    name: str | None = None
    decision_day: date | None = None        # the day the plan was made: its prices are in the share basis of that day (only a split between that day and the fill changes anything)


@dataclass
class Lot:
    """One purchase, open. The `entry_*`, `stop`, `target` and `time_stop` are in the share basis of the ENTRY (what `follow` reads); `shares`, `basis`, `stop_now` and `target_now` are in the basis of `applied` (after the splits so far)."""
    symbol: str
    name: str | None
    record_id: str | None
    key: str | None
    trade_id: str
    entry_day: date
    entry_price: Decimal
    entry_shares: Decimal
    cost: Decimal                           # what the fill cost: the fill's own amount
    stop: Decimal | None
    target: Decimal | None
    time_stop: str | None
    plan_day: date                          # the day the plan was made (the entry day when nobody said)
    shares: Decimal
    basis: Decimal
    stop_now: Decimal | None
    target_now: Decimal | None
    applied: date
    last_close: Decimal | None = None
    last_day: date | None = None
    trail: Decimal | None = None            # the distance a TRAILING plan's stop keeps under the highest close, in the share basis of the ENTRY (None: a fixed stop)
    highest: Decimal | None = None          # the highest close since the entry through the last mark, in the share basis of the ENTRY (only a lot that trails keeps it)


class SimPortfolio:
    def __init__(self, cash, user: str = "Proving Ground", splits: dict[str, splits_mod.SplitTable] | None = None, slippage_bps: Decimal | None = None, split_suspect_pct: Decimal | None = None):
        """`cash`: the starting cash (dollars to the cent, zero or more). `user`: the name the desk's portfolio carries (the account's profile has the same). `splits`: {symbol: its SplitTable} for the symbols that have one.
        `slippage_bps`: what `follow` is given (the Trader's settings' own, 0, when left out). `split_suspect_pct`: the suspect rule's move (the Proving Ground's settings' own, 0.40, when left out)."""
        self.starting_cash = non_negative_money(cash)
        self.user, self.splits = user, dict(splits or {})
        self.slippage_bps = load_trader_config().costs.slippage_bps if slippage_bps is None else slippage_bps
        self.split_suspect_pct = load_config().split_suspect_pct if split_suspect_pct is None else split_suspect_pct
        self.warnings: list[str] = []          # why a position was not followed (a suspected split with no table), each said once
        self._cash, self._spent, self._proceeds, self._peak = self.starting_cash, ZERO, ZERO, self.starting_cash
        self.lots: list[Lot] = []
        self.closed: list[Exit] = []

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def realized(self) -> Decimal:
        """The sum of the gains of the positions closed so far, to the cent."""
        return sum((e.gain for e in self.closed), ZERO)

    # ---------------------------------------------------------------- the fills

    def apply_fills(self, trades, info: dict[str, LotInfo] | None = None) -> list[Lot]:
        """Put the Trader's FILLED purchases into the account: each costs its fill's own `amount` (never recomputed) out of the cash and becomes a lot at the fill's price, shares and date. `trades` are `Trade`s or the dicts the Trader wrote on the work
        order (they go through the Trade model's doors); a trade that did not fill is left out; a SALE raises ValueError (a position leaves only by its plan). `info` is {record_id: LotInfo}: the plan that will close each purchase (none: no rule fires,
        the lot stays open), its outcome's key and the company's name. Raises IdentityError for a fill whose amount is not `trade_amount(shares, price)` or that the cash does not cover. Returns the new lots."""
        found = []
        for raw in trades:
            trade = raw if isinstance(raw, Trade) else Trade.model_validate(raw)
            if trade.status != "filled" or trade.fill is None:
                continue
            if trade.side != "buy":
                raise ValueError(f"{trade.trade_id}: the Time Machine's account never sells but by the plan: a sale is refused")
            fill = trade.fill
            if fill.amount != trade_amount(fill.shares, fill.price):
                raise IdentityError(f"{trade.trade_id}: the fill's amount {fill.amount} is not {trade_amount(fill.shares, fill.price)}, the cost of {sh.text(fill.shares)} shares at {fill.price}")
            if fill.amount > self._cash:
                raise IdentityError(f"{trade.trade_id}: the fill costs ${fill.amount:,.2f} and the cash is ${self._cash:,.2f}: the Account Manager approves nothing the cash cannot pay for")
            about = (info or {}).get(trade.record_id) if trade.record_id else None
            plan = about.plan if about and about.plan else PlanView()
            day = date.fromisoformat(fill.date)
            made = about.decision_day if about and about.decision_day else day
            table = self.splits.get(trade.symbol)
            factor = table.factor_between(made, day) if table is not None else Decimal(1)                 # a split between the day the plan was made and the day of the fill: the plan's prices into the fill's basis
            stop, target, trail = (None if x is None else x if factor == 1 else price_scale(EXACT.divide(x, factor)) for x in (plan.stop, plan.target, plan.trail_distance if plan.trailing else None))
            self._cash -= fill.amount
            self._spent += fill.amount
            lot = Lot(symbol=trade.symbol, name=about.name if about else None, record_id=trade.record_id, key=about.key if about else None, trade_id=trade.trade_id, entry_day=day, entry_price=fill.price,
                      entry_shares=fill.shares, cost=fill.amount, stop=stop, target=target, time_stop=plan.time_stop_date.isoformat() if plan.time_stop_date else None, plan_day=made, shares=fill.shares, basis=fill.price,
                      stop_now=stop, target_now=target, applied=day, trail=trail)
            self.lots.append(lot)
            found.append(lot)
        self.check()
        return found

    # ---------------------------------------------------------------- the day

    def _roll(self, lot: Lot, day: date) -> None:
        """Put a lot through every split of its symbol that executed after it was last rolled and on or before `day` (`SplitTable.adjust`: the shares multiplied, the cost basis and the plan prices divided)."""
        table = self.splits.get(lot.symbol)
        if table is not None:
            for split in table.splits:
                if lot.applied < split.execution_date <= day:
                    lot.shares, kept = table.adjust(lot.shares, {"basis": lot.basis, "stop": lot.stop_now, "target": lot.target_now}, split.execution_date)
                    lot.basis, lot.stop_now, lot.target_now = kept["basis"], kept["stop"], kept["target"]
        lot.applied = max(lot.applied, day)

    def _series(self, lot: Lot, bars: dict[str, list[dict]], day: date) -> list[dict]:
        """The stored bars of a lot's symbol up to `day`: all of them, unless the symbol has no split table and a close-to-open move past the suspect rule lies between the day the plan was made and `day`: then the bars from that day
        on are left out (and `warnings` says so, once)."""
        series = [b for b in bars.get(lot.symbol) or [] if isinstance(b, dict) and isinstance(b.get("date"), str) and b["date"] <= day.isoformat()]
        if lot.symbol not in self.splits:
            moved = splits_mod.suspect_move(series, lot.plan_day, day, self.split_suspect_pct)
            if moved is not None:
                said = f"{lot.symbol} moved {moved[1] * 100:+.1f}% from one close to the next open on {moved[0].isoformat()}, which looks like a split, and no split table is stored for it: the position is not followed past that day"
                if said not in self.warnings:
                    self.warnings.append(said)
                series = [b for b in series if b["date"] < moved[0].isoformat()]
        return series

    def step(self, day: date, bars: dict[str, list[dict]]) -> list[Exit]:
        """P5 for `day`, on the day's bar, for every open position that was bought on or before it: `simulator.follow` over the position's bars from its entry to `day`; a position it closes is sold at its price, the proceeds go to the cash and the
        gain is booked. `bars` is {symbol: [bar, ...]} (the stored bars as the Data Clerk gives them; the held symbols are enough). Returns the exits, in the order of the lots. The identities are checked at the end."""
        exits = []
        for lot in list(self.lots):
            if lot.entry_day > day:
                continue
            table = self.splits.get(lot.symbol)
            window = splits_mod.to_basis(self._series(lot, bars, day), table, lot.entry_day)
            outcome = simulator.follow(lot.entry_price, lot.entry_shares, lot.entry_day, window, lot.stop, lot.target, lot.time_stop, self.slippage_bps, trail_distance=lot.trail)
            if outcome.status != "closed":
                self._roll(lot, day)
                continue
            exits.append(self._sell(lot, outcome, table))
        self.check()
        return exits

    def _sell(self, lot: Lot, outcome, table: splits_mod.SplitTable | None) -> Exit:
        """Close a lot at the exit `follow` found: the proceeds, the gain (which must be `follow`'s own), the cash, the books."""
        when = date.fromisoformat(outcome.exit_date)
        self._roll(lot, when)
        proceeds = trade_amount(lot.entry_shares, outcome.exit_price)
        gain = proceeds - lot.cost
        if gain != outcome.realized_gain:
            raise IdentityError(f"{lot.symbol}: the gain of the sale is {gain} but `follow` says {outcome.realized_gain}: there must be one exit rule")
        factor = table.factor_between(lot.entry_day, when) if table is not None else Decimal(1)
        price = outcome.exit_price if factor == 1 else price_scale(EXACT.divide(outcome.exit_price, factor))        # shown in the share basis of the day it closed
        exit = Exit(date=when, symbol=lot.symbol, shares=lot.shares, price=price, amount=proceeds, cost=lot.cost, gain=gain, reason=outcome.exit_reason, sessions_held=outcome.sessions_held, entry_date=lot.entry_day,
                    record_id=lot.record_id, key=lot.key)
        self._cash += proceeds
        self._proceeds += proceeds
        self.closed.append(exit)
        self.lots.remove(lot)
        return exit

    def mark(self, day: date, bars: dict[str, list[dict]]) -> EquityPoint:
        """The account at the close of `day`: the cash and each open position at its close (`trade_amount(shares, close)`). A held symbol with no bar on `day` keeps its previous close (and the point names it in `stale`); with no close
        at all yet it stays at what it cost. The equity is cash + the positions; the drawdown is against the highest equity so far (the starting cash counts)."""
        invested, stale = ZERO, []
        for lot in self.lots:
            if lot.entry_day > day:
                continue
            self._roll(lot, day)
            bar = simulator.bar_on(self._series(lot, bars, day), day)
            close = simulator.bar_price(bar, "close") if bar else None
            if close is not None:
                lot.last_close, lot.last_day = close, day
                if lot.trail is not None:        # a trailing stop's base: the highest close since the entry, in the share basis of the ENTRY (the stored bars after a split are put back into it)
                    table = self.splits.get(lot.symbol)
                    window = splits_mod.to_basis(self._series(lot, bars, day), table, lot.entry_day)
                    closes = [simulator.bar_price(b, "close") for b in window if isinstance(b.get("date"), str) and b["date"] >= lot.entry_day.isoformat()]
                    lot.highest = max((c for c in closes if c is not None), default=None)
            else:
                stale.append(lot.symbol)
            invested += trade_amount(lot.shares, lot.last_close if lot.last_close is not None else lot.basis)
        equity = self._cash + invested
        self._peak = max(self._peak, equity)
        point = EquityPoint(date=day, equity=equity, cash=self._cash, invested=invested, drawdown_pct=simulator._fraction(equity, self._peak) if self._peak > 0 else 0, stale=sorted(set(stale)))
        self.check()
        if sum((p.value for p in self.positions(day)), ZERO) != invested or point.equity != point.cash + point.invested:
            raise IdentityError("the equity of the day is not the cash plus the positions at the close")
        return point

    # ---------------------------------------------------------------- reading it

    def positions(self, day: date | None = None) -> list[Position]:
        """The open positions in the share basis of today, marked at the last close `mark` found (what it cost before the first mark). With `day`, a lot bought after that day is left out and a position whose last close is not from that day is `stale`."""
        out = []
        for lot in self.lots:
            if day is not None and lot.entry_day > day:
                continue
            price = lot.last_close if lot.last_close is not None else lot.basis
            value = trade_amount(lot.shares, price)
            stop = lot.stop_now
            if lot.trail is not None:          # the stop in force, worked in the basis of the entry (where the highest close is) and shown in the basis of today
                table = self.splits.get(lot.symbol)
                level = simulator.trailing_stop(lot.stop, lot.highest, lot.trail)
                factor = table.factor_between(lot.entry_day, lot.applied) if table is not None else Decimal(1)
                stop = level if level is None or factor == 1 else price_scale(EXACT.divide(level, factor))
            out.append(Position(symbol=lot.symbol, name=lot.name, shares=lot.shares, cost_basis=lot.basis, entry_date=lot.entry_day, record_id=lot.record_id, key=lot.key, stop=stop, target=lot.target_now, trailing=lot.trail is not None,
                                time_stop_date=date.fromisoformat(lot.time_stop) if lot.time_stop else None, price=price, value=value, gain=value - trade_amount(lot.shares, lot.basis),
                                stale=day is not None and lot.last_day != day))
        return out

    def as_desk_portfolio(self) -> Portfolio:
        """The portfolio the desk reads (`common.portfolio.Portfolio`: the user, the cash, one position per symbol). Lots of one symbol are one position, at the weighted average cost, by the desk's own rules (`apply_fill`)."""
        held = Portfolio(user=self.user, cash_avail=sum((trade_amount(lot.shares, lot.basis) for lot in self.lots), ZERO))      # cash enough to put each lot in with the desk's own `apply_fill`; the real cash is put back at the end
        for lot in sorted(self.lots, key=lambda lot: lot.entry_day):
            held = apply_fill(held, lot.symbol, "buy", lot.shares, lot.basis, lot.entry_day, lot.name)
        return held.model_copy(update={"cash_avail": self._cash})

    def check(self) -> None:
        """The identities of the account (see the module text). Raises IdentityError, naming the one that broke."""
        if self._cash < 0:
            raise IdentityError(f"the cash is negative: {self._cash}")
        if self._cash != self.starting_cash - self._spent + self._proceeds:
            raise IdentityError("the cash is not the starting cash less what was paid plus what came in")
        proceeds, cost, gains = (sum((getattr(e, name) for e in self.closed), ZERO) for name in ("amount", "cost", "gain"))
        if gains != proceeds - cost or proceeds != self._proceeds:
            raise IdentityError("the realized gains are not the proceeds less the cost of the closed lots")
        if self._spent != cost + sum((lot.cost for lot in self.lots), ZERO):
            raise IdentityError("what was paid is not the cost of the open lots plus the cost of the closed ones")
