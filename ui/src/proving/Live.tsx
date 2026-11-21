// The Live view (spec 13.2 and rule P11): the paper account marked to live quotes, with every open position drawn against its plan. It WATCHES and tells: a stop or a target that was hit is an alert
// (a stop is red, a target green; yellow is only "this waits for you"), and nothing on this page can sell or place anything. The curve is the account's equity from its first day with the live point
// on its end; the tiles are market value, gain, how each position moved today, and cash.
import { useMemo } from "react";
import { EquityChart } from "../charts/EquityChart";
import { PlanLadder } from "../charts/PlanLadder";
import { Sparkline } from "../charts/Sparkline";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import { money, pct } from "../format";
import type { EquityPoint, LiveSnapshot } from "./types";
import { useLive } from "./useProvingData";

/** The history of the account plus the live point, as the points the equity chart draws (the book has no benchmark and no drawdown figure for them). */
export function livePoints(s: LiveSnapshot): EquityPoint[] {
  const history: EquityPoint[] = s.history.map((h) => ({ date: h.date, equity: h.equity, cash: h.cash, invested: h.invested, benchmark: null, drawdown_pct: 0, stale: h.stale }));
  const today = s.as_of_utc.slice(0, 10);
  const live: EquityPoint = { date: today, equity: s.equity, cash: s.cash, invested: s.market_value, benchmark: null, drawdown_pct: 0, stale: [] };
  return [...history.filter((h) => h.date < today), live];
}

export function Live({ accountId, enabled }: { accountId: number | null; enabled: boolean }) {
  const live = useLive(accountId, enabled);
  const s = live.data;
  const points = useMemo(() => (s ? livePoints(s) : []), [s]);

  if (accountId === null) {
    return (
      <Card title="Live">
        <div className="empty">
          <b>Open an account first</b>The Live view marks one account&apos;s positions to live quotes.
        </div>
      </Card>
    );
  }
  if (!s) {
    return (
      <Card title="Live">
        <div className="empty">
          <b>{live.error ?? "Reading the account and its quotes…"}</b>
          {live.error ? "Quotes come from Finnhub; a stored close is used when the clock is simulated." : ""}
        </div>
      </Card>
    );
  }

  const anySimulated = s.positions.some((p) => p.simulated);
  const alerts = s.alerts;
  return (
    <div className="live">
      <section className="card live-head" aria-label="Account now">
        <div className="live-title">
          <h2>{s.name ?? `Account #${s.account_id}`}</h2>
          <span className="muted">
            as of <span className="num">{s.as_of_utc.replace("T", " ").replace("Z", " UTC")}</span>
          </span>
          <Badge tone={s.session.open ? "pos" : "neutral"} title={s.session.reason || undefined}>
            {s.session.open ? "MARKET OPEN" : `MARKET CLOSED${s.session.next_open_utc ? ` · opens ${s.session.next_open_utc.slice(0, 16).replace("T", " ")} UTC` : ""}`}
          </Badge>
          {anySimulated && <Badge tone="warn" title="The prices are stored closes of a simulated clock, not live quotes">SIMULATED QUOTES</Badge>}
          <span className="live-watch muted">This view only watches. It never sells or places an order.</span>
        </div>
        <div className="kpis live-kpis" role="group" aria-label="Account figures">
          <div className="kpi neutral big">
            <span className="kpi-label">Market value</span>
            <b className="kpi-value">{money(s.market_value)}</b>
            <span className="kpi-sub">
              {money(s.invested_at_cost)} at cost · equity {money(s.equity)}
            </span>
            <span className="kpi-spark">
              <Sparkline values={points.map((p) => p.equity)} label="Account equity" color="var(--text)" />
            </span>
          </div>
          <div className={`kpi ${s.gain > 0 ? "pos" : s.gain < 0 ? "neg" : "neutral"}`}>
            <span className="kpi-label">Gain</span>
            <b className="kpi-value">
              {s.gain >= 0 ? "+" : "−"}
              {money(Math.abs(s.gain))}
            </b>
            <span className="kpi-sub">{pct(s.gain_pct, 2)} on what was paid</span>
          </div>
          <div className="kpi neutral">
            <span className="kpi-label">Today</span>
            <b className="kpi-value kpi-small">
              {s.positions.length === 0
                ? "—"
                : s.positions.map((p) => (
                    <span key={p.symbol} className={`mover num ${p.day_change_pct !== null && p.day_change_pct < 0 ? "neg" : "pos"}`}>
                      {p.symbol} {pct(p.day_change_pct, 2)}
                    </span>
                  ))}
            </b>
            <span className="kpi-sub">each position&apos;s move since yesterday&apos;s close</span>
          </div>
          <div className="kpi neutral">
            <span className="kpi-label">Cash</span>
            <b className="kpi-value">{money(s.cash)}</b>
            <span className="kpi-sub">not invested</span>
          </div>
        </div>
      </section>

      {alerts.length > 0 && (
        <ul className="alerts" aria-label="Alerts">
          {alerts.map((a) => (
            <li key={`${a.symbol}-${a.kind}`} className={`alert ${a.kind === "target_hit" ? "pos" : a.kind === "stop_hit" ? "neg" : "neutral"}`} role="status">
              <Badge tone={a.kind === "target_hit" ? "pos" : a.kind === "stop_hit" ? "neg" : "neutral"}>{a.kind.replace(/_/g, " ").toUpperCase()}</Badge>
              <span>{a.sentence}</span>
            </li>
          ))}
        </ul>
      )}

      <Card title="The account, from its first day" right={live.streaming ? "updates while the market is open" : `stored closes through ${s.last_bar_day ?? "—"}, then the live point`}>
        <div className="card-b">
          <EquityChart points={points} shown={points.length} sweeping={false} events={[]} days={[]} research={false} startCash={s.history[0]?.equity ?? s.cash + s.invested_at_cost} baselineWord="first day" seriesName="The account" domain={null} underwater={false} />
          {s.history.length === 0 && <p className="muted live-note">This account was opened today: the curve has its live point only.</p>}
        </div>
      </Card>

      <Card title="Open positions against their plans" right={`${s.positions.length} position${s.positions.length === 1 ? "" : "s"}`}>
        <div className="card-b">
          {s.positions.length === 0 ? (
            <div className="empty">
              <b>No open positions</b>Buy something from the Desk tab and it appears here with its plan.
            </div>
          ) : (
            <PlanLadder positions={s.positions} />
          )}
          {s.positions.some((p) => p.evidence) && (
            <ul className="evidence-list muted">
              {s.positions
                .filter((p) => p.evidence && p.status !== "in_plan" && p.status !== "no_plan")
                .map((p) => (
                  <li key={p.symbol}>
                    <b className="tick">{p.symbol}</b>: {p.status.replace(/_/g, " ")} on {p.evidence}
                  </li>
                ))}
            </ul>
          )}
          {s.positions.flatMap((p) => p.notes.map((n) => ({ symbol: p.symbol, n }))).map((x) => (
            <p key={`${x.symbol}${x.n}`} className="muted live-note">
              <b className="tick">{x.symbol}</b> {x.n}
            </p>
          ))}
          {s.notes.map((n) => (
            <p key={n} className="muted live-note">
              {n}
            </p>
          ))}
          {s.clock_note && <p className="muted live-note">{s.clock_note}</p>}
        </div>
      </Card>
    </div>
  );
}
