import type { AccountView } from "../api/types";
import { clockOfIso, money, percent, shares as shareText } from "../format";
import { changeBadge } from "../state/selectors";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

export interface PortfolioChanges {
  changes: Record<string, number>;
  cash_change: number;
  cash_before: number;
}

interface Row {
  symbol: string;
  name: string | null;
  shares: number;
  value: number;
  change: number | null;
  /** The shares a waiting sale has set aside. */
  reserved: number;
}

/** The holdings, the cash and the totals (spec 5.3 (1)). Value is shares x average cost: the book knows no market price (D11). */
export function PortfolioCard({ view, changed, before }: { view: AccountView | null; changed: PortfolioChanges | null; before: AccountView | null }) {
  const positions = view?.positions ?? [];
  const changes = changed?.changes ?? {};
  const rows: Row[] = positions.map((p) => ({ symbol: p.symbol, name: p.name, shares: p.shares, value: p.value, change: changes[p.symbol] ?? null, reserved: p.reserved ?? 0 }));
  // A position the run sold out is no longer on the account: keep its row, with the change, until the next run.
  for (const [symbol, change] of Object.entries(changes)) {
    if (!positions.some((p) => p.symbol === symbol) && change < 0) {
      rows.push({ symbol, name: before?.positions.find((p) => p.symbol === symbol)?.name ?? null, shares: 0, value: 0, change, reserved: 0 });
    }
  }
  const invested = view?.invested ?? 0;
  const cash = view?.cash ?? 0;
  // WAITING ORDERS (entry 140): the cash is all of it; what waiting orders have set aside is not available to anything else.
  const setAside = view?.reserved_cash ?? 0;
  const available = view?.available ?? cash;
  const total = view?.total ?? 0;
  const investedPct = percent(invested, total);
  const cashChanged = changed !== null && Math.abs(changed.cash_change) >= 0.005;
  const updated = clockOfIso(view?.as_of_utc);

  return (
    <Card title="Portfolio" className="portfolio" right={updated ? <>updated <span className="num">{updated}</span></> : undefined}>
      <div className="card-b pf-body">
        <div className="alloc" title="Invested against cash" role="img" aria-label={`${investedPct.toFixed(1)}% invested, ${(100 - investedPct).toFixed(1)}% cash`}>
          <i style={{ width: `${investedPct}%` }} />
        </div>
        <div className="alloc-cap">
          <span>
            <b style={{ color: "var(--text)" }}>{investedPct.toFixed(1)}%</b> invested
          </span>
          <span>{view ? (100 - investedPct).toFixed(1) : "100.0"}% cash</span>
        </div>
        <div className="pf-scroll scroll">
          <table className="tbl">
            <thead>
              <tr>
                <th>Symbol</th>
                <th className="r">Shares</th>
                <th className="r">Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const badge = r.change !== null ? changeBadge(r.change, r.shares) : null;
                return (
                  <tr key={r.symbol} className={r.change !== null ? "flash" : undefined}>
                    <td>
                      <span className="tick">{r.symbol}</span> {badge && <Badge tone={badge.tone}>{badge.text}</Badge>}
                      <span className="sub" title={r.name ?? undefined}>
                        {r.name ?? " "}
                      </span>
                      {r.reserved > 0 && <span className="sub aside">{shareText(r.reserved)} set aside for a waiting sale</span>}
                    </td>
                    <td className="r num">{shareText(r.shares)}</td>
                    <td className="r num">{money(r.value)}</td>
                  </tr>
                );
              })}
              {rows.length === 0 && view && (
                <tr className="empty-row">
                  <td colSpan={3}>
                    <b>No holdings yet.</b> Add funds and ask the desk to buy.
                  </td>
                </tr>
              )}
              <tr className="cash">
                <td>
                  <b>Cash</b>
                  <span className="sub">{setAside > 0 ? `${money(setAside)} set aside for waiting orders · ${money(available)} available to trade` : "available to trade"}</span>
                </td>
                <td className="r num muted">—</td>
                <td className="r num">{money(cash)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <table className="tbl pf-foot">
          <tbody>
            <tr>
              <td>Total invested</td>
              <td className="r num">{money(invested)}</td>
            </tr>
            <tr>
              <td>
                Funds available
                {cashChanged && changed ? <span className="was">was {money(changed.cash_before)}</span> : null}
              </td>
              <td className="r num">{money(available)}</td>
            </tr>
            <tr className="grand">
              <td>Account value</td>
              <td className="r num">{money(total)}</td>
            </tr>
          </tbody>
        </table>
        <p className="hint">Value is shares × average cost.</p>
      </div>
    </Card>
  );
}
