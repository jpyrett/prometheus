import type { OpenOrder, OrderNotice, OrdersView } from "../api/types";
import { money, shares as shareText } from "../format";
import { Button } from "./ui/Button";
import { Card } from "./ui/Card";

/** "4:00 pm ET Mon 2026-11-16": when a day order ends, in the market's own time. */
export function untilText(iso: string): string {
  const moment = new Date(iso);
  if (Number.isNaN(moment.getTime())) return iso;
  const part = (options: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", ...options }).format(moment);
  const clock = part({ hour: "numeric", minute: "2-digit", hour12: true }).toLowerCase();
  const weekday = part({ weekday: "short" });
  const ymd = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(moment);
  return `${clock} ET ${weekday} ${ymd}`;
}

/** What the order is to do: "buy 10 MU at $90.00 or less" (a purchase has a ceiling), "sell 5 MSFT at $600.00 or more" (a sale has a floor). */
export function orderWords(order: OpenOrder): string {
  const what = order.dollars !== null ? `buy ${money(order.dollars)} of ${order.symbol}` : `${order.side} ${shareText(order.shares ?? 0)} ${order.symbol}`;
  return `${what} at ${money(order.limit_price)} ${order.side === "buy" ? "or less" : "or more"}`;
}

export function heldWords(order: OpenOrder): string {
  return order.side === "buy" ? `${money(order.reserved_cash)} set aside` : `${shareText(order.reserved_shares)} ${order.reserved_shares === 1 ? "share" : "shares"} set aside`;
}

/** What a purchase the user permitted to raise funds says (design choices entry 143): holdings are sold for the rest only when its price is reached, never while it waits. null for every other order (a sale needs no money). */
export function raiseWords(order: OpenOrder): string | null {
  if (order.side !== "buy" || !order.raise_funds_by) return null;
  return `sells holdings for the rest only if the price is reached · ${order.raise_funds_by === "setting" ? "the setting ALWAYS_RAISE_FUNDS is on" : "you said to raise funds"}`;
}

/** The account's waiting limit orders (design choices entry 140): each holds its cash or its shares, ends at its session's close, and can be cancelled. Nothing is drawn when there is none and nothing was said. */
export function OpenOrdersCard({ view, notices, error, cancelling, onCancel, readOnlyReason }: { view: OrdersView | null; notices: OrderNotice[]; error: string | null; cancelling: number | null; onCancel: (id: number) => void; readOnlyReason: string | null }) {
  const orders = view?.orders ?? [];
  if (orders.length === 0 && notices.length === 0 && !error) return null;
  return (
    <Card title="Waiting orders" right={orders.length > 0 ? `${money(view?.reserved_cash ?? 0)} set aside` : undefined} className="orders">
      <div className="card-b">
        {orders.length > 0 && (
          <table className="tbl orders-tbl">
            <tbody>
              {orders.map((order) => (
                <tr key={order.order_id}>
                  <td>
                    <span className="tick">{order.symbol}</span> <span className="num">#{order.order_id}</span>
                    <span className="sub">{orderWords(order)}</span>
                    <span className="sub">
                      {heldWords(order)} · until {untilText(order.expires_utc)}
                    </span>
                    {raiseWords(order) && <span className="sub">{raiseWords(order)}</span>}
                  </td>
                  <td className="r">
                    <Button variant="ghost" onClick={() => onCancel(order.order_id)} disabled={cancelling !== null || readOnlyReason !== null} title={readOnlyReason ?? "Cancel this order: what it set aside is available again"} aria-label={`Cancel order ${order.order_id}`}>
                      {cancelling === order.order_id ? "Cancelling…" : "Cancel"}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {error && (
          <p className="hint err" role="alert">
            {error}
          </p>
        )}
        {notices.map((notice) => (
          <p key={`${notice.type}-${notice.order_id}-${notice.ts}`} className="hint notice">
            {notice.text}
          </p>
        ))}
        {orders.length > 0 && (
          <p className="hint">
            A waiting order is a day order: it ends at its session&apos;s close, and it is watched only while this page&apos;s server runs
            {view && !view.watching ? " — and no watcher is running right now (start the server without --no-watch)." : "."}
          </p>
        )}
      </div>
    </Card>
  );
}
