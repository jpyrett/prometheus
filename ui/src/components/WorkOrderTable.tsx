import { useRef } from "react";
import type { WorkRow } from "../api/types";
import type { RunState } from "../state/runReducer";
import { orderWords, statusBadge, tradeWords } from "../state/selectors";
import { Badge, type BadgeTone } from "./ui/Badge";
import { Card } from "./ui/Card";

export function stanceTone(stance: string | null | undefined): BadgeTone {
  return stance === "bullish" ? "pos" : stance === "bearish" ? "neg" : "neutral";
}

export interface WorkOrderTableProps {
  state: RunState;
  selectedKey: string | null;
  onSelect: (key: string) => void;
}

/** The stocks on the work order, one row each (spec 5.3 (10)). Click or the arrow keys select a row; the Recommendation panel shows it. */
export function WorkOrderTable({ state, selectedKey, onSelect }: WorkOrderTableProps) {
  const order = state.workOrder;
  const rows = order?.rows ?? [];
  const body = useRef<HTMLTableSectionElement>(null);
  const status = statusBadge(state);

  const move = (from: number, delta: number) => {
    const target = rows[Math.max(0, Math.min(rows.length - 1, from + delta))];
    if (!target) return;
    onSelect(target.key);
    requestAnimationFrame(() => Array.from(body.current?.querySelectorAll<HTMLElement>("[data-key]") ?? []).find((el) => el.dataset.key === target.key)?.focus());
  };

  return (
    <Card
      title="Work order"
      className="workorder"
      badge={order?.request_id ? <Badge num>{order.request_id}</Badge> : undefined}
      right={status ? <Badge tone={status.tone}>{status.text}</Badge> : undefined}
    >
      <div className="card-b scroll">
        {rows.length === 0 ? (
          <div className="empty">
            <b>No work order yet</b>The stocks the desk works on appear here as soon as the Data Clerk finds them.
          </div>
        ) : (
          <table className="tbl pick" role="grid" aria-label="Stocks on the work order">
            <thead>
              <tr>
                <th>#</th>
                <th>Stock</th>
                <th>Advisor</th>
                <th>Critic</th>
                <th className="r">Order</th>
                <th className="r">Trade</th>
              </tr>
            </thead>
            <tbody ref={body}>
              {rows.map((row, i) => (
                <Row key={row.key} row={row} selected={row.key === selectedKey || (selectedKey === null && i === 0)} state={state} onSelect={onSelect} onMove={(d) => move(i, d)} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  );
}

function Row({ row, selected, state, onSelect, onMove }: { row: WorkRow; selected: boolean; state: RunState; onSelect: (key: string) => void; onMove: (delta: number) => void }) {
  const scored = row.group === "scored";
  const direct = row.group === "direct";
  const trade = tradeWords(row, state);
  const order = orderWords(row);
  const stance = row.advisor?.stance;
  return (
    <tr
      className={`${selected ? "sel" : ""}${scored ? " dim" : ""}`.trim() || undefined}
      data-key={row.key}
      data-group={row.group}
      role="row"
      aria-selected={selected}
      tabIndex={selected ? 0 : -1}
      onClick={() => onSelect(row.key)}
      onKeyDown={(e) => {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          onMove(1);
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          onMove(-1);
        } else if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(row.key);
        }
      }}
    >
      <td className="num muted">{direct ? "▸" : (row.rank ?? "")}</td>
      <td>
        <span className="tick">{row.symbol}</span>
        <span className="sub" title={row.company ?? undefined}>
          {row.company ?? " "}
        </span>
      </td>
      <td>
        {row.advisor && row.advisor.score !== null ? (
          <>
            <span className="num">{Math.round(row.advisor.score)}</span> {stance && <Badge tone={stanceTone(stance)}>{stance.toUpperCase()}</Badge>}
          </>
        ) : direct ? (
          <Badge tone="info">YOUR ORDER</Badge>
        ) : (
          <span className="muted">—</span>
        )}
      </td>
      {scored ? (
        <td colSpan={3} className="muted">
          {row.reason}
        </td>
      ) : (
        <>
          <td>
            {row.critic ? (
              <Badge tone={row.critic.approved ? "pos" : "neg"}>
                {row.critic.approved ? "APPROVE" : "DENY"} {row.critic.score} · {row.critic.grade}
              </Badge>
            ) : row.group === "neutral" ? (
              <Badge tone="info">NEUTRAL PASS</Badge>
            ) : (
              <span className="muted">—</span>
            )}
          </td>
          <td className="r num">
            {order ? (
              <>
                {order.text}
                <span className="sub" style={{ maxWidth: "none" }}>
                  {order.dollars}
                </span>
              </>
            ) : (
              <span className="muted">—</span>
            )}
          </td>
          <td className="r">{trade ? <Badge tone={trade.tone}>{trade.text}</Badge> : <span className="muted">—</span>}</td>
        </>
      )}
    </tr>
  );
}
