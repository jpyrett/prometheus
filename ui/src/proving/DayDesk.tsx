// The day's desk (spec 13.2 (5)): the Desk tab's own WorkOrderTable and RecommendationPanel, fed by one decision day's work-order view (`pv_decision`). The same components, not copies.
import { useEffect, useMemo, useState } from "react";
import type { RawRecords, WorkOrderView } from "../api/types";
import { RecommendationPanel } from "../components/RecommendationPanel";
import { WorkOrderTable } from "../components/WorkOrderTable";
import { shortDate } from "../charts/scale";
import { deskState } from "./selectors";

export function DayDesk({ day, view, onClose, loadRecords }: { day: string; view: WorkOrderView; onClose: () => void; loadRecords?: (key: string) => Promise<RawRecords> }) {
  const state = useMemo(() => deskState(view), [view]);
  const first = view.rows.find((r) => r.group !== "found")?.key ?? view.rows[0]?.key ?? null;
  const [selected, setSelected] = useState<string | null>(first);
  useEffect(() => setSelected(first), [first, day]);
  const row = view.rows.find((r) => r.key === selected) ?? null;
  return (
    <section className="dayDesk" aria-label={`The desk on ${day}`}>
      <div className="dayDesk-h">
        <h3>The desk on {shortDate(day, true)}</h3>
        <span className="muted">what the Data Clerk, the Advisor and the Critic knew that day, and what the Account Manager and the Trader did with it</span>
        <button type="button" className="btn ghost" onClick={onClose}>
          Close
        </button>
      </div>
      <div className="dayDesk-b">
        <WorkOrderTable state={state} selectedKey={selected} onSelect={setSelected} />
        <RecommendationPanel row={row} runId={null} waitingForAdvisor={false} loadRecords={loadRecords} />
      </div>
    </section>
  );
}
