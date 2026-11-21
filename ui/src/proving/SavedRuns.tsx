// The saved runs under the launch pad (spec 13.2): small cards with their own sparkline, return and excess. Open shows the run as it was; "Watch it again" plays it through the same events (free).
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Sparkline } from "../charts/Sparkline";
import { shortDate } from "../charts/scale";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import { pct, points } from "../format";
import { SpeedBadge } from "./SpeedBadge";
import type { RunIndexRow } from "./types";

export interface SavedRunsProps {
  rows: RunIndexRow[] | null;
  error: string | null;
  disabled: boolean;
  onOpen: (id: string) => void;
  onWatch: (id: string) => void;
  onDelete: (id: string) => Promise<unknown> | void;
}

// The index of saved runs carries no curve, so each card reads its run once for its own sparkline (free) and keeps it for the session.
const curves = new Map<string, { desk: number[]; bench: number[] } | null>();
function useCurve(id: string, enabled: boolean) {
  const [curve, setCurve] = useState(curves.get(id) ?? null);
  useEffect(() => {
    if (!enabled || curves.has(id)) {
      setCurve(curves.get(id) ?? null);
      return;
    }
    let live = true;
    api
      .provingRun(id)
      .then((run) => {
        const c = { desk: run.equity.map((p) => p.equity), bench: run.equity.map((p) => p.benchmark).filter((v): v is number => v !== null) };
        curves.set(id, c);
        if (live) setCurve(c);
      })
      .catch(() => curves.set(id, null));
    return () => {
      live = false;
    };
  }, [id, enabled]);
  return curve;
}

/** The newest saved runs shown before "Show all": a store that has seen many tests is a long list, and the launch pad is what this page is for. */
const FIRST = 6;

export function SavedRuns({ rows, error, disabled, onOpen, onWatch, onDelete }: SavedRunsProps) {
  const [sure, setSure] = useState<string | null>(null);
  const [all, setAll] = useState(false);
  const saved = (rows ?? []).filter((r) => r.kind === "time_machine");
  const runs = all ? saved : saved.slice(0, FIRST);
  return (
    <Card title="Saved runs" right={rows ? `${saved.length} saved` : undefined}>
      <div className="card-b">
        {error ? (
          <p className="err">{error}</p>
        ) : rows === null ? (
          <p className="muted">Reading the saved runs…</p>
        ) : saved.length === 0 ? (
          <div className="empty">
            <b>No saved run yet</b>A test you launch is saved here, and can be opened or watched again for free.
          </div>
        ) : (
          <ul className="saved">
            {runs.map((r, i) => (
              <SavedCard key={r.run_id} r={r} first={i < 12} disabled={disabled} sure={sure === r.run_id} setSure={setSure} onOpen={onOpen} onWatch={onWatch} onDelete={onDelete} />
            ))}
          </ul>
        )}
        {saved.length > FIRST && (
          <div className="saved-more">
            <button type="button" className="btn" aria-expanded={all} onClick={() => setAll((v) => !v)}>
              {all ? `Show only the newest ${FIRST}` : `Show all ${saved.length} saved runs`}
            </button>
          </div>
        )}
      </div>
    </Card>
  );
}

function SavedCard({ r, first, disabled, sure, setSure, onOpen, onWatch, onDelete }: { r: RunIndexRow; first: boolean; disabled: boolean; sure: boolean; setSure: (id: string | null) => void } & Pick<SavedRunsProps, "onOpen" | "onWatch" | "onDelete">) {
  const curve = useCurve(r.run_id, first && !r.spark);
  const spark = r.spark ?? curve?.desk;
  return (
      <li className="saved-card">
        <div className="saved-h">
          <span className="num muted">{r.range ? `${shortDate(r.range.first_day, true)} to ${shortDate(r.range.last_day, true)}` : r.created_utc.slice(0, 10)}</span>
          <SpeedBadge speed={r.speed} />
          {typeof r.headline === "object" && r.headline && r.headline.tone !== "neutral" && <Badge tone={r.headline.tone}>{r.headline.tone === "pos" ? "BEAT" : "TRAILED"}</Badge>}
        </div>
        <p className="saved-prompt" title={r.prompt}>
          {r.prompt}
        </p>
        <div className="saved-m">
          {spark && spark.length > 1 ? <Sparkline values={spark} other={curve?.bench} label="Equity curve of the run" color="var(--text)" width={96} /> : <span className="spark-none" aria-hidden="true" />}
          <div>
            <b className="num">{pct(r.return_pct)}</b>
            <small className="muted num">{points(r.excess_return_pct)} vs SPY</small>
          </div>
        </div>
        <div className="saved-a">
          <button type="button" className="btn" onClick={() => onOpen(r.run_id)}>
            Open
          </button>
          <button type="button" className="btn" disabled={disabled} title={disabled ? "a run is active" : "Plays the saved run again (free: no model is called)"} onClick={() => onWatch(r.run_id)}>
            Watch it again
          </button>
          {sure ? (
            <button
              type="button"
              className="btn no"
              onClick={async () => {
                await onDelete(r.run_id);
                setSure(null);
              }}
            >
              Delete this run?
            </button>
          ) : (
            <button type="button" className="btn ghost" onClick={() => setSure(r.run_id)} aria-label={`Delete the saved run of ${r.range?.first_day ?? r.created_utc}`}>
              Delete
            </button>
          )}
        </div>
      </li>
  );
}
