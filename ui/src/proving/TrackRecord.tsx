// The Track Record (spec 10 and 13.2): every recommendation the desk already made in real runs, scored against the bars since. The same hero, outcome wall and quality row as a Time Machine run, from
// `/api/proving/track`, with "$1 in every traded recommendation" against SPY as its curve and a filter on where each outcome came from. Free: stored bars only.
import { useMemo, useState } from "react";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { ChipGroup } from "./Chips";
import { CriticRules } from "./CriticRules";
import { RunView } from "./RunView";
import { stateOfRun } from "./provingReducer";
import type { TrackCoverage } from "./types";
import { useTrack } from "./useProvingData";

type Source = "all" | "run" | "fixture";

export function TrackRecord({ enabled, onTimeMachine, minSample = 20 }: { enabled: boolean; onTimeMachine: () => void; minSample?: number }) {
  const track = useTrack(enabled);
  const [source, setSource] = useState<Source>("all");
  const state = useMemo(() => (track.data ? stateOfRun(track.data) : null), [track.data]);
  const coverage = track.data?.coverage as Partial<TrackCoverage> | undefined;

  if (!state) {
    return (
      <Card title="Track Record">
        <div className="empty">
          <b>{track.error ?? "Scoring the recommendations made so far…"}</b>
          {track.error ? "The Track Record reads the saved sessions and the stored bars; it costs nothing." : ""}
        </div>
      </Card>
    );
  }
  const outcomes = state.outcomes;
  const count = (k: Source) => (k === "all" ? outcomes.length : outcomes.filter((o) => state.sources[o.key] === k).length);
  const wall = source === "all" ? outcomes : outcomes.filter((o) => state.sources[o.key] === source);
  const few = (state.stats?.outcomes.closed ?? 0) < minSample;

  return (
    <div className="track">
      <div className="track-bar card">
        <div className="track-bar-l">
          <b>{track.data?.understood}</b>
          <span className="muted">
            scored against stored bars through <span className="num">{coverage?.last_bar_day ?? track.data?.range?.last_bar_day ?? "—"}</span>
          </span>
        </div>
        <ChipGroup<Source>
          label="Where the recommendations came from"
          value={source}
          onChange={setSource}
          options={[
            { value: "all", label: "Everything", count: count("all") },
            { value: "run", label: "Your runs", count: count("run"), title: "Runs of this page and of the session command" },
            { value: "fixture", label: "Recorded sessions", count: count("fixture"), title: "The project's own recorded sessions, so the page is not empty on day one" },
          ]}
        />
      </div>
      {few && (
        <div className="track-few" role="note">
          <span>
            Only <b className="num">{state.stats?.outcomes.closed ?? 0}</b> of these recommendations have closed, and a verdict needs {minSample}. The Time Machine replays the desk over months of history and gives a much bigger sample.
          </span>
          <Button onClick={onTimeMachine}>Open the Time Machine</Button>
        </div>
      )}
      <RunView state={state} minSample={minSample} wallOutcomes={wall} chartNote={coverage?.index ?? null} kicker={`${count("all")} recommendations scored`} />
      {source !== "all" && <p className="muted track-filter-note">The wall shows {wall.length} of {outcomes.length}. The figures and charts above count every source.</p>}
      <CriticRules enabled={enabled} />
    </div>
  );
}
