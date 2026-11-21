// "How long it took" (spec 13.2): a compact card of the run view with the server's own timing: the days researched at once, the days that came from the cache over all of them, the wall time, and the chart
// of where the time went (the run's phases, the work of each desk stage, the time in each model), whose table twin is the per-model table. Shown only when the run carries timing.
import { TimingChart } from "../charts/TimingChart";
import { Card } from "../components/ui/Card";
import { duration } from "../format";
import type { ProvingState } from "./provingReducer";
import { SpeedBadge } from "./SpeedBadge";

export function TimingCard({ state }: { state: ProvingState }) {
  const t = state.timing;
  if (!t || (t.research_seconds === undefined && t.walk_seconds === undefined)) return null;
  const total = state.plannedDays.length || state.decisionDays.length;
  return (
    <Card title="How long it took" className="timing-card" right={state.seconds !== null ? <span className="num">{duration(state.seconds)} wall time</span> : undefined}>
      <div className="card-b">
        <dl className="facts">
          <div>
            <dt>Speed</dt>
            <dd>{state.speed ? <SpeedBadge speed={state.speed} /> : <span className="muted">not recorded</span>}</dd>
          </div>
          {t.parallel_days !== undefined && (
            <div>
              <dt>Days researched at once</dt>
              <dd className="num">{t.parallel_days}</dd>
            </div>
          )}
          {t.cached_days !== undefined && (
            <div>
              <dt>Days from the cache</dt>
              <dd className="num">
                {t.cached_days} of {total}
              </dd>
            </div>
          )}
        </dl>
        <TimingChart timing={t} />
      </div>
    </Card>
  );
}
