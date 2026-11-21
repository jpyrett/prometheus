// The hero of a run (spec 13.2 (1)): the headline strip with the server's own sentence in large type and its tone, and five KPI tiles. The numbers are the server's (P12); a tile counts up
// only when its figure arrives from a live stream. Under the minimum sample a rate or an average carries a mark and no colour of success or failure (P9).
import { useEffect, useRef, useState, type ReactNode } from "react";
import { money } from "../format";
import { prefersReducedMotion } from "../charts/useSize";
import { Sparkline } from "../charts/Sparkline";
import { shortDate } from "../charts/scale";
import { Badge } from "../components/ui/Badge";
import type { ProvingState } from "./provingReducer";
import { kpis, type Kpi } from "./selectors";
import { SpeedBadge } from "./SpeedBadge";

/** A figure that rises from zero to its value over 700 ms when `animate`; the value itself, at once, otherwise (and under reduced motion). */
function useCountUp(target: number | null, animate: boolean): number | null {
  const [shown, setShown] = useState<number | null>(animate && !prefersReducedMotion() ? 0 : target);
  const from = useRef<number | null>(null);
  useEffect(() => {
    if (target === null || !animate || prefersReducedMotion()) {
      setShown(target);
      return;
    }
    const start = from.current ?? 0;
    const t0 = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const k = Math.min(1, (now - t0) / 700);
      const ease = 1 - (1 - k) ** 3;
      setShown(start + (target - start) * ease);
      if (k < 1) frame = requestAnimationFrame(tick);
      else from.current = target;
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, animate]);
  return shown;
}

function Tile({ kpi, animate, spark }: { kpi: Kpi; animate: boolean; spark?: ReactNode }) {
  const value = useCountUp(kpi.raw, animate);
  const text = kpi.raw === null || value === null ? kpi.value : kpi.format(value);
  return (
    <div className={`kpi ${kpi.tone}${kpi.big ? " big" : ""}${kpi.small ? " small" : ""}`} data-kpi={kpi.key}>
      <span className="kpi-label">{kpi.label}</span>
      <b className="kpi-value">{text}</b>
      <span className="kpi-sub">{kpi.sub}</span>
      {kpi.small && (
        <Badge tone="neutral" title="Fewer outcomes than the minimum sample: a first look, not a verdict">
          SMALL SAMPLE
        </Badge>
      )}
      {spark && <span className="kpi-spark">{spark}</span>}
    </div>
  );
}

export interface HeroProps {
  state: ProvingState;
  /** The points the playhead has drawn so far (for the live readout). */
  shown: number;
  benchmark?: string;
  actions?: ReactNode;
  /** A line to show when there is nothing else (a saved run's date range). */
  kicker?: string;
}

export function Hero({ state, shown, benchmark = "SPY", actions, kicker }: HeroProps) {
  const tiles = kpis(state, benchmark);
  const tone = state.stats?.headline.tone ?? "neutral";
  const live = state.points[Math.max(0, Math.min(state.points.length, shown) - 1)];
  const running = state.phase === "running";
  const sentence = state.stats ? state.stats.headline.sentence : running ? (state.step === "research" ? "The desk is being replayed on the stored history, day by day." : "Walking the days in order: the account trades what each day found.") : state.failure ? state.failure : "Nothing to show yet.";
  const animate = state.streamed && state.phase === "finished";
  const eq = state.points.map((p) => p.equity);
  const bench = state.points.map((p) => p.benchmark).filter((v): v is number => v !== null);
  const dd = state.points.map((p) => p.drawdown_pct);

  return (
    <section className={`pv-hero card tone-${tone}`} aria-label="Result of the test">
      <div className="pv-headline">
        <div className="pv-kicker">
          <SpeedBadge speed={state.kind === "track" ? null : state.speed} />
          <span>
            {kicker ?? (state.range ? `${shortDate(state.range.first_day, true)} to ${shortDate(state.range.last_day, true)}` : "")}
            {state.understood ? <span className="muted"> · {state.understood}</span> : null}
          </span>
        </div>
        <h2 className="pv-sentence" aria-live="polite">
          {sentence}
        </h2>
        {running && live && (
          <div className="pv-readout num" aria-hidden="true">
            {shortDate(live.date, true)} · the desk {money(live.equity)}
            {live.benchmark !== null ? ` · ${benchmark} ${money(live.benchmark)}` : ""}
            <span className="muted"> · as it was walked</span>
          </div>
        )}
        {state.failure && state.status !== "finished" && (
          <div className="pv-note">
            <Badge tone="neg">{state.status === "stopped" ? "STOPPED" : "STOPPED EARLY"}</Badge> <span>{state.failure}</span>
          </div>
        )}
      </div>
      {actions && <div className="pv-hero-actions">{actions}</div>}
      <div className="kpis" role="group" aria-label="Key figures">
        {tiles ? (
          tiles.map((k) => (
            <Tile
              key={k.key}
              kpi={k}
              animate={animate}
              spark={
                k.key === "return" || k.key === "index" ? (
                  <Sparkline values={eq} label={`${k.label} curve`} color="var(--text)" />
                ) : k.key === "excess" || k.key === "bench" ? (
                  <Sparkline values={bench} other={eq} label={`${benchmark} against the desk`} color="var(--info)" />
                ) : k.key === "drawdown" ? (
                  <Sparkline values={dd} label="Drawdown curve" color="var(--neg)" />
                ) : undefined
              }
            />
          ))
        ) : (
          ["Return", `vs ${benchmark}`, "Win rate", "Max drawdown", "Recommendations"].map((label, i) => (
            <div key={label} className={`kpi pending${i === 1 ? " big" : ""}`} aria-busy={running}>
              <span className="kpi-label">{label}</span>
              <b className="kpi-value">{"—"}</b>
              <span className="kpi-sub">{running ? "arrives when the test is walked" : "no figures"}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
