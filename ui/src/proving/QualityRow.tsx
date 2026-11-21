// The quality row (spec 13.2 (4)): three cards that ask whether the desk's judgement held. "Was the Advisor's score right?" (calibration), "Was the Critic right?" (approved against denied) and
// "How did the trades end?" (the exit mix, wins against losses, the families). The sentences are the server's own.
import { CalibrationChart } from "../charts/CalibrationChart";
import { ExitMix } from "../charts/ExitMix";
import { VerdictStrip } from "../charts/VerdictStrip";
import { Badge } from "../components/ui/Badge";
import { Card } from "../components/ui/Card";
import type { RecOutcome, Stats } from "./types";

export function QualityRow({ outcomes, stats, minSample = 20, benchmark = "SPY" }: { outcomes: RecOutcome[]; stats: Stats | null; minSample?: number; benchmark?: string }) {
  if (!stats) {
    return (
      <div className="quality">
        {["Was the Advisor's score right?", "Was the Critic right?", "How did the trades end?"].map((t) => (
          <Card key={t} title={t}>
            <div className="empty">
              <b>Waiting for the outcomes</b>These charts fill in when the test has been walked.
            </div>
          </Card>
        ))}
      </div>
    );
  }
  const small = stats.outcomes.small_sample;
  const forwardKept = stats.by_score.some((b) => (b.forward ?? []).some((f) => f.sessions === 20 && f.n > 0)); // a run saved before the forward excess existed is drawn by its plan
  return (
    <div className="quality">
      <Card title="Was the Advisor's score right?" right={<Sample small={small} />}>
        <div className="card-b">
          <CalibrationChart outcomes={outcomes} buckets={stats.by_score} minSample={minSample} benchmarkName={benchmark} />
          <p className="q-note muted">
            {forwardKept
              ? `Each dot is a strategy-family pick: its score across, how it did against ${benchmark} up, 20 sessions after the decision day whatever its plan did. The line is the average of each score band; * marks a band under ${minSample} picks. Companies scored neutral are in the family table.`
              : `Each dot is a recommendation: its score across, how it did against ${benchmark} up. The line is the average of each score band; * marks a band under ${minSample} outcomes.`}
          </p>
        </div>
      </Card>
      <Card title="Was the Critic right?" right={<Sample small={stats.critic_value.denied_n < minSample || stats.critic_value.approved_n < minSample} />}>
        <div className="card-b">
          <VerdictStrip outcomes={outcomes} groups={stats.by_verdict} benchmarkName={benchmark} />
          <p className="q-sentence">{stats.critic_value.sentence}</p>
        </div>
      </Card>
      <Card title="How did the trades end?" right={<Sample small={small} />}>
        <div className="card-b">
          <ExitMix outcomes={stats.outcomes} families={stats.by_family} benchmarkName={benchmark} />
        </div>
      </Card>
    </div>
  );
}

function Sample({ small }: { small: boolean }) {
  return small ? (
    <Badge tone="neutral" title="Fewer outcomes than the minimum sample: shown with no colour of success or failure">
      SMALL SAMPLE
    </Badge>
  ) : null;
}
