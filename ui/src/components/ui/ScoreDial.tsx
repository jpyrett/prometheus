/** The Advisor's score as a ring: the conic gradient is filled to the score, the number sits in the middle. */
export function ScoreDial({ score, label }: { score: number; label?: string }) {
  const v = Math.max(0, Math.min(100, score));
  return (
    <div className="dial" style={{ ["--v" as string]: v }} role="img" aria-label={label ?? `Advisor score ${score} of 100`}>
      <b>{Math.round(score)}</b>
    </div>
  );
}
