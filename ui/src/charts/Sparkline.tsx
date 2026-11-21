// A sparkline: a small line with an end dot, in a KPI tile or a saved run's card. `values` are drawn as given (a second series, the benchmark, shares the same scale). Layout only (P12).
import { linePath } from "./path";
import { extent, linear } from "./scale";

export interface SparklineProps {
  values: number[];
  /** A second line on the same scale (the benchmark). */
  other?: number[] | null;
  width?: number;
  height?: number;
  color?: string;
  label: string;
}

export function Sparkline({ values, other, width = 92, height = 28, color = "var(--text-2)", label }: SparklineProps) {
  const range = extent([...values, ...(other ?? [])]);
  if (values.length < 2 || !range) {
    return <svg className="spark" width={width} height={height} role="img" aria-label={`${label}: nothing to draw yet`} />;
  }
  const pad = 4;
  const x = (i: number, of: number) => linear([0, Math.max(1, of - 1)], [pad, width - pad])(i);
  const y = linear(range, [height - pad, pad]);
  const line = linePath(values.map((v, i) => [x(i, values.length), y(v)] as const));
  const otherLine = other && other.length > 1 ? linePath(other.map((v, i) => [x(i, other.length), y(v)] as const)) : "";
  const end = values[values.length - 1]!;
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      {otherLine && <path d={otherLine} fill="none" stroke="var(--info)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.9" />}
      <path d={line} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={x(values.length - 1, values.length)} cy={y(end)} r={3} fill={color} stroke="var(--surface-1)" strokeWidth="2" />
    </svg>
  );
}
