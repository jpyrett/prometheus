// SVG path text for the charts: a line, an area, a step line. Pure; the numbers are pixels already (the scales did the work).

export type Point = readonly [number, number];
export type MaybePoint = readonly [number, number | null];

/** A pixel number as short text: two decimals, no trailing zeros ("12.5", "3"). */
export function n(value: number): string {
  return String(Math.round(value * 100) / 100);
}

/** M x y L x y ...; "" for no points. */
export function linePath(points: readonly Point[]): string {
  return points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${n(x)} ${n(y)}`).join("");
}

/** A line that breaks where a value is missing (null): one `M` per run of points that have a value. */
export function linePathWithGaps(points: readonly MaybePoint[]): string {
  let out = "";
  let pen = false;
  for (const [x, y] of points) {
    if (y === null || Number.isNaN(y)) {
      pen = false;
      continue;
    }
    out += `${pen ? "L" : "M"}${n(x)} ${n(y)}`;
    pen = true;
  }
  return out;
}

/** The area between a line and a horizontal baseline (a y in pixels), closed. "" for no points. */
export function areaPath(points: readonly Point[], baseline: number): string {
  if (points.length === 0) return "";
  const first = points[0]!;
  const last = points[points.length - 1]!;
  return `${linePath(points)}L${n(last[0])} ${n(baseline)}L${n(first[0])} ${n(baseline)}Z`;
}

/** A step line: the value holds until the next x ("after": the usual one for a bucket that runs from x to the next x). */
export function stepPath(points: readonly Point[], mode: "after" | "before" = "after"): string {
  let out = "";
  points.forEach(([x, y], i) => {
    if (i === 0) {
      out += `M${n(x)} ${n(y)}`;
      return;
    }
    const [px, py] = points[i - 1]!;
    out += mode === "after" ? `L${n(x)} ${n(py)}L${n(x)} ${n(y)}` : `L${n(px)} ${n(y)}L${n(x)} ${n(y)}`;
  });
  return out;
}

/** The area under a step line, closed on a baseline. */
export function stepAreaPath(points: readonly Point[], baseline: number): string {
  if (points.length === 0) return "";
  const first = points[0]!;
  const last = points[points.length - 1]!;
  return `${stepPath(points)}L${n(last[0])} ${n(baseline)}L${n(first[0])} ${n(baseline)}Z`;
}

/** An up-pointing triangle centred on (x, y), `size` pixels wide: the buy marker. */
export function triangleUp(x: number, y: number, size: number): string {
  const h = (size * 0.866);
  return `M${n(x)} ${n(y - (2 * h) / 3)}L${n(x + size / 2)} ${n(y + h / 3)}L${n(x - size / 2)} ${n(y + h / 3)}Z`;
}

/** A diamond centred on (x, y): the price marker of the plan ladder. */
export function diamond(x: number, y: number, r: number): string {
  return `M${n(x)} ${n(y - r)}L${n(x + r)} ${n(y)}L${n(x)} ${n(y + r)}L${n(x - r)} ${n(y)}Z`;
}
