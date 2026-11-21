// The scales of the Proving Ground's charts (spec 13): layout only. A scale turns a value into a pixel; it computes no statistic (P12).
// Pure: no DOM, no clock. Dates are ISO texts ("2026-08-10") and travel as whole days since 1970-01-01 (UTC), so a calendar gap is a gap on the axis.

export interface Scale {
  (value: number): number;
  domain: readonly [number, number];
  range: readonly [number, number];
  /** The value at a pixel (the pointer's x to a date, a handle's x to a day). */
  invert: (pixel: number) => number;
}

/** A linear scale. A domain of one point (min === max) maps everything to the middle of the range, so a flat series still draws. */
export function linear(domain: readonly [number, number], range: readonly [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  const scale = ((value: number) => (span === 0 ? (r0 + r1) / 2 : r0 + ((value - d0) / span) * (r1 - r0))) as Scale;
  scale.domain = domain;
  scale.range = range;
  scale.invert = (pixel: number) => (r1 === r0 ? d0 : d0 + ((pixel - r0) / (r1 - r0)) * span);
  return scale;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** [min, max] of the numbers that are numbers (null, undefined and NaN are skipped); null when there are none. */
export function extent(values: Iterable<number | null | undefined>): [number, number] | null {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return min === Infinity ? null : [min, max];
}

/** The step of a "nice" axis: 1, 2 or 5 times a power of ten, at most about `count` steps across the range. */
export function niceStep(span: number, count: number): number {
  if (!(span > 0) || count < 1) return 1;
  const raw = span / count;
  const exponent = Math.floor(Math.log10(raw));
  const fraction = raw / 10 ** exponent;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * 10 ** exponent;
}

function decimals(step: number): number {
  const text = String(step);
  if (text.includes("e-")) return Number(text.split("e-")[1]);
  const dot = text.indexOf(".");
  return dot < 0 ? 0 : text.length - dot - 1;
}

/** Round, clean ticks inside [min, max]: niceTicks(0, 100, 5) is 0, 20, 40, 60, 80, 100. A range of nothing gives the one value. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (max < min) [min, max] = [max, min];
  if (max === min) return [min];
  const step = niceStep(max - min, count);
  const places = decimals(step);
  const ticks: number[] = [];
  for (let t = Math.ceil(min / step - 1e-9) * step; t <= max + step * 1e-9; t += step) ticks.push(Number(t.toFixed(places)));
  return ticks;
}

/** The domain widened to its nice ticks, so an axis starts and ends on a labelled line. */
export function niceDomain(min: number, max: number, count = 5): [number, number] {
  if (max < min) [min, max] = [max, min];
  if (max === min) {
    const pad = Math.abs(min) * 0.01 || 1;
    return [min - pad, max + pad];
  }
  const step = niceStep(max - min, count);
  return [Math.floor(min / step + 1e-9) * step, Math.ceil(max / step - 1e-9) * step];
}

/** A domain padded by a fraction of its own span on each side, for a line that should not touch the frame. */
export function padDomain(domain: readonly [number, number], fraction: number): [number, number] {
  const pad = (domain[1] - domain[0]) * fraction || Math.abs(domain[0]) * fraction || fraction;
  return [domain[0] - pad, domain[1] + pad];
}

// ---------------------------------------------------------------- days

const DAY_MS = 86_400_000;

/** "2026-08-10" to whole days since 1970-01-01 (UTC). A text that is not a date gives NaN. */
export function dayNumber(iso: string): number {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!m) return NaN;
  return Math.round(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])) / DAY_MS);
}

export function isoOfDay(day: number): string {
  return new Date(day * DAY_MS).toISOString().slice(0, 10);
}

/** 0 = Sunday ... 6 = Saturday (a weekday is 1 to 5). */
export function weekdayOf(day: number): number {
  return new Date(day * DAY_MS).getUTCDay();
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function monthName(index: number): string {
  return MONTHS[index] ?? "";
}

/** "Aug 10", or "Aug 10, 2026" with `year`. */
export function shortDate(iso: string, year = false): string {
  const day = dayNumber(iso);
  if (Number.isNaN(day)) return iso;
  const d = new Date(day * DAY_MS);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}${year ? `, ${d.getUTCFullYear()}` : ""}`;
}

export interface TimeTick {
  day: number;
  label: string;
  /** A month start (or a year start): drawn a little stronger. */
  major: boolean;
}

/** Ticks for a span of days: month starts when the span is long, Mondays when it is short; at most `max` of them (every k-th kept when there would be more). */
export function timeTicks(first: number, last: number, max = 8): TimeTick[] {
  if (!(last >= first)) return [];
  const span = last - first;
  const all: TimeTick[] = [];
  if (span > 75) {
    const start = new Date(first * DAY_MS);
    let y = start.getUTCFullYear();
    let m = start.getUTCMonth();
    for (;;) {
      const day = Math.round(Date.UTC(y, m, 1) / DAY_MS);
      if (day > last) break;
      if (day >= first) all.push({ day, label: m === 0 ? String(y) : MONTHS[m]!, major: m === 0 });
      m += 1;
      if (m > 11) {
        m = 0;
        y += 1;
      }
    }
  } else if (span <= 14) {
    // a short stretch of days: every weekday is a tick
    for (let day = first; day <= last; day += 1) if (weekdayOf(day) >= 1 && weekdayOf(day) <= 5) all.push({ day, label: shortDate(isoOfDay(day)), major: false });
  } else {
    const step = span > 28 ? 14 : 7;
    for (let day = first; day <= last; day += 1) {
      if (weekdayOf(day) !== 1) continue;
      const index = all.length;
      if (index % (step / 7) === 0) all.push({ day, label: shortDate(isoOfDay(day)), major: false });
      else all.push({ day, label: "", major: false });
    }
  }
  if (all.length <= max) return all;
  const every = Math.ceil(all.length / max);
  return all.filter((_, i) => i % every === 0);
}

/** The index of the value in `sorted` (ascending) nearest to `x`; -1 for an empty list. */
export function nearestIndex(sorted: readonly number[], x: number): number {
  if (sorted.length === 0) return -1;
  let low = 0;
  let high = sorted.length - 1;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (sorted[mid]! < x) low = mid + 1;
    else high = mid;
  }
  if (low > 0 && Math.abs(sorted[low - 1]! - x) <= Math.abs(sorted[low]! - x)) return low - 1;
  return low;
}

// ---------------------------------------------------------------- the range picker's rules (the chart that picks a range)

export interface DayRange {
  start: number;
  end: number;
}

/**
 * Move one handle of the range to `day`: the day is snapped to one of `days` (the days that may be picked, ascending), kept inside the usable range
 * (days[0] to the last of them) and never lets the two handles cross (the start stays before the end by at least one pickable day). The other handle never moves.
 */
export function moveHandle(range: DayRange, which: "start" | "end", day: number, days: readonly number[]): DayRange {
  if (days.length < 2) return range;
  const snapped = days[nearestIndex(days, day)]!;
  const startIndex = nearestIndex(days, range.start);
  const endIndex = nearestIndex(days, range.end);
  if (which === "start") {
    const index = Math.min(nearestIndex(days, snapped), endIndex - 1);
    return { start: days[Math.max(0, index)]!, end: days[endIndex]! };
  }
  const index = Math.max(nearestIndex(days, snapped), startIndex + 1);
  return { start: days[startIndex]!, end: days[Math.min(days.length - 1, index)]! };
}

/** A handle moved by `steps` pickable days (the arrow keys); the same rules as `moveHandle`. */
export function stepHandle(range: DayRange, which: "start" | "end", steps: number, days: readonly number[]): DayRange {
  if (days.length < 2) return range;
  const from = nearestIndex(days, which === "start" ? range.start : range.end);
  const target = clamp(from + steps, 0, days.length - 1);
  return moveHandle(range, which, days[target]!, days);
}

/** The pickable days when the server sends none: every weekday between two days. (Holidays are the server's to refuse at the plan.) */
export function weekdaysBetween(first: number, last: number): number[] {
  const out: number[] = [];
  for (let d = first; d <= last; d += 1) if (weekdayOf(d) >= 1 && weekdayOf(d) <= 5) out.push(d);
  return out;
}
