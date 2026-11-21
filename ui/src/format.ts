// Money, numbers and times (spec 5.6): Intl.NumberFormat en-US USD, local HH:MM:SS.

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const usd4 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 4, maximumFractionDigits: 4 });
const int = new Intl.NumberFormat("en-US");
const shareFormat = new Intl.NumberFormat("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 6 });

export function money(value: number | null | undefined): string {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : usd.format(value);
}

/** A fill price keeps its four decimals (the Trader's own precision). */
export function price4(value: number | null | undefined): string {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : usd4.format(value);
}

export function count(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : int.format(value);
}

/** A share quantity has at most 6 decimals (the server's `shares.PLACES`): round to them before comparing two quantities, never compare raw floats (design choices entry 110). */
export function round6(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

/** A number of shares as the server prints it (`shares.text`): thousands separators, up to 6 decimals, no trailing zero: 81, 1,234, 81.004455, 0.5. `count` would show 81.004 here (3 decimals). */
export function shares(value: number | null | undefined): string {
  return value === null || value === undefined || Number.isNaN(value) ? "—" : shareFormat.format(round6(value));
}

/** The same with a sign and a real minus sign: +0.5, −81 (a change in a holding). */
export function signedShares(value: number): string {
  const v = round6(value);
  return v > 0 ? `+${shares(v)}` : v < 0 ? `−${shares(-v)}` : "0";
}

/** +8, −5 (a real minus sign, as the spec writes it). */
export function signed(value: number): string {
  return value > 0 ? `+${int.format(value)}` : value < 0 ? `−${int.format(-value)}` : "0";
}

/** +0.49, −0.12: the score bars' values. */
export function signedDecimal(value: number, digits = 2): string {
  const text = Math.abs(value).toFixed(digits);
  return value > 0 ? `+${text}` : value < 0 ? `−${text}` : text;
}

function two(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

/** Server epoch seconds to the local HH:MM:SS. */
export function clock(epochSeconds: number | null | undefined): string {
  if (epochSeconds === null || epochSeconds === undefined) return "";
  const d = new Date(epochSeconds * 1000);
  return `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
}

/** An ISO UTC text ("2026-11-13T20:32:14Z") to the local HH:MM:SS. */
export function clockOfIso(iso: string | null | undefined): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  return Number.isNaN(t) ? "" : clock(t / 1000);
}

/** 133 seconds to 02:13. */
export function mmss(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${two(Math.floor(s / 60))}:${two(s % 60)}`;
}

/** 37.24 to 37.2s. */
export function secs(seconds: number): string {
  return `${Math.max(0, seconds).toFixed(1)}s`;
}

/** 12400 to 12.4k. */
export function tokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

export function percent(part: number, whole: number): number {
  return whole > 0 ? (100 * part) / whole : 0;
}

export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  return ((parts[0]?.[0] ?? "") + (parts.length > 1 ? (parts[parts.length - 1]?.[0] ?? "") : "")).toUpperCase();
}

/** "yes_auto" style codes and snake_case words to readable text. */
export function words(text: string): string {
  return text.replace(/_/g, " ");
}

export function cap(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// ---------------------------------------------------------------- the Proving Ground (a `_pct` is a FRACTION: 0.0077 is 0.77%)

const MINUS = "−";
const DASH = "—";

/** A fraction as a percent with a real minus sign: 0.0321 to "+3.2%", -0.1 to "−10.0%". `sign: false` leaves out the plus. */
export function pct(fraction: number | null | undefined, digits = 1, sign = true): string {
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return DASH;
  const text = (Math.abs(fraction) * 100).toFixed(digits);
  const zero = Number(text) === 0;
  return `${fraction < 0 && !zero ? MINUS : sign && fraction > 0 && !zero ? "+" : ""}${text}%`;
}

/** A difference of two fractions as points: 0.032 to "+3.2 pts" (percentage points). */
export function points(fraction: number | null | undefined, digits = 1): string {
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return DASH;
  const text = (Math.abs(fraction) * 100).toFixed(digits);
  const zero = Number(text) === 0;
  return `${fraction < 0 && !zero ? MINUS : fraction > 0 && !zero ? "+" : ""}${text} pts`;
}

/** An axis money: $9,800 under 10,000; $10.4k above; $1.2M. One style per axis is the caller's job. */
export function moneyAxis(value: number): string {
  const a = Math.abs(value);
  const sign = value < 0 ? MINUS : "";
  if (a >= 1_000_000) return `${sign}$${(a / 1_000_000).toFixed(a >= 10_000_000 ? 0 : 1)}M`;
  if (a >= 100_000) return `${sign}$${Math.round(a / 1000)}k`;
  if (a >= 10_000) return `${sign}$${(a / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return `${sign}$${int.format(Math.round(a))}`;
}

/** A rate that is 0 to 1 as a whole percent: 0.583 to "58%". */
export function rate(fraction: number | null | undefined): string {
  return fraction === null || fraction === undefined || Number.isNaN(fraction) ? DASH : `${Math.round(fraction * 100)}%`;
}

/** 133 seconds to "2m 13s"; under a minute "37.2s". */
export function duration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  return `${m}m ${two(Math.round(seconds - m * 60))}s`;
}
