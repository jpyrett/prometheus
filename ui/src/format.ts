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
