// What the page says about time: the words of the two speeds, the desk's stages with their agent hues, and the seconds of a day. Every number is the server's (rule P12): a total of the days' seconds
// is a sum of figures the server sent, never a statistic of the page's own, and a share of a bar is drawn, never printed.
import type { PlanAnswer, RunTiming, Speed } from "./types";

export const SPEEDS: { value: Speed; label: string; what: string }[] = [
  { value: "fast", label: "Fast", what: "reasoning off for the Advisor" },
  { value: "faithful", label: "Faithful", what: "the live settings" },
];

export const SPEED_WORDS: Record<Speed, string> = { fast: "FAST", faithful: "FAITHFUL" };
export const SPEED_TITLE: Record<Speed, string> = {
  fast: "Made with the Advisor's reasoning switched off and the Critic's at a lower effort, so it finishes in minutes. The rules and the numbers are the same code as a faithful run.",
  faithful: "Made with the models asked exactly as the live desk asks them. It takes longer; the rules and the numbers are the same code as a fast run.",
};

/** The desk's stages in the order a day runs them, each with its agent's hue (the page's one use of the agent colours outside the Desk tab). */
export const STAGES: { key: string; label: string; color: string }[] = [
  { key: "account_manager", label: "Account Manager", color: "var(--am)" },
  { key: "data_clerk", label: "Data Clerk", color: "var(--clerk)" },
  { key: "advisor", label: "Advisor", color: "var(--advisor)" },
  { key: "critic", label: "Critic", color: "var(--critic)" },
];

/** The stages a day's `stages` object names, in desk order, with their seconds. A stage this page does not know is kept after the known ones, in grey. */
export function stagesOf(stages: Record<string, number> | null | undefined): { key: string; label: string; color: string; seconds: number }[] {
  if (!stages) return [];
  const known = STAGES.filter((s) => typeof stages[s.key] === "number").map((s) => ({ ...s, seconds: stages[s.key]! }));
  const others = Object.entries(stages)
    .filter(([k, v]) => !STAGES.some((s) => s.key === k) && typeof v === "number")
    .map(([k, v]) => ({ key: k, label: k.replace(/_/g, " "), color: "var(--neutral)", seconds: v }));
  return [...known, ...others];
}

const oneDecimal = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** A measured number of seconds as the page writes it: 12.35 is "12.4 s" (rounded as written, half up), 786.51 is "787 s". */
export function secondsText(seconds: number): string {
  return `${seconds >= 100 ? whole.format(seconds) : oneDecimal.format(seconds)} s`;
}

/** "90 s", "6 min": a rough duration for what the plan expects (to the nearest 5 seconds under two minutes, to the minute above). */
export function roughSeconds(seconds: number): string {
  if (seconds < 120) return `${Math.max(5, Math.round(seconds / 5) * 5)} s`;
  return `${Math.round(seconds / 60)} min`;
}

/** The seconds of a run's three phases (setup, research, walk), when the server measured them. A run saved before the speeds has research and walk only. */
export function phasesOf(timing: RunTiming | null | undefined): { key: "setup" | "research" | "walk"; label: string; seconds: number }[] {
  if (!timing) return [];
  const parts: { key: "setup" | "research" | "walk"; label: string; seconds: number | undefined }[] = [
    { key: "setup", label: "Setup", seconds: timing.setup_seconds },
    { key: "research", label: "Research", seconds: timing.research_seconds },
    { key: "walk", label: "Walk", seconds: timing.walk_seconds },
  ];
  return parts.filter((p): p is { key: "setup" | "research" | "walk"; label: string; seconds: number } => typeof p.seconds === "number");
}

/** What the plan line says of the time and the one line under each speed on the switch, from that speed's own plan answer. `null` when there is no answer for it yet. */
export function planTimeWords(plan: PlanAnswer | null): string | null {
  if (!plan) return null;
  const t = plan.timing;
  if (!t) return null;
  if (t.paid_days === 0 && t.cached_days > 0) return "from the cache, in seconds";
  if (t.estimate_seconds === null || t.estimate_seconds === undefined) return "no estimate yet";
  return `about ${roughSeconds(t.estimate_seconds)}`;
}
