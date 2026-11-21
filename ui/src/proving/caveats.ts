// The twelve caveats of design/proving_ground_spec.md section 8.4, `speed.py` and Critic v3 (the analysts' counts), each ONE sentence, shown by the page as written (P3: never hidden). The server sends the codes; the sentences are
// prometheus/proving/record.py `CAVEATS`, copied word for word, and caveats.test.ts reads that file and fails if a sentence here differs from it.
import type { CaveatCode, CaveatEntry, CaveatItem } from "./types";

export const CAVEAT_SENTENCES: Record<CaveatCode, string> = {
  fundamentals_not_point_in_time: "Company fundamentals (earnings, margins, ratios) are today's figures from Finnhub, not the figures known on each past decision day.",
  symbol_directory_is_todays: "Companies are found with today's symbol directory, so a company that was delisted or renamed since a decision day cannot be found.",
  price_returns_no_dividends: "Returns are price returns with no dividends, for the stocks and for the benchmark alike.",
  fills_at_next_open_no_slippage: "Every purchase fills at the next regular open of the stored daily bars, with no slippage and no commission.",
  daily_bars_stop_before_target: "Exits are judged on daily bars: a day that touches both the stop and the target counts as the stop, and on the day of the entry only the stop can fire.",
  news_coverage_thin: "On some decision days few stored articles were found for the companies, so the research of those days saw less than it would have live.",
  small_sample: "There are fewer outcomes than the minimum sample, so every rate and average is a first look, not a verdict.",
  open_positions_marked: "A position that no rule has closed is open: it is marked at the last stored close and is never counted as a win.",
  split_adjusted: "A position that crossed a stock split was adjusted on the split's date from the stored split table.",
  model_knowledge_cutoff: "The models know the world only up to their own training cutoff, which can be later than a past decision day: they may remember what happened next.",
  analysts_not_point_in_time: "The analysts' recommendation counts are each month's last revision as stored, not the counts that were known on the decision day.",
  fast_models: "The Advisor's model ran with its reasoning switched off and the Critic's at reduced reasoning effort to make this test fast; a faithful run uses the live settings and takes longer.",
};

/** The sentence of a code the server sent; a code this page does not know is shown as its own words rather than dropped. */
export function caveatSentence(code: string): string {
  return (CAVEAT_SENTENCES as Record<string, string>)[code] ?? code.replace(/_/g, " ");
}

/** The caveats of a run as `{code, sentence}`: the server's own sentence when it sent one, else the sentence of the code (the recorded Track Record sends codes only). Order and repeats are kept out: a code once. */
export function caveatItems(entries: CaveatEntry[] | undefined | null): CaveatItem[] {
  const seen = new Set<string>();
  const out: CaveatItem[] = [];
  for (const e of entries ?? []) {
    const item = typeof e === "string" ? { code: e, sentence: caveatSentence(e) } : { code: e.code, sentence: e.sentence || caveatSentence(e.code) };
    if (seen.has(item.code)) continue;
    seen.add(item.code);
    out.push(item);
  }
  return out;
}
