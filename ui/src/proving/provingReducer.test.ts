import { describe, expect, it } from "vitest";
import type { ServerEvent } from "../api/types";
import { rate } from "../format";
import { fastRun, shortRun, trackViewRun } from "../test/proving";
import { fastEvents, shortEvents } from "../test/provingStreams";
import { initialProvingState, provingReducer, replayProving, stateOfRun } from "./provingReducer";
import { kpis } from "./selectors";

describe("the reducer loaded with the Track Record (the route's view)", () => {
  const s = stateOfRun(trackViewRun);
  it("is finished, keeps the server's statistics and counts, and reads the curve as an index", () => {
    expect(s.phase).toBe("finished");
    expect(s.streamed).toBe(false);
    expect(s.stats).toEqual(trackViewRun.stats);
    expect(s.points).toHaveLength(trackViewRun.equity.length);
    expect(s.outcomes).toHaveLength(trackViewRun.outcomes.length);
    expect(s.unit).toBe("index");
    expect(s.caveats.map((c) => c.code)).toEqual(trackViewRun.caveat_codes);
    expect(s.caveats.every((c) => c.sentence.length > 20)).toBe(true);
  });
  it("shows the server's own counts in the recommendations tile and marks the win rate a small sample", () => {
    const tiles = kpis(s)!;
    const r = trackViewRun.kpis!.recommendations;
    expect(tiles.find((k) => k.key === "recs")!.sub).toBe(`${r.traded} traded · ${r.denied} denied · ${r.open} open`);
    const win = tiles.find((k) => k.key === "win")!;
    expect(win.small).toBe(true);
    expect(win.tone).toBe("neutral"); // no colour of success or failure under the minimum sample (P9)
    expect(win.value).toBe(rate(trackViewRun.stats.outcomes.win_rate));
  });
  it("reads where each recommendation came from off the outcome itself, as the source filter needs", () => {
    const own = Object.fromEntries(trackViewRun.outcomes.map((o) => [o.key, o.source]));
    expect(Object.values(own).every((v) => v === "run" || v === "fixture")).toBe(true);
    expect(s.sources).toEqual(own);
    // and the same from the older map, for a run saved before the outcomes said so
    const older = stateOfRun({ ...trackViewRun, outcomes: trackViewRun.outcomes.map((o) => ({ ...o, source: null })) });
    expect(older.sources).toEqual(own);
  });
  it("a run of enough outcomes carries no small-sample mark", () => {
    const big = stateOfRun({ ...trackViewRun, stats: { ...trackViewRun.stats, outcomes: { ...trackViewRun.stats.outcomes, small_sample: false } } });
    expect(kpis(big)!.find((k) => k.key === "win")!.small).toBe(false);
  });
});

describe("events that are not this reducer's are left alone", () => {
  it("a Desk run's start, log and finish change nothing while idle", () => {
    const desk = { seq: 1, type: "run_started", data: { run_id: "d1", mode: "live", account_id: 1, prompt: "x", options: { strict: false, neutral: true, fetch: false }, trade_mode: "auto", started_utc: "", stages: [], ts: 1 } } as ServerEvent;
    const log = { seq: 2, type: "log", data: { agent: "system", level: "info", text: "hello", block: false, ts: 2 } } as ServerEvent;
    expect(replayProving([desk, log])).toBe(initialProvingState);
    expect(provingReducer(initialProvingState, { type: "reset" })).toBe(initialProvingState);
  });
});

// The two REAL recorded streams (scripts/record_proving.mjs, from the replay server playing the saved runs of tests/fixtures/proving_real/): each must end where the saved run ends.
describe.each([
  ["the real fast six-month run (23 weekly decision days)", fastEvents, fastRun],
  ["the real short run (4 decision days, all from the cache)", shortEvents, shortRun],
])("the stream of %s", (_name, events, saved) => {
  const s = replayProving(events);
  it("ends in the recorded statistics, curve, markers, outcomes and days", () => {
    expect(s.phase).toBe("finished");
    expect(s.streamed).toBe(true);
    expect(s.savedId).toBe(saved.run_id);
    expect(s.stats).toEqual(saved.stats);
    expect(s.points).toEqual(saved.equity);
    expect(s.events).toEqual(saved.events);
    expect(s.decisionDays).toEqual(saved.decision_days);
    expect(s.outcomes.map((o) => o.key).sort()).toEqual(saved.outcomes.map((o) => o.key).sort());
    expect(s.plannedDays).toEqual(saved.decision_days.map((d) => d.day));
  });
  it("keeps what the server measured: the speed, the timing, the caveats", () => {
    expect(s.speed).toBe(saved.speed);
    // a run saved before the speeds sends `timing: {}` and keeps its phases in `coverage.timing`: the finished event carries them either way
    expect(s.timing).toEqual(Object.keys(saved.timing ?? {}).length > 0 ? saved.timing : (saved.coverage as { timing: unknown }).timing);
    expect(s.caveats.map((c) => c.code)).toEqual(saved.caveat_codes);
  });
  it("is the same state as the saved run opened whole, in what is drawn", () => {
    const whole = stateOfRun(saved);
    expect(s.points).toEqual(whole.points);
    expect(s.events).toEqual(whole.events);
    expect(s.stats).toEqual(whole.stats);
    expect(s.speed).toBe(whole.speed);
    expect(s.timing).toEqual(whole.timing); // the saved run opened whole reads the same timing as the stream carried
    expect(s.range).toEqual(whole.range);
  });
});

describe("the fast six-month run, as recorded", () => {
  const s = replayProving(fastEvents);
  it("says it was fast, from its start to its end, and carries the caveat that says so", () => {
    expect(replayProving(fastEvents.slice(0, 4)).speed).toBe("fast"); // run_started, log, pv_started, the first pv_research
    expect(s.speed).toBe("fast");
    expect(s.caveats.map((c) => c.code)).toContain("fast_models");
    expect(replayProving(shortEvents).caveats.map((c) => c.code)).not.toContain("fast_models");
  });
  it("has real buys and exits on its curve, in the order the walk made them", () => {
    expect(s.events.filter((e) => e.kind === "buy")).toHaveLength(14);
    expect(s.events.filter((e) => e.kind === "exit")).toHaveLength(5);
    expect(s.events.map((e) => e.date)).toEqual([...s.events.map((e) => e.date)].sort());
    expect(s.points).toHaveLength(117);
  });
  it("knows each decision day's seconds and its split by stage once the run has finished", () => {
    const day = s.timing!.per_day!["2026-06-08"]!;
    expect(day.seconds).toBe(73.26);
    expect(day.stages).toEqual({ account_manager: 0.17, data_clerk: 38.54, advisor: 12.35, critic: 22.19 });
    expect(s.research["2026-06-08"]!.stages).toEqual(day.stages);
  });
});
