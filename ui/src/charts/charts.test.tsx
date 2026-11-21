import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { stateOfRun } from "../proving/provingReducer";
import forwardRunFile from "../test/proving_forward_run.json";
import { coverageFixture, fastRun, livePlanSnapshot, trackRun } from "../test/proving";
import type { ProvingRun } from "../proving/types";
import { CalibrationChart } from "./CalibrationChart";
import { EquityChart } from "./EquityChart";
import { ExitMix } from "./ExitMix";
import { PlanChart } from "./PlanChart";
import { PlanLadder } from "./PlanLadder";
import { quickRange, RangePicker, pickableDays } from "./RangePicker";
import { dayNumber, isoOfDay } from "./scale";
import { VerdictStrip } from "./VerdictStrip";

// Every chart has a table twin (the same data as a `.tbl`): the tests open it and read the recorded numbers (the Track Record and the live snapshots of tests/fixtures/proving_real/).
const showTable = () => fireEvent.click(screen.getAllByRole("button", { name: /^(Show as table|Table)$/ })[0]!);
const cells = (row: HTMLElement) => within(row).getAllByRole("cell").map((c) => c.textContent);
const rowsOf = () => within(screen.getByRole("table")).getAllByRole("row").slice(1);

describe("the equity chart", () => {
  const s = stateOfRun(trackRun);
  const draw = (props: Partial<Parameters<typeof EquityChart>[0]> = {}) =>
    render(<EquityChart points={s.points} shown={s.points.length} sweeping={false} events={s.events} days={[]} research={false} startCash={100} domain={null} unit="index" maxDrawdown={s.stats!.portfolio?.max_drawdown_pct ?? null} {...props} />);

  it("is an image with its headline numbers, and its table lists every day of the curve as recorded", () => {
    draw();
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("went from 100 to 94.97 by 2026-11-19, SPY to 95.74");
    showTable();
    const rows = rowsOf();
    expect(rows).toHaveLength(trackRun.equity.length);
    expect(cells(rows[0]!).slice(0, 3)).toEqual(["2026-11-13", "99.98", "99.25"]);
    expect(cells(rows[4]!).slice(0, 3)).toEqual(["2026-11-19", "94.97", "95.74"]);
    expect(cells(rows[4]!)[5]).toBe("−5.0%"); // the server's drawdown_pct (a fraction), as a percent
  });

  // The real fast six-month run (tests/fixtures/proving_real/oil_weekly_6m_fast.json): 117 walk days, 23 decision days, 14 buys and 5 exits, 91 recommendations followed.
  const fast = stateOfRun(fastRun);
  const drawFast = (props: Partial<Parameters<typeof EquityChart>[0]> = {}) =>
    render(<EquityChart points={fast.points} shown={fast.points.length} sweeping={false} events={fast.events} days={fast.decisionDays.map((d) => ({ day: d.day, state: "done" as const, recommended: d.recommended, approved: d.approved, denied: d.denied, traded: d.traded, hasDesk: true }))} research={false} startCash={10000} domain={null} maxDrawdown={fast.stats!.portfolio!.max_drawdown_pct} {...props} />);

  it("a stock the desk did not buy is not in the equity table: only the buys and exits of the account are", () => {
    const bought = new Set(fastRun.events.map((e) => e.symbol));
    const onlyHypothetical = [...new Set(fastRun.outcomes.filter((o) => o.hypothetical).map((o) => o.symbol))].filter((s) => !bought.has(s));
    expect(onlyHypothetical.length).toBeGreaterThan(0);
    drawFast();
    showTable();
    const text = screen.getByRole("table").textContent!;
    expect(text).toContain("bought OXY");
    expect(text).toContain("sold (target) CVX");
    for (const symbol of onlyHypothetical) expect(text).not.toContain(` ${symbol} `);
  });

  it("marks every recorded buy and exit on the curve: one triangle under it for the buys of a day, one dot on it for the exits of a day and a reason, each with its count", () => {
    const { container } = drawFast();
    const buys = fastRun.events.filter((e) => e.kind === "buy");
    const exits = fastRun.events.filter((e) => e.kind === "exit");
    const groups = (kind: string) => [...container.querySelectorAll<HTMLElement>(".mk-group")].filter((g) => g.querySelector(`.mk.${kind}`));
    // the markers say every trade of the recording, grouped by day (and by reason for an exit)
    expect(groups("buy").reduce((n, g) => n + Number(g.dataset.count), 0)).toBe(buys.length);
    expect(groups("buy")).toHaveLength(new Set(buys.map((e) => e.date)).size);
    expect(groups("exit").reduce((n, g) => n + Number(g.dataset.count), 0)).toBe(exits.length);
    expect(groups("exit")).toHaveLength(new Set(exits.map((e) => `${e.date}|${e.reason}`)).size);
    expect(container.querySelectorAll(".mk.exit")).toHaveLength(groups("exit").length);
    expect([...container.querySelectorAll(".mk.exit")].every((m) => m.getAttribute("fill") === "var(--pos)")).toBe(true); // every exit of this run was a target
    // the three buys of 2026-06-09 are one triangle marked x3
    const first = groups("buy")[0]!;
    expect(first.dataset.count).toBe("3");
    expect(first.querySelector(".mk-count")!.textContent).toBe("×3");
  });

  it("puts every decision day on the axis as a tick that opens its desk, 23 of them, and still reads each one", () => {
    const open = vi.fn();
    drawFast({ onOpenDay: open });
    const ticks = within(screen.getByRole("group", { name: "Decision days" })).getAllByRole("button");
    expect(ticks).toHaveLength(23);
    fireEvent.click(ticks[0]!);
    expect(open).toHaveBeenCalledWith("2026-06-08");
    expect(ticks[0]!.getAttribute("aria-label")).toBe("Jun 8, 2026: decision day, open its desk");
  });

  it("the research strip says what each decision day is (a tile can be opened only when its desk exists), and the playhead shows while sweeping", () => {
    const days = s.points.slice(0, 4).map((p, i) => ({ day: p.date, state: (["done", "cached", "busy", "failed"] as const)[i]!, recommended: i < 2 ? 5 : 0, approved: i === 0 ? 3 : 0, denied: 0, traded: 0, hasDesk: i === 0 }));
    const open = vi.fn();
    const { container } = draw({ days, research: true, sweeping: true, onOpenDay: open });
    expect(container.querySelector(".pv-playhead")).toBeTruthy();
    const tiles = within(screen.getByRole("group", { name: "Research of each decision day" }));
    expect(tiles.getByRole("button", { name: /3 of 5 recommended stocks approved/ })).toBeTruthy();
    expect(tiles.getByRole("img", { name: /from the cache/ }) ?? null).toBeTruthy();
    expect(tiles.getByRole("img", { name: /being researched/ })).toBeTruthy();
    expect(tiles.getByRole("img", { name: /the research failed/ })).toBeTruthy();
    fireEvent.click(tiles.getByRole("button", { name: /3 of 5/ }));
    expect(open).toHaveBeenCalledWith(days[0]!.day);
  });

  it("a research tile of a wide chart says its approved over recommended and the seconds the day took, from the recorded timing", () => {
    const days = ["2026-06-08", "2026-06-15"].map((day) => {
      const d = fastRun.decision_days.find((x) => x.day === day)!;
      const t = fastRun.timing!.per_day![day]!;
      return { day, state: "done" as const, recommended: d.recommended, approved: d.approved, denied: d.denied, traded: d.traded, hasDesk: false, seconds: t.seconds, stages: t.stages, thinNews: day === "2026-06-15", articles: 2 };
    });
    const { container } = draw({ days, research: true, fallbackWidth: 1400, domain: { first: "2026-06-08", last: "2026-11-09" } });
    const tiles = container.querySelectorAll(".rs-tile");
    expect(tiles).toHaveLength(2);
    expect(tiles[0]!.querySelector(".rs-text")!.textContent).toBe("2/3");
    expect(tiles[0]!.querySelector(".rs-sub")!.textContent).toBe("73 s");
    expect(tiles[1]!.querySelector(".rs-thin")).toBeTruthy(); // few stored articles that day
    expect(tiles[1]!.getAttribute("aria-label")).toContain("few stored articles (2)");
    expect(tiles[0]!.querySelector(".rs-thin")).toBeNull();
  });

  it("its keyboard slider walks the days and says each one", () => {
    draw();
    const slider = screen.getByRole("slider", { name: /Move along the equity chart/ });
    fireEvent.focus(slider);
    expect(slider.getAttribute("aria-valuetext")).toContain("Nov 19, 2026");
    fireEvent.keyDown(slider, { key: "Home" });
    expect(slider.getAttribute("aria-valuetext")).toContain("Nov 13, 2026: desk 99.98");
  });
});

describe("the plan chart of an outcome", () => {
  it("lists the recorded path in its table, and a traded one is not drawn as not traded", () => {
    const o = trackRun.outcomes.find((x) => x.result === "won")!; // AMD, hit its target on 2026-11-19 (a stock the Critic denied: followed, not bought)
    const { container } = render(<PlanChart outcome={o} />);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("Exited at the target on Nov 19, 2026");
    expect(container.querySelector(".planchart.hypo")).toBeTruthy();
    showTable();
    const rows = rowsOf();
    expect(rows).toHaveLength(o.path.length);
    expect(cells(rows[0]!)).toEqual(["2026-11-16", "$504.2800", "+0.2%", "−1.3%"]);
    expect(cells(rows[3]!)[1]).toBe("$489.1900");
  });
  it("an outcome with no bar says why instead of drawing a path", () => {
    const o = trackRun.outcomes.find((x) => x.result === "no_data")!;
    render(<PlanChart outcome={o} />);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("No data:");
  });
});

describe("the quality charts", () => {
  const stats = trackRun.stats;
  it("the calibration table has a row for every score band, with n, the mean excess and the small-sample mark", () => {
    render(<CalibrationChart outcomes={trackRun.outcomes} buckets={stats.by_score} minSample={20} />);
    showTable();
    const rows = rowsOf().map(cells);
    expect(rows).toHaveLength(6);
    expect(rows[3]).toEqual(["60–69", "19", "−1.2 pts", "33%", "small sample (under 20)"]);
    expect(rows[0]).toEqual(["0–39", "0", "—", "—", "none"]);
  });
  it("the Critic's strip lists approved, denied and neutral passes with the server's n and means", () => {
    render(<VerdictStrip outcomes={trackRun.outcomes} groups={stats.by_verdict} />);
    showTable();
    const rows = rowsOf().map(cells);
    expect(rows.map((r) => [r[0], r[1]])).toEqual([["Approved", "12"], ["Denied", "9"], ["Neutral pass", "15"]]);
    expect(rows[0]![2]).toBe("−0.1 pts");
    expect(rows[0]![4]).toBe("small sample");
  });
  it("the exit mix lists the exits and the average win and loss as recorded", () => {
    render(<ExitMix outcomes={stats.outcomes} families={stats.by_family} />);
    showTable();
    const rows = rowsOf().map(cells);
    expect(rows.slice(0, 4).map((r) => [r[0], r[1]])).toEqual([["Target", "1"], ["Stop", "16"], ["Time stop", "0"], ["Still open", "19"]]);
    expect(rows[4]).toEqual(["Average win", "1", "+7.7%"]);
    expect(rows[5]).toEqual(["Average loss", "16", "−3.3%"]);
    expect(rows.some((r) => r[0] === "Family: trend" && r[2]!.includes("small sample"))).toBe(true);
  });
});

// The recorded fast six-month oil run walked again from its research cache with the code of design choices entry 139 (the forward excess, the new score buckets, the neutral row of the family table): its statistics as the server computed them, and its 91
// outcomes without the day-by-day path (src/test/proving_forward_run.json). Every number below is the server's.
const forwardRun = forwardRunFile as unknown as Pick<ProvingRun, "stats" | "outcomes">;

describe("the family and score tables read the forward excess", () => {
  const stats = forwardRun.stats;
  it("the family table lists the forward excess after each horizon with its n, standard error and the chance band, and the win rate is 'won by its plan'", () => {
    render(<ExitMix outcomes={stats.outcomes} families={stats.by_family} />);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("Exit mix. 74 trades followed: 41 target, 11 stop, 0 time stop, 22 still open");
    showTable();
    const rows = rowsOf().map(cells);
    const trend = rows.find((r) => r[0] === "Family: trend")!;
    expect(trend[1]).toBe("28"); // the plan's n
    expect(trend[2]).toContain("after the decision day, against SPY: 5 sessions +0.7 pts (se +0.7 pts, n 35, within chance); 10 sessions +1.3 pts (se +1.0 pts, n 35, within chance); 20 sessions +3.2 pts (se +1.4 pts, n 29); 40 sessions +8.7 pts (se +1.6 pts, n 19)");
    expect(trend[2]).toContain("won by its plan 68% of 28");
    expect(rows.map((r) => r[0]).filter((x) => x!.startsWith("Family:"))).toEqual(["Family: conviction", "Family: trend", "Family: neutral"]);
  });
  it("the family bars are the forward excess after 20 sessions, grey for a mean that chance alone gives", () => {
    const { container } = render(<ExitMix outcomes={stats.outcomes} families={stats.by_family} />);
    expect(container.textContent).toContain("Mean excess over SPY 20 sessions after the decision day, by strategy family");
    expect(container.textContent).toContain("n=29"); // the trend's forward n, not the plan's 28
    const bars = [...container.querySelectorAll("rect.hbar")].map((r) => r.getAttribute("fill"));
    expect(bars.slice(-3)).toEqual(["var(--pos)", "var(--pos)", "var(--neutral)"]); // conviction and trend beyond chance; neutral (+3.2 points, 1.7 standard errors) within it
  });
  it("the exit mix shows a trailing-stop part only when a trade ended by one", () => {
    const withTrailing = { ...stats.outcomes, exits: { ...stats.outcomes.exits, trailing_stop: 3 } };
    const { unmount } = render(<ExitMix outcomes={stats.outcomes} families={[]} />);
    expect(screen.queryByText(/Trailing stop/)).toBeNull();
    unmount();
    render(<ExitMix outcomes={withTrailing} families={[]} />);
    showTable();
    expect(rowsOf().map(cells).slice(0, 5).map((r) => r[0])).toEqual(["Target", "Stop", "Trailing stop", "Time stop", "Still open"]);
  });
  it("the score table lists the new buckets with the forward excess after 20 sessions, its n and standard error, and says which are within chance or small", () => {
    render(<CalibrationChart outcomes={forwardRun.outcomes} buckets={stats.by_score} minSample={20} />);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("picks by Advisor score against excess return over SPY 20 sessions after the decision day");
    showTable();
    const rows = rowsOf().map(cells);
    expect(rows.map((r) => r[0])).toEqual(["0–59", "60–64", "65–69", "70–74", "75–100"]);
    expect(rows[3]).toEqual(["70–74", "28", "+2.1 pts", "+1.4 pts", "82%", "within chance"]);
    expect(rows[0]).toEqual(["0–59", "0", "—", "—", "—", "none"]);
    expect(rows[2]![5]).toBe("small sample (under 20)");
  });
  it("a run saved before the forward excess existed is drawn by the plan, as it was", () => {
    const old = trackRun.stats;
    render(<CalibrationChart outcomes={trackRun.outcomes} buckets={old.by_score} minSample={20} />);
    showTable();
    expect(screen.getByRole("table").querySelectorAll("th")).toHaveLength(5);
  });
});

describe("the plan ladder of a trailing position", () => {
  // The recorded CVX position of the live snapshot with ONE change, the plan the trend family has since Advisor config version 3: its stop trails (a starting stop, the stop in force now), and it has no target.
  const recorded = livePlanSnapshot.positions.find((p) => p.symbol === "CVX")!;
  const trailing = livePlanSnapshot.positions.map((p) =>
    p.symbol === "CVX" ? { ...p, status: "in_plan" as const, plan: { stop: p.plan!.stop, target: null, time_stop_date: "2026-12-21", trailing: true, trail_distance: 9.8, highest_close: 229.5, stop_now: 219.7 } } : p,
  );
  it("draws the stop in force as a TRAILING stop, never the starting one, and says there is no target", () => {
    const { container } = render(<PlanLadder positions={trailing} />);
    const cvx = [...container.querySelectorAll(".lad-row")].find((r) => r.textContent!.includes("CVX"))!;
    expect(cvx.textContent).toContain("trailing stop $219.70");
    expect(cvx.textContent).not.toContain(`$${recorded.plan!.stop!.toFixed(2)}`);
    expect(cvx.textContent).toContain("no target");
    expect(cvx.querySelector(".lad-cap.target")).toBeNull();
    showTable();
    expect(rowsOf().map(cells).find((r) => r[0] === "CVX")![4]).toBe("$219.70");
  });
});

describe("the plan ladder", () => {
  it("lists every position against its plan: the stops that were hit say so", () => {
    render(<PlanLadder positions={livePlanSnapshot.positions} />);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("CVX: $209.02, stop $215.48, target $224.74, stop hit");
    showTable();
    const rows = rowsOf().map(cells);
    expect(rows.map((r) => [r[0], r[4], r[5], r[8]])).toEqual([
      ["CVX", "$215.48", "$224.74", "STOP HIT"],
      ["EOG", "$150.03", "$162.26", "STOP HIT"],
      ["XOM", "$169.38", "$174.94", "STOP HIT"],
    ]);
  });
});

describe("the plan ladder of a company the Advisor scored neutral", () => {
  // The recorded CVX position of the live snapshot with ONE change: the Advisor's exit plan of a neutral company is a stop and a time stop and no target (Advisor config version 2), so the target is gone and the time stop is a date.
  const recorded = livePlanSnapshot.positions.find((p) => p.symbol === "CVX")!;
  const neutral = livePlanSnapshot.positions.map((p) => (p.symbol === "CVX" ? { ...p, plan: { stop: p.plan!.stop, target: null, time_stop_date: "2026-12-21" } } : p));

  it("draws its stop and says the time stop and that there is no target, never 'no plan'", () => {
    const { container } = render(<PlanLadder positions={neutral} />);
    const cvx = [...container.querySelectorAll(".lad-row")].find((r) => r.textContent!.includes("CVX"))!;
    expect(cvx.textContent).toContain(`stop $${recorded.plan!.stop!.toFixed(2)}`);
    expect(cvx.textContent).toContain("no target");
    expect(cvx.textContent).toMatch(/time stop \w+ 21/);
    expect(cvx.textContent).not.toContain("no plan");
    expect(cvx.querySelector(".lad-cap.stop")).not.toBeNull();
    expect(cvx.querySelector(".lad-cap.target")).toBeNull();
  });

  it("keeps a position with no plan at all as it was", () => {
    const bare = livePlanSnapshot.positions.map((p) => ({ ...p, plan: null }));
    const { container } = render(<PlanLadder positions={bare} />);
    expect(container.textContent).toContain("no plan: price against cost only");
  });
});

describe("the coverage range picker", () => {
  const days = pickableDays(coverageFixture);
  const first = coverageFixture.usable!.first_day!;
  const last = coverageFixture.usable!.last_day!;

  it("its table is what is stored, month by month, as recorded", () => {
    render(<RangePicker coverage={coverageFixture} range={{ start: "2026-09-01", end: "2026-11-02" }} onChange={() => undefined} decisionDays={[]} />);
    showTable();
    const june = rowsOf().map(cells).find((r) => r[0] === "2026-06")!;
    expect(june).toEqual(["2026-06", "21", "21411", "193", "162"]);
  });

  it("the handles are sliders that cannot leave the usable range or cross each other", () => {
    const onChange = vi.fn();
    render(<RangePicker coverage={coverageFixture} range={{ start: first, end: "2026-09-01" }} onChange={onChange} decisionDays={[]} />);
    const start = screen.getByRole("slider", { name: "Start of the test range" });
    const end = screen.getByRole("slider", { name: "End of the test range" });
    expect(start.getAttribute("aria-valuetext")).toBe(first);
    // one step before the first usable day: nothing moves
    fireEvent.keyDown(start, { key: "ArrowLeft" });
    fireEvent.keyDown(start, { key: "Home" });
    expect(onChange).not.toHaveBeenCalled();
    // the end handle sent to the end of everything stops on the last usable day
    fireEvent.keyDown(end, { key: "End" });
    expect(onChange).toHaveBeenLastCalledWith({ start: first, end: last });
    // the start handle pushed past the end stays one pickable day before it
    onChange.mockClear();
    fireEvent.keyDown(start, { key: "End" });
    const moved = onChange.mock.calls[0]![0] as { start: string; end: string };
    expect(moved.end).toBe("2026-09-01");
    expect(dayNumber(moved.start)).toBeLessThan(dayNumber(moved.end));
    expect(days.includes(dayNumber(moved.start))).toBe(true);
  });

  it("quick ranges end on the last usable day and start inside the usable range", () => {
    const month = quickRange(days, 1)!;
    expect(month.end).toBe(last);
    expect(isoOfDay(dayNumber(month.start))).toBe(month.start);
    expect(dayNumber(month.start)).toBeGreaterThanOrEqual(dayNumber(first));
    expect(quickRange(days, null)).toEqual({ start: first, end: last });
    expect(quickRange(days, 600)!.start).toBe(first); // longer than there is: all there is
  });
});
