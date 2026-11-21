import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { duration } from "../format";
import { fastRun, shortRun, thinNewsRun } from "../test/proving";
import { fastEvents, shortEvents } from "../test/provingStreams";
import { OutcomeWall } from "./OutcomeWall";
import { replayProving, stateOfRun } from "./provingReducer";
import { RunView } from "./RunView";

// The REAL Time Machine runs of tests/fixtures/proving_real/, as the replay server plays them (scripts/record_proving.mjs): the fast six-month run (23 weekly decision days, 14 buys and 5 exits,
// 91 recommendations followed) and the short run (4 days, all from the cache, played at the live settings).
afterEach(() => vi.unstubAllGlobals());
const cells = (row: HTMLElement) => within(row).getAllByRole("cell").map((c) => c.textContent);

describe("the run view over the recorded fast six-month run", () => {
  const state = () => replayProving(fastEvents);

  it("draws the hero from the server's headline, with the speed beside it", () => {
    const s = state();
    render(<RunView state={s} />);
    expect(screen.getByRole("heading", { name: s.stats!.headline.sentence })).toBeTruthy();
    const hero = within(screen.getByRole("region", { name: "Result of the test" }));
    expect(hero.getByText("FAST")).toBeTruthy();
    const r = fastRun.kpis!.recommendations;
    expect(hero.getByText(`${r.traded} traded · ${r.denied} denied · ${r.open} open`)).toBeTruthy(); // the Recommendations tile: the server's own counts
  });

  it("the wall counts what the server counted: won, lost, open and not filled, bought or not", () => {
    render(<RunView state={state()} />);
    const wall = within(screen.getByRole("heading", { name: /Every recommendation/ }).closest("section")!);
    const o = fastRun.stats.outcomes;
    expect(wall.getByRole("button", { name: new RegExp(`^Won\\s*${o.won}$`) })).toBeTruthy();
    expect(wall.getByRole("button", { name: new RegExp(`^Lost\\s*${o.lost}$`) })).toBeTruthy();
    expect(wall.getByRole("button", { name: new RegExp(`^Open\\s*${o.open}$`) })).toBeTruthy();
    expect(wall.getByRole("button", { name: new RegExp(`^Not filled\\s*${o.not_filled}$`) })).toBeTruthy();
    const bought = fastRun.outcomes.filter((x) => x.traded).length;
    expect(wall.getByRole("button", { name: new RegExp(`^Bought\\s*${bought}$`) })).toBeTruthy();
    expect(wall.getByRole("button", { name: new RegExp(`^Not bought\\s*${fastRun.outcomes.length - bought}$`) })).toBeTruthy();
  });

  it("the equity table lists every walk day, and its Trades column says what was bought and sold on it, as recorded", () => {
    render(<RunView state={state()} />);
    fireEvent.click(screen.getAllByRole("button", { name: "Show as table" })[0]!);
    const rows = within(screen.getAllByRole("table")[0]!).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(fastRun.equity.length);
    const on = (date: string) => cells(rows.find((r) => cells(r)[0] === date)!)[6]!;
    expect(on("2026-06-09")).toBe("bought OXY 17.568517 sh; bought CVX 5.305039 sh; bought COP 8.474576 sh");
    expect(on("2026-06-10")).toBe("sold (target) OXY 17.568517 sh; sold (target) CVX 5.305039 sh");
    expect(on("2026-06-11")).toBe("");
  });

  it("opens the day's desk from its research tile, with the Desk's own table", () => {
    render(<RunView state={state()} />);
    fireEvent.click(screen.getByRole("button", { name: /^Jun 8, 2026: 2 of 3 recommended stocks approved/ }));
    const desk = within(screen.getByRole("region", { name: /The desk on 2026-06-08/ }));
    const oxy = desk.getByRole("row", { name: /OXY/ });
    expect(within(oxy).getByText("FILLED")).toBeTruthy();
  });

  it("a research tile says its seconds and opens the split by stage on hover, with the recorded numbers", () => {
    render(<RunView state={state()} />);
    const tile = screen.getByRole("button", { name: /^Jun 8, 2026: 2 of 3/ });
    expect(tile.getAttribute("aria-label")).toContain("researched in 73.3 s: Account Manager 0.2 s, Data Clerk 38.5 s, Advisor 12.4 s, Critic 22.2 s");
    fireEvent.pointerEnter(tile);
    const tip = document.querySelector(".chart-tip.below") as HTMLElement;
    expect(tip).toBeTruthy();
    expect(within(tip).getByText("38.5 s").parentElement!.textContent).toContain("Data Clerk");
    expect(within(tip).getByText("12.4 s").parentElement!.textContent).toContain("Advisor");
    expect(within(tip).getByText("22.2 s").parentElement!.textContent).toContain("Critic");
    expect(tip.querySelectorAll(".stagebar > span")).toHaveLength(4);
    fireEvent.pointerLeave(tile);
    expect(document.querySelector(".chart-tip.below")).toBeNull();
  });

  it("the timing card says the days at once, the days from the cache over all, and the per-model table of the recording", () => {
    render(<RunView state={state()} />);
    const card = within(screen.getByRole("heading", { name: "How long it took" }).closest("section")!);
    expect(card.getByText("Days researched at once").nextElementSibling!.textContent).toBe(String(fastRun.timing!.parallel_days));
    expect(card.getByText("Days from the cache").nextElementSibling!.textContent).toBe("0 of 23");
    expect(card.getByText(`${duration(fastRun.seconds)} wall time`)).toBeTruthy();
    fireEvent.click(card.getByRole("button", { name: "Show as table" }));
    const rows = within(card.getByRole("table")).getAllByRole("row").slice(1).map(cells);
    expect(rows).toContainEqual(["Phase", "Research", "77.24", "—", "—", "—", "—"]);
    expect(rows).toContainEqual(["Phase", "Setup", "9.62", "—", "—", "—", "—"]);
    expect(rows).toContainEqual(["Phase", "Walk", "1.45", "—", "—", "—", "—"]);
    expect(rows).toContainEqual(["Model", "openai/gpt-oss-120b", "75.95", "116", "0", "381,549", "65,299"]);
    expect(rows).toContainEqual(["Model", "nvidia/nemotron-3-super-120b-a12b", "248.05", "60", "0", "276,550", "51,022"]);
    expect(rows).toContainEqual(["Model", "z-ai/glm-5.3", "130.01", "63", "0", "96,039", "25,837"]);
    expect(rows).toContainEqual(["Stage, all days added", "Data Clerk", "786.51", "—", "—", "—", "—"]);
  });

  it("the playhead sweeps the long curve while it is drawn, and Skip to the end draws all of it at once", async () => {
    const s = state();
    const { container } = render(<RunView state={s} />);
    const deskPath = () => container.querySelector(".line.desk")?.getAttribute("d") ?? "";
    await waitFor(() => expect(container.querySelector(".pv-playhead")).toBeTruthy());
    const skip = screen.getByRole("button", { name: "Skip to the end" });
    expect(deskPath().split("L").length).toBeLessThan(s.points.length); // not yet all of the 117 days
    await act(async () => {
      fireEvent.click(skip);
    });
    expect(screen.queryByRole("button", { name: "Skip to the end" })).toBeNull();
    expect(container.querySelector(".pv-playhead")).toBeNull();
    expect(deskPath().split("L")).toHaveLength(s.points.length); // M, then one L for each of the other days
  });
});

describe("the outcome wall over the 91 recorded recommendations", () => {
  it("says why a stock was not bought, in the server's own reasons, and shows a recorded cooldown as the Advisor's own", () => {
    const { container } = render(<OutcomeWall outcomes={fastRun.outcomes} desks={{}} />);
    // show every card
    while (screen.queryByRole("button", { name: /^Show \d+ more/ })) fireEvent.click(screen.getByRole("button", { name: /^Show \d+ more/ }));
    const cards = [...container.querySelectorAll<HTMLElement>(".oc")];
    expect(cards).toHaveLength(fastRun.outcomes.length);
    const why = (re: RegExp) => cards.filter((c) => re.test(c.querySelector(".oc-why")?.textContent ?? "")).length;
    const reason = (r: string | null) => fastRun.outcomes.filter((o) => o.not_traded_reason === r).length;
    expect(why(/^not traded: the Advisor's cooldown held it back$/)).toBe(reason("cooldown"));
    expect(why(/^not traded: no cash was left$/)).toBe(reason("no_cash"));
    expect(why(/^not traded: denied by the Critic$/)).toBe(reason("denied"));
    expect(why(/^not traded: the order was not filled$/)).toBe(reason("not_filled"));
    // a card the desk bought is not drawn as a hypothetical one; a stock it did not buy is grey and dashed
    expect(cards.filter((c) => c.classList.contains("hypo"))).toHaveLength(fastRun.outcomes.filter((o) => o.hypothetical).length);
    for (const result of ["won", "lost", "open", "not_filled"] as const) expect(cards.filter((c) => c.dataset.result === result)).toHaveLength(fastRun.outcomes.filter((o) => o.result === result).length);
  });
});

describe("a recorded fast run with decision days of thin news", () => {
  const thin = ["2026-10-05", "2026-10-12", "2026-10-19"];
  it("shows the news_coverage_thin caveat as the server's sentence", () => {
    render(<RunView state={stateOfRun(thinNewsRun)} />);
    expect(thinNewsRun.caveat_codes).toContain("news_coverage_thin");
    const item = document.querySelector('[data-caveat="news_coverage_thin"]')!;
    expect(item.textContent).toBe(thinNewsRun.caveats.map((c) => (typeof c === "string" ? c : c.code === "news_coverage_thin" ? c.sentence : "")).join(""));
    expect(item.textContent).toMatch(/few stored articles/);
  });
  it("marks exactly those days on the research strip, with the number of articles that were found", () => {
    render(<RunView state={stateOfRun(thinNewsRun)} />);
    const strip = within(screen.getByRole("group", { name: "Research of each decision day" }));
    const found = (thinNewsRun.coverage as { news_sources: Record<string, number> }).news_sources;
    const marked = strip.getAllByRole("button").filter((b) => /few stored articles/.test(b.getAttribute("aria-label")!));
    expect(marked).toHaveLength(thin.length);
    for (const day of thin) expect(marked.some((b) => b.getAttribute("aria-label")!.includes(`few stored articles (${found[day]})`))).toBe(true);
    expect(document.querySelectorAll(".rs-tile.thin")).toHaveLength(thin.length);
  });
});

describe("the run view over the recorded short run (every day from the cache, at the live settings)", () => {
  const state = () => replayProving(shortEvents);

  it("says FAITHFUL, and draws the tiles as from the cache with no seconds", () => {
    render(<RunView state={state()} />);
    expect(within(screen.getByRole("region", { name: "Result of the test" })).getByText("FAITHFUL")).toBeTruthy();
    const strip = within(screen.getByRole("group", { name: "Research of each decision day" }));
    expect(strip.getAllByRole("button", { name: /from the cache/ })).toHaveLength(shortRun.decision_days.length);
    expect(strip.queryByText(/ s$/)).toBeNull();
  });

  it("a cached tile opens a note and no stage bar", () => {
    render(<RunView state={state()} />);
    fireEvent.pointerEnter(screen.getAllByRole("button", { name: /from the cache/ })[0]!);
    const tip = document.querySelector(".chart-tip.below") as HTMLElement;
    expect(tip.textContent).toContain("Nothing was asked of a model");
    expect(tip.querySelector(".stagebar")).toBeNull();
  });

  it("the timing card of a run saved before the speeds has its phases and no per-model figures", () => {
    render(<RunView state={state()} />);
    const card = within(screen.getByRole("heading", { name: "How long it took" }).closest("section")!);
    expect(card.getByText("Days from the cache").nextElementSibling!.textContent).toBe("4 of 4");
    fireEvent.click(card.getByRole("button", { name: "Show as table" }));
    const rows = within(card.getByRole("table")).getAllByRole("row").slice(1).map(cells);
    expect(rows).toEqual([["Phase", "Research", "0.02", "—", "—", "—", "—"], ["Phase", "Walk", "0.53", "—", "—", "—", "—"]]);
  });
});
