import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { replayEvents, type RunState } from "../state/runReducer";
import { ClockContext } from "../state/useNow";
import { bobEvents, dollarsEvents, gatesEvents, isQuestion, replayAccount, stageEvent, through } from "../test/fixtures";
import { ActivityFeed } from "./ActivityFeed";
import { Pipeline } from "./Pipeline";
import { PortfolioCard } from "./PortfolioCard";
import { RecommendationPanel } from "./RecommendationPanel";
import { WorkOrderTable } from "./WorkOrderTable";

const bobDone = replayEvents(bobEvents);

describe("Pipeline: every stage state, in the spec's words", () => {
  const stateClasses = () => within(screen.getByRole("region", { name: "Desk pipeline" })).getAllByRole("listitem").map((li) => li.className.replace("step ", ""));

  it("busy: the current stage carries aria-current and counts seconds from the server's timestamp", () => {
    const state = replayEvents(through(bobEvents, stageEvent("trade", "busy")));
    const busySince = state.stages[5]!.since!;
    render(
      <ClockContext.Provider value={busySince + 3.4}>
        <Pipeline state={state} />
      </ClockContext.Provider>,
    );
    expect(stateClasses()).toEqual(["done", "done", "done", "done", "done", "busy"]);
    const items = screen.getAllByRole("listitem");
    expect(items[5]?.getAttribute("aria-current")).toBe("step");
    expect(items.filter((li) => li.getAttribute("aria-current")).length).toBe(1);
    expect(within(items[5]!).getByText("3.4s")).toBeTruthy();
    expect(within(items[0]!).getByText(/Account Manager/)).toBeTruthy();
  });

  it("a direct order shows the stages it skips; a stage still ahead says nothing while the run goes on, and 'not reached' once the run has ended", () => {
    const state = replayEvents(through(gatesEvents, stageEvent("gather", "skipped"))); // work order done, gather skipped, the run still going
    const { unmount } = render(<Pipeline state={state} />);
    expect(stateClasses()).toEqual(["done", "skipped", "pending", "pending", "pending", "pending"]);
    expect(screen.queryAllByText(/· not reached$/).length).toBe(0);
    unmount();
    render(<Pipeline state={{ ...state, phase: "finished" }} />); // the same stages on a run that ended there
    expect(screen.getAllByText(/· not reached$/).length).toBe(4);
  });

  it("a question turns the busy stage into 'waiting' with a question mark, and the run box says so", () => {
    const base = replayEvents(through(bobEvents, stageEvent("trade", "busy")));
    const waiting: RunState = { ...base, stages: base.stages.map((s) => (s.state === "busy" ? { ...s, state: "waiting" as const } : s)) };
    render(<Pipeline state={waiting} />);
    expect(screen.getAllByRole("listitem")[5]?.textContent).toContain("?");
    expect(screen.getByText("Waiting for you", { exact: false })).toBeTruthy();
  });

  it("idle says Ready; finished says the result's own headline and the total time", () => {
    const { unmount } = render(<Pipeline state={replayEvents([])} />);
    expect(screen.getByText("READY")).toBeTruthy();
    unmount();
    render(<Pipeline state={bobDone} />);
    expect(screen.getByText("COMPLETED")).toBeTruthy();
    expect(screen.getByText(bobDone.result!.headline)).toBeTruthy();
    expect(screen.getByText("12 model calls · 26.6k tokens")).toBeTruthy();
  });
});

describe("PortfolioCard", () => {
  const changed = { changes: bobDone.portfolio!.changes, cash_change: bobDone.portfolio!.cash_change, cash_before: bobDone.portfolio!.cash_before };

  it("empty: says so, and the Cash row and the totals still show", () => {
    render(<PortfolioCard view={replayAccount} changed={null} before={null} />);
    expect(screen.getByText(/No holdings yet\./).closest("td")?.textContent).toBe("No holdings yet. Add funds and ask the desk to buy.");
    expect(screen.getByText("Cash")).toBeTruthy();
    expect(screen.getByText("Account value").nextSibling?.textContent).toBe("$15,000.00");
  });

  it("after a run: +8 NEW on each new holding, the cash it was, and the totals; EOG has no invented name", () => {
    render(<PortfolioCard view={{ ...bobDone.portfolio! }} changed={changed} before={bobDone.portfolioBefore} />);
    expect(screen.getAllByText("+8 NEW")).toHaveLength(2);
    expect(screen.getByText("was $15,000.00")).toBeTruthy();
    expect(screen.getByText("Total invested").nextSibling?.textContent).toBe("$2,632.59");
    const eog = screen.getByText("EOG").closest("tr")!;
    expect(eog.querySelector(".sub")?.textContent?.trim()).toBe("");
    expect(screen.getByText("ExxonMobil")).toBeTruthy();
  });

  it("a position the run sold out keeps its row with the loss", () => {
    const before = bobDone.portfolio!;
    const view = { ...before, positions: before.positions.filter((p) => p.symbol !== "XOM") };
    render(<PortfolioCard view={view} changed={{ changes: { XOM: -8 }, cash_change: 1300, cash_before: 12000 }} before={before} />);
    expect(screen.getByText("−8 SOLD")).toBeTruthy();
    expect(screen.getByText("ExxonMobil")).toBeTruthy();
  });
});

describe("WorkOrderTable", () => {
  it("shows the Critic's words, the order and the trade, dims a scored row and gives it one reason cell", () => {
    const onSelect = vi.fn();
    render(<WorkOrderTable state={bobDone} selectedKey="XOM" onSelect={onSelect} />);
    const grid = within(screen.getByRole("grid"));
    const xom = grid.getByRole("row", { name: /^1 XOM/ });
    expect(xom.textContent).toContain("APPROVE 86 · B");
    expect(xom.textContent).toContain("Buy 8");
    expect(xom.textContent).toContain("$1,364.72");
    expect(grid.getByRole("row", { name: /^2 APA/ }).textContent).toContain("DENY 49 · F");
    const cvx = grid.getByRole("row", { name: /^4 CVX/ });
    expect(cvx.className).toBe("dim");
    expect(cvx.querySelectorAll("td")).toHaveLength(4); // # | stock | advisor | the reason across the last three
    expect(cvx.querySelector("td[colspan='3']")?.textContent).toMatch(/^not recommended: /);
    expect(within(screen.getByText("Work order").closest("section")!).getByText("TRADED")).toBeTruthy();
  });

  it("arrow keys move the selection", () => {
    const onSelect = vi.fn();
    render(<WorkOrderTable state={bobDone} selectedKey="XOM" onSelect={onSelect} />);
    const xom = within(screen.getByRole("grid")).getByRole("row", { name: /^1 XOM/ });
    fireEvent.keyDown(xom, { key: "ArrowDown" });
    expect(onSelect).toHaveBeenLastCalledWith("APA");
    fireEvent.keyDown(within(screen.getByRole("grid")).getByRole("row", { name: /^3 EOG/ }), { key: "ArrowUp" });
    expect(onSelect).toHaveBeenLastCalledWith("APA");
  });

  it("empty before the Data Clerk has found anything", () => {
    render(<WorkOrderTable state={replayEvents([])} selectedKey={null} onSelect={() => undefined} />);
    expect(screen.getByText("No work order yet")).toBeTruthy();
  });
});

describe("RecommendationPanel", () => {
  const row = (events: typeof bobEvents, key: string) => replayEvents(events).workOrder!.rows.find((r) => r.key === key)!;
  const sectionTitles = () => Array.from(document.querySelectorAll(".sec h4")).map((h) => h.firstChild?.textContent);

  it("a recommended stock shows every section of the spec, in order, and the plan says the Advisor sets no size", () => {
    render(<RecommendationPanel row={row(bobEvents, "XOM")} runId="r" waitingForAdvisor={false} />);
    expect(sectionTitles()).toEqual(["Thesis", "Bull case", "Bear case", "Evidence", "How the score was built", "Key numbers", "Trade plan", "Critic review", "Order and fill"]);
    expect(screen.getByText("40 / 30 / 20 / 10")).toBeTruthy();
    expect(screen.getByText("prices and dates only: the Advisor sets no size")).toBeTruthy();
    expect(document.querySelectorAll(".plan > div")).toHaveLength(6);
    const review = screen.getByText("Critic review").closest(".sec")!;
    expect(review.querySelectorAll(".bar")).toHaveLength(6);
    expect(within(review as HTMLElement).getByText("APPROVE")).toBeTruthy();
    expect(screen.getByText("Rank 1 of 15 scored · trend strategy · Advisor score 73 of 100")).toBeTruthy();
    expect(screen.getByLabelText("XOM: Advisor score 73 of 100")).toBeTruthy();
  });

  it("a review that asked for a person is a DENIAL: the panel says needs_human_review and never 'needs your eyes' (entry 103)", () => {
    const xom = row(bobEvents, "XOM");
    const asked = { ...xom, detail: { ...xom.detail, review: { ...xom.detail.review!, approved: false, requires_human: true } } };
    render(<RecommendationPanel row={asked} runId="r" waitingForAdvisor={false} />);
    const review = screen.getByText("Critic review").closest(".sec") as HTMLElement;
    expect(within(review).getByText("DENY")).toBeTruthy();
    expect(within(review).getByText("needs_human_review")).toBeTruthy();
    expect(within(review).getByText(/a finding asks for a person to look, so the Critic denied it and nothing is traded for it/)).toBeTruthy();
    expect(screen.queryByText("NEEDS YOUR EYES")).toBeNull();
  });

  it("a version-3 review shows the three groups, bad or clean, and the analysts' counts in one line; an older review shows neither (Critic v3)", () => {
    const xom = row(bobEvents, "XOM");
    render(<RecommendationPanel row={xom} runId="r" waitingForAdvisor={false} />);
    expect(document.querySelector(".groups")).toBeNull(); // the capture's review was made before the groups: nothing is drawn for them
    expect(screen.queryByText("ANALYSTS")).toBeNull();
    cleanup();
    const groups = [
      { name: "news", label: "News", bad: false, codes: [] },
      { name: "performance", label: "Performance", bad: true, codes: ["trend_not_confirmed"] },
      { name: "price", label: "Price", bad: false, codes: [] },
    ];
    const analysts = "6 strong buy, 14 buy, 14 hold, 1 sell, 0 strong sell in 2026-11-01 (35 analysts): mean 3.71 of 5, down 0.04 since 2026-08-01";
    const v3 = { ...xom, detail: { ...xom.detail, review: { ...xom.detail.review!, groups, analysts } } };
    render(<RecommendationPanel row={v3} runId="r" waitingForAdvisor={false} />);
    const review = screen.getByText("Critic review").closest(".sec") as HTMLElement;
    const shown = within(review).getByRole("group", { name: /three groups of judgment/ });
    expect(Array.from(shown.querySelectorAll(".group")).map((g) => g.textContent)).toEqual(["NEWS CLEAN", "PERFORMANCE BAD trend_not_confirmed", "PRICE CLEAN"]);
    expect(shown.querySelectorAll(".badge.neg")).toHaveLength(1);
    expect(shown.querySelectorAll(".badge.pos")).toHaveLength(2);
    expect(within(review).getByText("ANALYSTS")).toBeTruthy();
    expect(within(review).getByText(analysts)).toBeTruthy();
  });

  it("a stock that was scored and not recommended has its reason and no thesis, plan or review", () => {
    render(<RecommendationPanel row={row(bobEvents, "CVX")} runId="r" waitingForAdvisor={false} />);
    expect(sectionTitles()).toContain("Not recommended");
    expect(sectionTitles()).not.toContain("Thesis");
    expect(sectionTitles()).not.toContain("Critic review");
    expect(sectionTitles()).not.toContain("Trade plan");
  });

  it("a direct order says it is the user's own: no recommendation and no Critic", () => {
    render(<RecommendationPanel row={row(gatesEvents, replayEvents(gatesEvents).workOrder!.rows[0]!.key)} runId="r" waitingForAdvisor={false} />);
    expect(screen.getByText("Your own order: no recommendation is made and the Critic is not asked.")).toBeTruthy();
    expect(sectionTitles()).toContain("Order and fill");
    expect(sectionTitles()).not.toContain("Critic review");
  });

  // Back on fixtures/dollars.json (a real run of 2026-11-16): the test that fixtures/pam.json carried until Jacob deleted it.
  it("a neutral pass says the Advisor found no strategy and that it counts as approved", () => {
    render(<RecommendationPanel row={row(dollarsEvents, replayEvents(dollarsEvents).workOrder!.rows.find((r) => r.symbol === "OXY")!.key)} runId="r" waitingForAdvisor={false} />);
    expect(screen.getByText(/The Advisor found no strategy for this company \(a neutral score\)\. A neutral recommendation counts as approved, so there is no Critic review\./)).toBeTruthy();
    expect(sectionTitles()).not.toContain("Critic review");
    expect(sectionTitles()).toContain("Order and fill");
  });

  it("an order in dollars says its dollars first and the shares as an estimate, and the fill says what was filled", () => {
    render(<RecommendationPanel row={row(dollarsEvents, replayEvents(dollarsEvents).workOrder!.rows.find((r) => r.symbol === "EOG")!.key)} runId="r" waitingForAdvisor={false} />);
    const order = screen.getByText("Order and fill").closest(".sec") as HTMLElement;
    expect(order.textContent).toContain("buy $6,000.00, about 37.897928 shares at $158.32");
    expect(order.textContent).toContain("37.897928 @ $158.3200 = $6,000.00");
  });

  it("with no row it asks to pick one, and says when the Advisor has not answered", () => {
    const { unmount } = render(<RecommendationPanel row={null} runId={null} waitingForAdvisor={false} />);
    expect(screen.getByText("Pick a stock")).toBeTruthy();
    unmount();
    render(<RecommendationPanel row={null} runId={null} waitingForAdvisor />);
    expect(screen.getByText("The Advisor has not answered yet.")).toBeTruthy();
  });
});

describe("ActivityFeed", () => {
  const rowOf = () => undefined;
  const answer = vi.fn().mockResolvedValue(undefined);

  it("says nothing yet before a run", () => {
    render(<ActivityFeed state={replayEvents([])} answer={answer} rowOf={rowOf} />);
    expect(screen.getByText("Nothing yet.")).toBeTruthy();
    expect(screen.getByRole("log")).toBeTruthy();
  });

  it("keeps a report block's line breaks in a monospace block, and colours each line by its agent", () => {
    render(<ActivityFeed state={bobDone} answer={answer} rowOf={rowOf} />);
    const block = Array.from(document.querySelectorAll("pre.block")).find((b) => b.textContent?.includes("yes (AUTO)"))!;
    expect(block.textContent?.split("\n").length).toBeGreaterThan(3);
    const decided = block.closest(".line")!;
    expect(decided.getAttribute("data-agent")).toBe("account_manager");
    expect(screen.getAllByText("Trader").length).toBeGreaterThan(0);
  });

  it("an open question pins its card at the bottom and the header says Waiting for you", () => {
    const asked = replayEvents(through(gatesEvents, isQuestion));
    render(<ActivityFeed state={asked} answer={answer} rowOf={rowOf} />);
    expect(screen.getByText("Waiting for you")).toBeTruthy();
    expect(screen.getByRole("group", { name: "Confirm what was understood" })).toBeTruthy();
  });
});

describe("ProfileCard: the trade mode is the request card's Mode switch, not this card's", () => {
  const actions = { create: vi.fn(async () => undefined), patch: vi.fn(async () => undefined), funds: vi.fn(async () => undefined) };

  it("has no Trades control on an account, and opens an account in the mode the switch shows", async () => {
    const { ProfileCard } = await import("./ProfileCard");
    const { unmount } = render(<ProfileCard view={replayAccount} running={false} readOnlyReason={null} loadError={null} actions={actions} newMode="auto" />);
    expect(screen.queryByText("Trades")).toBeNull();
    expect(screen.queryByRole("radiogroup")).toBeNull();
    expect(screen.getByLabelText("Risk level")).toBeTruthy();
    unmount();
    render(<ProfileCard view={null} running={false} readOnlyReason={null} loadError={null} actions={actions} newMode="auto" />);
    expect(screen.queryByText("Trades")).toBeNull();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Ada Lovelace" } });
    fireEvent.change(screen.getByLabelText("Age"), { target: { value: "41" } });
    fireEvent.click(screen.getByRole("button", { name: "Create account" }));
    expect(actions.create).toHaveBeenCalledWith({ name: "Ada Lovelace", age: 41, risk: "moderate", mode: "auto", cash: "10000.00" });
  });
});
