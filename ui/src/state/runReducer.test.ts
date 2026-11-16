import { describe, expect, it } from "vitest";
import type { ServerEvent } from "../api/types";
import { approveQuestion, bobEvents, dollarsEvents, gatesEvents, isQuestion, ofType, stageEvent, through } from "../test/fixtures";
import { initialRunState, replayEvents, runReducer } from "./runReducer";
import { shares as shareText } from "../format";
import { busyStage, orderWords, runBoxView, tradeWords, whatIsHappening } from "./selectors";

describe("runReducer over the recorded replay of bob_jones_oil (a research run in Auto mode)", () => {
  const final = replayEvents(bobEvents);

  it("ends with six done stages, the nodes that ran, and the run's own seconds", () => {
    expect(final.phase).toBe("finished");
    expect(final.stages.map((s) => [s.key, s.state, s.node])).toEqual([
      ["work_order", "done", "account_manager"],
      ["gather", "done", "data_clerk"],
      ["recommend", "done", "advisor"],
      ["critique", "done", "critic"],
      ["prepare", "done", "account_manager_decision"],
      ["trade", "done", "trader"],
    ]);
    const clerk = ofType(bobEvents, "stage").find((e) => e.data.key === "gather" && e.data.state === "done");
    expect(final.stages[1]?.seconds).toBe(clerk?.data.seconds);
    expect(final.stages.every((s) => s.since === null)).toBe(true);
  });

  it("keeps the whole work-order view of the last event: the three recommendations, then the scored rest", () => {
    const rows = final.workOrder?.rows ?? [];
    expect(rows.slice(0, 3).map((r) => [r.symbol, r.group, r.critic?.approved, r.trade?.status ?? null])).toEqual([
      ["XOM", "recommended", true, "filled"],
      ["APA", "recommended", false, null],
      ["EOG", "recommended", true, "filled"],
    ]);
    expect(rows[0]?.critic).toMatchObject({ score: 86, grade: "B" });
    expect(rows[1]?.decision?.will_trade).toBe("no_system_denied");
    expect(rows.slice(3).every((r) => r.group === "scored" && r.reason?.startsWith("not recommended: "))).toBe(true);
    expect(rows).toHaveLength(15);
    expect(final.workOrder?.status).toBe("traded");
  });

  it("starts from the account before the run and reports what changed", () => {
    expect(final.portfolioBefore?.cash).toBe(15000);
    expect(final.portfolioBefore?.positions).toEqual([]);
    expect(final.portfolio?.changes).toEqual({ EOG: 8, XOM: 8 });
    expect(final.portfolio?.cash).toBe(12367.41);
    expect(final.portfolio?.cash_change).toBeCloseTo(-2632.59, 2);
    expect(final.portfolio?.cash_before).toBe(15000);
  });

  it("keeps the last activity totals, the result and the log in order", () => {
    expect(final.activity).toMatchObject({ calls: 12, node: "trader" });
    expect(final.result?.status).toBe("traded");
    expect(final.result?.headline).toContain("$12,367.41");
    expect(final.log.map((l) => l.id)).toEqual([...final.log.map((l) => l.id)].sort((a, b) => a - b));
    expect(final.log.some((l) => l.block && l.agent === "trader")).toBe(true);
    expect(final.lastSeq).toBe(bobEvents[bobEvents.length - 1]?.seq);
  });

  it("counts from the server's own timestamps: the busy stage's `since` is its busy event's ts", () => {
    const upToTrader = replayEvents(through(bobEvents, stageEvent("trade", "busy")));
    const busy = busyStage(upToTrader);
    expect(busy?.key).toBe("trade");
    expect(busy?.since).toBe(ofType(bobEvents, "stage").find((e) => e.data.key === "trade" && e.data.state === "busy")?.data.ts);
    expect(upToTrader.phase).toBe("running");
    expect(whatIsHappening(upToTrader)).toBe("Trader is placing trade 1 of 2");
    expect(runBoxView(upToTrader, (upToTrader.startedTs ?? 0) + 133).timer).toBe("02:13");
  });

  it("ignores an event it already has (a reconnect replays) and is pure", () => {
    const again = replayEvents(bobEvents.slice(0, 10), final);
    expect(again).toBe(final);
    const a = replayEvents(bobEvents.slice(0, 20));
    const b = replayEvents(bobEvents.slice(0, 20));
    expect(b).toEqual(a);
    expect(runReducer(initialRunState, bobEvents[1] as ServerEvent).lastSeq).toBe(2);
  });

  it("a run_started of ANOTHER run replaces everything", () => {
    const other = { ...bobEvents[0]!, seq: 1, data: { ...(bobEvents[0]!.data as object), run_id: "another" } } as ServerEvent;
    const restarted = runReducer(final, other);
    expect(restarted.phase).toBe("running");
    expect(restarted.log).toEqual([]);
    expect(restarted.workOrder).toBeNull();
    expect(restarted.stages.every((s) => s.state === "pending")).toBe(true);
  });

  it("words the Trade column: FILLED, NO TRADE, and PLACING… while the trader works", () => {
    const rows = final.workOrder!.rows;
    expect(tradeWords(rows[0]!, final)?.text).toBe("FILLED");
    expect(tradeWords(rows[1]!, final)?.text).toBe("NO TRADE");
    const firstFill = through(bobEvents, (e) => e.type === "portfolio" && Object.keys(e.data.changes).length > 0);
    const mid = replayEvents(firstFill.slice(0, firstFill.findIndex((e) => e.type === "work_order" && e.data.rows[0]?.trade?.status === "filled") + 1)); // XOM is filled, EOG is next
    expect(tradeWords(mid.workOrder!.rows[0]!, mid)?.text).toBe("FILLED");
    expect(tradeWords(mid.workOrder!.rows[2]!, mid)?.text).toBe("PLACING…");
  });
});

describe("runReducer over a direct order in Ask mode (bill_gates_direct_order_funded)", () => {
  const confirmAt = gatesEvents.findIndex(isQuestion);

  it("opens the confirm question and closes it with the server's question_closed", () => {
    const open = replayEvents(gatesEvents.slice(0, confirmAt + 1));
    expect(open.question).toMatchObject({ kind: "confirm", title: "Confirm what was understood" });
    expect(open.question?.text.endsWith("Go ahead?")).toBe(true);
    const closed = replayEvents(gatesEvents.slice(0, confirmAt + 2), open);
    expect(closed.question).toBeNull();
    expect(ofType(gatesEvents, "question_closed")[0]?.data).toMatchObject({ answer: true, by: "you" });
  });

  it("skips the three stages a direct order never runs and shows a direct row", () => {
    const final = replayEvents(gatesEvents);
    expect(final.stages.map((s) => s.state)).toEqual(["done", "skipped", "skipped", "skipped", "done", "done"]);
    expect(final.workOrder?.kind).toBe("direct");
    expect(final.workOrder?.rows[0]).toMatchObject({ symbol: "MSFT", group: "direct", advisor: null, critic: null });
    expect(final.result?.status).toBe("traded");
  });

  it("marks the busy stage as waiting while a question is open, and busy again after", () => {
    const withQuestion = replayEvents(through(bobEvents, stageEvent("trade", "busy")));
    const next = withQuestion.lastSeq + 1;
    const asked = runReducer(withQuestion, { seq: next, type: "question", data: { ...approveQuestion, ts: 0 } });
    expect(asked.stages.find((s) => s.key === "trade")?.state).toBe("waiting");
    expect(runBoxView(asked, 0).label).toBe("WAITING FOR YOU");
    const closed = runReducer(asked, { seq: next + 1, type: "question_closed", data: { question_id: approveQuestion.question_id, answer: true, by: "you", ts: 0 } });
    expect(closed.question).toBeNull();
    expect(closed.stages.find((s) => s.key === "trade")?.state).toBe("busy");
    expect(closed.stages.find((s) => s.key === "trade")?.since).toBe(withQuestion.stages.find((s) => s.key === "trade")?.since);
  });
});

// Back on fixtures/dollars.json (a real run of 2026-11-16 under today's rules): the test that fixtures/pam.json carried until Jacob deleted it ("has neutral rows whose decision says where they came from").
describe("runReducer over a neutral pass: has neutral rows whose decision says where they came from", () => {
  const final = replayEvents(dollarsEvents);

  it("the two neutral companies that filled the number are neutral rows with no Critic, and their decision says origin neutral", () => {
    const neutral = (final.workOrder?.rows ?? []).filter((r) => r.group === "neutral");
    expect(neutral.map((r) => r.symbol)).toEqual(["OXY", "META"]);
    expect(neutral.every((r) => r.critic === null || r.critic === undefined)).toBe(true);
    expect(neutral.map((r) => [r.decision?.origin, r.decision?.will_trade, r.decision?.dollars, r.trade?.status])).toEqual([
      ["neutral", "yes_auto", 6000, "filled"],
      ["neutral", "yes_auto", 6000, "filled"],
    ]);
  });

  it("every purchase is an order of $6,000.00 and the five fills add up to the budget of $30,000.00, not the cash of $40,000.00", () => {
    const rows = (final.workOrder?.rows ?? []).filter((r) => r.decision?.will_trade === "yes_auto");
    expect(rows.map((r) => r.symbol)).toEqual(["EOG", "XOM", "CVX", "OXY", "META"]);
    expect(rows.map((r) => orderWords(r))).toEqual(rows.map((r) => ({ text: "Buy $6,000.00", dollars: `about ${shareText(r.decision!.shares)} sh` })));
    expect(rows.reduce((sum, r) => sum + (r.trade?.amount ?? 0), 0)).toBe(30000);
    expect(final.portfolio?.cash).toBe(10000);
    expect(final.portfolio?.cash_before).toBe(40000);
  });
});
