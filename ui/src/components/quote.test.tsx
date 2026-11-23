import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { QuoteView, RunStartedData, ServerEvent } from "../api/types";
import { EVENT_TYPES } from "../api/types";
import { initialRunState, replayEvents, runReducer } from "../state/runReducer";
import { FakeEventSource, stubFetch } from "../test/fakes";
import { health, newsEvents, ofType, quoteEvents, replayAccount } from "../test/fixtures";
import { Pipeline } from "./Pipeline";

const quote = ofType(quoteEvents, "quote")[0]!.data as QuoteView & { ts: number };
const QUOTE_RUN = (quoteEvents[0]!.data as RunStartedData).run_id;
const PRICE_LINE = "TSLA  $388.42  +42.60 (+12.32%)  live quote, last trade 2026-11-23 15:26 ET";

afterEach(cleanup);

describe("the run state of a request that only asked for the current price (the real replay of the live session)", () => {
  it("knows the quote event, finishes with the status quote and keeps the price as a line of the output, with no work order and no news", () => {
    expect(EVENT_TYPES).toContain("quote");
    expect(quote.stocks).toHaveLength(1);
    expect(quote.stocks[0]).toMatchObject({ symbol: "TSLA", price: 388.42, change: 42.6, simulated: false, when: "live quote, last trade 2026-11-23 15:26 ET" });
    const state = replayEvents(quoteEvents);
    expect(state.phase).toBe("finished");
    expect(state.result?.status).toBe("quote");
    expect(state.result?.headline).toBe("TSLA $388.42 (+12.32%)");
    expect(state.workOrder).toBeNull();
    expect(state.news).toBeNull();
    expect(state.stages.map((s) => s.state)).toEqual(["done", "done", "pending", "pending", "pending", "pending"]);
    const blocks = state.log.filter((l) => l.block && l.agent === "data_clerk");
    expect(blocks.map((l) => l.text)).toEqual([PRICE_LINE]); // one line for the stock, written by the server (views.quote_lines)
    expect(state.log[state.log.length - 1]!.text).toBe("TSLA $388.42 (+12.32%)"); // the output ends with the price
  });

  it("ignores a quote event it has already applied and keeps the news of an earlier run viewable", () => {
    const state = replayEvents(newsEvents);
    const next = replayEvents(quoteEvents, state);
    expect(next.runId).toBe(QUOTE_RUN);
    expect(next.news).toBe(state.news); // the last news stays, as for any run
    const again = runReducer(next, ofType(quoteEvents, "quote")[0]!);
    expect(again).toBe(next);
    expect(runReducer(initialRunState, { seq: 9, type: "quote", data: quote } as ServerEvent).lastSeq).toBe(9);
  });
});

describe("the pipeline of a price run", () => {
  it("shows the Account Manager and the Data Clerk only once the run has finished, and all six stages while it goes on", () => {
    render(<Pipeline state={replayEvents(quoteEvents)} />);
    const items = within(screen.getByRole("region", { name: "Desk pipeline" })).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]!.textContent).toContain("Account Manager");
    expect(items[1]!.textContent).toContain("Data Clerk");
    cleanup();
    render(<Pipeline state={replayEvents(quoteEvents.slice(0, 4))} />);
    expect(within(screen.getByRole("region", { name: "Desk pipeline" })).getAllByRole("listitem")).toHaveLength(6);
  });
});

describe("the page: the price is said in the main output and the right column is not switched", () => {
  beforeEach(() => {
    FakeEventSource.reset();
    vi.stubGlobal("EventSource", FakeEventSource);
  });
  afterEach(() => vi.unstubAllGlobals());

  const feed = (events: ServerEvent[]) =>
    act(() => {
      for (const e of events) FakeEventSource.last.emit(e);
    });

  it("writes the price in the Activity output, shows the status in the run box, keeps the toggle on Work order and still lets the user open News", async () => {
    stubFetch({ "GET /api/health": health, "GET /api/accounts/0": replayAccount, "POST /api/runs": () => ({ run_id: QUOTE_RUN }) });
    render(<App />);
    await waitFor(() => expect((screen.getByRole("button", { name: "Go!" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    expect(screen.getByRole("tab", { name: "Work order" }).getAttribute("aria-selected")).toBe("true");
    feed(quoteEvents);
    // the main output: the stock's line, then the run's last line, the price
    const output = screen.getByRole("log", { name: "What the desk reports" });
    expect(within(output).getByText(PRICE_LINE.replace(/\s+/g, " "))).toBeTruthy(); // the text matcher collapses runs of spaces; the block keeps them
    expect(within(output).getByText("TSLA $388.42 (+12.32%)")).toBeTruthy();
    expect(screen.getByText("COMPLETED")).toBeTruthy();
    // no third view, no pane of its own: the toggle is News | Work order and the quote run did not switch it
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["News", "Work order"]);
    expect(screen.getByRole("tab", { name: "Work order" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.queryByRole("region", { name: "News summary and articles" })).toBeNull();
    expect(screen.getByText("No work order yet")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "News" }));
    expect(screen.getByText(/Ask for a stock's news/)).toBeTruthy();
  });

  it("does not take the user off the News view they opened when a price run starts after a news run", async () => {
    stubFetch({ "GET /api/health": health, "GET /api/accounts/0": replayAccount, "POST /api/runs": () => ({ run_id: QUOTE_RUN }) });
    render(<App />);
    await waitFor(() => expect((screen.getByRole("button", { name: "Go!" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    feed(quoteEvents.slice(0, 1));
    fireEvent.click(screen.getByRole("tab", { name: "News" }));
    feed(quoteEvents.slice(1));
    expect(screen.getByRole("tab", { name: "News" }).getAttribute("aria-selected")).toBe("true"); // the price event changes nothing in the right column
  });
});
