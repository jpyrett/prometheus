import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { RunStartedData, ServerEvent } from "./api/types";
import { FakeEventSource, stubFetch } from "./test/fakes";
import { bobEvents, health, replayAccount, stageEvent, through } from "./test/fixtures";

beforeEach(() => {
  FakeEventSource.reset();
  vi.stubGlobal("EventSource", FakeEventSource);
});
afterEach(() => vi.unstubAllGlobals());

const RUN_ID = (bobEvents[0]!.data as RunStartedData).run_id; // the id the recorded stream carries

function feed(events: ServerEvent[]) {
  act(() => {
    for (const e of events) FakeEventSource.last.emit(e);
  });
}

describe("the page against the recorded replay of bob_jones_oil", () => {
  it("loads the replay account and the default prompt, posts Go!, plays the stream, shows the result and lets go of the stream", async () => {
    const calls = stubFetch({
      "GET /api/health": health,
      "GET /api/accounts/0": replayAccount,
      "POST /api/runs": { run_id: RUN_ID },
      [`GET /api/runs/${RUN_ID}/records/XOM`]: { recommendation: { symbol: "XOM" }, decision: { will_trade: "yes_auto" } },
    });
    render(<App />);

    // Idle: the replay's account, the pill, the default prompt, nothing run yet.
    const box = (await screen.findByRole("textbox", { name: "Your request" })) as HTMLTextAreaElement;
    expect(box.value).toBe(health.defaults.prompt);
    expect(screen.getByText("REPLAY")).toBeTruthy();
    expect(screen.getByText("PAPER TRADING · SIMULATED BROKER")).toBeTruthy();
    expect(await screen.findByText("Bob Jones")).toBeTruthy();
    expect(screen.getByText(/No holdings yet/)).toBeTruthy();
    expect(screen.getByText(/Nothing yet\./)).toBeTruthy();

    // Go! posts the payload and opens the stream of the run it was given.
    await waitFor(() => expect((screen.getByRole("button", { name: "Go!" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(calls.find((c) => c.method === "POST")?.body).toEqual({ account_id: 0, prompt: health.defaults.prompt, strict: false, neutral: true, fetch: false, raise_funds: false, mode: "auto" });
    expect(FakeEventSource.last.url).toBe(`/api/runs/${RUN_ID}/events?after=0`);

    // Mid-run: Running…, the trade stage is current, Activity fills.
    const midRun = through(bobEvents, stageEvent("trade", "busy"));
    feed(midRun);
    expect(screen.getByRole("button", { name: /Running/ })).toBeTruthy();
    const steps = within(screen.getByRole("region", { name: "Desk pipeline" })).getAllByRole("listitem");
    expect(steps[5]?.getAttribute("aria-current")).toBe("step");
    expect(steps[0]?.className).toContain("done");
    expect(within(screen.getByRole("log")).getByText(/read the request and made the work order/)).toBeTruthy();

    // Finished: the portfolio changes, the table shows the fills, the first row is selected and its recommendation shows.
    feed(bobEvents.slice(midRun.length));
    expect(FakeEventSource.last.closed).toBe(true);
    expect(screen.getAllByText("+8 NEW")).toHaveLength(2);
    expect(screen.getAllByText("$12,367.41").length).toBeGreaterThan(0);
    const table = within(screen.getByRole("grid", { name: "Stocks on the work order" }));
    const xom = table.getByRole("row", { name: /^1 XOM/ });
    expect(xom.getAttribute("aria-selected")).toBe("true");
    expect(within(xom).getByText("FILLED")).toBeTruthy();
    expect(screen.getByText("Rank 1 of 15 scored · trend strategy · Advisor score 73 of 100")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Go!" }) as HTMLButtonElement).disabled).toBe(false);

    // Selecting another row (the user's own choice) shows its recommendation; the Raw record fold reads the server's record.
    fireEvent.click(table.getByRole("row", { name: /^2 APA/ }));
    expect(table.getByRole("row", { name: /^2 APA/ }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("group", { name: "Recommendation for APA" })).toBeTruthy();
    fireEvent.click(xom);
    const fold = screen.getByText("Raw record").closest("details")!;
    fold.open = true; // what a click on the summary does; jsdom does not fire the toggle event itself
    fireEvent(fold, new Event("toggle"));
    await waitFor(() => expect(document.querySelector(".rawjson")?.textContent).toContain('"yes_auto"'));
  });
});
