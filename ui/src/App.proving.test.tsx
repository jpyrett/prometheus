import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { ServerEvent } from "./api/types";
import { shortDate } from "./charts/scale";
import { FakeEventSource, stubFetch } from "./test/fakes";
import { bobEvents, health, ofType, replayAccount, stageEvent, through } from "./test/fixtures";
import { coverageFixture, fastRun, planFaithfulFixture, planFaithfulRequest, planFixture, planRequest, trackViewRun } from "./test/proving";
import { fastEvents } from "./test/provingStreams";

// The page's tabs and the Proving Ground's three views, over the recorded answers of the server (fetch and EventSource are the only stand-ins: the browser's transport).
beforeEach(() => {
  FakeEventSource.reset();
  vi.stubGlobal("EventSource", FakeEventSource);
  window.location.hash = "";
  try {
    localStorage.setItem("prometheus.account", "1");
  } catch {
    /* no storage */
  }
});
afterEach(() => {
  vi.unstubAllGlobals();
  window.location.hash = "";
  localStorage.clear();
});

const live = { ...health, mode: "live" as const, clock_note: "", clock_pill: "" };
const routes = {
  "GET /api/health": live,
  "GET /api/accounts/1": replayAccount,
  "GET /api/proving/coverage": coverageFixture,
  "GET /api/proving/runs": [],
  // the server's own answer for each speed (recorded): the same request, planned fast and planned faithful
  "POST /api/proving/plan": (body: unknown) => ((body as { speed?: string }).speed === "faithful" ? planFaithfulFixture : planFixture),
  "GET /api/proving/track": trackViewRun,
};
const launch = () => screen.getByRole("button", { name: /Launch|Running/ }) as HTMLButtonElement;
const feed = (events: ServerEvent[]) => act(() => events.forEach((e) => FakeEventSource.last.emit(e)));

describe("the tabs (spec 13.1)", () => {
  it("opens on the Desk, which is untouched; the Proving Ground is a tab beside it and its address is the hash", async () => {
    stubFetch(routes);
    render(<App />);
    expect(await screen.findByRole("textbox", { name: "Your request" })).toBeTruthy();
    const nav = screen.getByRole("navigation", { name: "Sections" });
    expect(within(nav).getByRole("link", { name: "Desk" }).getAttribute("aria-current")).toBe("page");
    expect(screen.queryByRole("navigation", { name: "Proving Ground views" })).toBeNull(); // not even mounted before its first visit

    fireEvent.click(within(nav).getByRole("link", { name: /Proving Ground/ }));
    expect(window.location.hash).toBe("#proving");
    expect(await screen.findByRole("navigation", { name: "Proving Ground views" })).toBeTruthy();
    expect(screen.queryByRole("textbox", { name: "Your request" })).toBeNull(); // the Desk is hidden, not gone
    fireEvent.click(within(nav).getByRole("link", { name: "Desk" }));
    expect(await screen.findByRole("textbox", { name: "Your request" })).toBeTruthy();
    expect(window.location.hash).toBe("");
  });

  it("the PAPER TRADING pill stays on the Proving Ground, and no control on it sends an order", async () => {
    stubFetch(routes);
    window.location.hash = "#proving/live";
    render(<App />);
    expect(screen.getByText("PAPER TRADING · SIMULATED BROKER")).toBeTruthy();
    const view = await screen.findByRole("navigation", { name: "Proving Ground views" });
    expect(view).toBeTruthy();
    for (const name of [/sell/i, /place/i, /buy/i, /order/i]) expect(screen.queryByRole("button", { name })).toBeNull();
  });
});

describe("the busy rule between the two kinds of run", () => {
  it("Launch is disabled, with its reason, while a desk run is active", async () => {
    const calls = stubFetch({ ...routes, "POST /api/runs": { run_id: "d1" } });
    render(<App />);
    await waitFor(() => expect((screen.getByRole("button", { name: "Go!" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    feed(through(bobEvents, stageEvent("trade", "busy")));

    fireEvent.click(screen.getByRole("link", { name: /Proving Ground/ }));
    await waitFor(() => expect(launch()).toBeTruthy());
    expect(launch().disabled).toBe(true);
    expect(screen.getByText(/A desk run is active/)).toBeTruthy();
    expect(within(screen.getByRole("navigation", { name: "Sections" })).getByTitle("a desk run is active")).toBeTruthy(); // the Desk tab shows its run is going
    expect(calls.some((c) => c.method === "POST" && c.path === "/api/proving/runs")).toBe(false);
  });

  it("the Desk's Go! is disabled, with its reason, while a Time Machine test runs", async () => {
    stubFetch({ ...routes, "POST /api/proving/runs": { run_id: "p1" } });
    window.location.hash = "#proving";
    render(<App />);
    await waitFor(() => expect(launch().disabled).toBe(false), { timeout: 4000 });
    fireEvent.click(launch());
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    feed([fastEvents[0]!]); // the real run_started of a Time Machine run

    fireEvent.click(screen.getByRole("link", { name: "Desk" }));
    const go = await screen.findByRole("button", { name: "Go!" });
    expect((go as HTMLButtonElement).disabled).toBe(true);
    expect(go.getAttribute("title")).toContain("Proving Ground test is running");
  });
});

describe("the Time Machine's launch pad and run", () => {
  it("opens on the warm request, asks for the plan (free), launches it as it stands, and draws the research strip as the days come back", async () => {
    const calls = stubFetch({ ...routes, "POST /api/proving/runs": { run_id: "p1" } });
    window.location.hash = "#proving";
    render(<App />);
    await waitFor(() => expect(launch().disabled).toBe(false), { timeout: 4000 });
    // the plan line, in the server's words: 23 decision days, every one of them from the cache, in seconds
    const line = document.querySelector(".lp-plan")!.textContent!;
    expect(line).toContain(`${planFixture.decision_days.length} decision days, ${planFixture.cached_days.length} from the cache, 0 paid desk runs`);
    expect(line).toContain("from the cache, in seconds");
    const planCall = calls.find((c) => c.path === "/api/proving/plan" && (c.body as { speed: string }).speed === "fast")!;
    expect(planCall.body).toEqual(planRequest);

    fireEvent.click(launch());
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/api/proving/runs")).toBe(true));
    expect(calls.find((c) => c.path === "/api/proving/runs" && c.method === "POST")!.body).toEqual(planRequest);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(FakeEventSource.last.url).toBe("/api/runs/p1/events?after=0");

    // the stream's first events, as recorded: the run's start, its plan (the same days as the plan route named), and the first three days' research
    const research = ofType(fastEvents, "pv_research").slice(0, 3);
    const started = ofType(fastEvents, "pv_started")[0]!;
    expect(started.data.decision_days).toEqual(planFixture.decision_days);
    feed([fastEvents[0]!, started, ...research]);
    const strip = await screen.findByRole("group", { name: "Research of each decision day" });
    for (const r of research) expect(within(strip).getByRole("img", { name: new RegExp(`^${shortDate(r.data.day, true)}: ${r.data.approved} of ${r.data.recommended} recommended stocks approved`) })).toBeTruthy();
    expect(within(strip).getAllByRole("img", { name: /being researched/ })).toHaveLength(planFixture.decision_days.length - research.length);
    expect(screen.getByText(`Researching ${planFixture.decision_days.length} decision days, ${started.data.parallel_days} at a time: ${research.length} done`, { exact: false })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Stop the test" })).toBeTruthy();
  });

  it("the speed switch sends the speed with the plan and with Launch, and the plan line follows it", async () => {
    const calls = stubFetch({ ...routes, "POST /api/proving/runs": { run_id: "p1" } });
    window.location.hash = "#proving";
    render(<App />);
    await waitFor(() => expect(launch().disabled).toBe(false), { timeout: 4000 });
    const fast = screen.getByRole("radio", { name: /^Fast/ });
    const faithful = screen.getByRole("radio", { name: /^Faithful/ });
    expect(fast.getAttribute("aria-checked")).toBe("true");
    // each side says its own time, from its own plan answer (the other speed is planned too, free)
    await waitFor(() => expect(faithful.textContent).toContain("about 6 min · the live settings"));
    expect(fast.textContent).toContain("from the cache, in seconds · reasoning off for the Advisor");

    const planCalls = () => calls.filter((c) => c.path === "/api/proving/plan").length;
    const before = planCalls();
    fireEvent.click(faithful);
    // both speeds were planned already: the plan line follows the switch at once, and the switch asks for nothing new
    expect(document.querySelector(".lp-plan")!.textContent).toContain("about 6 min");
    await new Promise((r) => setTimeout(r, 450));
    expect(planCalls()).toBe(before);
    const line = document.querySelector(".lp-plan")!.textContent!;
    expect(line).toContain(`${planFaithfulFixture.paid_days} paid desk runs`);
    expect(line).toContain(`about $${planFaithfulFixture.cost_estimate_usd}`);
    const asked = calls.filter((c) => c.path === "/api/proving/plan" && (c.body as { speed: string }).speed === "faithful");
    expect(asked.length).toBeGreaterThan(0);
    expect(asked.at(-1)!.body).toEqual(planFaithfulRequest);

    await waitFor(() => expect(launch().disabled).toBe(false));
    fireEvent.click(launch());
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/api/proving/runs")).toBe(true));
    expect(calls.find((c) => c.path === "/api/proving/runs" && c.method === "POST")!.body).toEqual(planFaithfulRequest);
  });

  it("the first quick prompt fills the whole pad with the request whose research is warm, whatever was changed", async () => {
    const calls = stubFetch(routes);
    window.location.hash = "#proving";
    render(<App />);
    await waitFor(() => expect(launch().disabled).toBe(false), { timeout: 4000 });
    // change the cash, the speed and the words
    fireEvent.change(screen.getByRole("textbox", { name: "Starting cash" }), { target: { value: "500" } });
    fireEvent.click(screen.getByRole("radio", { name: /^Faithful/ }));
    fireEvent.click(screen.getByRole("button", { name: "AI stocks, top 3" }));
    expect((screen.getByRole("textbox", { name: "Your request, in your own words" }) as HTMLTextAreaElement).value).toContain("AI stocks");
    // the first chip puts all of it back
    fireEvent.click(screen.getByRole("button", { name: "Oil, $3,000 split evenly" }));
    expect((screen.getByRole("textbox", { name: "Your request, in your own words" }) as HTMLTextAreaElement).value).toBe(planRequest.prompt);
    expect((screen.getByRole("textbox", { name: "Starting cash" }) as HTMLInputElement).value).toBe(String(planRequest.cash));
    expect(screen.getByRole("radio", { name: /^Fast/ }).getAttribute("aria-checked")).toBe("true");
    await waitFor(() => expect(calls.filter((c) => c.path === "/api/proving/plan" && JSON.stringify(c.body) === JSON.stringify(planRequest)).length).toBeGreaterThan(0));
    await waitFor(() => expect(document.querySelector(".lp-plan")!.textContent).toContain("from the cache, in seconds"));
  });

  it("a page loaded in the middle of a Time Machine test attaches it by the health answer's active_kind, and the Desk waits", async () => {
    stubFetch({ ...routes, "GET /api/health": { ...live, active_run: "p1", active_kind: "proving" } });
    render(<App />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(FakeEventSource.last.url).toBe("/api/runs/p1/events?after=0");
    feed([fastEvents[0]!]);
    const go = await screen.findByRole("button", { name: "Go!" });
    await waitFor(() => expect((go as HTMLButtonElement).disabled).toBe(true));
    expect(go.getAttribute("title")).toContain("Proving Ground test is running");
  });
});

describe("a saved run opened by its address", () => {
  it("is shown whole, and Watch it again plays its stream over it once, without the address opening the saved run a second time", async () => {
    const id = fastRun.run_id;
    const calls = stubFetch({ ...routes, [`GET /api/proving/runs/${id}`]: fastRun, [`POST /api/proving/runs/${id}/replay`]: { run_id: "r1" } });
    window.location.hash = `#proving/run/${id}`;
    render(<App />);
    expect(await screen.findByRole("heading", { name: fastRun.stats.headline.sentence })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Skip to the end" })).toBeNull(); // opened whole: nothing is drawn at the page's pace
    fireEvent.click(screen.getByRole("button", { name: "Watch it again" }));
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(FakeEventSource.last.url).toBe("/api/runs/r1/events?after=0");
    feed(fastEvents.slice(0, 30));
    await screen.findByRole("group", { name: "Research of each decision day" });
    await new Promise((r) => setTimeout(r, 50));
    expect(calls.filter((c) => c.method === "GET" && c.path === `/api/proving/runs/${id}`)).toHaveLength(1); // the address did not open it again
    expect(screen.getByRole("button", { name: "Stop the test" })).toBeTruthy(); // the stream is what is on show: a test being played
  });
});

describe("the Track Record view", () => {
  it("shows the recorded Track Record: its sentence, its small-sample marks and a filter on where the outcomes came from", async () => {
    stubFetch(routes);
    window.location.hash = "#proving/track";
    render(<App />);
    expect(await screen.findByText(trackViewRun.stats.headline.sentence)).toBeTruthy();
    expect(screen.getAllByText("SMALL SAMPLE").length).toBeGreaterThan(2);
    expect(screen.queryByText(/^(FAST|FAITHFUL)$/)).toBeNull(); // the Track Record is not a Time Machine run: it has no speed
    const filter = screen.getByRole("group", { name: "Where the recommendations came from" });
    const sources = Object.values((trackViewRun.coverage as { sources: Record<string, string> }).sources);
    const of = (kind: string) => sources.filter((s) => s === kind).length;
    expect(within(filter).getByRole("button", { name: new RegExp(`Everything\\s*${trackViewRun.outcomes.length}$`) })).toBeTruthy();
    expect(within(filter).getByRole("button", { name: new RegExp(`Your runs\\s*${of("run")}$`) })).toBeTruthy();
    expect(within(filter).getByRole("button", { name: new RegExp(`Recorded sessions\\s*${of("fixture")}$`) })).toBeTruthy();
    expect(document.querySelector(".track-few")!.textContent).toContain(`Only ${trackViewRun.stats.outcomes.closed} of these recommendations have closed, and a verdict needs 20`);
    // every caveat the server named, as a sentence
    for (const c of trackViewRun.caveats) expect(screen.getByText(typeof c === "string" ? c : c.sentence)).toBeTruthy();
  });
});
