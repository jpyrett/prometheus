import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { NewsView, RunStartedData, ServerEvent } from "../api/types";
import { initialRunState, replayEvents, runReducer } from "../state/runReducer";
import { FakeEventSource, stubFetch } from "../test/fakes";
import { bobEvents, health, newsEvents, ofType, replayAccount } from "../test/fixtures";
import { NewsPanel, SummaryText, newsHeader, safeUrl } from "./NewsPanel";
import { Pipeline } from "./Pipeline";

const news = ofType(newsEvents, "news")[0]!.data as NewsView & { ts: number };
const NEWS_RUN = (newsEvents[0]!.data as RunStartedData).run_id;
const BOB_RUN = (bobEvents[0]!.data as RunStartedData).run_id;

afterEach(cleanup);

describe("the run state of a request that only asked for the news (the real replay of the live session)", () => {
  it("keeps the news, finishes with the status news and shows no work order; a run that follows keeps the news viewable", () => {
    const state = replayEvents(newsEvents);
    expect(state.phase).toBe("finished");
    expect(state.result?.status).toBe("news");
    expect(state.result?.headline).toBe("The news of TSLA for the last 7 days");
    expect(state.workOrder).toBeNull();
    expect(state.news?.items).toHaveLength(12);
    expect(state.news?.items.filter((i) => i.cited)).toHaveLength(9);
    expect(state.stages.map((s) => s.state)).toEqual(["done", "done", "pending", "pending", "pending", "pending"]);
    const next = runReducer(state, bobEvents[0]!); // the next run starts: its own state is fresh, the last news stays
    expect(next.runId).toBe(BOB_RUN);
    expect(next.news).toBe(state.news);
    expect(next.workOrder).toBeNull();
  });
});

describe("the pipeline of a news run", () => {
  it("shows the Account Manager and the Data Clerk only once the run has finished, and all six stages while it goes on", () => {
    render(<Pipeline state={replayEvents(newsEvents)} />);
    const items = within(screen.getByRole("region", { name: "Desk pipeline" })).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]!.textContent).toContain("Account Manager");
    expect(items[1]!.textContent).toContain("Data Clerk");
    cleanup();
    render(<Pipeline state={replayEvents(newsEvents.slice(0, 4))} />);
    expect(within(screen.getByRole("region", { name: "Desk pipeline" })).getAllByRole("listitem")).toHaveLength(6);
  });
});

describe("NewsPanel", () => {
  it("says what to ask when there is no news yet", () => {
    render(<NewsPanel news={null} />);
    expect(screen.getByText(/Ask for a stock's news, e\.g\. "What's Tesla doing the last week\?"/)).toBeTruthy();
  });

  it("shows the stocks and the window in its header, the failed fetch in words, the summary as text and the articles with the cited ones first", () => {
    render(<NewsPanel news={news} />);
    expect(newsHeader(news)).toBe("TSLA · Tesla Inc · last 7 days · stored news");
    expect(screen.getByText("TSLA · Tesla Inc · last 7 days · stored news")).toBeTruthy();
    expect(screen.getByRole("status").textContent).toMatch(/^could not fetch the latest news \(.*\): showing the stored news$/);
    expect(screen.getByText(/Tesla \(TSLA\) news summary/)).toBeTruthy();
    const items = within(document.querySelector<HTMLElement>(".news-items")!).getAllByRole("listitem");
    expect(items).toHaveLength(12);
    expect(within(items[0]!).getByText("Forget SpaceX: This Stock Could Be the Real Winner of the Space Race")).toBeTruthy();
    expect(within(items[0]!).getByText("cited")).toBeTruthy();
    expect(within(items[11]!).getByText("not cited")).toBeTruthy();
    const link = within(items[0]!).getByRole("link", { name: /Forget SpaceX/ });
    expect(link.getAttribute("href")).toMatch(/^https:\/\/www\.fool\.com\//);
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
  });

  it("links each [n] of the summary to the article with that number, and a click lights it", () => {
    render(<NewsPanel news={news} />);
    const marks = screen.getAllByRole("link").filter((a) => a.classList.contains("news-mark"));
    expect(marks.map((a) => a.textContent)).toContain("9"); // every mark of the summary that has an article is a link
    const first = marks.find((a) => a.textContent === "1")!;
    expect(first.getAttribute("href")).toBe("#news-item-1");
    fireEvent.click(first);
    expect(document.getElementById("news-item-1")!.className).toContain("lit");
    expect(window.location.hash).toBe(""); // the page's own hash (its tabs) is not touched
    expect(document.getElementById("news-item-2")!.className).not.toContain("lit");
  });

  it("joins the chunk numbers of one article: a mark of any of them finds it", () => {
    const items = [{ ...news.items[0]!, number: 1, numbers: [1, 13] }];
    render(<SummaryText text="Two chunks [13] and a number nobody has [99]." items={items} onJump={() => undefined} />);
    expect(screen.getByRole("link", { name: "13" }).getAttribute("href")).toBe("#news-item-1");
    expect(screen.queryByRole("link", { name: "99" })).toBeNull();
  });

  it("shows the model's bullets as a list and its **bold** as bold, with the marks still linked and no asterisk left", () => {
    const items = [{ ...news.items[0]!, number: 1, numbers: [1] }];
    const { container } = render(<SummaryText text={"**Tesla news**\n\n- A **robotaxi** forecast [1].\n- A plant expansion."} items={items} onJump={() => undefined} />);
    expect([...container.querySelectorAll("li")].map((li) => li.textContent)).toEqual(["A robotaxi forecast [1].", "A plant expansion."]);
    expect([...container.querySelectorAll("strong")].map((b) => b.textContent)).toEqual(["Tesla news", "robotaxi"]);
    expect(container.textContent).not.toContain("*");
    expect(screen.getByRole("link", { name: "1" }).getAttribute("href")).toBe("#news-item-1");
  });

  it("renders a headline with a script tag as text, links nothing but a web address, and the source never uses dangerouslySetInnerHTML", () => {
    const hostile: NewsView = {
      ...news,
      summary: "<img src=x onerror=alert(1)> [1]",
      items: [
        { ...news.items[0]!, headline: "<script>alert('x')</script>Tesla", url: "javascript:alert(1)" },
        { ...news.items[1]!, number: 2, numbers: [2], headline: "Second", url: "https://example.com/a" },
      ],
      problems: ["<b>problem</b>"],
      notes: ["<i>note</i>"],
    };
    const { container } = render(<NewsPanel news={hostile} />);
    expect(screen.getByText("<script>alert('x')</script>Tesla")).toBeTruthy();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b, i")).toBeNull();
    expect(screen.queryByRole("link", { name: /alert/ })).toBeNull(); // the javascript: address is not a link
    expect(safeUrl("javascript:alert(1)")).toBeNull();
    expect(safeUrl(" https://example.com/x ")).toBe("https://example.com/x");
    expect(readFileSync("src/components/NewsPanel.tsx", "utf8")).not.toContain("dangerouslySetInnerHTML");
  });

  it("says a theme and a fetch that worked in the header", () => {
    expect(newsHeader({ ...news, topic: "quantum computing", stocks: [], fetched: false })).toBe("quantum computing · last 7 days · stored news");
    expect(newsHeader({ ...news, fetched: true, fetch_note: null })).toBe("TSLA · Tesla Inc · last 7 days · fetched now");
  });
});

describe("the page: the right column toggles between the News and the work order and keeps both", () => {
  beforeEach(() => {
    FakeEventSource.reset();
    vi.stubGlobal("EventSource", FakeEventSource);
  });
  afterEach(() => vi.unstubAllGlobals());

  const feed = (events: ServerEvent[]) =>
    act(() => {
      for (const e of events) FakeEventSource.last.emit(e);
    });

  it("selects News when a news run's news arrives, shows the work order again when another run starts, and keeps the last news for the toggle", async () => {
    let posts = 0;
    stubFetch({ "GET /api/health": health, "GET /api/accounts/0": replayAccount, "POST /api/runs": () => ({ run_id: posts++ === 0 ? NEWS_RUN : BOB_RUN }) });
    render(<App />);
    const go = async () => {
      await waitFor(() => expect((screen.getByRole("button", { name: "Go!" }) as HTMLButtonElement).disabled).toBe(false));
      fireEvent.click(screen.getByRole("button", { name: "Go!" }));
      await waitFor(() => expect(FakeEventSource.instances.length).toBe(posts));
    };
    await go();
    // Before any news: Work order is shown and the News view says what to ask.
    expect(screen.getByRole("tab", { name: "Work order" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.queryByRole("region", { name: "News summary and articles" })).toBeNull();
    feed(newsEvents);
    // The news run: News is selected, its card is shown and the work order table is not.
    expect(screen.getByRole("tab", { name: "News" }).getAttribute("aria-selected")).toBe("true");
    const card = screen.getByRole("region", { name: "News summary and articles" });
    expect(within(card).getByText("TSLA · Tesla Inc · last 7 days · stored news")).toBeTruthy();
    expect(screen.queryByRole("grid", { name: "Stocks on the work order" })).toBeNull();
    expect(screen.getByText("COMPLETED")).toBeTruthy();
    expect(screen.getByText("The news of TSLA for the last 7 days")).toBeTruthy();
    // The user switches to the work order and back; both keep their content.
    fireEvent.click(screen.getByRole("tab", { name: "Work order" }));
    expect(screen.getByText("No work order yet")).toBeTruthy(); // a news run has no stocks on a work order table
    expect(screen.queryByRole("region", { name: "News summary and articles" })).toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "News" }));
    expect(screen.getByRole("region", { name: "News summary and articles" })).toBeTruthy();
    // Another run starts: Work order is selected, and the news is still there to look at.
    await go();
    feed([bobEvents[0]!]);
    expect(screen.getByRole("tab", { name: "Work order" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.queryByRole("region", { name: "News summary and articles" })).toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "News" }));
    expect(within(screen.getByRole("region", { name: "News summary and articles" })).getByText("TSLA · Tesla Inc · last 7 days · stored news")).toBeTruthy();
  });

  it("starts with the work order and a News view that says what to ask", async () => {
    stubFetch({ "GET /api/health": health, "GET /api/accounts/0": replayAccount });
    render(<App />);
    await screen.findByRole("textbox", { name: "Your request" });
    expect(screen.getByRole("tab", { name: "Work order" }).getAttribute("aria-selected")).toBe("true");
    fireEvent.click(screen.getByRole("tab", { name: "News" }));
    expect(screen.getByText(/Ask for a stock's news/)).toBeTruthy();
  });
});

describe("the initial state", () => {
  it("has no news", () => {
    expect(initialRunState.news).toBeNull();
  });
});
