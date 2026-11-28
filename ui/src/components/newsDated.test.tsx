import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { NewsView } from "../api/types";
import { newsEvents, ofType } from "../test/fixtures";
import { NewsPanel, newsHeader, newsWindow } from "./NewsPanel";

// The news of a past day (design choices entry 151, D8): the page's header says the days asked for in place of "last 7 days". The rolling view is the real replay of the live session (it has no `window`).
const rolling = ofType(newsEvents, "news")[0]!.data as NewsView & { ts: number };
const oneDay: NewsView = { ...rolling, days: null, window: { from: "2026-11-20", to: "2026-11-20" } };
const span: NewsView = { ...rolling, days: null, window: { from: "2026-11-03", to: "2026-11-05" }, fetched: true, fetch_note: null };

afterEach(cleanup);

describe("the header of a news view that has a dated window", () => {
  it("says 'on <day>' for one day and 'from <day> to <day>' for a span, in place of the last N days", () => {
    expect(newsHeader(oneDay)).toBe("TSLA · Tesla Inc · on 2026-11-20 · stored news");
    expect(newsHeader(span)).toBe("TSLA · Tesla Inc · from 2026-11-03 to 2026-11-05 · fetched now");
    expect(newsWindow(oneDay)).toBe("on 2026-11-20");
    expect(newsHeader(oneDay)).not.toContain("last");
  });

  it("is what it was for the rolling window the live session replayed (the event of a run before this change has no window at all)", () => {
    expect("window" in rolling).toBe(false);
    expect(newsHeader(rolling)).toBe("TSLA · Tesla Inc · last 7 days · stored news");
    expect(newsWindow({ ...rolling, window: null })).toBe("last 7 days");
    expect(newsWindow({ ...rolling, days: null })).toBeNull();
  });

  it("is shown by the panel, and a day with no stored article says so with the day in its words", () => {
    render(<NewsPanel news={{ ...oneDay, items: [], summary: "", no_supporting_text: true }} />);
    expect(screen.getByText("TSLA · Tesla Inc · on 2026-11-20 · stored news")).toBeTruthy();
    expect(screen.getByText("The stored news has nothing about this on 2026-11-20.")).toBeTruthy();
    cleanup();
    render(<NewsPanel news={{ ...rolling, items: [], summary: "", no_supporting_text: true }} />);
    expect(screen.getByText("The stored news has nothing about this in the window.")).toBeTruthy();
  });
});
