import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { stubFetch } from "../test/fakes";
import { savedRunsIndex } from "../test/proving";
import { SavedRuns } from "./SavedRuns";
import { SPEED_WORDS } from "./timing";

afterEach(() => vi.unstubAllGlobals());

// The index of saved runs as the replay server sent it (scripts/record_proving.mjs): 15 real runs, fast and faithful, the older ones with no speed on their row.
describe("the saved runs", () => {
  const rows = savedRunsIndex.filter((r) => r.kind === "time_machine");
  const badge = (card: Element) => card.querySelector(".badge")?.textContent ?? null;

  it("shows each run's speed beside its range, as its row says, and none for a row that has none", () => {
    stubFetch({});
    const { container } = render(<SavedRuns rows={rows} error={null} disabled={false} onOpen={() => undefined} onWatch={() => undefined} onDelete={() => undefined} />);
    expect(rows.some((r) => r.speed === "fast") && rows.some((r) => r.speed === "faithful")).toBe(true);
    const cards = [...container.querySelectorAll(".saved-card")];
    cards.forEach((c, i) => expect(badge(c)).toBe(rows[i]!.speed ? SPEED_WORDS[rows[i]!.speed!] : null));
  });

  it("lists the newest six and the rest only when asked for, so a store that has seen many tests does not bury the launch pad", () => {
    stubFetch({});
    const { container } = render(<SavedRuns rows={rows} error={null} disabled={false} onOpen={() => undefined} onWatch={() => undefined} onDelete={() => undefined} />);
    expect(container.querySelectorAll(".saved-card")).toHaveLength(6);
    expect(screen.getByText(`${rows.length} saved`)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: `Show all ${rows.length} saved runs` }));
    expect(container.querySelectorAll(".saved-card")).toHaveLength(rows.length);
    const faithful = [...container.querySelectorAll(".saved-card")].filter((c) => badge(c) === "FAITHFUL").length;
    expect(faithful).toBe(rows.filter((r) => r.speed === "faithful").length);
    fireEvent.click(screen.getByRole("button", { name: /^Show only the newest/ }));
    expect(container.querySelectorAll(".saved-card")).toHaveLength(6);
    expect(within(container.querySelector(".saved-card")!).getByRole("button", { name: "Watch it again" })).toBeTruthy();
  });
});
