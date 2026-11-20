import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { replayEvents } from "../state/runReducer";
import { orderWords } from "../state/selectors";
import { fracEvents } from "../test/fixtures";
import { PortfolioCard } from "./PortfolioCard";

// The recorded replay of auto_even_split_fractions.json: a real Auto session (live quotes, 2026-11-20) that split $30,000 evenly and bought 72.912869 COP, 47.725862 CVX and 220.799293 APA (design choices entries 110 and 123).
describe("a run that bought fractions of a share", () => {
  const done = replayEvents(fracEvents);

  it("the portfolio shows every fraction in full, the NEW badge and the cash the run spent", () => {
    const changed = { changes: done.portfolio!.changes, cash_change: done.portfolio!.cash_change, cash_before: done.portfolio!.cash_before };
    render(<PortfolioCard view={{ ...done.portfolio! }} changed={changed} before={done.portfolioBefore} />);
    const row = (symbol: string) => screen.getByText(symbol).closest("tr")!;
    expect(within(row("COP")).getByText("72.912869")).toBeTruthy();
    expect(within(row("CVX")).getByText("47.725862")).toBeTruthy();
    expect(within(row("APA")).getByText("220.799293")).toBeTruthy();
    expect(screen.getByText("+72.912869 NEW")).toBeTruthy();
    expect(screen.getByText("+220.799293 NEW")).toBeTruthy();
    expect(screen.getByText("was $31,000.00")).toBeTruthy();
  });

  it("the decisions of the work order say the order in dollars to the cent and its fraction as the estimate (entry 111: an order in dollars)", () => {
    const rows = done.workOrder?.rows ?? [];
    const words = rows.map((r) => [r.symbol, orderWords(r)]).filter(([, w]) => w !== null);
    expect(words).toEqual([
      ["COP", { text: "Buy $10,000.00", dollars: "about 72.912869 sh" }],
      ["CVX", { text: "Buy $10,000.00", dollars: "about 47.725862 sh" }],
      ["APA", { text: "Buy $10,000.00", dollars: "about 220.799293 sh" }],
    ]);
  });
});
