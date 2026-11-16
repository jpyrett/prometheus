import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { replayEvents } from "../state/runReducer";
import { orderWords } from "../state/selectors";
import { fracEvents } from "../test/fixtures";
import { PortfolioCard } from "./PortfolioCard";

// The recorded replay of auto_even_split_fractions.json: a real Auto session that split $30,000 evenly and bought 63.163213 EOG, 45.477284 CVX and 58.620083 XOM (design choices entry 110).
describe("a run that bought fractions of a share", () => {
  const done = replayEvents(fracEvents);

  it("the portfolio shows every fraction in full, the NEW badge and the cash the run spent", () => {
    const changed = { changes: done.portfolio!.changes, cash_change: done.portfolio!.cash_change, cash_before: done.portfolio!.cash_before };
    render(<PortfolioCard view={{ ...done.portfolio! }} changed={changed} before={done.portfolioBefore} />);
    const row = (symbol: string) => screen.getByText(symbol).closest("tr")!;
    expect(within(row("EOG")).getByText("63.163213")).toBeTruthy();
    expect(within(row("CVX")).getByText("45.477284")).toBeTruthy();
    expect(within(row("XOM")).getByText("58.620083")).toBeTruthy();
    expect(screen.getByText("+63.163213 NEW")).toBeTruthy();
    expect(screen.getByText("+58.620083 NEW")).toBeTruthy();
    expect(screen.getByText("was $31,000.00")).toBeTruthy();
  });

  it("the decisions of the work order say the order with its fraction and the dollars to the cent", () => {
    const rows = done.workOrder?.rows ?? [];
    const words = rows.map((r) => [r.symbol, orderWords(r)]).filter(([, w]) => w !== null);
    expect(words).toEqual([
      ["EOG", { text: "Buy 63.163213", dollars: "$10,000.00" }],
      ["CVX", { text: "Buy 45.477284", dollars: "$10,000.00" }],
      ["XOM", { text: "Buy 58.620083", dollars: "$10,000.00" }],
    ]);
  });
});
