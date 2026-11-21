import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { points } from "../format";
import { criticReportFixture as report } from "../test/proving";
import { CriticRulesCard, rowsOfVersion } from "./CriticRules";

// The report card of the Critic's rules (Critic v2 rule C1): the page draws what GET /api/proving/critic-report answered (recorded from the replay server over the saved real runs). Every figure, reading and sentence below is
// compared with the recording, never typed: the page computes nothing (P12).
const newest = report.config_versions[report.config_versions.length - 1]!;
const card = report.by_config_version[newest]!;
const showTable = () => fireEvent.click(screen.getByRole("button", { name: /^Show as table$/ }));
const cells = (row: HTMLElement) => within(row).getAllByRole("cell").map((c) => c.textContent);
const dataRows = () => within(screen.getAllByRole("table")[0]!).getAllByRole("row").slice(1);

describe("the report card of the Critic's rules", () => {
  it("is the recording of two versions of the Critic, each with its verdict row and one row for every finding code", () => {
    expect(report.config_versions.length).toBeGreaterThan(1);
    for (const v of report.config_versions) {
      const c = report.by_config_version[v]!;
      expect(c.verdict.kind).toBe("verdict");
      expect(c.rows.length).toBeGreaterThan(3);
    }
  });

  it("draws the verdict and a row for each rule of the newest version on one axis, zero marked, and says so in its label", () => {
    const { container } = render(<CriticRulesCard report={report} />);
    const rows = container.querySelectorAll(".rule-row");
    expect(rows).toHaveLength(1 + card.rows.length);
    expect(container.querySelectorAll(".grid.vert.strong")).toHaveLength(1); // zero, marked
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain(`${rows.length} rows`);
    expect(screen.getByText(card.verdict.sentence)).toBeTruthy(); // the server's own sentence
    expect(container.textContent).toContain(`${card.reviewed} reviewed recommendations with an outcome in version ${card.config_version}`);
  });

  it("colours a row by the server's reading and by nothing else: green picks losers beyond chance, red denies winners beyond chance, grey the rest, and a suitability rule is greyed with no dot", () => {
    const { container } = render(<CriticRulesCard report={report} />);
    const wanted = (r: { sentence_code: string }) => (r.sentence_code === "picks_losers" ? "var(--pos)" : r.sentence_code === "denies_winners" ? "var(--neg)" : "var(--neutral)");
    for (const r of [card.verdict, ...card.rows]) {
      const row = container.querySelector(`.rule-row[data-key="${r.code}"]`)!;
      expect(row.getAttribute("data-reading")).toBe(r.sentence_code);
      const dot = row.querySelector("circle.gap-dot");
      if (r.suitability) {
        expect(row.classList.contains("greyed")).toBe(true);
        expect(dot).toBeNull();
      } else if (r.gap !== null && r.se !== null) {
        expect(dot!.getAttribute("fill")).toBe(wanted(r));
        expect(row.querySelector(".gap-bar")!.getAttribute("stroke")).toBe(wanted(r));
        expect(row.classList.contains("greyed")).toBe(false);
      }
    }
  });

  it("has a table twin with every figure as the server gave it: the fired count, the won share, the two means, the gap and its two-standard-error bar", () => {
    render(<CriticRulesCard report={report} />);
    showTable();
    const rows = dataRows();
    const all = [card.verdict, ...card.rows];
    expect(rows).toHaveLength(all.length);
    const rule = card.rows.find((r) => !r.suitability && r.gap !== null)!;
    const c = cells(rows[all.indexOf(rule)]!);
    expect(c[0]).toBe(rule.code);
    expect(c[2]).toBe(String(rule.fired));
    expect(c[3]).toBe(rule.hit.closed ? `${rule.hit.won}/${rule.hit.closed}` : "—");
    expect(c.slice(4, 8)).toEqual([points(rule.hit.mean_excess), points(rule.rest.mean_excess), points(rule.gap), points(report.chance_band * rule.se!)]);
    const suitability = card.rows.find((r) => r.suitability);
    if (suitability) expect(cells(rows[all.indexOf(suitability)]!)).toEqual(expect.arrayContaining(["suitability", "—"]));
  });

  it("never mixes the versions of the Critic in one row: a chip switches the whole card to the other version", () => {
    const { container } = render(<CriticRulesCard report={report} />);
    const other = report.config_versions[0]!;
    expect(other).not.toBe(newest);
    fireEvent.click(screen.getByRole("button", { name: new RegExp(`Critic version ${other}`) }));
    const old = report.by_config_version[other]!;
    expect(container.querySelectorAll(".rule-row")).toHaveLength(1 + old.rows.length);
    expect(screen.getByText(old.verdict.sentence)).toBeTruthy();
    expect(container.textContent).toContain(`in version ${other}`);
    expect(rowsOfVersion(old, null).map((r) => r.key)).toEqual([old.verdict.code, ...old.rows.map((r) => r.code)]);
  });

  it("shows one theme when a theme chip is pressed: the rules that fired in it, with that theme's own numbers", () => {
    const { container } = render(<CriticRulesCard report={report} />);
    const theme = report.themes.find((t) => card.rows.some((r) => r.by_theme[t]))!;
    fireEvent.click(screen.getByRole("button", { name: theme }));
    const rows = rowsOfVersion(card, theme);
    expect(rows.length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".rule-row")).toHaveLength(rows.length);
    showTable();
    const first = card.rows.find((r) => r.by_theme[theme] && !r.suitability)!;
    const cell = first.by_theme[theme]!;
    const index = rows.findIndex((r) => r.key === first.code);
    expect(cells(dataRows()[index]!).slice(0, 3)).toEqual([first.code, first.kind, String(cell.n)]);
  });

  it("says plainly when no review has an outcome yet", () => {
    render(<CriticRulesCard report={{ ...report, config_versions: [], by_config_version: {}, themes: [] }} />);
    expect(screen.getByText(/No review has an outcome yet/)).toBeTruthy();
  });
});
