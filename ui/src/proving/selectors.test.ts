import { describe, expect, it } from "vitest";
import { planFixture } from "../test/proving";
import { initialProvingState, type ProvingState } from "./provingReducer";
import { dayMarks, researchLine } from "./selectors";

// The days are the real decision days of the recorded plan (the oil request over the usable range); the research states laid over them are the cases the strip has to draw.
const days = planFixture.decision_days;
const base: ProvingState = { ...initialProvingState, phase: "running", step: "research", plannedDays: days, parallelDays: 8 };

describe("the research strip of a run in its first phase", () => {
  it("the first `parallel_days` days still waiting are the ones being researched; the rest are waiting", () => {
    const marks = dayMarks(base);
    expect(marks.map((m) => m.state).slice(0, 10)).toEqual([...Array(8).fill("busy"), "pending", "pending"]);
  });
  it("a day that came back is done (approved over recommended), from the cache, or failed, and frees a place for the next", () => {
    const research = {
      [days[0]!]: { day: days[0]!, status: "done" as const, recommended: 5, approved: 3, denied: 2, cached: false },
      [days[1]!]: { day: days[1]!, status: "cached" as const, recommended: 4, approved: 0, denied: 4, cached: true },
      [days[2]!]: { day: days[2]!, status: "failed" as const, reason: "a model error", cached: false },
    };
    const marks = dayMarks({ ...base, research });
    expect(marks.slice(0, 3).map((m) => [m.state, m.approved, m.recommended])).toEqual([["done", 3, 5], ["cached", 0, 4], ["failed", 0, 0]]);
    expect(marks.filter((m) => m.state === "busy")).toHaveLength(8);
    expect(marks[3]!.state).toBe("busy");
    expect(marks[10]!.state).toBe("busy");
    expect(marks[11]!.state).toBe("pending");
  });
  it("says how many are done, how many came from the cache, and how many are researched at a time", () => {
    const research = { [days[0]!]: { day: days[0]!, status: "cached" as const, cached: true, recommended: 1, approved: 1, denied: 0 }, [days[1]!]: { day: days[1]!, status: "done" as const, recommended: 1, approved: 0, denied: 1 } };
    expect(researchLine({ ...base, research })).toBe(`Researching ${days.length} decision days, 8 at a time: 2 done, 1 from the cache`);
    expect(researchLine({ ...base, step: "walk" })).toContain("Walking the days in order");
  });
  it("a day opens only when it has a desk to show", () => {
    const marks = dayMarks({ ...base, deskDays: [days[0]!] });
    expect(marks[0]!.hasDesk).toBe(true);
    expect(marks[1]!.hasDesk).toBe(false);
  });
});
