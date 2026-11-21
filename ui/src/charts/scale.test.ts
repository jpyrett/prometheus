import { describe, expect, it } from "vitest";
import { areaPath, linePath, linePathWithGaps, stepPath, triangleUp } from "./path";
import { dayNumber, extent, isoOfDay, linear, moveHandle, nearestIndex, niceDomain, niceTicks, stepHandle, timeTicks, weekdayOf, weekdaysBetween } from "./scale";

describe("linear scales on exact inputs", () => {
  it("maps the domain onto the range, and back", () => {
    const x = linear([0, 100], [10, 210]);
    expect(x(0)).toBe(10);
    expect(x(50)).toBe(110);
    expect(x(100)).toBe(210);
    expect(x.invert(110)).toBe(50);
  });
  it("an inverted range draws a y axis (large values up)", () => {
    const y = linear([9000, 11000], [200, 0]);
    expect(y(9000)).toBe(200);
    expect(y(11000)).toBe(0);
    expect(y(10000)).toBe(100);
  });
  it("a domain of one point maps to the middle, so a flat series still draws", () => {
    expect(linear([5, 5], [0, 100])(5)).toBe(50);
  });
  it("extent skips what is not a number and is null for nothing", () => {
    expect(extent([3, null, 9, undefined, NaN, -2])).toEqual([-2, 9]);
    expect(extent([])).toBeNull();
  });
});

describe("nice ticks", () => {
  it("are round numbers inside the range", () => {
    expect(niceTicks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(niceTicks(9870, 10420, 5)).toEqual([10000, 10200, 10400])  // 550 over 5 steps is 110 a step, rounded up to 200;
    expect(niceTicks(-0.08, 0.02, 5)).toEqual([-0.08, -0.06, -0.04, -0.02, 0, 0.02]);
  });
  it("a flat range is the one value; a widened domain starts and ends on a tick", () => {
    expect(niceTicks(7, 7)).toEqual([7]);
    expect(niceDomain(9870, 10420, 5)).toEqual([9800, 10600]);
  });
});

describe("days", () => {
  it("count whole days since 1970-01-01 and come back as the same text", () => {
    expect(dayNumber("1970-01-02")).toBe(1);
    expect(isoOfDay(dayNumber("2026-08-10"))).toBe("2026-08-10");
    expect(weekdayOf(dayNumber("2026-08-10"))).toBe(1); // a Monday
    expect(dayNumber("not a date")).toBeNaN();
  });
  it("month ticks for a long span, one label per month, the year at January", () => {
    const ticks = timeTicks(dayNumber("2025-11-15"), dayNumber("2026-03-20"), 12);
    expect(ticks.map((t) => t.label)).toEqual(["Dec", "2026", "Feb", "Mar"]);
    expect(ticks.map((t) => isoOfDay(t.day))).toEqual(["2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01"]);
  });
  it("Monday ticks for a short span, and never more than asked for", () => {
    const ticks = timeTicks(dayNumber("2026-08-03"), dayNumber("2026-08-31"), 10);
    expect(ticks.map((t) => isoOfDay(t.day))).toEqual(["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31"]);
    expect(timeTicks(dayNumber("2026-08-03"), dayNumber("2026-08-31"), 2).length).toBeLessThanOrEqual(3);
  });
  it("finds the nearest value in a sorted list", () => {
    expect(nearestIndex([1, 4, 9], 5)).toBe(1);
    expect(nearestIndex([1, 4, 9], 100)).toBe(2);
    expect(nearestIndex([1, 4, 9], -3)).toBe(0);
    expect(nearestIndex([], 3)).toBe(-1);
  });
});

describe("the range picker's rules", () => {
  const days = weekdaysBetween(dayNumber("2026-06-01"), dayNumber("2026-06-30")); // Mon 1 June ... Tue 30 June
  const first = days[0]!;
  const last = days[days.length - 1]!;
  const range = { start: dayNumber("2026-06-08"), end: dayNumber("2026-06-22") };

  it("weekdays only", () => {
    expect(days.every((d) => weekdayOf(d) >= 1 && weekdayOf(d) <= 5)).toBe(true);
    expect(isoOfDay(first)).toBe("2026-06-01");
    expect(isoOfDay(last)).toBe("2026-06-30");
  });
  it("a handle dragged past the usable range stops at its edge, on a pickable day", () => {
    expect(moveHandle(range, "start", dayNumber("2026-01-01"), days).start).toBe(first);
    expect(moveHandle(range, "end", dayNumber("2026-12-31"), days).end).toBe(last);
  });
  it("a handle dropped on a Saturday snaps to a pickable day", () => {
    const moved = moveHandle(range, "end", dayNumber("2026-06-20"), days); // a Saturday
    expect(weekdayOf(moved.end)).toBeGreaterThanOrEqual(1);
    expect(weekdayOf(moved.end)).toBeLessThanOrEqual(5);
  });
  it("the two handles cannot cross: the start stays at least one day before the end, the other handle never moves", () => {
    const pushed = moveHandle(range, "start", dayNumber("2026-06-29"), days);
    expect(pushed.end).toBe(range.end);
    expect(pushed.start).toBeLessThan(pushed.end);
    const pulled = moveHandle(range, "end", dayNumber("2026-06-02"), days);
    expect(pulled.start).toBe(range.start);
    expect(pulled.end).toBeGreaterThan(pulled.start);
  });
  it("arrow keys step by pickable days and stop at the edge", () => {
    expect(isoOfDay(stepHandle(range, "end", 1, days).end)).toBe("2026-06-23");
    expect(stepHandle({ start: first, end: range.end }, "start", -1, days).start).toBe(first);
    expect(stepHandle({ start: range.start, end: last }, "end", 5, days).end).toBe(last);
  });
});

describe("path text", () => {
  it("a line, an area and a step", () => {
    expect(linePath([[0, 10], [5, 20.123]])).toBe("M0 10L5 20.12");
    expect(areaPath([[0, 10], [5, 20]], 30)).toBe("M0 10L5 20L5 30L0 30Z");
    expect(stepPath([[0, 10], [5, 20], [9, 5]])).toBe("M0 10L5 10L5 20L9 20L9 5");
    expect(linePath([])).toBe("");
  });
  it("a missing value breaks the line", () => {
    expect(linePathWithGaps([[0, 1], [1, null], [2, 3], [3, 4]])).toBe("M0 1M2 3L3 4");
  });
  it("a triangle has three corners", () => {
    expect(triangleUp(10, 10, 8).match(/[ML]/g)).toHaveLength(3);
  });
});
