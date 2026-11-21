import { describe, expect, it } from "vitest";
import { duration, moneyAxis, pct, points, rate } from "./format";

describe("the Proving Ground's number words (a _pct is a fraction)", () => {
  it("a fraction is a percent, with a real minus and no sign on zero", () => {
    expect(pct(0.0321)).toBe("+3.2%");
    expect(pct(-0.1)).toBe("−10.0%");
    expect(pct(0)).toBe("0.0%");
    expect(pct(-0.00001)).toBe("0.0%");
    expect(pct(0.0321, 1, false)).toBe("3.2%");
    expect(pct(null)).toBe("—");
  });
  it("a difference of fractions is percentage points", () => {
    expect(points(0.032)).toBe("+3.2 pts");
    expect(points(-0.0216)).toBe("−2.2 pts");
  });
  it("axis money, rates and durations", () => {
    expect(moneyAxis(9800)).toBe("$9,800");
    expect(moneyAxis(10400)).toBe("$10.4k");
    expect(moneyAxis(10000)).toBe("$10k");
    expect(moneyAxis(1_250_000)).toBe("$1.3M");
    expect(rate(0.583)).toBe("58%");
    expect(duration(133)).toBe("2m 13s");
    expect(duration(37.24)).toBe("37.2s");
  });
});
