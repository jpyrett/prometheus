import { describe, expect, it } from "vitest";
import { count, round6, shares, signedShares } from "./format";
import { changeBadge, orderWords } from "./state/selectors";
import type { WorkRow } from "./api/types";

// A share quantity has at most 6 decimals (the server's shares.PLACES); `count` has the default 3 and would silently show 81.004 (design choices entry 110).
describe("shares: a number of shares as the server prints it", () => {
  it("keeps up to six decimals and drops the trailing zeros, with thousands separators", () => {
    expect(shares(81)).toBe("81");
    expect(shares(1234)).toBe("1,234");
    expect(shares(81.004455)).toBe("81.004455");
    expect(shares(31.581606)).toBe("31.581606");
    expect(shares(0.5)).toBe("0.5");
    expect(shares(0.000001)).toBe("0.000001");
    expect(shares(1234567.5)).toBe("1,234,567.5");
    expect(count(81.004455)).toBe("81.004");
    expect(shares(null)).toBe("—");
  });

  it("rounds the arithmetic noise of a float away and signs a change with a real minus", () => {
    expect(round6(0.1 + 0.2)).toBe(0.3);
    expect(shares(0.1 + 0.2)).toBe("0.3");
    expect(signedShares(0.5)).toBe("+0.5");
    expect(signedShares(-81)).toBe("−81");
    expect(signedShares(0.004455)).toBe("+0.004455");
    expect(signedShares(1e-9)).toBe("0");
  });

  it("calls a holding NEW or SOLD by quantities at six places, not by raw floats", () => {
    expect(changeBadge(0.1 + 0.2, 0.3)).toEqual({ text: "+0.3 NEW", tone: "pos" });
    expect(changeBadge(31.581606, 31.581606)).toEqual({ text: "+31.581606 NEW", tone: "pos" });
    expect(changeBadge(-0.3, 0.1 + 0.2 - 0.3)).toEqual({ text: "−0.3 SOLD", tone: "neg" });
    expect(changeBadge(-0.222222, 81)).toEqual({ text: "−0.222222", tone: "neg" });
    expect(changeBadge(8, 50)).toEqual({ text: "+8", tone: "pos" });
  });

  it("writes the order of a decision with its fraction", () => {
    const row = { decision: { shares: 22.738642, est_cost: 5000, will_trade: "yes_auto", action: "open" } } as unknown as WorkRow;
    expect(orderWords(row)).toEqual({ text: "Buy 22.738642", dollars: "$5,000.00" });
  });
});
