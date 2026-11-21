import { describe, expect, it } from "vitest";
import { formatHash, parseHash } from "./hash";

describe("the page's address (no router: the hash)", () => {
  it("every route round-trips", () => {
    for (const hash of ["", "#proving", "#proving/track", "#proving/live", "#proving/run/tm-20261120-1430"]) {
      expect(formatHash(parseHash(hash))).toBe(hash);
    }
  });
  it("the Desk is the default, and anything unknown is the Desk", () => {
    expect(parseHash("")).toEqual({ tab: "desk" });
    expect(parseHash("#desk")).toEqual({ tab: "desk" });
    expect(parseHash("#nonsense/here")).toEqual({ tab: "desk" });
  });
  it("a saved run's id is kept whole, and a broken one falls back to the Time Machine", () => {
    expect(parseHash("#proving/run/a%2Fb")).toEqual({ tab: "proving", view: "run", runId: "a/b" });
    expect(parseHash("#proving/run/%E0%A4%A")).toEqual({ tab: "proving", view: "time_machine" });
    expect(parseHash("#proving/run/")).toEqual({ tab: "proving", view: "time_machine" });
  });
});
