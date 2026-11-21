import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { CAVEAT_SENTENCES, caveatItems } from "./caveats";

describe("the caveats are shown as written (P3)", () => {
  it("every sentence is word for word prometheus/proving/record.py's CAVEATS", () => {
    const text = readFileSync(resolve(process.cwd(), "..", "prometheus", "proving", "record.py"), "utf8");
    const block = text.slice(text.indexOf("CAVEATS: dict[str, str] = {"), text.indexOf("assert set(get_args(CaveatCode))"));
    const written = Object.fromEntries([...block.matchAll(/^\s+"(\w+)": "(.*)",$/gm)].map((m) => [m[1], m[2]]));
    expect(Object.keys(written)).toHaveLength(11);
    expect(CAVEAT_SENTENCES).toEqual(written);
  });
  it("the server's own sentence wins over the page's copy, a bare code gets the sentence of its code, a code twice is shown once", () => {
    expect(caveatItems([{ code: "small_sample", sentence: "the server's words" }, "small_sample", "price_returns_no_dividends"]).map((c) => [c.code, c.sentence])).toEqual([
      ["small_sample", "the server's words"],
      ["price_returns_no_dividends", CAVEAT_SENTENCES.price_returns_no_dividends],
    ]);
  });
});
