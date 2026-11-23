import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { TradeMode } from "../api/types";
import { health } from "../test/fixtures";
import { RequestCard, type RequestCardProps } from "./RequestCard";

// Spec 5.3: the starting text, exactly.
const DEFAULT_PROMPT = "Buy an even mixture of 3 most recommended stocks in the pizza restaurants using $3000 of my available money";

function setup(over: Partial<RequestCardProps> = {}) {
  const onGo = vi.fn();
  const onStop = vi.fn();
  const onPrompt = vi.fn();
  const onMode = vi.fn(async (_mode: TradeMode) => undefined);
  const props: RequestCardProps = { prompt: health.defaults.prompt, onPrompt, maxChars: health.limits.prompt_chars, accountId: 7, running: false, starting: false, defaults: health.defaults, onGo, onStop, error: null, mode: "ask", onMode, ...over };
  render(<RequestCard {...props} />);
  return { onGo, onStop, onPrompt, onMode };
}

const strict = () => screen.getByLabelText("Strict recommendations") as HTMLInputElement;
const neutral = () => screen.getByLabelText("Neutral recommendations as passes") as HTMLInputElement;
const refresh = () => screen.getByLabelText("Refresh market data first") as HTMLInputElement;
const raiseFunds = () => screen.getByLabelText("Always raise funds") as HTMLInputElement;
const ignoreCritic = () => screen.getByLabelText("Ignore Critic") as HTMLInputElement;
const modeSwitch = () => screen.getByRole("switch") as HTMLButtonElement;

describe("RequestCard", () => {
  it("starts with the server's default text, which is the spec's exact sentence, and a counter", () => {
    setup();
    const box = screen.getByRole("textbox", { name: "Your request" }) as HTMLTextAreaElement;
    expect(box.value).toBe(DEFAULT_PROMPT);
    expect(box.maxLength).toBe(600);
    expect(screen.getByText(`${DEFAULT_PROMPT.length} / 600`)).toBeTruthy();
  });

  it("has Strict off, Neutral on and Refresh on by default (its label has no (paid): Jacob, 2026-11-23), all three in view, with the tooltip that says it writes the data files", () => {
    setup();
    expect([strict().checked, neutral().checked, refresh().checked]).toEqual([false, true, true]);
    expect(document.querySelector("details")).toBeNull();                                       // no "More" fold any more
    expect(refresh().closest("label")?.getAttribute("title")).toBe("Fetches fresh bars, news and filings with the paid APIs and writes the data files. Untick to use the stored data.");
    expect(refresh().closest("label")?.querySelector(".paid")).toBeNull();
  });

  it("has the box Always raise funds OFF by default (entry 142, R12); ticking it sends raise_funds true, and it is locked while a run is active", () => {
    const { onGo } = setup();
    expect(raiseFunds().checked).toBe(false);
    expect(raiseFunds().closest("label")?.getAttribute("title")).toContain("ALWAYS_RAISE_FUNDS");
    fireEvent.click(raiseFunds());
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    expect(onGo).toHaveBeenLastCalledWith(expect.objectContaining({ raise_funds: true }));
  });

  it("has the box Ignore Critic OFF by default (entry 148); ticking it sends ignore_critic true, unticks and disables Strict with the reason, and unticking brings Strict back", () => {
    const { onGo } = setup();
    expect(ignoreCritic().checked).toBe(false);
    expect(ignoreCritic().closest("label")?.getAttribute("title")).toContain("Skip the Critic for this run");
    fireEvent.click(strict());                                                                  // Strict on, then Ignore Critic on: the Critic does not run, so Strict goes
    fireEvent.click(ignoreCritic());
    expect(strict().checked).toBe(false);
    expect(strict().disabled).toBe(true);
    expect(strict().closest("label")?.getAttribute("title")).toBe("Ignore Critic: the Critic does not run");
    expect(neutral().checked).toBe(true);                                                       // the Neutral box keeps its own meaning
    expect(neutral().disabled).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    expect(onGo).toHaveBeenLastCalledWith({ account_id: 7, prompt: DEFAULT_PROMPT, strict: false, neutral: true, fetch: true, raise_funds: false, ignore_critic: true, mode: "ask" });
    fireEvent.click(ignoreCritic());
    expect(strict().checked).toBe(true);
    expect(strict().disabled).toBe(false);
  });

  it("locks Ignore Critic while a run is active", () => {
    setup({ running: true });
    expect(ignoreCritic().disabled).toBe(true);
  });

  it("shows Refresh unticked and disabled in a recorded session, and Go! then sends fetch false", () => {
    const { onGo } = setup({ replay: true, modeLockedReason: "a recorded session: its account is read-only" });
    expect(refresh().checked).toBe(false);
    expect(refresh().disabled).toBe(true);
    expect(refresh().closest("label")?.getAttribute("title")).toBe("a recorded session: nothing is fetched");
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    expect(onGo).toHaveBeenLastCalledWith(expect.objectContaining({ fetch: false }));
  });

  it("Clear empties the box and puts the cursor in it", () => {
    const { onPrompt } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(onPrompt).toHaveBeenCalledWith("");
    expect(document.activeElement).toBe(screen.getByRole("textbox", { name: "Your request" }));
  });

  it("Strict unticks and disables Neutral (with the reason) and brings it back when Strict goes", () => {
    setup();
    fireEvent.click(strict());
    expect(neutral().checked).toBe(false);
    expect(neutral().disabled).toBe(true);
    expect(neutral().closest("label")?.getAttribute("title")).toBe("Strict: only what the Critic approves trades");
    fireEvent.click(strict());
    expect(neutral().checked).toBe(true);
    expect(neutral().disabled).toBe(false);
  });

  it("Go! posts {account_id, prompt, strict, neutral, fetch, mode}; Refresh goes as true unless unticked, and with Strict on neutral goes as false", () => {
    const { onGo } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    expect(onGo).toHaveBeenLastCalledWith({ account_id: 7, prompt: DEFAULT_PROMPT, strict: false, neutral: true, fetch: true, raise_funds: false, ignore_critic: false, mode: "ask" });
    fireEvent.click(strict());
    fireEvent.click(refresh());
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    expect(onGo).toHaveBeenLastCalledWith({ account_id: 7, prompt: DEFAULT_PROMPT, strict: true, neutral: false, fetch: false, raise_funds: false, ignore_critic: false, mode: "ask" });
  });

  it("the Mode switch is off for Ask and on for Auto, with the chosen word bold, and flipping it saves {mode} at once and Go! then carries it", async () => {
    const { onMode, onGo } = setup();
    expect(modeSwitch().getAttribute("aria-checked")).toBe("false");
    expect(screen.getByText("Ask").className).toContain("on");
    fireEvent.click(modeSwitch());
    expect(modeSwitch().getAttribute("aria-checked")).toBe("true");                              // it moves first
    expect(onMode).toHaveBeenCalledWith("auto");
    expect(screen.getByText("Auto").className).toContain("on");
    expect(modeSwitch().closest(".mode")?.getAttribute("title")).toBe("Auto: the desk trades what the Critic approves without asking.");
    fireEvent.click(screen.getByRole("button", { name: "Go!" }));
    expect(onGo).toHaveBeenLastCalledWith(expect.objectContaining({ mode: "auto" }));
    fireEvent.keyDown(modeSwitch(), { key: " " });                                                // Space and Enter flip it too, once each
    await waitFor(() => expect(modeSwitch().getAttribute("aria-checked")).toBe("false"));
    fireEvent.keyDown(modeSwitch(), { key: "Enter" });
    expect(onMode.mock.calls.map((c) => c[0])).toEqual(["auto", "ask", "auto"]);
  });

  it("moves the switch back and shows the server's message under it when the save is refused", async () => {
    const onMode = vi.fn(async (_mode: TradeMode) => {
      throw new Error("sample accounts are read-only here");
    });
    setup({ onMode });
    fireEvent.click(modeSwitch());
    expect(modeSwitch().getAttribute("aria-checked")).toBe("true");
    await waitFor(() => expect(modeSwitch().getAttribute("aria-checked")).toBe("false"));
    expect(screen.getByRole("alert").textContent).toBe("sample accounts are read-only here");
  });

  it("shows the mode it is given (the account's own) and flipping it from Auto asks for Ask", () => {
    const { onMode } = setup({ mode: "auto" });
    expect(modeSwitch().getAttribute("aria-checked")).toBe("true");
    fireEvent.click(modeSwitch());
    expect(onMode).toHaveBeenCalledWith("ask");
  });

  it("Ctrl+Enter in the box is Go!", () => {
    const { onGo } = setup();
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Your request" }), { key: "Enter", ctrlKey: true });
    expect(onGo).toHaveBeenCalledTimes(1);
  });

  it("is disabled while a run is active: the gear and Running…, a quiet Stop beside it", () => {
    const { onGo, onStop } = setup({ running: true });
    const go = screen.getByRole("button", { name: /Running/ }) as HTMLButtonElement;
    expect(go.disabled).toBe(true);
    fireEvent.click(go);
    fireEvent.keyDown(screen.getByRole("textbox", { name: "Your request" }), { key: "Enter", ctrlKey: true });
    expect(onGo).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("keeps the Mode switch and the checkboxes disabled while a run is active, with the reason as the tooltip", () => {
    const { onMode } = setup({ running: true, mode: "auto" });
    expect(modeSwitch().disabled).toBe(true);
    expect(modeSwitch().closest(".mode")?.getAttribute("title")).toBe("a run is active: wait for it to finish");
    expect([strict().disabled, neutral().disabled, refresh().disabled]).toEqual([true, true, true]);
    fireEvent.click(modeSwitch());
    expect(onMode).not.toHaveBeenCalled();
  });

  it("keeps the Mode switch disabled for a sample account or a recorded session, which show their own mode", () => {
    setup({ modeLockedReason: "sample accounts are read-only here", mode: "auto" });
    expect(modeSwitch().disabled).toBe(true);
    expect(modeSwitch().getAttribute("aria-checked")).toBe("true");
    expect(modeSwitch().closest(".mode")?.getAttribute("title")).toBe("sample accounts are read-only here");
  });

  it("cannot go without an account or without text", () => {
    setup({ accountId: null });
    expect((screen.getByRole("button", { name: "Go!" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows a refusal of the server under the box", () => {
    setup({ error: "a run is already active: wait for it to finish, or stop it" });
    expect(screen.getByRole("alert").textContent).toContain("a run is already active");
  });
});
