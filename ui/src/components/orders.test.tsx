import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AccountView, OpenOrder, OrderNotice, OrdersView } from "../api/types";
import { replayAccount } from "../test/fixtures";
import { OpenOrdersCard, heldWords, orderWords, raiseWords, untilText } from "./OpenOrdersCard";
import { PortfolioCard } from "./PortfolioCard";

// Waiting limit orders (design choices entry 140): the numbers are the server's own (a real account's two orders, as `AccountStore.place_open_order` wrote them).
const buy: OpenOrder = {
  order_id: 5, account_id: 9, symbol: "MU", company: "Micron Technology Inc", side: "buy", shares: 10, dollars: null, limit_price: 90, reserved_cash: 900, reserved_shares: 0,
  request: "Buy 10 shares of MU at $90 or less", placed_utc: "2026-11-16T15:30:00Z", expires_utc: "2026-11-16T21:00:00Z", status: "open", reason: null,
};
const sale: OpenOrder = {
  order_id: 6, account_id: 9, symbol: "MSFT", company: "Microsoft Corp", side: "sell", shares: 5, dollars: null, limit_price: 600, reserved_cash: 0, reserved_shares: 5,
  request: "Sell 5 shares of MSFT at $600 or more", placed_utc: "2026-11-13T22:00:00Z", expires_utc: "2026-11-16T21:00:00Z", status: "open", reason: null,
};
const dollars: OpenOrder = { ...buy, order_id: 7, shares: null, dollars: 5000, reserved_cash: 5000 };
const view = (orders: OpenOrder[], watching = true): OrdersView => ({ orders, reserved_cash: orders.reduce((n, o) => n + o.reserved_cash, 0), available: 4000, watching });

describe("the words of a waiting order", () => {
  it("a purchase has a ceiling and holds cash, a sale has a floor and holds shares, an order in dollars says its dollars", () => {
    expect(orderWords(buy)).toBe("buy 10 MU at $90.00 or less");
    expect(orderWords(sale)).toBe("sell 5 MSFT at $600.00 or more");
    expect(orderWords(dollars)).toBe("buy $5,000.00 of MU at $90.00 or less");
    expect(heldWords(buy)).toBe("$900.00 set aside");
    expect(heldWords(sale)).toBe("5 shares set aside");
    expect(heldWords({ ...sale, reserved_shares: 1 })).toBe("1 share set aside");
  });
  it("a day order ends at its session's close, in the market's own time", () => {
    expect(untilText("2026-11-16T21:00:00Z")).toBe("4:00 pm ET Mon 2026-11-16");
    expect(untilText("2026-11-27T18:00:00Z")).toBe("1:00 pm ET Fri 2026-11-27"); // an early-close day
  });
});

describe("a waiting purchase the user permitted to raise funds (design choices entry 143)", () => {
  // The server's own order: ANT at $85 or less, $60.14 set aside of $170.00, "raise funds if needed" in the words.
  const funded: OpenOrder = { ...buy, order_id: 8, symbol: "ANT", shares: 2, limit_price: 85, reserved_cash: 60.14, request: "Buy 2 shares of ANT at $85 or less and raise funds if needed", raise_funds_by: "words" };
  it("says holdings are sold only if the price is reached, and how the user permitted it", () => {
    expect(raiseWords(funded)).toBe("sells holdings for the rest only if the price is reached · you said to raise funds");
    expect(raiseWords({ ...funded, raise_funds_by: "setting" })).toBe("sells holdings for the rest only if the price is reached · the setting ALWAYS_RAISE_FUNDS is on");
  });
  it("says nothing for an order with no permission, an order the server sent without the field, or a sale", () => {
    expect(raiseWords(buy)).toBeNull();
    expect(raiseWords({ ...buy, raise_funds_by: null })).toBeNull();
    expect(raiseWords({ ...sale, raise_funds_by: "words" })).toBeNull(); // a sale needs no money: never funded
  });
  it("is drawn under the order on the card, and only there", () => {
    render(<OpenOrdersCard view={view([funded, buy])} notices={[]} error={null} cancelling={null} onCancel={() => undefined} readOnlyReason={null} />);
    expect(screen.getAllByText(/sells holdings for the rest only if the price is reached/)).toHaveLength(1);
    expect(screen.getByText("$60.14 set aside · until 4:00 pm ET Mon 2026-11-16")).toBeTruthy();
  });
});

describe("OpenOrdersCard", () => {
  it("lists each order with what it holds and until when, and Cancel names the order", () => {
    const onCancel = vi.fn();
    render(<OpenOrdersCard view={view([buy, sale])} notices={[]} error={null} cancelling={null} onCancel={onCancel} readOnlyReason={null} />);
    const card = screen.getByRole("region", { name: "Waiting orders" });
    expect(within(card).getByText("buy 10 MU at $90.00 or less")).toBeTruthy();
    expect(within(card).getByText("$900.00 set aside · until 4:00 pm ET Mon 2026-11-16")).toBeTruthy();
    expect(within(card).getByText("sell 5 MSFT at $600.00 or more")).toBeTruthy();
    expect(within(card).getByText("$900.00 set aside")).toBeTruthy(); // the card's right-hand side: the cash the orders hold
    fireEvent.click(within(card).getByRole("button", { name: "Cancel order 6" }));
    expect(onCancel).toHaveBeenCalledWith(6);
  });
  it("says it is a day order, watched only while the server runs, and says when no watcher is running", () => {
    render(<OpenOrdersCard view={view([buy], false)} notices={[]} error={null} cancelling={null} onCancel={() => undefined} readOnlyReason={null} />);
    expect(screen.getByText(/A waiting order is a day order/)).toBeTruthy();
    expect(screen.getByText(/no watcher is running right now/)).toBeTruthy();
    cleanup();
    render(<OpenOrdersCard view={view([buy], true)} notices={[]} error={null} cancelling={null} onCancel={() => undefined} readOnlyReason={null} />);
    expect(screen.queryByText(/no watcher is running right now/)).toBeNull();
  });
  it("a cancel in progress and a read-only account disable the buttons, and a refusal is shown", () => {
    const { unmount } = render(<OpenOrdersCard view={view([buy])} notices={[]} error={null} cancelling={5} onCancel={() => undefined} readOnlyReason={null} />);
    expect((screen.getByRole("button", { name: "Cancel order 5" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Cancelling…")).toBeTruthy();
    unmount();
    render(<OpenOrdersCard view={view([buy])} notices={[]} error="the order 5 is filled: it cannot be cancelled" cancelling={null} onCancel={() => undefined} readOnlyReason="sample accounts are read-only here" />);
    expect((screen.getByRole("button", { name: "Cancel order 5" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole("alert").textContent).toBe("the order 5 is filled: it cannot be cancelled");
  });
  it("shows what the watcher said, and draws nothing when there is no order and nothing was said", () => {
    const notice: OrderNotice = { type: "filled", order_id: 5, account_id: 9, symbol: "MU", side: "buy", text: "Order #5 filled: bought 10 MU at $89.50 ($895.00), the waiting order you left: buy 10 MU at $90.00 or less.", at: "2026-11-16T16:05:00Z", ts: 1 };
    const { container, unmount } = render(<OpenOrdersCard view={view([])} notices={[notice]} error={null} cancelling={null} onCancel={() => undefined} readOnlyReason={null} />);
    expect(screen.getByText(notice.text)).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
    unmount();
    const empty = render(<OpenOrdersCard view={view([])} notices={[]} error={null} cancelling={null} onCancel={() => undefined} readOnlyReason={null} />);
    expect(empty.container.firstChild).toBeNull();
    expect(container).toBeTruthy();
  });
});

describe("PortfolioCard with cash set aside", () => {
  const held: AccountView = { ...replayAccount, cash: 10000, available: 4900, reserved_cash: 5100, invested: 400, total: 10400, positions: [{ symbol: "MSFT", name: "Microsoft Corp", shares: 4, cost_basis: 100, value: 400, reserved: 3 }] };
  it("the cash is all of it, the funds available are what may be spent, and a waiting sale's shares are named", () => {
    render(<PortfolioCard view={held} changed={null} before={null} />);
    expect(screen.getByText("$5,100.00 set aside for waiting orders · $4,900.00 available to trade")).toBeTruthy();
    const row = screen.getByText("Funds available").closest("tr")!;
    expect(within(row).getByText("$4,900.00")).toBeTruthy();
    expect(screen.getByText("3 set aside for a waiting sale")).toBeTruthy();
  });
  it("with nothing set aside it reads as it always did", () => {
    render(<PortfolioCard view={replayAccount} changed={null} before={null} />);
    expect(screen.getByText("available to trade")).toBeTruthy();
    expect(screen.queryByText(/set aside/)).toBeNull();
  });
});
