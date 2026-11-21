// The account's waiting limit orders (design choices entry 140): loaded from GET /api/accounts/{id}/orders when the account loads, when a run ends and whenever the watcher says something; the notices are the lines of
// GET /api/orders/events (a fill, an expiry, a cancel) that came after this page was opened.
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { OrderNotice, OrdersView } from "../api/types";

const KEEP_NOTICES = 4;

export function useOrders(accountId: number | null, enabled: boolean, refreshKey: string) {
  const [view, setView] = useState<OrdersView | null>(null);
  const [notices, setNotices] = useState<OrderNotice[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<number | null>(null);
  const openedAt = useRef(Date.now() / 1000);
  const live = enabled && accountId !== null && accountId > 0;

  const load = useCallback(async () => {
    if (!live || accountId === null) {
      setView(null);
      return;
    }
    try {
      setView(await api.orders(accountId));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "the waiting orders could not be read");
    }
  }, [live, accountId]);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  // What the watcher (or a cancel) says: shown once, and the list is read again. A notice made before this page was opened is not shown.
  useEffect(() => {
    if (!live || typeof EventSource === "undefined") return;
    const source = new EventSource("/api/orders/events");
    const onNotice = (message: Event) => {
      let data: OrderNotice;
      try {
        data = JSON.parse((message as MessageEvent).data) as OrderNotice;
      } catch {
        return;
      }
      if (data.account_id !== accountId) return;
      if (data.ts >= openedAt.current) setNotices((now) => [...now, data].slice(-KEEP_NOTICES));
      void load();
    };
    source.addEventListener("order", onNotice);
    return () => source.close();
  }, [live, accountId, load]);

  const cancel = useCallback(
    async (orderId: number) => {
      setCancelling(orderId);
      try {
        await api.cancelOrder(orderId);
        setError(null);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "the order could not be cancelled");
      } finally {
        setCancelling(null);
        await load();
      }
    },
    [load],
  );

  return { view, notices, error, cancelling, cancel, reload: load };
}
