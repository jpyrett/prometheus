"""The watcher of the waiting limit orders, as the web server runs it (design choices entry 140, W5 and W6).

    watch = OrderWatch(store)            # the live watcher of `prometheus/agents/account_manager/orders.py`, with the page's notices: built by `__main__`, started and stopped with the app (the lifespan of `create_app`)
    watch.notices.add({...})              # an event the page shows: a fill, an expiry, a cancel, a rejection
    batch, over = watch.notices.read_from(after, wait=1.0)

The watcher checks the waiting orders of every account while the server runs (`Watcher`: expire what is over, quote the symbols once in the regular session, fill what the quotes reached through the Trader funnel). It is NEVER started by
`create_app` by itself: a test, or a replay, builds the app without one, and a test that wants one starts it explicitly with a fake clock. `Notices` is the page's log of what the watcher (and the cancel route) did: numbered events a client
reads from `/api/orders/events` the way it reads a run's events, so the page can say "your order for MU was filled" without a run being open.
"""
import threading
import time

from prometheus.agents.account_manager.orders import Watcher

KEEP_NOTICES = 200      # the last notices are kept in memory (a page that reconnects reads them again)


class Notices:
    def __init__(self, keep: int = KEEP_NOTICES):
        self.events: list[dict] = []        # {seq, type, data}
        self.cond = threading.Condition()
        self.keep = keep
        self._first = 1                     # the number of the first event still kept

    def add(self, data: dict) -> int:
        """Append one event (numbered from 1) and wake the readers. Returns its number."""
        with self.cond:
            seq = self._first + len(self.events)
            self.events.append({"seq": seq, "type": "order", "data": {**data, "ts": time.time_ns() // 1_000_000 / 1000}})
            if len(self.events) > self.keep:
                drop = len(self.events) - self.keep
                del self.events[:drop]
                self._first += drop
            self.cond.notify_all()
            return seq

    def read_from(self, after: int = 0, wait=0) -> tuple[list[dict], bool]:
        """The events with a number above `after`, waiting up to `wait` seconds for one. (The second answer is always False: the stream of notices never ends by itself.)"""
        with self.cond:
            if wait > 0 and self._last() <= after:
                self.cond.wait(timeout=wait)
            return [e for e in self.events if e["seq"] > after], False

    def _last(self) -> int:
        return self._first + len(self.events) - 1


class OrderWatch:
    def __init__(self, store, notices: Notices | None = None, **options):
        """`store` is the account book; `options` go to the `Watcher` (clerk, clock, trader_factory, audit_path, seconds, lock). Its events are the page's notices."""
        self.notices = notices or Notices()
        self.watcher = Watcher(store, notify=self.notices.add, **options)

    def start(self) -> None:
        self.watcher.start()

    def stop(self) -> None:
        self.watcher.stop()
