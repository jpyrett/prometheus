import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { bobEvents } from "../test/fixtures";
import { FakeEventSource } from "../test/fakes";
import { connectEvents } from "./events";
import type { ServerEvent } from "./types";

beforeEach(() => {
  FakeEventSource.reset();
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

const factory = (url: string) => new FakeEventSource(url) as unknown as EventSource;

describe("connectEvents", () => {
  it("reconnects with the last seq it has, applies nothing twice, and closes itself at run_finished", () => {
    const seen: ServerEvent[] = [];
    const dropped = vi.fn();
    connectEvents("r1", { onEvent: (e) => seen.push(e), onReconnect: dropped, factory, retryMs: 500 });
    const first = FakeEventSource.last;
    expect(first.url).toBe("/api/runs/r1/events?after=0");

    for (const e of bobEvents.slice(0, 5)) first.emit(e);
    expect(seen.map((e) => e.seq)).toEqual([1, 2, 3, 4, 5]);

    first.onerror?.(); // the connection drops
    expect(first.closed).toBe(true);
    expect(dropped).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(500);
    const second = FakeEventSource.last;
    expect(second).not.toBe(first);
    expect(second.url).toBe("/api/runs/r1/events?after=5");

    for (const e of bobEvents.slice(3, 8)) second.emit(e); // the server repeats 4 and 5: they are not applied again
    expect(seen.map((e) => e.seq)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);

    const finished = bobEvents[bobEvents.length - 1]!;
    for (const e of bobEvents.slice(8)) second.emit(e);
    expect(seen[seen.length - 1]).toEqual(finished);
    expect(second.closed).toBe(true); // otherwise the browser would reconnect for ever
    second.onerror?.();
    vi.advanceTimersByTime(5000);
    expect(FakeEventSource.instances).toHaveLength(2);
  });
});
