// The EventSource of one run (spec 4.2, 5.4). Every message is `id: <seq>`, `event: <type>`, `data: <json>`.
// The connection is closed when `run_finished` arrives (the server ends the stream after the last event and a browser would reconnect for ever),
// and when it drops mid-run it is opened again with `?after=<last seq>` so no event is applied twice or missed.
import { EVENT_TYPES, type ServerEvent } from "./types";

export interface EventConnection {
  close(): void;
}

export interface ConnectOptions {
  /** The last event the page already has. */
  after?: number;
  onEvent: (event: ServerEvent) => void;
  /** Called when the connection drops and a new one is about to open. */
  onReconnect?: () => void;
  retryMs?: number;
  /** A stand-in for EventSource (tests). */
  factory?: (url: string) => EventSource;
}

export function connectEvents(runId: string, options: ConnectOptions): EventConnection {
  const make = options.factory ?? ((url: string) => new EventSource(url));
  let last = options.after ?? 0;
  let source: EventSource | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;

  const open = () => {
    if (closed) return;
    const es = make(`/api/runs/${runId}/events?after=${last}`);
    source = es;
    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (message) => {
        const seq = Number((message as MessageEvent).lastEventId);
        if (!Number.isFinite(seq) || seq <= last) return;
        last = seq;
        let data: unknown;
        try {
          data = JSON.parse((message as MessageEvent).data);
        } catch {
          return;
        }
        options.onEvent({ seq, type, data } as ServerEvent);
        if (type === "run_finished") close();
      });
    }
    es.onerror = () => {
      // The browser would reconnect by itself with Last-Event-ID; doing it here keeps `last` the only authority and lets the page tell the user.
      if (closed) return;
      es.close();
      source = null;
      options.onReconnect?.();
      timer = setTimeout(open, options.retryMs ?? 1000);
    };
  };

  function close() {
    closed = true;
    if (timer) clearTimeout(timer);
    source?.close();
    source = null;
  }

  open();
  return { close };
}
