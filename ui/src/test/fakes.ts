// A stand-in for the BROWSER's transport only (EventSource and fetch), never for an agent or for stock data: what flows through is the recorded stream.
import { vi } from "vitest";
import type { ServerEvent } from "../api/types";

export class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  closed = false;
  onerror: (() => void) | null = null;
  private listeners: Record<string, ((m: MessageEvent) => void)[]> = {};
  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }
  addEventListener(type: string, fn: (m: MessageEvent) => void) {
    (this.listeners[type] ??= []).push(fn);
  }
  close() {
    this.closed = true;
  }
  emit(event: ServerEvent) {
    for (const fn of this.listeners[event.type] ?? []) fn(new MessageEvent(event.type, { data: JSON.stringify(event.data), lastEventId: String(event.seq) }));
  }
  static reset() {
    FakeEventSource.instances = [];
  }
  static get last(): FakeEventSource {
    const es = FakeEventSource.instances[FakeEventSource.instances.length - 1];
    if (!es) throw new Error("no EventSource was opened");
    return es;
  }
}

export interface Call {
  method: string;
  path: string;
  body: unknown;
}

/** fetch answering with `routes` ("GET /api/health": payload); every call is recorded. A route may be a function of the body. */
export function stubFetch(routes: Record<string, unknown | ((body: unknown) => unknown)>) {
  const calls: Call[] = [];
  const fn = vi.fn(async (path: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, path, body });
    const route = routes[`${method} ${path}`];
    if (route === undefined) return new Response(JSON.stringify({ error: `no route ${method} ${path}` }), { status: 404 });
    const payload = typeof route === "function" ? (route as (b: unknown) => unknown)(body) : route;
    return new Response(JSON.stringify(payload), { status: method === "POST" && path === "/api/runs" ? 202 : 200 });
  });
  vi.stubGlobal("fetch", fn);
  return calls;
}
