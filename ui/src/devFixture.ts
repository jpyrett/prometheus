// A recorded event stream fed to the page without a run (the screenshots, a quick look):
//   /?fixture=bob              the whole recorded run
//   /?fixture=bob&at=30        the first 30 events, the clock frozen 3 s after the last one (&dt=5 for another gap)
//   /?fixture=bob&at=29&question=approve   one more event: an open question taken from fixtures/questions.json
//   &theme=light               the theme for this view only (nothing is remembered)
// It is harmless in every build: without `fixture` in the address nothing here runs. The streams are RECORDED from the server's replay (ui/scripts/record_fixture.mjs), never typed.
import type { ServerEvent } from "./api/types";

export interface FixtureFile {
  _note?: string;
  source?: string;
  events: ServerEvent[];
}

export interface FixtureParams {
  name: string;
  at: number | null;
  dt: number;
  question: string | null;
  theme: "dark" | "light" | null;
}

export function parseFixtureParams(search: string): FixtureParams | null {
  const q = new URLSearchParams(search);
  const name = q.get("fixture");
  if (!name || !/^[\w-]+$/.test(name)) return null;
  const at = q.get("at");
  const theme = q.get("theme");
  return {
    name,
    at: at !== null && /^\d+$/.test(at) ? Number(at) : null,
    dt: Number(q.get("dt") ?? 3) || 0,
    question: q.get("question"),
    theme: theme === "light" || theme === "dark" ? theme : null,
  };
}

const loaders = import.meta.glob("./fixtures/*.json", { import: "default" }) as Record<string, () => Promise<unknown>>;

export async function loadFixture(params: FixtureParams): Promise<{ events: ServerEvent[]; now: number } | null> {
  const load = loaders[`./fixtures/${params.name}.json`];
  if (!load) return null;
  const file = (await load()) as FixtureFile;
  let events = file.events.slice(0, params.at ?? file.events.length);
  if (params.question) {
    const questions = (await (loaders["./fixtures/questions.json"]?.() ?? Promise.resolve(null))) as Record<string, ServerEvent> | null;
    const q = questions?.[params.question];
    const last = events[events.length - 1];
    if (q && last) events = [...events, { ...q, seq: last.seq + 1, data: { ...q.data, ts: last.data.ts } } as ServerEvent];
  }
  const last = events[events.length - 1];
  return { events, now: (last ? last.data.ts : Date.now() / 1000) + params.dt };
}
