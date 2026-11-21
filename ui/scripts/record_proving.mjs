#!/usr/bin/env node
// Record the Proving Ground's answers and event streams from a running REPLAY server into ui/src/fixtures/proving_*.json (never typed by hand). Free: the replay server calls no model, no quote and writes no account.
//
//   ~/miniconda3/envs/cmu/bin/python -m prometheus.application.web --replay --port 8011
//   node ui/scripts/record_proving.mjs [--server http://127.0.0.1:8011] [--out ui/src/fixtures] [--only critic]      (--only critic records the report card of the Critic's rules and nothing else: the other fixtures hold the recordings the tests count)
//
// WHICH RUNS: the two real recordings of tests/fixtures/proving_real/ (RECORDINGS below: the real fast six-month run, and the short run). Their run ids are read from those files, so the script records
// exactly the runs that were really made, not whatever else the replay server's store holds. It writes, for each recording <name> (the file's name without .json):
//   proving_run_view_<name>.json        GET  /api/proving/runs/<id>                         the saved run as the page's view
//   proving_day_<name>_<day>.json       GET  /api/proving/runs/<id>/days/<day>              the desk of the run's first decision day, and of the first one that bought (when it is another)
//   proving_stream_<name>.json          POST /api/proving/runs/<id>/replay?speed=0          the event stream as {_note, source, events}, one event a line
//   proving_run_view_thin_news.json     GET  /api/proving/runs/<id> of one more real run of the replay server's store (EXTRA_VIEWS below): the view only
// and once:
//   proving_coverage.json               GET  /api/proving/coverage
//   proving_plan.json / proving_plan_faithful.json   POST /api/proving/plan for the page's first quick prompt in each speed (`request` is the body that was sent)
//   proving_runs_index.json             GET  /api/proving/runs
//   proving_track_view.json             GET  /api/proving/track
//   proving_critic_report.json          GET  /api/proving/critic-report                     the report card of the Critic's rules (Critic v2): every saved run of the replay server's store, the recorded ones and the Track Record
//   proving_live_replay.json            GET  /api/proving/live/8277
// The server must be in --replay mode: this script refuses otherwise (a live server's Launch is a paid run, and this script never presses it).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
const server = args.get("server") ?? "http://127.0.0.1:8011";
const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(args.get("out") ?? join(here, "..", "src", "fixtures"));
const real = join(here, "..", "..", "tests", "fixtures", "proving_real");
mkdirSync(out, { recursive: true });

/** The recordings the page's tests use: [file of tests/fixtures/proving_real, why]. */
const RECORDINGS = [
  ["oil_weekly_6m_fast.json", "the real fast six-month run: 23 weekly decision days, buys, exits"],
  ["oil_weekly_short.json", "the real short run: 4 decision days"],
];

/** Saved runs of the replay server's own store that the tests also need, recorded as the page's VIEW only (no stream): [run id, name, why]. This one has decision days with thin news, which no recording of tests/fixtures/proving_real has. */
const EXTRA_VIEWS = [["tm-20261120-222822-1221", "thin_news", "a later real fast run of the same request: three decision days with thin stored news (the caveat news_coverage_thin)"]];

/** The request the page's FIRST quick prompt fills in (src/proving/LaunchPad.tsx QUICK_PROMPTS[0], and a test says the two are one): the research of these 23 days is in the cache, so Launch replays from it in seconds. */
const WARM = { prompt: "Buy the 3 most recommended oil stocks with $3,000, split evenly", cash: 10000, start: "2026-06-08", end: "2026-11-09", cadence: "weekly", risk: "moderate", neutral: true, strict: false };

const health = await (await fetch(`${server}/api/health`)).json();
if (health.mode !== "replay") {
  console.error("refusing: this server is not in replay mode (a live run is a paid run)");
  process.exit(2);
}

const note = (source) => `Recorded from the replay server by scripts/record_proving.mjs (${source}). The answer in \`data\` is exactly what the page received.`;
const save = (name, source, data, extra = {}) => {
  writeFileSync(join(out, name), JSON.stringify({ _note: note(source), source, ...extra, data }, null, 1) + "\n");
  console.log(`wrote ${name}`);
};
const get = async (path) => {
  const r = await fetch(`${server}${path}`);
  return r.ok ? await r.json() : null;
};

if (args.get("only") === "critic") {
  const card = await get("/api/proving/critic-report");
  if (card) save("proving_critic_report.json", "GET /api/proving/critic-report", card);
  else console.error("the replay server has no report card");
  process.exit(card ? 0 : 1);
}

const coverage = await get("/api/proving/coverage");
if (coverage) save("proving_coverage.json", "GET /api/proving/coverage", coverage);

for (const [speed, name] of [["fast", "proving_plan.json"], ["faithful", "proving_plan_faithful.json"]]) {
  const request = { ...WARM, speed };
  const r = await fetch(`${server}/api/proving/plan`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(request) });
  if (r.ok) save(name, `POST /api/proving/plan (${speed})`, await r.json(), { request });
  else console.error(`the plan for ${speed} was refused: ${r.status}`);
}

const index = await get("/api/proving/runs");
if (index) save("proving_runs_index.json", "GET /api/proving/runs", index);
const track = await get("/api/proving/track");
if (track) save("proving_track_view.json", "GET /api/proving/track", track);
const critic = await get("/api/proving/critic-report");
if (critic) save("proving_critic_report.json", "GET /api/proving/critic-report", critic);

async function stream(id) {
  const started = await fetch(`${server}/api/proving/runs/${encodeURIComponent(id)}/replay?speed=0`, { method: "POST" });
  if (!started.ok) throw new Error(`could not replay ${id}: ${started.status}`);
  const { run_id: runId } = await started.json();
  const response = await fetch(`${server}/api/runs/${runId}/events`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const events = [];
  let buffer = "";
  let finished = false;
  while (!finished) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const fields = {};
      for (const line of block.split("\n")) {
        if (line.startsWith(":")) continue;
        const colon = line.indexOf(": ");
        if (colon > 0) fields[line.slice(0, colon)] = line.slice(colon + 2);
      }
      if (!fields.data) continue;
      const event = { seq: Number(fields.id), type: fields.event, data: JSON.parse(fields.data) };
      events.push(event);
      if (event.type === "run_finished") finished = true;
    }
  }
  await reader.cancel().catch(() => undefined);
  return events;
}

for (const [file, why] of RECORDINGS) {
  const path = join(real, file);
  if (!existsSync(path)) {
    console.log(`skipped ${file}: not recorded yet`);
    continue;
  }
  const id = JSON.parse(readFileSync(path, "utf8")).run_id;
  const name = file.replace(/\.json$/, "");
  const view = await get(`/api/proving/runs/${encodeURIComponent(id)}`);
  if (!view) {
    console.error(`the replay server has no saved run ${id} (${file})`);
    continue;
  }
  save(`proving_run_view_${name}.json`, `GET /api/proving/runs/${id}`, view, { recording: `tests/fixtures/proving_real/${file}`, why });
  // the desk of the first decision day, and of the first day that bought something when that is another day
  const desks = view.desks ?? [];
  const bought = (view.decision_days ?? []).find((d) => d.traded > 0 && desks.includes(d.day))?.day;
  for (const day of [...new Set([desks[0], bought].filter(Boolean))]) {
    const d = await get(`/api/proving/runs/${encodeURIComponent(id)}/days/${day}`);
    if (d) save(`proving_day_${name}_${day}.json`, `GET /api/proving/runs/${id}/days/${day}`, d);
  }
  const events = await stream(id);
  const header = { _note: `The event stream of POST /api/proving/runs/${id}/replay?speed=0 on the replay server, recorded by scripts/record_proving.mjs from tests/fixtures/proving_real/${file}. Every event is exactly what the page received.`, source: `saved run ${id} (${file})` };
  const text = `{${Object.entries(header).map(([k, v]) => `${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n")},\n"events": [\n${events.map((e) => JSON.stringify(e)).join(",\n")}\n]}\n`;
  writeFileSync(join(out, `proving_stream_${name}.json`), text);
  console.log(`wrote proving_stream_${name}.json: ${events.length} events, ${(text.length / 1e6).toFixed(2)} MB`);
}

for (const [id, name, why] of EXTRA_VIEWS) {
  const view = await get(`/api/proving/runs/${encodeURIComponent(id)}`);
  if (view) save(`proving_run_view_${name}.json`, `GET /api/proving/runs/${id}`, view, { why });
  else console.error(`the replay server has no saved run ${id}`);
}

const live = await get("/api/proving/live/8277");
if (live) save("proving_live_replay.json", "GET /api/proving/live/8277", live);
