#!/usr/bin/env node
// Record the event stream of one run from a running replay server into a fixture file (never typed by hand).
//
//   1. start the replay server (free: no key, no network, no database), for example
//        ~/miniconda3/envs/cmu/bin/python -m prometheus.application.web --replay tests/fixtures/session_real/bob_jones_oil.json --replay-speed 1.0 --port 8100
//   2. node scripts/record_fixture.mjs --server http://127.0.0.1:8100 --out src/fixtures/bob.json
//
// With --api src/fixtures/api.json it also saves what GET /api/health and GET /api/accounts/0 answer (the tests mock fetch with them).
// A question the replay asks (Ask mode: the confirmation of what was understood) is answered with --answer yes (default) or --answer no.
// The file is {_note, source, events: [{seq, type, data}]}: exactly what the page received.
import { writeFileSync } from "node:fs";

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
const server = args.get("server") ?? "http://127.0.0.1:8100";
const out = args.get("out");
const apiOut = args.get("api");
const answerYes = (args.get("answer") ?? "yes") !== "no";
if (!out) {
  console.error("usage: node scripts/record_fixture.mjs --server http://127.0.0.1:8100 --out src/fixtures/NAME.json [--answer yes|no]");
  process.exit(2);
}

const health = await (await fetch(`${server}/api/health`)).json();
if (health.mode !== "replay") {
  console.error("refusing: this server is not in replay mode (a live run is a paid run)");
  process.exit(2);
}
if (apiOut) {
  const account = await (await fetch(`${server}/api/accounts/0`)).json();
  writeFileSync(apiOut, JSON.stringify({ _note: `GET /api/health and GET /api/accounts/0 of the replay server over ${health.replay_file}, saved by scripts/record_fixture.mjs.`, health, account }, null, 1) + "\n");
  console.log(`wrote ${apiOut}`);
}
const started = await fetch(`${server}/api/runs`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
if (!started.ok) {
  console.error("could not start a run:", started.status, await started.text());
  process.exit(1);
}
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
    if (event.type === "question") {
      const r = await fetch(`${server}/api/runs/${runId}/answers`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question_id: event.data.question_id, answer: answerYes }) });
      if (!r.ok) console.error("the answer was refused:", r.status);
    }
    if (event.type === "run_finished") finished = true;
  }
}
await reader.cancel().catch(() => undefined);

const source = events.find((e) => e.type === "run_started")?.data.source ?? health.replay_file;
const file = { _note: `Recorded from the replay server over ${source} by scripts/record_fixture.mjs (run ${runId}). Every event is exactly what the page received; nothing was edited.`, source, events };
writeFileSync(out, JSON.stringify(file, null, 1) + "\n");
console.log(`wrote ${out}: ${events.length} events from ${source}`);
