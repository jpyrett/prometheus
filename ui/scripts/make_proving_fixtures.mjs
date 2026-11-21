#!/usr/bin/env node
// Build ui/src/fixtures/proving_*.json from the REAL recordings of the Proving Ground (tests/fixtures/proving_real/), never typed by hand.
//
//   node ui/scripts/make_proving_fixtures.mjs            (every recording that exists is copied; one that does not exist yet is reported and skipped)
//
// A single JSON answer (the Track Record, a live snapshot, a saved run) is wrapped as {_note, source, data}; the answer itself is not edited.
// The runs, their days' desks, the plans, the coverage, the index and the event streams are recorded from the replay server by scripts/record_proving.mjs (the streams as {_note, source, events} like the Desk's).
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const from = join(repo, "tests", "fixtures", "proving_real");
const to = join(here, "..", "src", "fixtures");

const WANTED = [
  ["track.json", "proving_track.json", "GET /api/proving/track"],
  ["live_snapshot_oil.json", "proving_live_oil.json", "GET /api/proving/live/{account_id}: three neutral companies, no plan"],
  ["live_snapshot_plan.json", "proving_live_plan.json", "GET /api/proving/live/{account_id}: three positions with a plan, every stop hit"],
];
// The Time Machine runs themselves (the real fast six-month run and the short run) are NOT copied from here: scripts/record_proving.mjs records their view, the desk of a day and the event stream from the replay server, which is
// what the page really receives, as proving_run_view_*.json, proving_day_*.json and proving_stream_*.json.

for (const [name, out, what] of WANTED) {
  const file = join(from, name);
  if (!existsSync(file)) {
    console.log(`skipped ${name}: not recorded yet`);
    continue;
  }
  const data = JSON.parse(readFileSync(file, "utf8"));
  const wrapped = { _note: `${what}. Copied by scripts/make_proving_fixtures.mjs from tests/fixtures/proving_real/${name}; the answer in \`data\` is not edited.`, source: `tests/fixtures/proving_real/${name}`, data };
  writeFileSync(join(to, out), JSON.stringify(wrapped, null, 1) + "\n");
  console.log(`wrote src/fixtures/${out}`);
}
