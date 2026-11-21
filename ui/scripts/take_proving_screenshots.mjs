#!/usr/bin/env node
// Take docs/ui_screenshots/proving_*.png of the Proving Ground against the REPLAY server (free: no key, no network, no database) with the built page (corepack npm@10.9.2 run build).
//
//   ~/miniconda3/envs/cmu/bin/python -m prometheus.application.web --replay --port 8011
//   node ui/scripts/take_proving_screenshots.mjs [--server http://127.0.0.1:8011] [--out docs/ui_screenshots] [--only name]
//
// Every shot is a real page in a real headless Chrome, in both themes. The run shots are the REAL fast six-month run of tests/fixtures/proving_real/oil_weekly_6m_fast.json (its id is read from that file, and the
// replay server serves it from the saved runs):
//   proving_launchpad     the launch pad with the speed switch, the warm request and the saved runs
//   proving_run           the whole run view of the saved run
//   proving_run_chart     the equity card with the markers, and the stage split of a research tile on hover
//   proving_run_research  "Watch it again" a moment after it starts: the days coming back, the playhead not yet moving
//   proving_run_timing    the "How long it took" card with a bar on hover
//   proving_run_thin_news the equity card of a later real fast run that has decision days of thin news (marked under their tiles)
//   proving_track         the Track Record's quality row; proving_live  the Live view; proving_states  the states sheet
// A view whose route answers an error is shot as it shows that error, and reported.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { launch } from "./cdp.mjs";

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
const server = args.get("server") ?? "http://127.0.0.1:8011";
const only = args.get("only");
const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(args.get("out") ?? join(here, "..", "..", "docs", "ui_screenshots"));
const real = join(here, "..", "..", "tests", "fixtures", "proving_real");
const runId = JSON.parse(readFileSync(join(real, "oil_weekly_6m_fast.json"), "utf8")).run_id;
mkdirSync(out, { recursive: true });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
let port = 9600;

/** Move the mouse to the middle of the first element `sel` matches (a hover, as a person's pointer would). */
async function hover(b, sel) {
  const p = await b.eval(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return null; const r = e.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; })()`);
  if (!p) throw new Error(`nothing to hover: ${sel}`);
  await b.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: p.x, y: p.y });
  await wait(350);
}

/** The replay server runs one thing at a time: a "Watch it again" of an earlier shot keeps playing there after its browser is closed, and the next page would show a test running. Wait until it is done. */
async function idle() {
  for (let i = 0; i < 120; i++) {
    const h = await (await fetch(`${server}/api/health`)).json();
    if (!h.active_run) return;
    await wait(500);
  }
  throw new Error("the replay server is still busy");
}

async function shot(name, url, { dark = true, width = 1440, height = 900, full = true, ready, after, hoverOn, selector, clipOnly = false } = {}) {
  if (only && !name.includes(only)) return;
  await idle();
  const b = await launch({ dark, width, height, port: port++ });
  try {
    await b.goto(`${server}/${url}`);
    if (ready) await b.waitFor(ready, 30000);
    await wait(700);
    if (after) await after(b);
    if (full) {
      const h = await b.eval(selector && !clipOnly ? `Math.ceil(document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect().bottom + scrollY) + 24` : `Math.max(document.documentElement.scrollHeight, innerHeight)`);
      await b.resize(width, Math.min(h, 12000));
      await wait(400);
    }
    // a hover is made once the page has its full height (a resize would lose the pointer)
    if (hoverOn) await hover(b, hoverOn);
    const file = join(out, `${name}_${dark ? "dark" : "light"}.png`);
    if (clipOnly && selector) {
      // the element alone, not the page around it
      const r = await b.eval(`(() => { const q = document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect(); return { x: q.left + scrollX, y: q.top + scrollY, w: q.width, h: q.height }; })()`);
      const res = await b.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: Math.max(0, r.x - 10), y: Math.max(0, r.y - 10), width: r.w + 20, height: r.h + 20, scale: 1 } });
      writeFileSync(file, Buffer.from(res.data, "base64"));
    } else {
      await b.shot(file);
    }
    console.log(`${name}_${dark ? "dark" : "light"}${b.logs.length ? `  (console: ${b.logs.slice(0, 2).join(" | ")})` : ""}`);
  } finally {
    await b.close();
  }
}

const thinId = JSON.parse(readFileSync(join(here, "..", "src", "fixtures", "proving_run_view_thin_news.json"), "utf8")).data.run_id;
const runUrl = `#proving/run/${runId}`;
const runReady = `!!document.querySelector('.runview .wall-grid') && !!document.querySelector('.rs-tile')`;

for (const dark of [true, false]) {
  const theme = dark ? "dark" : "light";
  // the sheet of states: every chart from the real recordings
  await shot("proving_states", `?states&theme=${theme}`, { dark, ready: `document.querySelectorAll('.pv-states .card').length >= 6` });
  // the launch pad: the real coverage in the range picker, the speed switch with each speed's own time (from the plans), the saved runs
  await shot("proving_launchpad", `#proving`, { dark, ready: `!!document.querySelector('.rangepicker .rp-handle') && !!document.querySelector('.lp-plan b') && [...document.querySelectorAll('.speedsw-line')].every((e) => !e.textContent.includes('reading'))`, selector: ".proving", clipOnly: true });
  // the whole saved run
  await shot("proving_run", runUrl, { dark, ready: runReady, selector: ".runview", clipOnly: true });
  // the equity card: markers on the curve, and what a research tile opens on hover
  await shot("proving_run_chart", runUrl, { dark, ready: runReady, selector: ".equity-card", clipOnly: true, hoverOn: ".rs-tile.open" });
  // "Watch it again", a moment in: the days coming back, some still being researched
  await shot("proving_run_research", runUrl, {
    dark,
    ready: runReady,
    selector: ".equity-card",
    clipOnly: true,
    full: false,
    after: async (b) => {
      await b.eval(`[...document.querySelectorAll('button')].find((x) => x.textContent.trim() === 'Watch it again').click()`);
      // the replay sends a day's research every 50 ms: watch for the moment in the page itself, not from here
      await b.eval(`new Promise((res, rej) => { const t0 = Date.now(); const t = setInterval(() => { if (document.querySelectorAll('.rs-tile.done, .rs-tile.cached').length >= 9 && document.querySelectorAll('.rs-tile.busy').length >= 3) { clearInterval(t); res(true); } else if (Date.now() - t0 > 15000) { clearInterval(t); rej(new Error("no research in sight")); } }, 15); })`);
    },
  });
  // a real run with thin news on three days: their tiles carry a mark, and the hover says how many articles were found
  await shot("proving_run_thin_news", `#proving/run/${thinId}`, { dark, ready: runReady, selector: ".equity-card", clipOnly: true, hoverOn: ".rs-tile.thin" });
  // how long it took
  await shot("proving_run_timing", runUrl, { dark, ready: `${runReady} && !!document.querySelector('.timing-card .tseg')`, selector: ".timing-card", clipOnly: true, hoverOn: ".timing-card .tseg:nth-of-type(2)" });
  // the Track Record from the route
  await shot("proving_track", `#proving/track`, { dark, ready: `!!document.querySelector('.wall-grid')`, selector: ".quality" });
  // Live from the recorded snapshot
  await shot("proving_live", `#proving/live`, { dark, ready: `!!document.querySelector('.ladder svg')`, selector: ".live" });
}
