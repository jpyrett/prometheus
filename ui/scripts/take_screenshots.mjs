#!/usr/bin/env node
// Retake docs/ui_screenshots/*.png against a REPLAY server (free: no key, no network, no database) with the built page (corepack npm@10.9.2 run build).
//
//   ~/miniconda3/envs/cmu/bin/python -m prometheus.application.web --replay tests/fixtures/session_real/bob_jones_oil.json --replay-speed 0.5 --port 8100
//   node ui/scripts/take_screenshots.mjs [--server http://127.0.0.1:8100] [--ask-server http://127.0.0.1:8101] [--out docs/ui_screenshots]
//
// --ask-server (optional) is a second replay server over an Ask-mode capture (bill_gates_direct_order_funded.json): it adds 03b, the REAL confirm question of a run (the replay asks it before any stage starts, so no stage waits).
//
// 01, 02, 04, 05, 06, 07 are REAL runs of the replay through a real browser (Go! is pressed, the SSE stream is read). 03 and 08 use the page's dev views:
// ?fixture=bob&question=approve (a recorded stream with a recorded question) and ?states (the sheet of component states).
import { readFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { launch } from "./cdp.mjs";

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
const server = args.get("server") ?? "http://127.0.0.1:8100";
const askServer = args.get("ask-server");
const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(args.get("out") ?? join(here, "..", "..", "docs", "ui_screenshots"));
mkdirSync(out, { recursive: true });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function page(opts, fn) {
  const b = await launch(opts);
  try {
    await fn(b);
    if (b.logs.length) console.log("  console:", b.logs.slice(0, 3));
  } finally {
    await b.close();
  }
}
const ready = (b) => b.waitFor(`!!document.querySelector('textarea') && !!document.querySelector('.btn.go') && !document.querySelector('.btn.go').disabled`);
const completed = (b) => b.waitFor(`document.querySelector('.runbox .rb-label')?.textContent === 'COMPLETED'`, 120000);
async function fullPage(b, w) {
  const h = await b.eval(`Math.max(document.documentElement.scrollHeight, innerHeight)`);
  await b.resize(w, h);
  await wait(400);
}

// 01: idle, the default prompt
await page({ dark: true, port: 9551 }, async (b) => {
  await b.goto(`${server}/`);
  await ready(b);
  await b.shot(join(out, "01_idle_dark.png"));
  console.log("01_idle_dark");
});

// 02: a stage busy in the middle of a real run, Activity filling
await page({ dark: true, port: 9552 }, async (b) => {
  await b.goto(`${server}/`);
  await ready(b);
  await b.eval(`document.querySelector('.btn.go').click()`);
  await b.waitFor(`document.querySelector('.step.busy .label')?.textContent.includes('Critique')`, 60000);
  await wait(1100);
  await b.shot(join(out, "02_busy_dark.png"));
  console.log("02_busy_dark");
});

// 03: a question open (the Account Manager asks to approve a trade): the stage waits, Activity says so
{
  const bob = JSON.parse(readFileSync(join(here, "..", "src", "fixtures", "bob.json"), "utf8"));
  const at = bob.events.find((e) => e.type === "stage" && e.data.key === "trade" && e.data.state === "busy").seq;
  await page({ dark: true, port: 9553 }, async (b) => {
    await b.goto(`${server}/?fixture=bob&at=${at}&question=approve&theme=dark&dt=2.4`);
    await b.waitFor(`!!document.querySelector('.qcard')`, 15000);
    await wait(600);
    await b.shot(join(out, "03_question_dark.png"));
    console.log("03_question_dark");
  });
}

// 03b: the real confirm question of an Ask-mode run (optional)
if (askServer) {
  await page({ dark: true, port: 9563 }, async (b) => {
    await b.goto(`${askServer}/`);
    await ready(b);
    await b.eval(`document.querySelector('.btn.go').click()`);
    await b.waitFor(`!!document.querySelector('.qcard')`, 15000);
    await wait(500);
    await b.shot(join(out, "03b_question_confirm_real_dark.png"));
    console.log("03b_question_confirm_real_dark");
  });
}

// 04 and 05: finished (real runs), a row selected, the recommendation showing; dark and light
for (const [name, dark, port] of [["04_finished_dark", true, 9554], ["05_finished_light", false, 9555]]) {
  await page({ dark, port }, async (b) => {
    await b.goto(`${server}/`);
    await ready(b);
    await b.eval(`document.querySelector('.btn.go').click()`);
    await completed(b);
    await wait(900);
    await b.shot(join(out, `${name}.png`));
    console.log(name);
  });
}

// 06 and 07: the same at 1100 and 800 px wide (the page scrolls: the whole page is taken)
for (const [name, width, port] of [["06_finished_1100", 1100, 9556], ["07_finished_800", 800, 9557]]) {
  await page({ dark: true, port, width, height: 900 }, async (b) => {
    await b.goto(`${server}/`);
    await ready(b);
    await b.eval(`document.querySelector('.btn.go').click()`);
    await completed(b);
    await wait(900);
    await fullPage(b, width);
    await b.shot(join(out, `${name}.png`));
    console.log(name);
  });
}

// 08: the sheet of component states
await page({ dark: true, port: 9558 }, async (b) => {
  await b.goto(`${server}/?states&theme=dark`);
  await b.waitFor(`document.querySelectorAll('.gal .card').length >= 6`, 15000);
  await wait(500);
  await fullPage(b, 1440);
  await b.shot(join(out, "08_states_dark.png"));
  console.log("08_states_dark");
});
