/* SPDX-License-Identifier: MIT
 *
 * Does the episode viewer actually animate?
 *
 * A still render is not a test of playback. Two viewers in this project
 * shipped frozen while every screenshot looked perfect, because the harness
 * that produced the screenshots drove frames by hand and never exercised the
 * requestAnimationFrame loop. So this script loads the real page from the real
 * server, watches the step readout move on its own, and then exercises Pause
 * and Play.
 *
 * It is not a pytest test: it needs a Chrome binary and a GPU-ish context,
 * neither of which the Python CI image has. Run it by hand against a served
 * results directory:
 *
 *   pomdp-report serve <runs_dir> --port 8765 &
 *   node scripts/check_viewer_playback.mjs http://127.0.0.1:8765/<episode-url>
 *
 * Exits non-zero on a frozen page or any console error.
 */
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [, , pageUrl] = process.argv;
if (!pageUrl) {
  console.error("usage: node check_viewer_playback.mjs <episode page url>");
  process.exit(2);
}

const CHROME = process.env.CHROME_BIN
  || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9000 + Math.floor(Math.random() * 350);
const userDir = mkdtempSync(join(tmpdir(), "pomdp-play-"));

const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${userDir}`,
  "--window-size=1280,720", "--hide-scrollbars",
  // Headless Chrome has no GPU, so the composite runs on SwiftShader. It is
  // dimmer and rougher than a real GPU; judge motion here, not fine filtering.
  "--enable-unsafe-swiftshader", "--use-gl=angle",
  pageUrl
], { stdio: ["ignore", "ignore", "pipe"] });
let stderr = "";
chrome.stderr.on("data", (d) => { stderr += d.toString(); });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let tab;
for (let i = 0; i < 80 && !tab; i++) {
  try {
    const tabs = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    tab = tabs.find((x) => x.type === "page" && x.webSocketDebuggerUrl);
  } catch { /* Chrome is not up yet */ }
  if (!tab) await sleep(250);
}
if (!tab) {
  console.error("Chrome never opened a debuggable tab.\n" + stderr.slice(-1200));
  process.exit(1);
}

const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let nextId = 1;
const pending = new Map();
ws.onmessage = (event) => {
  const m = JSON.parse(event.data);
  if (m.id && pending.has(m.id)) {
    const p = pending.get(m.id); pending.delete(m.id);
    m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result);
  }
};
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = nextId++; pending.set(id, { resolve, reject });
  ws.send(JSON.stringify({ id, method, params }));
});
await send("Runtime.enable");
await send("Log.enable").catch(() => {});

const problems = [];
ws.addEventListener("message", (event) => {
  const m = JSON.parse(event.data);
  if (m.method === "Runtime.exceptionThrown") {
    const d = m.params.exceptionDetails;
    problems.push("EXCEPTION: " + (d.exception?.description ?? d.text));
  }
  if (m.method === "Log.entryAdded" && m.params.entry.level === "error") {
    problems.push("CONSOLE ERROR: " + m.params.entry.text);
  }
});

async function ev(expression) {
  const r = await send("Runtime.evaluate", {
    expression, returnByValue: true, awaitPromise: true
  });
  if (r.exceptionDetails) {
    throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  }
  return r.result.value;
}

const readout = () => ev(`(function () {
  return {
    step: document.getElementById("hud-step").textContent,
    scrub: document.getElementById("scrub").value,
    play: document.getElementById("play").textContent,
    ret: document.getElementById("hud-return").textContent,
    belief: document.getElementById("hud-belief").textContent,
    status: document.getElementById("viewer-status").textContent,
    failed: document.getElementById("viewer-status").classList.contains("failed")
  };
})()`);

// Wait for the trace fetch and the scene build.
for (let i = 0; i < 100; i++) {
  const r = await readout();
  if (r.failed || !/Loading/.test(r.status)) break;
  await sleep(200);
}

const fails = [];

const t0 = await readout();
console.log("at load   ", t0);
if (t0.failed) fails.push("viewer reported a failure: " + t0.status);

// 1. Does it move on its own? This is the check the still harness cannot make.
await sleep(2600);
const t1 = await readout();
console.log("after 2.6s", t1);
if (Number(t1.scrub) <= Number(t0.scrub)) {
  fails.push(`playback did not advance: scrub ${t0.scrub} -> ${t1.scrub}`);
}
if (t1.step === t0.step) fails.push(`step readout did not change: still "${t1.step}"`);

/* A short episode can run to the end inside the window above and settle on
 * Replay, so the transport tests rewind first and only then assert. Without
 * this the harness reports a Pause bug on any episode shorter than a few
 * seconds, which is most of them. */
async function rewindAndPlay() {
  await ev(`(function () {
    var s = document.getElementById("scrub");
    s.value = "0";
    s.dispatchEvent(new Event("input"));           // seeking also pauses
    if (document.getElementById("play").textContent !== "Pause") {
      document.getElementById("play").click();
    }
  })()`);
  await sleep(300);
}

// 2. Does Pause stop it?
await rewindAndPlay();
await ev(`document.getElementById("play").click()`);
const p0 = await readout();
await sleep(1200);
const p1 = await readout();
console.log("paused    ", p1);
if (p1.scrub !== p0.scrub) fails.push(`Pause did not stop playback: ${p0.scrub} -> ${p1.scrub}`);
if (p1.play !== "Play") fails.push(`paused button reads "${p1.play}", expected "Play"`);

// 3. Does Play resume?
await ev(`document.getElementById("play").click()`);
const r0 = await readout();
await sleep(600);
const r1 = await readout();
console.log("resumed   ", r1);
if (Number(r1.scrub) <= Number(r0.scrub) && r1.play !== "Replay") {
  fails.push(`Play did not resume: ${r0.scrub} -> ${r1.scrub}`);
}

/* 4. Is the canvas drawing anything at all? A loop that runs while the
 *    composite outputs black is still a broken viewer.
 *
 *    The measurement is a page screenshot, not gl.readPixels: the drawing
 *    buffer is not preserved after a frame is presented, so reading it from
 *    outside the animation callback returns zeros whether the render worked
 *    or not — which reads as a failure on a page that is perfectly fine.
 */
await send("Page.enable").catch(() => {});
const shot = await send("Page.captureScreenshot", { format: "png" });
const png = Buffer.from(shot.data, "base64");
// A blank dark page compresses to almost nothing; a lit 3D scene with grain
// and bloom does not. Crude, but it separates "drew a scene" from "drew black".
console.log("screenshot bytes", png.length);
if (png.length < 20000) {
  fails.push(`canvas looks blank: screenshot is only ${png.length} bytes`);
}

console.log(problems.length ? "\nPAGE PROBLEMS:\n" + problems.slice(0, 8).join("\n") : "\nconsole clean");
console.log(fails.length ? "PLAYBACK FAILURES:\n" + fails.join("\n") : "PLAYBACK OK");
ws.close();
chrome.kill();
process.exit(fails.length || problems.length ? 1 : 0);
