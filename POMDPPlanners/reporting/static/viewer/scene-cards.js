/* SPDX-License-Identifier: MIT
 *
 * The scene on every episode card — a still by default, all of them playing
 * at once when the page is switched to live.
 *
 * One WebGL context for the whole page, not one per card: browsers cap live
 * contexts at around sixteen and silently drop the oldest, so a grid of
 * viewers would start blanking out as it grew. Every episode is built once
 * into its own detached group; to draw one, its group is attached to the
 * shared scene, rendered on one offscreen canvas, and the pixels are copied
 * into that card's own 2D canvas. A 2D canvas costs nothing to keep, so all
 * of them can hold a frame at the same time.
 *
 * Live mode draws a few cards per frame rather than all of them, and each one
 * advances by the time since it was last drawn. Twelve episodes therefore run
 * slower per card but stay in step with the clock, instead of dropping the
 * page's frame rate to a slideshow.
 *
 * A card that cannot be drawn — no WebGL, no scene module for its payload
 * kind, a trace that fails to load — keeps the recorded image it was served
 * with. The recording is only hidden once a frame has actually been drawn
 * over it, so a failure leaves the page as it was.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var cards = Array.prototype.slice.call(document.querySelectorAll("[data-thumb-trace]"));
  if (!cards.length || !global.THREE || !V) return;

  // Drawn larger than shown, so the copy stays sharp on a dense screen.
  var W = 640;
  var H = 480;
  // How many cards advance per animation frame. Above this the frame budget,
  // not the episode, decides what the reader sees.
  var PER_FRAME = 3;
  var STEPS_PER_SECOND = 1.6;

  var stage = document.createElement("div");
  stage.style.cssText =
    "position:fixed;left:-10000px;top:0;width:" + W + "px;height:" + H + "px;pointer-events:none";
  var canvas = document.createElement("canvas");
  stage.appendChild(canvas);
  document.body.appendChild(stage);

  var core = V.createCore({ canvas: canvas, stage: stage, preserveDrawingBuffer: true });
  if (!core) {
    document.body.removeChild(stage);
    return;
  }
  core.resize();

  var built = [];
  var live = false;
  var cursor = 0;

  function attach(entry) {
    core.composite.uniforms.exposure.value = entry.exposure;
    core.scene.add(entry.group);
  }

  function detach(entry) {
    core.scene.remove(entry.group);
  }

  function drawFrame(entry, t, dt, elapsed) {
    attach(entry);
    var hud = entry.scene.update(t, dt, elapsed, live);
    entry.rig.update(dt, hud && hud.follow);
    core.renderComposite(elapsed);
    entry.ctx.drawImage(canvas, 0, 0, W, H);
    detach(entry);

    if (entry.card.hidden) {
      entry.card.hidden = false;
      var recorded = entry.card.parentNode.querySelector(".thumb-recorded");
      if (recorded) recorded.hidden = true;
    }
  }

  function prepare(card) {
    return fetch(card.dataset.thumbTrace, { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (trace) {
        var module = V.scenes[trace.payload_kind];
        if (!module) return;

        var exposure = module.exposure === undefined ? 0.155 : module.exposure;
        core.composite.uniforms.exposure.value = exposure;

        // A scene module adds to core.scene directly, so what it added is
        // whatever is new afterwards; those objects move into a group that can
        // be attached and detached as a unit.
        var before = core.scene.children.slice();
        var scene = module.build(core, trace);
        var group = new global.THREE.Group();
        core.scene.children.slice().forEach(function (child) {
          if (before.indexOf(child) === -1) group.add(child);
        });

        card.width = W;
        card.height = H;
        var entry = {
          card: card,
          ctx: card.getContext("2d"),
          scene: scene,
          group: group,
          exposure: exposure,
          rig: V.createCameraRig(core, canvas, scene.camera || module.camera),
          steps: Math.max(1, scene.steps),
          t: 0,
          elapsed: 0,
          last: 0
        };

        // Warm up on a middle frame: the first step is the agent at its start
        // with nothing shown yet, the least informative frame there is.
        var still = (entry.steps - 1) * 0.5;
        for (var frame = 0; frame < 40; frame++) {
          attach(entry);
          var hud = entry.scene.update(still, 1 / 30, frame / 30, false);
          entry.rig.update(1 / 30, hud && hud.follow);
          detach(entry);
        }
        entry.t = still;
        entry.elapsed = 40 / 30;
        drawFrame(entry, still, 1 / 30, entry.elapsed);
        built.push(entry);
      })
      .catch(function () {
        /* The card keeps the recorded image it was served with. */
      });
  }

  var queue = cards.slice();
  function prepareNext() {
    var card = queue.shift();
    if (!card) return;
    prepare(card).then(function () {
      (global.requestIdleCallback || global.requestAnimationFrame)(prepareNext);
    });
  }
  prepareNext();

  var previous = 0;
  function tick(now) {
    if (!live) return;
    global.requestAnimationFrame(tick);
    if (!built.length) return;

    var wall = previous ? Math.min((now - previous) / 1000, 0.25) : 0;
    previous = now;

    for (var drawn = 0; drawn < Math.min(PER_FRAME, built.length); drawn++) {
      var entry = built[cursor % built.length];
      cursor++;
      // Each card advances by the time since it was last drawn, not by this
      // frame's delta, so a card drawn every fourth frame still plays at speed.
      var dt = entry.last ? Math.min((now - entry.last) / 1000, 0.25) : wall;
      entry.last = now;
      entry.elapsed += dt;
      entry.t += dt * STEPS_PER_SECOND;
      // On repeat, and from the start: an episode's opening is where the
      // belief is widest, which is the part worth seeing again.
      if (entry.t > entry.steps - 1) entry.t = 0;
      drawFrame(entry, entry.t, dt, entry.elapsed);
    }
  }

  function setLive(on) {
    live = on;
    document.querySelectorAll("[data-live]").forEach(function (button) {
      button.setAttribute("aria-pressed", on ? "true" : "false");
    });
    if (on) {
      previous = 0;
      built.forEach(function (entry) { entry.last = 0; });
      global.requestAnimationFrame(tick);
    }
  }

  Array.prototype.slice.call(document.querySelectorAll("[data-live]")).forEach(function (button) {
    button.addEventListener("click", function () { setLive(!live); });
  });

  // A page left running in a background tab burns a GPU for nobody.
  document.addEventListener("visibilitychange", function () {
    if (document.hidden && live) setLive(false);
  });
})(window);
