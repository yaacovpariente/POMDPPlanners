/* SPDX-License-Identifier: MIT
 *
 * Episode cards showing the 3D scene rather than the recorded GIF.
 *
 * One WebGL context for the whole page, not one per card: browsers cap the
 * number of live contexts at around sixteen and drop the oldest without
 * warning, so a grid of viewers would start blanking out as it grew. Each card
 * is drawn in turn on one offscreen canvas and the pixels are copied into the
 * card's own 2D canvas, which costs nothing to keep.
 *
 * A card that cannot be drawn — no WebGL, no scene module for that payload
 * kind, a trace that fails to load — keeps the recorded GIF it was served
 * with. The GIF is never removed until a frame has actually been copied over
 * it, so a failure leaves the page as it was rather than empty.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var cards = Array.prototype.slice.call(document.querySelectorAll("[data-thumb-trace]"));
  if (!cards.length || !global.THREE || !V) return;

  // Drawn larger than it is shown, so the copy stays sharp on a dense screen.
  var W = 640;
  var H = 480;

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

  function clearScene() {
    // The scene is reused across cards; three keeps no registry of what a
    // module added, so everything is removed by hand between builds.
    for (var i = core.scene.children.length - 1; i >= 0; i--) {
      core.scene.remove(core.scene.children[i]);
    }
  }

  function drawOne(card) {
    return fetch(card.dataset.thumbTrace, { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (trace) {
        var module = V.scenes[trace.payload_kind];
        if (!module) return;

        clearScene();
        core.composite.uniforms.exposure.value =
          module.exposure === undefined ? 0.155 : module.exposure;

        var scene = module.build(core, trace);
        var rig = V.createCameraRig(core, canvas, scene.camera || module.camera);

        // Midway through the episode: the first step is the agent at its start
        // with nothing yet shown, which is the least informative frame there is.
        var t = (scene.steps - 1) * 0.5;
        var hud;
        for (var frame = 0; frame < 40; frame++) {
          hud = scene.update(t, 1 / 30, frame / 30, false);
          rig.update(1 / 30, hud && hud.follow);
        }
        core.renderComposite(1.0);

        var target = card;
        target.width = W;
        target.height = H;
        target.getContext("2d").drawImage(canvas, 0, 0, W, H);
        target.hidden = false;
        var gif = target.parentNode.querySelector(".thumb-recorded");
        if (gif) gif.hidden = true;
      })
      .catch(function () {
        /* The card keeps the recorded image it was served with. */
      });
  }

  // One card at a time: forty warm-up frames each, and running them in
  // parallel would only contend for the same context.
  var queue = cards.slice();
  function next() {
    var card = queue.shift();
    if (!card) {
      document.body.removeChild(stage);
      return;
    }
    drawOne(card).then(function () {
      (global.requestIdleCallback || global.requestAnimationFrame)(next);
    });
  }
  next();
})(window);
