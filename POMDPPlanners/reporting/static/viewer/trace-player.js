/* SPDX-License-Identifier: MIT
 *
 * Wires a trace.json to the renderer core and a scene module.
 *
 * There is no fallback episode. If the trace does not load, or its payload
 * kind has no scene module, the page says so and draws nothing. A viewer that
 * quietly plays something invented would be worse than an empty one: these
 * pages are a record of what a planner actually did, including the episodes
 * where it did badly, and nothing in the UI may suggest otherwise.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var root = document.getElementById("viewer");
  if (!root) return;

  var statusEl = document.getElementById("viewer-status");
  function status(message, failed) {
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.classList.toggle("failed", !!failed);
  }

  if (!global.THREE || !V) {
    status("The 3D renderer could not start in this browser.", true);
    return;
  }

  var canvas = document.getElementById("viewer-canvas");
  var el = {
    play: document.getElementById("play"),
    scrub: document.getElementById("scrub"),
    speed: document.getElementById("speed"),
    step: document.getElementById("hud-step"),
    action: document.getElementById("hud-action"),
    pos: document.getElementById("hud-pos"),
    reward: document.getElementById("hud-reward"),
    ret: document.getElementById("hud-return"),
    belief: document.getElementById("hud-belief")
  };

  fetch(root.dataset.trace, { cache: "no-store" })
    .then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    })
    .then(start)
    .catch(function (error) {
      status("Could not load this episode's trace: " + error.message, true);
    });

  function start(trace) {
    var module = V.scenes[trace.payload_kind];
    if (!module) {
      status(
        "No viewer for payload kind “" + trace.payload_kind + "” yet.",
        true
      );
      return;
    }

    var core = V.createCore({
      canvas: canvas,
      stage: root,
      exposure: module.exposure
    });
    if (!core) {
      status("WebGL is unavailable, so this episode cannot be drawn.", true);
      return;
    }

    var scene;
    try {
      scene = module.build(core, trace);
    } catch (error) {
      status("This trace could not be drawn: " + error.message, true);
      return;
    }

    core.resize();
    global.addEventListener("resize", core.resize);

    // The scene frames itself: the board's size comes from the trace.
    var rig = V.createCameraRig(core, canvas, scene.camera || module.camera);
    var camButtons = Array.prototype.slice.call(root.querySelectorAll("[data-cam]"));
    rig.onModeChange = function (mode) {
      camButtons.forEach(function (button) {
        button.setAttribute("aria-pressed", button.dataset.cam === mode ? "true" : "false");
      });
    };
    camButtons.forEach(function (button) {
      button.addEventListener("click", function () { rig.setMode(button.dataset.cam); });
    });

    var player = V.createPlayer({
      steps: scene.steps,
      onFrame: function (t, dt, elapsed) {
        var hud = scene.update(t, dt, elapsed, player.state.playing);
        rig.update(dt, hud.follow);
        core.renderComposite(elapsed);
        updateHud(hud, t);
      }
    });

    el.scrub.max = String(Math.max(0, scene.steps - 1));
    el.scrub.step = "0.01";
    el.play.textContent = player.state.playing ? "Pause" : "Play";
    player.onEnd = function () { el.play.textContent = "Replay"; };

    el.play.addEventListener("click", function () {
      player.toggle();
      el.play.textContent = player.state.playing ? "Pause" : "Play";
    });
    el.scrub.addEventListener("input", function () {
      player.seek(parseFloat(el.scrub.value));
      player.pause();
      el.play.textContent = "Play";
    });
    el.speed.addEventListener("change", function () {
      player.setSpeed(parseFloat(el.speed.value));
    });

    function updateHud(hud, t) {
      if (!player.state.playing) {
        // Leave the scrubber alone while it is being dragged: writing to it
        // during the drag fights the pointer.
      } else {
        el.scrub.value = String(t);
      }
      el.step.textContent = "step " + hud.step + " / " + (scene.steps - 1);
      el.action.textContent = "action " + hud.action;
      /* A scene may name its own position readout. "x 3.00  y 5.00" is right
         for a continuous world and reads oddly on a grid, where the honest
         labels are row and column. A scene returns `hud.pos` when it wants to
         say it itself; `hud.x`/`hud.y` stay the default so existing scenes and
         continuous worlds are unaffected. */
      el.pos.textContent = hud.pos !== undefined && hud.pos !== null
        ? hud.pos
        : "x " + hud.x.toFixed(2) + "  y " + hud.y.toFixed(2);
      el.reward.textContent = hud.reward === null || hud.reward === undefined
        ? "reward —"
        : "reward " + hud.reward.toFixed(2);
      el.ret.textContent = "return " + hud.ret.toFixed(2);
      el.belief.textContent = "belief: " + hud.belief;
    }

    var outcome = trace.reach_terminal_state ? "ended in a terminal state" : "ran out of steps";
    status(
      "Replaying episode " + trace.episode_index + " of " + (trace.policy || "an unnamed policy") +
      " — " + trace.num_steps + " steps, discounted return " +
      trace.discounted_return.toFixed(2) + ", " + outcome + "."
    );

    // The loop always runs. A still render is not a test of playback: a viewer
    // that only draws on demand looks perfect in a screenshot and ships frozen.
    player.start();
  }
})(window);
