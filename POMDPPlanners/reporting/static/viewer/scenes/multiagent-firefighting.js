/* SPDX-License-Identifier: MIT
 *
 * Multi-agent firefighting scene module.
 *
 * Builds the world from a trace's `payload.world` block and moves it from the
 * trace's recorded fire maps, robot poses and beliefs. Nothing here is
 * invented: there is no fallback episode, no hand-placed robot and no
 * synthetic belief. If the player hands this module no trace, it draws
 * nothing and says so.
 *
 * Three things are specific to this environment and are the reason this file
 * is not Light-Dark's:
 *
 *  1. **The hidden wind is the POMDP.** One wind value -- four directions
 *     crossed with two strengths -- drives the flame lean, the smoke drift,
 *     the ember tracks and the canopy sway, because in the model it drives
 *     which neighbour catches. Nothing in an observation mentions it. So the
 *     belief over it gets a titled panel of its own under the viewport, as a
 *     compass rose plus eight labelled bars, computed from the run's own
 *     particles at the indices the payload names. A belief a reader has to go
 *     looking for is a belief that was not shown.
 *  2. **The fire is discrete in the model and stays so.** A cell is alight or
 *     it is not and a hose flips it in one step. What is tweened is the
 *     *rendering* across the step -- a cell catching accelerates in, a cell
 *     knocked down gutters and steams, a burnt-out cell smoulders and fades --
 *     while every count and every readout reports the step's true category.
 *  3. **Two engines, not one.** The core rig's chase mode frames one agent
 *     with a heading, which puts the other engine off screen. This module
 *     supplies its own crew chase, and leaves the rig itself alone.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* The five category codes, in FireCategory order. */
  var UNBURNT = 0, SMOLDERING = 1, BURNING = 2, BURNT = 3, WET = 4;
  /* SUPPRESS is the last per-robot action, as FirefightingAction declares. */
  var SUPPRESS = 4;

  /* Engine trim colours. Helmet and pack panel only: a figure painted head to
     toe in a team colour is a game piece. Cycled, so a run with more than two
     robots still gives each one an identity. */
  var CREW_COLORS = [
    { helmet: 0xE8641E, panel: 0xD4581A, tag: "#FF8E5E" },
    { helmet: 0xE8C022, panel: 0xD4A81A, tag: "#FFD24C" },
    { helmet: 0x4FB0E8, panel: 0x3C90C8, tag: "#7FD0FF" },
    { helmet: 0x74D06A, panel: 0x5CB054, tag: "#A8E8A0" }
  ];

  /* The decal a category puts on the ground: colour, opacity and roughness.
     Wet is dark and smooth because water on soil is dark and smooth; char is
     dark and utterly matte. Unburnt puts nothing down at all -- the painted
     terrain is what an unburnt cell looks like. */
  var DECAL = [
    null,
    { color: 0x2A1E14, opacity: 0.55, rough: 0.92 },
    { color: 0x1E1410, opacity: 0.72, rough: 0.90 },
    { color: 0x0E0C0B, opacity: 0.97, rough: 0.98 },
    { color: 0x22262A, opacity: 0.80, rough: 0.14 }
  ];

  var WOOD_P = [0x3A2A1C, 0x54402A, 0x6E563A, 0x8A7050];
  var LEAF_LIVE = [0x2A2E1A, 0x353A22, 0x40462A, 0x4C5234];
  var LEAF_CURED = [0x4A3F28, 0x584C30, 0x685A3A, 0x786A48];
  var LEAF_CHAR = [0x1A1614, 0x241E1A, 0x2E2622, 0x3A302A];

  var FUEL = ["#3A3A22", "#454528", "#4F4D2C", "#5A5632", "#656038", "#726B40", "#7F7749", "#8C8455"];
  var CURED = ["#93875A", "#9E9263", "#A99C6C", "#B4A878"];
  var EARTH = ["#4C3D2A", "#5A4A32", "#6A583E", "#7A684A", "#8A7858"];

  /* Smoke-loaded air: warmer and dirtier than a clear sky, which is what a
     working fire does to the light for a kilometre around it. */
  var HAZE = 0xBE9C78;

  /* Late afternoon, twenty-one degrees up. The sun rakes across the run of the
     fire rather than along it, which is what makes the smoke column read. */
  var SUN_ELEVATION = 21 * Math.PI / 180;

  var GRAV = 9.4;   // the jet's gravity, shared by the core and the droplets
  var JET_T = 0.36; // flight time of the throw, shared by both

  function smooth(e0, e1, x) {
    var t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  /* --------------------------------------------------------------- shaders
     Flame is layered additive billboards with animated noise, not a cone and
     not a particle spray with hard edges. `lean` is the hidden wind: it skews
     the column with height, so the flame bends downwind -- which, with the
     smoke and the embers, is all the evidence a viewer gets about which of the
     eight winds is blowing. */
  var FLAME_VERT = [
    "varying vec2 vUv;",
    "void main() {",
    "  vUv = uv;",
    "  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);",
    "}"
  ].join("\n");

  var FLAME_FRAG = [
    "varying vec2 vUv;",
    "uniform sampler2D noiseTex;",
    "uniform float time, intensity, seed, lean, gain;",
    "void main() {",
    "  vec2 uv = vUv;",
    "  uv.x -= lean * pow(uv.y, 1.55);",
    "  float w = mix(0.37, 0.10, pow(uv.y, 0.70));",
    "  float d = abs(uv.x - 0.5) / w;",
    "  float body = 1.0 - smoothstep(0.28, 1.0, d);",
    "  body *= smoothstep(0.0, 0.06, uv.y);",
    "  if (body <= 0.0) discard;",
    // Two octaves scrolling up at different rates: turbulence, not a conveyor.
    "  float n1 = texture2D(noiseTex, vec2(uv.x * 1.25 + seed, uv.y * 0.72 - time * 0.62 + seed)).r;",
    "  float n2 = texture2D(noiseTex, vec2(uv.x * 2.90 - seed * 1.7, uv.y * 1.55 - time * 1.24)).r;",
    "  float n = n1 * 0.62 + n2 * 0.38;",
    "  float h = 1.0 - uv.y;",
    "  float f = body * smoothstep(0.54 - 0.40 * h * intensity, 1.06, n + h * 0.34 * intensity);",
    "  f *= 1.0 - smoothstep(0.55, 1.0, uv.y);",
    "  if (f <= 0.002) discard;",
    // Radiance ramp: deep red edge, orange body, near-white core at the base.
    "  vec3 col = mix(vec3(1.00, 0.09, 0.01), vec3(1.00, 0.38, 0.05), smoothstep(0.0, 0.56, f));",
    "  col = mix(col, vec3(1.00, 0.70, 0.30), smoothstep(0.56, 0.95, f) * (1.0 - uv.y * 0.60));",
    "  col = mix(col, vec3(1.00, 0.88, 0.62), smoothstep(0.86, 1.0, f) * (1.0 - uv.y) * 0.55);",
    "  gl_FragColor = vec4(col * f * gain, f);",
    "}"
  ].join("\n");

  var SMOKE_FRAG = [
    "varying vec2 vUv;",
    "uniform sampler2D noiseTex;",
    "uniform float time, intensity, seed, lean, litAmount;",
    "void main() {",
    "  vec2 uv = vUv;",
    "  uv.x -= lean * pow(uv.y, 1.25);",
    "  float w = mix(0.16, 0.40, pow(uv.y, 0.7));",
    "  float d = abs(uv.x - 0.5) / w;",
    "  float body = 1.0 - smoothstep(0.0, 1.0, d);",
    "  body *= smoothstep(0.0, 0.26, uv.y) * (1.0 - smoothstep(0.42, 1.0, uv.y));",
    "  body *= 1.0 - smoothstep(0.24, 0.50, abs(vUv.x - 0.5));",
    "  if (body <= 0.0) discard;",
    "  float n1 = texture2D(noiseTex, vec2(uv.x * 0.80 + seed, uv.y * 0.50 - time * 0.16 + seed)).r;",
    "  float n2 = texture2D(noiseTex, vec2(uv.x * 1.90 - seed, uv.y * 1.10 - time * 0.34)).r;",
    "  float n = n1 * 0.66 + n2 * 0.34;",
    "  float a = body * smoothstep(0.22, 0.88, n) * (0.30 + 0.80 * intensity);",
    "  if (a <= 0.003) discard;",
    // The underside of a smoke column is lit by the fire beneath it; the top is
    // just dirty air. A column that ignores its own fire is the loudest tell.
    "  vec3 cool = mix(vec3(0.07, 0.065, 0.06), vec3(0.30, 0.27, 0.24), uv.y);",
    "  vec3 lit = mix(vec3(1.20, 0.42, 0.10), vec3(0.34, 0.24, 0.20), smoothstep(0.0, 0.42, uv.y));",
    "  vec3 col = mix(cool, lit, litAmount * (1.0 - smoothstep(0.0, 0.50, uv.y)));",
    "  gl_FragColor = vec4(col, a);",
    "}"
  ].join("\n");

  /* ------------------------------------------------------------- textures */

  /** A soft radial sprite: contact shadows, light pools, droplets, embers. */
  function radialTexture(inner, falloff) {
    var s = 128;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grad.addColorStop(0, "rgba(255,255,255," + inner + ")");
    grad.addColorStop(falloff, "rgba(255,255,255," + (inner * 0.32) + ")");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, s, s);
    return new THREE.CanvasTexture(cv);
  }

  /* A tileable value-noise texture. The flame and the smoke both read from it
     at two scales and two speeds; one scrolling layer reads as a conveyor
     belt, two beating against each other read as turbulence. */
  function noiseTexture() {
    var N = 64, S = 256;
    var rnd = mulberry(9021);
    var grid = new Float32Array(N * N);
    for (var i = 0; i < N * N; i++) grid[i] = rnd();
    var cv = document.createElement("canvas");
    cv.width = cv.height = S;
    var g = cv.getContext("2d");
    var img = g.createImageData(S, S);
    function at(x, y) { return grid[((y % N) + N) % N * N + (((x % N) + N) % N)]; }
    function fade(t) { return t * t * (3 - 2 * t); }
    function value(x, y, f) {
      var xi = Math.floor(x * f), yi = Math.floor(y * f);
      var xf = fade(x * f - xi), yf = fade(y * f - yi);
      return lerp(lerp(at(xi, yi), at(xi + 1, yi), xf),
                  lerp(at(xi, yi + 1), at(xi + 1, yi + 1), xf), yf);
    }
    for (var y = 0; y < S; y++) {
      for (var x = 0; x < S; x++) {
        var u = x / S * N, v = y / S * N;
        // Three octaves, each an exact divisor of the grid so the result tiles.
        var n = value(u, v, 0.25) * 0.55 + value(u, v, 0.5) * 0.30 + value(u, v, 1) * 0.15;
        var k = (y * S + x) * 4;
        img.data[k] = img.data[k + 1] = img.data[k + 2] = clamp(n, 0, 1) * 255;
        img.data[k + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    var tex = new THREE.CanvasTexture(cv);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }

  /* Sobel over a blurred copy of a painted height pass. Cheap, runs once, and
     it is the difference between ground that looks mown and ground that looks
     printed. */
  function normalMapFrom(renderer, cv, strength) {
    var s = cv.width;
    var soft = document.createElement("canvas");
    soft.width = soft.height = s;
    var sctx = soft.getContext("2d");
    sctx.filter = "blur(1.2px)";
    sctx.drawImage(cv, 0, 0);
    sctx.filter = "none";
    var data = sctx.getImageData(0, 0, s, s).data;
    var dst = document.createElement("canvas");
    dst.width = dst.height = s;
    var dctx = dst.getContext("2d");
    var img = dctx.createImageData(s, s);
    var o = img.data;
    function h(x, y) {
      x = (x + s) % s; y = (y + s) % s;
      return data[((y * s + x) << 2)] / 255;
    }
    for (var y = 0; y < s; y++) {
      for (var x = 0; x < s; x++) {
        var nx = -(h(x + 1, y) - h(x - 1, y)) * strength;
        var ny = -(h(x, y + 1) - h(x, y - 1)) * strength;
        var len = Math.sqrt(nx * nx + ny * ny + 1);
        var i = (y * s + x) * 4;
        o[i] = (nx / len * 0.5 + 0.5) * 255;
        o[i + 1] = (ny / len * 0.5 + 0.5) * 255;
        o[i + 2] = (1 / len * 0.5 + 0.5) * 255;
        o[i + 3] = 255;
      }
    }
    dctx.putImageData(img, 0, 0);
    var tex = new THREE.CanvasTexture(dst);
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  }

  /* Four ragged cell masks, so no two burnt cells are the same shape. */
  function cellMasks() {
    var out = [];
    for (var v = 0; v < 4; v++) {
      var S = 128;
      var cv = document.createElement("canvas");
      cv.width = cv.height = S;
      var g = cv.getContext("2d");
      var rnd = mulberry(500 + v * 37);
      for (var i = 0; i < 16; i++) {
        var a = rnd() * 6.28, d = Math.pow(rnd(), 0.6) * S * 0.20;
        var cx = S / 2 + Math.cos(a) * d, cy = S / 2 + Math.sin(a) * d;
        var r = S * (0.13 + rnd() * 0.20);
        var gr = g.createRadialGradient(cx, cy, 0, cx, cy, r);
        gr.addColorStop(0, "rgba(255,255,255,0.75)");
        gr.addColorStop(0.55, "rgba(255,255,255,0.34)");
        gr.addColorStop(1, "rgba(255,255,255,0)");
        g.fillStyle = gr;
        g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fill();
      }
      // Speckle at the edge, so the boundary is not a clean gradient.
      for (var k = 0; k < 400; k++) {
        var a2 = rnd() * 6.28, d2 = (0.22 + rnd() * 0.22) * S;
        g.globalAlpha = 0.10 + rnd() * 0.4;
        g.fillStyle = "#FFFFFF";
        g.beginPath();
        g.arc(S / 2 + Math.cos(a2) * d2, S / 2 + Math.sin(a2) * d2, 1 + rnd() * 3, 0, Math.PI * 2);
        g.fill();
      }
      g.globalAlpha = 1;
      out.push(new THREE.CanvasTexture(cv));
    }
    return out;
  }

  /* ---------------------------------------------------------- belief panel
     The wind posterior gets a panel of its own directly under the viewport,
     not a corner of the HUD. It is the POMDP: the one quantity the crew never
     observes, inferred from exactly the flame lean the viewer is watching.
     The site's own HUD carries one line per field and has no room for eight
     hypotheses or for a second engine, so the panel is built here, from this
     module, rather than by changing a page every environment shares. */
  var PANEL_CSS = [
    ".ff-panel{margin-top:14px;border:1px solid var(--line);border-left:3px solid var(--accent);",
    "  border-radius:6px;background:var(--panel);padding:16px 18px;display:grid;gap:14px 26px;",
    "  grid-template-columns:auto 1fr;align-items:start}",
    "@media (max-width:640px){.ff-panel{grid-template-columns:1fr}}",
    ".ff-panel h2{grid-column:1/-1;margin:0;font-size:16px}",
    ".ff-panel h2 small{display:block;font-weight:400;font-size:13px;color:var(--dim);",
    "  margin-top:5px;max-width:84ch;line-height:1.5}",
    ".ff-rose{display:grid;justify-items:center;gap:6px}",
    ".ff-rose b{font-size:12px;color:var(--dim);font-weight:500;text-align:center}",
    ".ff-bars{display:grid;grid-template-columns:repeat(2,1fr);gap:5px 20px;align-self:center}",
    "@media (max-width:520px){.ff-bars{grid-template-columns:1fr}}",
    ".ff-b{display:grid;grid-template-columns:4.4em 1fr 3.2em;align-items:center;gap:9px;",
    "  font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}",
    ".ff-b .ff-t{height:9px;background:var(--line);border-radius:2px;overflow:hidden}",
    ".ff-b .ff-t i{display:block;height:100%;background:var(--bar)}",
    ".ff-b.ff-true .ff-t i{background:#E84E3E}",
    ".ff-b em{font-style:normal;color:var(--ink);text-align:right}",
    ".ff-foot{grid-column:1/-1;display:grid;gap:10px}",
    ".ff-crew{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:12px;",
    "  font-variant-numeric:tabular-nums;color:var(--dim)}",
    ".ff-crew span{color:var(--ink)}",
    ".ff-tag{display:inline-block;min-width:2.2em;text-align:center;border-radius:3px;",
    "  padding:1px 5px;color:#14110F;font-weight:600;margin-right:6px}",
    ".ff-toggles{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:var(--dim)}",
    ".ff-toggles label{display:inline-flex;align-items:center;gap:6px;cursor:pointer}",
    ".ff-note{font-size:12px;color:var(--dim);line-height:1.5;max-width:92ch;margin:0}",
    ".ff-note strong{color:var(--ink);font-weight:500}"
  ].join("\n");

  var SVG_NS = "http://www.w3.org/2000/svg";
  /* N E S W as they are drawn on screen: the rose is a map, so north is up. */
  var DIR_ANGLE = [-Math.PI / 2, 0, Math.PI / 2, Math.PI];

  function svgEl(name, attrs) {
    var node = document.createElementNS(SVG_NS, name);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, String(attrs[key]));
    });
    return node;
  }

  function wedgePath(cx, cy, r0, r1, a0, a1) {
    var x0 = cx + Math.cos(a0) * r0, y0 = cy + Math.sin(a0) * r0;
    var x1 = cx + Math.cos(a1) * r0, y1 = cy + Math.sin(a1) * r0;
    var x2 = cx + Math.cos(a1) * r1, y2 = cy + Math.sin(a1) * r1;
    var x3 = cx + Math.cos(a0) * r1, y3 = cy + Math.sin(a0) * r1;
    return "M" + x0 + "," + y0 + " A" + r0 + "," + r0 + " 0 0 1 " + x1 + "," + y1 +
           " L" + x2 + "," + y2 + " A" + r1 + "," + r1 + " 0 0 0 " + x3 + "," + y3 + " Z";
  }

  /**
   * Build the wind panel and put it under the viewer.
   *
   * @param {Object} world   The trace's world block, for the labels.
   * @param {number} robots  How many engines to give a crew line to.
   * @returns {Object} Handles the scene writes into each step.
   */
  function buildPanel(world, robots) {
    if (!document.getElementById("ff-panel-style")) {
      var style = document.createElement("style");
      style.id = "ff-panel-style";
      style.textContent = PANEL_CSS;
      document.head.appendChild(style);
    }

    var labels = world.wind_labels || [];
    var panel = document.createElement("section");
    panel.className = "ff-panel";

    var title = document.createElement("h2");
    title.textContent = "Belief over the hidden wind";
    var sub = document.createElement("small");
    sub.textContent =
      "Four directions crossed with two strengths: eight values, drawn uniformly at reset, " +
      "never observed and never changing. Nothing in an observation mentions the wind, so the " +
      "only evidence is where the fire grew — the same flame lean, smoke drift and ember " +
      "tracks on screen. The inner wedge of each arm is low strength, the outer is high. These " +
      "are this run's own particles, read at the wind fields of the state vector.";
    title.appendChild(sub);
    panel.appendChild(title);

    var roseWrap = document.createElement("div");
    roseWrap.className = "ff-rose";
    var svg = svgEl("svg", {
      width: 208, height: 208, viewBox: "0 0 104 104", "aria-label": "Wind posterior"
    });
    var wedges = [];
    for (var v = 0; v < 8; v++) {
      var path = svgEl("path", { stroke: "rgba(10,8,6,0.35)", "stroke-width": 0.7 });
      svg.appendChild(path);
      wedges.push(path);
    }
    ["N", "E", "S", "W"].forEach(function (label, d) {
      var text = svgEl("text", {
        x: 52 + Math.cos(DIR_ANGLE[d]) * 45,
        y: 52 + Math.sin(DIR_ANGLE[d]) * 45 + 4,
        "text-anchor": "middle", "font-size": 11, fill: "currentColor"
      });
      text.textContent = label;
      svg.appendChild(text);
    });
    var trueMark = svgEl("circle", { r: 3, fill: "#E84E3E", opacity: 0 });
    svg.appendChild(trueMark);
    svg.style.color = "var(--dim)";
    roseWrap.appendChild(svg);
    var summary = document.createElement("b");
    roseWrap.appendChild(summary);
    panel.appendChild(roseWrap);

    var bars = document.createElement("div");
    bars.className = "ff-bars";
    var barRows = [];
    for (var k = 0; k < 8; k++) {
      var row = document.createElement("div");
      row.className = "ff-b";
      var name = document.createElement("span");
      name.textContent = labels[k] || String(k);
      var track = document.createElement("div");
      track.className = "ff-t";
      var fill = document.createElement("i");
      track.appendChild(fill);
      var num = document.createElement("em");
      num.textContent = "0.00";
      row.appendChild(name); row.appendChild(track); row.appendChild(num);
      bars.appendChild(row);
      barRows.push({ row: row, fill: fill, num: num });
    }
    panel.appendChild(bars);

    var foot = document.createElement("div");
    foot.className = "ff-foot";

    var crew = document.createElement("div");
    crew.className = "ff-crew";
    var crewCells = [];
    for (var r = 0; r < robots; r++) {
      var item = document.createElement("div");
      var tag = document.createElement("i");
      tag.className = "ff-tag";
      tag.style.background = CREW_COLORS[r % CREW_COLORS.length].tag;
      tag.textContent = "E" + (r + 1);
      var value = document.createElement("span");
      item.appendChild(tag);
      item.appendChild(value);
      crew.appendChild(item);
      crewCells.push(value);
    }
    foot.appendChild(crew);

    var toggles = document.createElement("div");
    toggles.className = "ff-toggles";
    var boxes = {};
    [
      ["truewind", "True wind arrow", false],
      ["sense", "Sensing footprint", true],
      ["smoke", "Smoke", true],
      ["lattice", "Cell lattice", false]
    ].forEach(function (spec) {
      var label = document.createElement("label");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.checked = spec[2];
      label.appendChild(box);
      label.appendChild(document.createTextNode(spec[1]));
      toggles.appendChild(label);
      boxes[spec[0]] = box;
    });
    foot.appendChild(toggles);

    var note = document.createElement("p");
    note.className = "ff-note";
    note.innerHTML =
      "<strong>The model is per-cell and discrete:</strong> a cell is alight or it is not, and " +
      "one spray flips it in a single step. The flames are <strong>interpolated between steps " +
      "for legibility</strong> — a cell catching grows in over its step, a cell knocked " +
      "down gutters out into steam, a burnt-out cell smoulders and fades. The readouts, the " +
      "counts and the return always report the true discrete state of the step on screen, " +
      "never the tween.";
    foot.appendChild(note);
    panel.appendChild(foot);

    var anchor = document.getElementById("viewer-status") || document.getElementById("viewer");
    if (anchor && anchor.parentNode) {
      anchor.parentNode.insertBefore(panel, anchor.nextSibling);
    }

    return {
      wedges: wedges, bars: barRows, summary: summary, trueMark: trueMark,
      crew: crewCells, toggles: boxes, panel: panel
    };
  }

  /* ------------------------------------------------------- wind posterior
     A firefighting particle is a whole state vector -- both engines, the wind
     and the hundred cells -- and core writes particles verbatim, so the
     marginal over the eight winds is a projection taken here from the run's
     own cloud, at the indices `world.state_layout` names. It is not a second
     belief: it is the recorded one, summed.

     Every other belief kind is reported as what it is rather than drawn. A
     rose over a Gaussian or over a batch of beliefs would be a picture of
     something nobody held. */
  function windMarginal(belief, layout, strengths) {
    if (!belief || belief.kind !== "particles") {
      return { mass: null, why: belief ? notDrawable(belief) : "—" };
    }
    var particles = belief.particles || [], weights = belief.weights || [];
    var mass = [0, 0, 0, 0, 0, 0, 0, 0];
    var counted = 0;
    for (var i = 0; i < particles.length; i++) {
      var particle = particles[i];
      // A particle that is not a state vector cannot carry a wind. Saying so
      // beats plotting whatever the first two entries happen to be.
      if (!particle || particle.length <= layout.wind_strength_index) continue;
      var direction = Math.round(particle[layout.wind_direction_index]);
      var strength = Math.round(particle[layout.wind_strength_index]);
      if (!(direction >= 0 && direction < 4) || !(strength >= 0 && strength < strengths)) continue;
      mass[direction * strengths + strength] += weights[i] === undefined ? 0 : weights[i];
      counted++;
    }
    if (!counted) return { mass: null, why: "no wind in these particles" };
    var total = mass.reduce(function (a, b) { return a + b; }, 0);
    if (total > 0) for (var k = 0; k < 8; k++) mass[k] /= total;
    return { mass: mass, why: null, particles: belief.num_particles, written: belief.num_written };
  }

  function notDrawable(belief) {
    if (belief.kind === "particle_batch") {
      // A batch is several beliefs held together for a vectorized planner. It
      // is not one episode's belief, and merging its members would show a
      // posterior that was never anyone's.
      return "batch of " + belief.batch_size + " beliefs, not drawn";
    }
    if (belief.kind === "gaussian" || belief.kind === "gaussian_mixture") {
      // The wind is one of eight labels. A density over a continuous state has
      // no marginal over them that this viewer could honestly take.
      return belief.kind + " belief, no wind marginal";
    }
    return "not recorded (" + (belief.belief_class || belief.kind) + ")";
  }

  function entropyBits(mass) {
    var bits = 0;
    for (var k = 0; k < mass.length; k++) {
      if (mass[k] > 1e-12) bits -= mass[k] * Math.log(mass[k]) / Math.LN2;
    }
    return bits;
  }

  /**
   * Repaint the rose and the bars. Only when the numbers actually change: the
   * belief moves once per step, and rewriting sixteen SVG paths every frame is
   * a layout pass per frame for a picture that changes a few dozen times in a
   * whole episode.
   */
  function drawRose(panel, mass, labels, trueIndex, showTrue) {
    var peak = 0;
    for (var p = 0; p < mass.length; p++) peak = Math.max(peak, mass[p]);
    for (var d = 0; d < 4; d++) {
      for (var s = 0; s < 2; s++) {
        var k = d * 2 + s;
        var share = mass[k] / Math.max(peak, 1e-6);
        var a0 = DIR_ANGLE[d] - 0.38, a1 = DIR_ANGLE[d] + 0.38;
        // Inner wedge is low strength, outer is high, so direction reads as
        // direction and strength as distance from the middle.
        var r0 = s === 0 ? 9 : 22;
        var r1 = s === 0 ? 9 + 11 * clamp(share, 0.03, 1) : 22 + 16 * clamp(share, 0.03, 1);
        panel.wedges[k].setAttribute("d", wedgePath(52, 52, r0, Math.max(r0 + 1.2, r1), a0, a1));
        // One hue with the value carrying the mass: two hues would imply two
        // kinds of hypothesis where there is one.
        panel.wedges[k].setAttribute("fill", showTrue && k === trueIndex ? "#E84E3E" : "#C9564A");
        panel.wedges[k].setAttribute("fill-opacity", String(0.16 + 0.80 * share));
        panel.bars[k].fill.style.width = (100 * clamp(mass[k], 0.004, 1)).toFixed(1) + "%";
        panel.bars[k].num.textContent = mass[k].toFixed(2);
        // The true value is marked only when the reader asks for it. A bar
        // permanently coloured red would hand over the answer the page is
        // about, which is worse than not drawing the belief at all.
        panel.bars[k].row.className = "ff-b" + (showTrue && k === trueIndex ? " ff-true" : "");
      }
    }
    if (showTrue && trueIndex >= 0) {
      panel.trueMark.setAttribute("cx", String(52 + Math.cos(DIR_ANGLE[trueIndex >> 1]) * 30));
      panel.trueMark.setAttribute("cy", String(52 + Math.sin(DIR_ANGLE[trueIndex >> 1]) * 30));
    }
    panel.trueMark.setAttribute("opacity", showTrue && trueIndex >= 0 ? "1" : "0");

    var best = 0;
    for (var b = 0; b < mass.length; b++) if (mass[b] > mass[best]) best = b;
    panel.summary.textContent =
      "argmax " + (labels[best] || best) + " " + mass[best].toFixed(2) +
      "  ·  entropy " + entropyBits(mass).toFixed(2) + " bits of 3.00";
    return { best: best, bits: entropyBits(mass) };
  }

  function clearRose(panel, why) {
    for (var k = 0; k < 8; k++) {
      panel.wedges[k].setAttribute("d", "");
      panel.bars[k].fill.style.width = "0%";
      panel.bars[k].num.textContent = "—";
      panel.bars[k].row.className = "ff-b";
    }
    panel.trueMark.setAttribute("opacity", "0");
    panel.summary.textContent = why;
  }

  /* ----------------------------------------------------------- the figures
     Built by sweeping cross-sections along a chain, not by stacking boxes. A
     torso is one surface whose section runs from a narrow waist out to the
     shoulders and back in at the neck; a limb is a tapering tube threaded
     through its own joint path, swelling at the knee and elbow rather than two
     cylinders butted together. Boxes and flat shading are what made the first
     version read as a voxel character standing in a wildfire, and no amount of
     lighting fixes a silhouette made of cubes.

     The figure faces its own local +x: brim and visor forward, line carried
     forward, pack on the back. */

  /* Sweep an elliptical section along a chain of points. The frame is carried
     along by parallel transport rather than rebuilt from a fixed "up" at each
     point, which is what stops a limb twisting where its chain bends back on
     itself. */
  function sweepChain(points, radii, seg, mat, capEnds) {
    var N = points.length, S = seg;
    var P = points.map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); });
    var T = [];
    for (var i = 0; i < N; i++) {
      var a = P[Math.max(0, i - 1)], b = P[Math.min(N - 1, i + 1)];
      var t = b.clone().sub(a);
      if (t.lengthSq() < 1e-12) t.set(0, 1, 0);
      T.push(t.normalize());
    }
    var ref = new THREE.Vector3(1, 0, 0);
    if (Math.abs(T[0].dot(ref)) > 0.9) ref.set(0, 0, 1);
    var U = ref.clone().addScaledVector(T[0], -ref.dot(T[0])).normalize();
    var pos = [], idx = [];
    for (var i2 = 0; i2 < N; i2++) {
      if (i2 > 0) {
        var axis = new THREE.Vector3().crossVectors(T[i2 - 1], T[i2]);
        var s = axis.length();
        if (s > 1e-8) U.applyAxisAngle(axis.divideScalar(s), Math.atan2(s, T[i2 - 1].dot(T[i2])));
        U.addScaledVector(T[i2], -U.dot(T[i2])).normalize();
      }
      var W = new THREE.Vector3().crossVectors(T[i2], U).normalize();
      for (var s2 = 0; s2 < S; s2++) {
        var ang = s2 / S * Math.PI * 2;
        var q = P[i2].clone()
          .addScaledVector(U, Math.cos(ang) * radii[i2][0])
          .addScaledVector(W, Math.sin(ang) * radii[i2][1]);
        pos.push(q.x, q.y, q.z);
      }
    }
    for (var i3 = 0; i3 < N - 1; i3++) {
      for (var s3 = 0; s3 < S; s3++) {
        var a3 = i3 * S + s3, b3 = i3 * S + (s3 + 1) % S;
        var c3 = (i3 + 1) * S + (s3 + 1) % S, d3 = (i3 + 1) * S + s3;
        idx.push(a3, b3, c3, a3, c3, d3);
      }
    }
    if (capEnds !== false) {
      var base = pos.length / 3;
      pos.push(P[0].x, P[0].y, P[0].z);
      for (var s4 = 0; s4 < S; s4++) idx.push(base, (s4 + 1) % S, s4);
      var base2 = pos.length / 3;
      pos.push(P[N - 1].x, P[N - 1].y, P[N - 1].z);
      var off = (N - 1) * S;
      for (var s5 = 0; s5 < S; s5++) idx.push(base2, off + s5, off + (s5 + 1) % S);
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    var mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  /* A band of reflective tape round a limb or a hem. */
  function tapeBand(centre, r0, r1, halfHeight, mat, seg) {
    return sweepChain(
      [[centre[0], centre[1] - halfHeight, centre[2]], [centre[0], centre[1] + halfHeight, centre[2]]],
      [[r0, r1], [r0, r1]], seg || 12, mat, false);
  }

  /**
   * One firefighter in turnout gear, carrying a charged line.
   *
   * @param {Object} spec     Helmet and panel colours for this engine.
   * @param {Object} shared   Shared materials and the contact-shadow sprite.
   * @returns {THREE.Group} The figure, with its joints in userData.
   */
  function buildFirefighter(spec, shared) {
    var g = new THREE.Group();
    // Weathered, desaturated gear. The colour budget goes to the helmet, the
    // collar and the pack panel: those are what a crew is told apart by.
    var coat = new THREE.MeshStandardMaterial({ color: 0x8A7A5E, roughness: 0.86, metalness: 0.02 });
    var coatDark = new THREE.MeshStandardMaterial({ color: 0x6E6046, roughness: 0.88, metalness: 0.02 });
    var trouser = new THREE.MeshStandardMaterial({ color: 0x5E523C, roughness: 0.9, metalness: 0.0 });
    // Retroreflective tape: low roughness and a strong environment response,
    // so it lights up when there is a fire beside it and reads at distance.
    var bandMat = new THREE.MeshStandardMaterial({
      color: 0xE8E4C8, roughness: 0.26, metalness: 0.12, envMapIntensity: 2.6
    });
    var helmetMat = new THREE.MeshStandardMaterial({ color: spec.helmet, roughness: 0.42, metalness: 0.08 });
    var trimMat = new THREE.MeshStandardMaterial({ color: spec.panel, roughness: 0.55, metalness: 0.06 });
    var gloveMat = new THREE.MeshStandardMaterial({ color: 0x3A3028, roughness: 0.92, metalness: 0.0 });
    var skinMat = new THREE.MeshStandardMaterial({ color: 0x9A7A5C, roughness: 0.8, metalness: 0.0 });

    // Legs. Thigh and shin are separate sweeps because they articulate, but
    // each tapers and swells at the joint, so a knee reads as a knee instead
    // of as the seam between two cylinders. Adult proportions: legs a shade
    // under half the standing height, head about a seventh of it.
    var legs = new THREE.Group();
    g.add(legs);
    g.userData.legs = [];
    g.userData.knees = [];
    [-0.056, 0.056].forEach(function (off) {
      var hip = new THREE.Group();
      hip.position.set(0, 0.262, off);
      hip.add(sweepChain(
        [[0, 0.004, 0], [0, -0.050, 0], [0, -0.102, 0], [0, -0.142, 0]],
        [[0.047, 0.049], [0.042, 0.044], [0.038, 0.040], [0.039, 0.041]],
        10, trouser));
      var knee = new THREE.Group();
      knee.position.y = -0.142;
      hip.add(knee);
      knee.add(sweepChain(
        [[0, 0.006, 0], [0, -0.034, 0], [0, -0.080, 0], [0, -0.112, 0]],
        [[0.039, 0.041], [0.035, 0.037], [0.030, 0.032], [0.029, 0.031]],
        10, trouser));
      knee.add(tapeBand([0, -0.094, 0], 0.032, 0.034, 0.013, bandMat, 12));
      // The boot: a sole and a toe overhanging FORWARD of the ankle, so which
      // way the figure points survives even when it is standing still.
      knee.add(sweepChain(
        [[-0.030, -0.118, 0], [-0.010, -0.140, 0], [0.028, -0.150, 0], [0.060, -0.148, 0]],
        [[0.031, 0.033], [0.032, 0.036], [0.028, 0.034], [0.017, 0.028]],
        10, new THREE.MeshStandardMaterial({ color: 0x1C1A18, roughness: 0.88, metalness: 0.02 })));
      legs.add(hip);
      g.userData.legs.push(hip);
      g.userData.knees.push(knee);
    });

    // The coat: one swept surface from hem to collar, elliptical at every
    // height, which is the whole difference between a person and a crate.
    g.add(sweepChain(
      [[0, 0.244, 0], [0, 0.286, 0], [0, 0.336, 0], [0, 0.388, 0],
       [0, 0.440, 0], [0, 0.474, 0], [0, 0.498, 0]],
      [[0.067, 0.090], [0.063, 0.084], [0.057, 0.077], [0.065, 0.094],
       [0.073, 0.120], [0.069, 0.116], [0.043, 0.052]],
      14, coat));
    g.add(tapeBand([0, 0.262, 0], 0.069, 0.092, 0.013, bandMat, 14));
    g.add(tapeBand([0, 0.380, 0], 0.067, 0.095, 0.012, bandMat, 14));
    g.add(tapeBand([0, 0.490, 0], 0.051, 0.062, 0.014, trimMat, 14));

    // Air pack: a rounded shell on the back with the cylinder strapped to it.
    g.add(sweepChain(
      [[-0.087, 0.318, 0], [-0.097, 0.352, 0], [-0.101, 0.406, 0], [-0.093, 0.450, 0]],
      [[0.024, 0.052], [0.034, 0.068], [0.036, 0.072], [0.026, 0.056]], 12, coatDark));
    g.add(sweepChain(
      [[-0.125, 0.326, 0], [-0.131, 0.362, 0], [-0.133, 0.416, 0], [-0.127, 0.446, 0]],
      [[0.016, 0.016], [0.030, 0.030], [0.031, 0.031], [0.018, 0.018]], 12, shared.steel));
    g.add(sweepChain(
      [[-0.115, 0.370, 0], [-0.119, 0.406, 0]],
      [[0.010, 0.030], [0.010, 0.030]], 10, trimMat, false));
    [-0.058, 0.058].forEach(function (sz) {
      g.add(sweepChain(
        [[-0.069, 0.464, sz], [0.010, 0.470, sz * 1.15], [0.053, 0.412, sz * 1.25], [0.045, 0.352, sz * 1.2]],
        [[0.012, 0.016], [0.012, 0.018], [0.011, 0.017], [0.010, 0.015]], 8, coatDark));
    });

    // Head, hood and helmet. The brim that sweeps out and the flap down the
    // back of the neck are the silhouette that says firefighter rather than
    // soldier, and they make the front of the head unmistakably the front.
    g.add(sweepChain([[0, 0.498, 0], [0, 0.522, 0]],
      [[0.029, 0.033], [0.031, 0.035]], 10, skinMat, false));
    var head = new THREE.Mesh(new THREE.SphereGeometry(0.044, 16, 12), skinMat);
    head.position.set(0.002, 0.556, 0);
    head.scale.set(0.96, 1.06, 0.94);
    head.castShadow = true;
    g.add(head);
    var hood = new THREE.Mesh(new THREE.SphereGeometry(0.049, 16, 12),
      new THREE.MeshStandardMaterial({ color: 0x2E2A26, roughness: 0.95 }));
    hood.position.set(-0.004, 0.552, 0);
    g.add(hood);
    var mask = new THREE.Mesh(new THREE.SphereGeometry(0.035, 14, 10),
      new THREE.MeshStandardMaterial({
        color: 0x1A1E22, roughness: 0.16, metalness: 0.25, envMapIntensity: 2.0
      }));
    mask.position.set(0.030, 0.550, 0);
    mask.scale.set(0.72, 0.92, 0.94);
    g.add(mask);
    var dome = new THREE.Mesh(
      new THREE.SphereGeometry(0.054, 18, 12, 0, Math.PI * 2, 0, Math.PI * 0.56), helmetMat);
    dome.position.set(-0.004, 0.570, 0);
    dome.scale.set(1.04, 0.95, 1.02);
    dome.castShadow = true;
    g.add(dome);
    var brim = sweepChain([[-0.004, 0.568, 0], [-0.004, 0.577, 0]],
      [[0.080, 0.070], [0.072, 0.064]], 18, helmetMat, false);
    brim.rotation.z = -0.13;
    g.add(brim);
    g.add(sweepChain([[-0.058, 0.566, 0], [-0.071, 0.546, 0], [-0.075, 0.524, 0]],
      [[0.008, 0.058], [0.008, 0.054], [0.007, 0.047]], 10, coatDark, false));
    g.add(sweepChain([[0.048, 0.586, 0], [0.054, 0.604, 0]],
      [[0.007, 0.026], [0.006, 0.022]], 10, trimMat, false));

    // The line. The nozzle is carried forward on the right and both hands are
    // ON it. A figure spraying a jet from an empty fist is half of what makes
    // one of these read as a game character.
    var upper = new THREE.Group();
    g.add(upper);
    g.userData.upper = upper;
    var nozzle = new THREE.Group();
    nozzle.position.set(0.126, 0.344, 0.088);
    upper.add(nozzle);
    nozzle.add(sweepChain(
      [[-0.115, 0, 0], [-0.030, 0, 0], [0.055, 0, 0], [0.110, 0, 0]],
      [[0.020, 0.020], [0.024, 0.024], [0.018, 0.018], [0.013, 0.013]], 12, shared.steel));
    var tip = new THREE.Object3D();
    tip.position.set(0.130, 0, 0);
    nozzle.add(tip);
    g.userData.nozzleTip = tip;
    g.userData.nozzle = nozzle;
    upper.add(sweepChain(
      [[0.016, 0.338, 0.090], [-0.058, 0.296, 0.112], [-0.130, 0.232, 0.132],
       [-0.212, 0.146, 0.142], [-0.302, 0.074, 0.132]],
      [[0.020, 0.020], [0.021, 0.021], [0.022, 0.022], [0.022, 0.022], [0.022, 0.022]],
      10, new THREE.MeshStandardMaterial({ color: 0x4A3C2C, roughness: 0.92 })));

    // Arms, swept from the shoulder through a real elbow to the hand on the
    // line: the elbow is where the chain bends and the section swells.
    function arm(shoulder, elbow, wrist) {
      upper.add(sweepChain(
        [shoulder,
         [lerp(shoulder[0], elbow[0], 0.55), lerp(shoulder[1], elbow[1], 0.55), lerp(shoulder[2], elbow[2], 0.55)],
         elbow,
         [lerp(elbow[0], wrist[0], 0.5), lerp(elbow[1], wrist[1], 0.5), lerp(elbow[2], wrist[2], 0.5)],
         wrist],
        [[0.034, 0.036], [0.030, 0.032], [0.030, 0.032], [0.026, 0.028], [0.024, 0.026]],
        10, coat));
      upper.add(tapeBand([wrist[0], wrist[1] + 0.004, wrist[2]], 0.027, 0.029, 0.010, bandMat, 10));
      var glove = new THREE.Mesh(new THREE.SphereGeometry(0.029, 12, 10), gloveMat);
      glove.position.set(wrist[0] + 0.012, wrist[1] - 0.004, wrist[2]);
      glove.scale.set(1.15, 0.9, 0.85);
      glove.castShadow = true;
      upper.add(glove);
    }
    arm([0.000, 0.462, 0.102], [0.060, 0.392, 0.132], [0.182, 0.348, 0.094]);
    arm([0.000, 0.462, -0.102], [0.044, 0.384, -0.108], [0.034, 0.340, 0.054]);

    var blob = new THREE.Mesh(new THREE.PlaneGeometry(0.46, 0.46),
      new THREE.MeshBasicMaterial({
        map: shared.pool, color: 0x000000, transparent: true, opacity: 0.30, depthWrite: false
      }));
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.012;
    g.add(blob);

    g.userData.soot = [coat, coatDark, trouser];
    return g;
  }

  /* Scrub oak and pine: a tapered trunk that bends, real branches, and a
     canopy of clusters hung near the ENDS of limbs with nothing filling the
     middle. Foliage at the limb tips is the whole difference between a tree
     and a broccoli floret, and the gaps are what break up the shadow it
     throws. Every canopy keeps its cluster materials, so the tree can be
     charred when its own cell burns. */
  function bentTrunk(r0, r1, h, bend, mat) {
    var geo = new THREE.CylinderGeometry(r0, r1, h, 9, 6);
    var pos = geo.attributes.position;
    for (var i = 0; i < pos.count; i++) {
      var y = pos.getY(i);
      var f = (y + h / 2) / h;
      pos.setX(i, pos.getX(i) * (1 + 0.22 * Math.sin(f * 5.3)) + bend * f * f * h);
      pos.setZ(i, pos.getZ(i) * (1 + 0.18 * Math.cos(f * 4.1 + 1.2)));
    }
    geo.computeVertexNormals();
    var mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  function buildTree(seed, species) {
    var rnd = mulberry(seed);
    var g = new THREE.Group();
    var bark = new THREE.MeshStandardMaterial({
      color: new THREE.Color(WOOD_P[1]).multiplyScalar(0.86 + rnd() * 0.3),
      roughness: 0.97, metalness: 0.0, flatShading: true
    });
    var tall = species === 2, scrub = species === 3;
    var h = scrub ? 0.44 : (tall ? 1.05 : 0.70 + rnd() * 0.22);
    var trunk = bentTrunk(tall ? 0.038 : 0.054, scrub ? 0.07 : 0.10, h, (rnd() - 0.5) * 0.22, bark);
    trunk.position.y = h / 2;
    g.add(trunk);

    // Roots flaring into the soil: a cylinder that just stops at the ground is
    // one of the tells that a tree was placed rather than grown.
    for (var r = 0; r < 5; r++) {
      var ra = (r / 5) * 6.28 + rnd() * 0.6;
      var root = new THREE.Mesh(new THREE.ConeGeometry(0.045 + rnd() * 0.03, 0.20, 5), bark);
      root.position.set(Math.cos(ra) * 0.085, 0.05, Math.sin(ra) * 0.085);
      root.rotation.set(Math.cos(ra) * 0.9, 0, -Math.sin(ra) * 0.9);
      root.castShadow = true;
      g.add(root);
    }

    var canopy = new THREE.Group();
    canopy.position.y = h * (scrub ? 0.55 : 0.80);
    g.add(canopy);

    var limbs = [];
    var nLimb = scrub ? 4 : (tall ? 6 : 7);
    for (var li = 0; li < nLimb; li++) {
      var la = (li / nLimb) * 6.28 + rnd() * 0.9;
      var reach = (scrub ? 0.22 : (tall ? 0.24 : 0.38)) * (0.45 + rnd() * 1.15);
      var rise = (scrub ? 0.10 : (tall ? 0.30 : 0.16)) * (0.3 + rnd() * 1.4) - 0.04;
      limbs.push([Math.cos(la) * reach, rise, Math.sin(la) * reach]);
      var limb = new THREE.Mesh(new THREE.CylinderGeometry(0.010, 0.026, reach * 1.9, 5), bark);
      limb.position.set(Math.cos(la) * reach * 0.5, rise * 0.5 - 0.02, Math.sin(la) * reach * 0.5);
      limb.lookAt(new THREE.Vector3(Math.cos(la) * reach, rise, Math.sin(la) * reach));
      limb.rotateX(Math.PI / 2);
      limb.castShadow = true;
      canopy.add(limb);
    }

    var mats = [];
    var nClust = limbs.length * (scrub ? 3 : 5);
    for (var c = 0; c < nClust; c++) {
      var end = limbs[c % limbs.length];
      var back = Math.pow(rnd(), 2.4) * 0.40;
      var spread = scrub ? 0.12 : 0.24;
      var cxp = end[0] * (1 - back) + (rnd() - 0.5) * spread;
      var cyp = end[1] * (1 - back) + (rnd() - 0.5) * spread * 0.9;
      var czp = end[2] * (1 - back) + (rnd() - 0.5) * spread;
      var up = clamp(0.5 + cyp * 2.0, 0, 1);
      var outlier = rnd() < 0.22;
      var size = (outlier ? 0.046 + rnd() * 0.034 : 0.076 + rnd() * 0.060) * (scrub ? 0.85 : 1);
      var shade = up > 0.50 ? 2 + (rnd() < 0.4 ? 1 : 0) : (rnd() < 0.35 ? 1 : 2);
      var cured = rnd() < 0.42;                 // some of the stand is already dead
      var mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color((cured ? LEAF_CURED : LEAF_LIVE)[shade]).multiplyScalar(0.92 + rnd() * 0.22),
        roughness: 0.94, metalness: 0.0, flatShading: true
      });
      mat.userData = { base: mat.color.clone(), char: new THREE.Color(LEAF_CHAR[shade]) };
      mats.push(mat);
      var blob = new THREE.Mesh(new THREE.IcosahedronGeometry(size, 1), mat);
      blob.position.set(cxp * (outlier ? 1.35 : 1), cyp, czp * (outlier ? 1.35 : 1));
      blob.scale.set(1 + rnd() * 0.6, 0.62 + rnd() * 0.36, 1 + rnd() * 0.6);
      blob.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
      blob.castShadow = true;
      blob.receiveShadow = true;
      canopy.add(blob);
    }

    g.userData.canopy = canopy;
    g.userData.leafMats = mats;
    g.userData.phase = rnd() * 6.28;
    return g;
  }

  /**
   * Build the firefighting world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json of kind multiagent_firefighting.v1.
   * @returns {Object} The scene module the player drives.
   */
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;

    var ROWS = world.num_rows, COLS = world.num_cols;
    var CELLS = ROWS * COLS;
    var ROBOTS = world.num_robots;
    var STEPS = payload.fires.length;
    var LAYOUT = world.state_layout;
    var STRENGTHS = world.num_wind_strengths || 2;
    var SENSE = world.sensing_radius;
    var MAX_TANK = world.max_tank, MAX_HEALTH = world.max_health;

    // The environment addresses cells as (row, col) with row increasing SOUTH
    // and col increasing EAST, so the scene maps col -> +x and row -> +z and
    // the grid centre sits at the origin. That is exactly what
    // DIRECTION_OFFSETS means, and the payload carries the offsets so the two
    // cannot drift.
    var HR = (ROWS - 1) / 2, HC = (COLS - 1) / 2;
    var DIR = world.direction_offsets;
    function wx(col) { return col - HC; }
    function wz(row) { return row - HR; }
    var SPAN = Math.max(ROWS, COLS);

    var obstacleAt = {};
    world.obstacle_cells.forEach(function (cell) { obstacleAt[cell[0] + ":" + cell[1]] = true; });
    var DEPOT = world.depot_cell;

    /* A shallow roll over the whole slope. Every prop is planted with this, so
       nothing floats, and the low sun reads the relief. */
    function groundY(row, col) {
      return Math.sin(col * 0.52 + 1.1) * 0.026
           + Math.sin(row * 0.67 - 0.5) * 0.022
           + Math.sin((row + col) * 0.29) * 0.017
           - row * 0.020;                       // the stand falls away to the south
    }

    /* ------------------------------------------------------- the episode
       Everything below is read off the trace. The fire maps are the recorded
       discrete categories; the poses, tanks and healths are the recorded robot
       block; the per-robot actions are the recorded joint action, decoded by
       the exporter. Nothing is smoothed here. */
    var FIRE = payload.fires;
    var WINDS = payload.winds;
    function catAt(t, row, col) {
      return FIRE[clamp(t, 0, STEPS - 1)][row * COLS + col];
    }
    function robotAt(t, r) { return payload.robots[clamp(t, 0, STEPS - 1)][r]; }
    function actionAt(t, r) {
      var acts = payload.robot_actions[clamp(t, 0, STEPS - 1)];
      return acts === null || acts === undefined ? null : acts[r];
    }
    /* Whether this robot's spray actually happens, by the model's own rule: a
       live robot, the SUPPRESS digit, and a non-empty tank. A robot choosing
       SUPPRESS on an empty tank does nothing and pays nothing, and drawing
       water for it would be drawing an action the model refused. */
    function isSpraying(t, r) {
      var fields = robotAt(t, r);
      return actionAt(t, r) === SUPPRESS && fields[2] > 0 && fields[3] > 0;
    }

    // Per-step counts and the running discounted return, both off the record.
    var ALIGHT = [], BURNT_COUNT = [], RETURN = [];
    var total = 0;
    for (var t0 = 0; t0 < STEPS; t0++) {
      var alight = 0, burnt = 0;
      for (var ci = 0; ci < CELLS; ci++) {
        var cat = FIRE[t0][ci];
        if (cat === SMOLDERING || cat === BURNING) alight++;
        else if (cat === BURNT) burnt++;
      }
      ALIGHT.push(alight);
      BURNT_COUNT.push(burnt);
      var reward = (trace.steps[t0] || {}).reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, t0);
      }
      RETURN.push(total);
    }

    // When each cell first became BURNT. A cell whose fuel has just gone does
    // not go cold on the step boundary -- it smoulders and fades -- so the
    // renderer needs to know how long ago that happened. Read off the maps; it
    // changes nothing about them.
    var burntAt = new Float32Array(CELLS).fill(Infinity);
    for (var tb = 0; tb < STEPS; tb++) {
      for (var cb = 0; cb < CELLS; cb++) {
        if (FIRE[tb][cb] === BURNT && burntAt[cb] === Infinity) burntAt[cb] = tb;
      }
    }

    /* ------------------------------------------------------------- scene */
    scene.background = new THREE.Color(HAZE).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(HAZE, 0.0125);
    scene.fog.color.convertSRGBToLinear();
    // The prototype's fires are bright but the sun is brighter; with the
    // core's fixed bright-pass threshold, a strong bloom would wash the sky.
    core.composite.uniforms.bloomStrength.value = 0.16;

    // The sun is in the west-north-west, so it rakes across the run of the
    // fire rather than along it: the crew working from upwind is lit while the
    // burn is backlit, which is what makes the smoke column read. A surface
    // facing straight up receives only sin(21 deg) of the irradiance, about a
    // stop and a half down, so the intensity is raised to match.
    var SUN_DIR = new THREE.Vector3(
      -Math.cos(SUN_ELEVATION) * 0.88,
      Math.sin(SUN_ELEVATION),
      -Math.cos(SUN_ELEVATION) * 0.47
    ).normalize();
    var sun = new THREE.DirectionalLight(0xFFC078, 11.0);
    sun.position.copy(SUN_DIR).multiplyScalar(30);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    var shadowSpan = SPAN * 2.2;
    sun.shadow.camera.left = -shadowSpan;
    sun.shadow.camera.right = shadowSpan;
    sun.shadow.camera.top = shadowSpan;
    sun.shadow.camera.bottom = -shadowSpan;
    sun.shadow.camera.near = 4;
    sun.shadow.camera.far = 84;
    sun.shadow.radius = 2.6;
    // normalBias, not a big negative bias: that is the fix for acne on a
    // displaced ground mesh lit from twenty degrees up.
    sun.shadow.normalBias = 0.030;
    sun.shadow.bias = -0.0003;
    scene.add(sun);
    scene.add(sun.target);
    // The sky is a light, not a backdrop: the hemisphere tops up what the
    // environment map supplies, so shadowed sides go warm-grey, not black.
    scene.add(new THREE.HemisphereLight(0xC8A488, 0x40341E, 0.30));

    var poolTex = radialTexture(0.85, 0.42);
    var partTex = radialTexture(0.95, 0.35);
    var noiseTex = noiseTexture();
    var maskTex = cellMasks();

    var steelMat = new THREE.MeshStandardMaterial({ color: 0x9BA0AA, roughness: 0.34, metalness: 0.88 });
    var rubberMat = new THREE.MeshStandardMaterial({ color: 0x16181A, roughness: 0.92, metalness: 0.0 });
    var woodMat = WOOD_P.map(function (c) {
      return new THREE.MeshStandardMaterial({ color: c, roughness: 0.95, metalness: 0.0 });
    });

    // A smoky sky, painted once and used three ways: the dome behind the
    // hills, the reflection every metal surface needs, and the gradient the
    // haze is keyed to. Without an environment map, PBR metal has nothing to
    // mirror and reads as flat grey plastic.
    var skyTex = (function buildSky() {
      var W = 1024, H = 512;
      var cv = document.createElement("canvas");
      cv.width = W; cv.height = H;
      var g = cv.getContext("2d");
      var grad = g.createLinearGradient(0, 0, 0, H);
      // The blue is knocked back and the horizon pushed to ochre: this is a
      // sky seen through the smoke of its own fire.
      [[0.00, "#2A5288"], [0.16, "#4C6E92"], [0.32, "#7E8AA4"], [0.40, "#A8A296"],
       [0.455, "#CCAE8E"], [0.487, "#EACB9C"], [0.50, "#F0C88C"], [0.53, "#94805A"],
       [0.72, "#5E5236"], [1.00, "#38321F"]].forEach(function (stop) {
        grad.addColorStop(stop[0], stop[1]);
      });
      g.fillStyle = grad;
      g.fillRect(0, 0, W, H);

      var sunU = Math.atan2(SUN_DIR.z, SUN_DIR.x) / (Math.PI * 2) + 0.5;
      var sunV = Math.asin(clamp(SUN_DIR.y, -1, 1)) / Math.PI + 0.5;
      var sx = sunU * W, sy = (1 - sunV) * H;
      var rnd = mulberry(4180);
      for (var i = 0; i < 280; i++) {
        var cy = rnd() * 0.47 * H;
        var band = 1 - cy / (0.46 * H);
        var cx = rnd() * W;
        var rx = 22 + rnd() * 130 * (0.4 + band);
        var ry = rx * (0.10 + rnd() * 0.18);
        var toSun = 1 - clamp(Math.hypot(((cx - sx + W * 1.5) % W) - W * 0.5, cy - sy) / 300, 0, 1);
        var warm = 0.35 + 0.65 * toSun;
        var a = (0.05 + rnd() * 0.22) * (0.45 + band);
        var cg = g.createRadialGradient(cx, cy, 0, cx, cy, rx);
        cg.addColorStop(0, "rgba(" + (244 + 10 * warm | 0) + "," + (224 + 22 * warm | 0) + "," +
          (196 + 24 * warm | 0) + "," + a.toFixed(3) + ")");
        cg.addColorStop(1, "rgba(208,190,176,0)");
        g.save();
        g.translate(cx, cy); g.scale(1, ry / rx); g.translate(-cx, -cy);
        g.fillStyle = cg;
        g.beginPath(); g.arc(cx, cy, rx, 0, Math.PI * 2); g.fill();
        g.restore();
      }
      for (var h = 0; h < 3; h++) {
        var hg = g.createRadialGradient(sx, sy, 0, sx, sy, [200, 96, 34][h]);
        hg.addColorStop(0, "rgba(255,226,180," + [0.18, 0.32, 0.55][h] + ")");
        hg.addColorStop(1, "rgba(255,216,164,0)");
        g.fillStyle = hg;
        g.beginPath(); g.arc(sx, sy, [200, 96, 34][h], 0, Math.PI * 2); g.fill();
      }
      g.fillStyle = "#FFEFC8";
      g.beginPath(); g.arc(sx, sy, 11, 0, Math.PI * 2); g.fill();

      var tex = new THREE.CanvasTexture(cv);
      tex.mapping = THREE.EquirectangularReflectionMapping;
      tex.encoding = THREE.sRGBEncoding;
      var pmrem = new THREE.PMREMGenerator(renderer);
      pmrem.compileEquirectangularShader();
      scene.environment = pmrem.fromEquirectangular(tex).texture;
      pmrem.dispose();
      return tex;
    })();

    // Inside the core camera's far plane, which stays where it is: the render
    // target's depth buffer is 16-bit and moving the far plane out is how a
    // scene starts z-fighting.
    var dome = new THREE.Mesh(
      new THREE.SphereGeometry(78, 48, 28),
      new THREE.MeshBasicMaterial({ map: skyTex, side: THREE.BackSide, fog: false, depthWrite: false })
    );
    scene.add(dome);

    /* ----------------------------------------------------------- terrain
       Dry standing fuel on a shallow slope: the thing that is about to burn.
       Painted with drawing operations rather than per-pixel noise -- at a
       hundred pixels per cell a per-pixel loop is too slow at load, and noise
       gives mottling where dry grass needs strokes.

       Desaturated and straw-coloured, never lawn green. This is chaparral in
       late summer: a vivid uniform green is the loudest "stylised" signal a
       daylight field can carry, and here it would also be a lie about the
       environment, whose whole premise is that every cell is fuel. */
    var TERRAIN = SPAN * 3.2;
    var CX0 = HC - TERRAIN / 2, CX1 = HC + TERRAIN / 2;
    var RY0 = HR - TERRAIN / 2, RY1 = HR + TERRAIN / 2;
    // The access track: the depot corner runs along the near edges, which is
    // how a crew actually gets to a stand. Painted, not modelled.
    var TRACKS = [
      [DEPOT[0] - 0.7, DEPOT[1] - 0.6, DEPOT[0] - 0.7, COLS - 0.6],
      [DEPOT[0] - 0.7, DEPOT[1] - 0.6, ROWS - 0.6, DEPOT[1] - 0.6]
    ];

    function paintTerrain(S, mode) {
      var cv = document.createElement("canvas");
      cv.width = cv.height = S;
      var g = cv.getContext("2d");
      var PPU = S / TERRAIN;
      var rnd = mulberry(20260919 + (mode === "height" ? 7 : mode === "rough" ? 13 : 0));
      function px(col) { return (col - CX0) * PPU; }
      function py(row) { return (row - RY0) * PPU; }

      g.fillStyle = mode === "albedo" ? "#5A5430" : (mode === "height" ? "#7E7E7E" : "#D2D2D2");
      g.fillRect(0, 0, S, S);

      // Stems. Short directional strokes, batched by colour so a hundred
      // thousand of them still paint in well under a second. This is the
      // detail that survives the close camera.
      var lanes = mode === "albedo" ? FUEL
        : ["#5A5A5A", "#6E6E6E", "#828282", "#969696", "#AAAAAA", "#BEBEBE", "#D2D2D2", "#E6E6E6"];
      var perLane = Math.round(9000 * (S / 2048) * (S / 2048));
      g.lineCap = "round";
      for (var L = 0; L < lanes.length; L++) {
        g.strokeStyle = lanes[L];
        g.globalAlpha = mode === "rough" ? 0.10 : 0.30 + (L / lanes.length) * 0.40;
        g.lineWidth = Math.max(1.8, PPU * (0.028 + rnd() * 0.026));
        g.beginPath();
        for (var b = 0; b < perLane; b++) {
          var bx = rnd() * S, by = rnd() * S;
          var ang = -1.35 + Math.sin(bx * 0.0016) * 0.5 + Math.cos(by * 0.0021) * 0.5 + (rnd() - 0.5) * 1.5;
          var len = PPU * (0.10 + rnd() * 0.17);
          g.moveTo(bx, by);
          g.lineTo(bx + Math.cos(ang) * len, by + Math.sin(ang) * len);
        }
        g.stroke();
      }
      g.globalAlpha = 1;

      // Two scales of smooth variation over the strokes, as upscaled low-res
      // noise rather than a scatter of radial gradients -- gradients leave
      // visible discs, and a field dappled with circles is worse than none.
      function noiseLayer(cells, alpha, op) {
        var small = document.createElement("canvas");
        small.width = small.height = cells;
        var sg = small.getContext("2d");
        var id = sg.createImageData(cells, cells);
        for (var n = 0; n < cells * cells; n++) {
          var v = 64 + Math.pow(rnd(), 0.78) * 122;
          id.data[n * 4] = id.data[n * 4 + 1] = id.data[n * 4 + 2] = v;
          id.data[n * 4 + 3] = 255;
        }
        sg.putImageData(id, 0, 0);
        g.save();
        g.globalCompositeOperation = op;
        g.globalAlpha = alpha;
        g.imageSmoothingEnabled = true;
        g.imageSmoothingQuality = "high";
        g.drawImage(small, 0, 0, S, S);
        g.restore();
      }
      noiseLayer(20, 0.80, "overlay");
      noiseLayer(62, 0.54, "overlay");
      noiseLayer(150, 0.22, "overlay");

      /* Cured ground. A masked tint rather than another overlay: overlay
         shifts value and leaves hue alone, and what a real stand has is whole
         regions of a different colour, not the same colour darker. */
      function dryLayer(cells, threshold, palette, alpha) {
        var small = document.createElement("canvas");
        small.width = small.height = cells;
        var sg = small.getContext("2d");
        var id = sg.createImageData(cells, cells);
        for (var n = 0; n < cells * cells; n++) {
          var t = rnd();
          var c = new THREE.Color(palette[Math.floor(rnd() * palette.length)]);
          id.data[n * 4] = c.r * 255;
          id.data[n * 4 + 1] = c.g * 255;
          id.data[n * 4 + 2] = c.b * 255;
          id.data[n * 4 + 3] = Math.pow(clamp((t - threshold) / (1 - threshold), 0, 1), 1.4) * 255;
        }
        sg.putImageData(id, 0, 0);
        g.save();
        g.globalAlpha = alpha;
        g.imageSmoothingEnabled = true;
        g.imageSmoothingQuality = "high";
        g.drawImage(small, 0, 0, S, S);
        g.restore();
      }
      if (mode === "albedo") {
        dryLayer(9, 0.50, CURED, 0.72);                      // whole cured stretches
        dryLayer(24, 0.70, CURED, 0.38);
        dryLayer(30, 0.76, ["#6E6048", "#5E5440"], 0.36);    // bare brown scuff
        dryLayer(16, 0.74, ["#4E4430", "#5C5038"], 0.40);    // old burn scar
        // Some stems belong to the cured ground, so a bone-dry patch is made
        // of bone-dry stems rather than green ones under a tan wash.
        g.lineCap = "round";
        for (var dl = 0; dl < 3; dl++) {
          g.strokeStyle = CURED[dl];
          g.globalAlpha = 0.22;
          g.lineWidth = Math.max(1.8, PPU * 0.034);
          g.beginPath();
          for (var db = 0; db < 5000; db++) {
            var dbx = rnd() * S, dby = rnd() * S;
            var dang = -1.35 + Math.sin(dbx * 0.0016) * 0.5 + Math.cos(dby * 0.0021) * 0.5 + (rnd() - 0.5) * 1.5;
            var dlen = PPU * (0.10 + rnd() * 0.17);
            g.moveTo(dbx, dby);
            g.lineTo(dbx + Math.cos(dang) * dlen, dby + Math.sin(dang) * dlen);
          }
          g.stroke();
        }
        g.globalAlpha = 1;
      }

      // Dry stalks, seed heads and bare speckle.
      for (var d = 0; d < 8000 * (S / 2048); d++) {
        var dx = rnd() * S, dy = rnd() * S;
        var pick = rnd();
        if (mode === "albedo") {
          g.fillStyle = pick < 0.44 ? "#9E9263"
            : (pick < 0.78 ? EARTH[1] : (pick < 0.93 ? "#B4A878" : (pick < 0.976 ? "#CFC6A2" : "#6E6048")));
        } else {
          var dv = mode === "height" ? (pick < 0.5 ? 60 : 180) : (pick < 0.5 ? 120 : 230);
          g.fillStyle = "rgb(" + dv + "," + dv + "," + dv + ")";
        }
        g.globalAlpha = 0.18 + rnd() * 0.5;
        g.fillRect(dx, dy, Math.max(1, PPU * 0.018), Math.max(1, PPU * 0.018));
      }
      g.globalAlpha = 1;

      // The access track, several passes of decreasing width so the edge
      // feathers into the fuel instead of stopping at a line.
      for (var w = 0; w < 5; w++) {
        g.lineWidth = PPU * (0.96 - w * 0.15);
        g.lineCap = "round";
        g.globalAlpha = 0.12 + w * 0.05;
        g.strokeStyle = mode === "albedo" ? EARTH[1 + (w % 3)] : (mode === "height" ? "#5E5E5E" : "#EFEFEF");
        TRACKS.forEach(function (seg) {
          g.beginPath();
          for (var s = 0; s <= 30; s++) {
            var f = s / 30;
            var row = lerp(seg[0], seg[2], f) + Math.sin(f * 6.3 + seg[1]) * 0.10;
            var col = lerp(seg[1], seg[3], f) + Math.sin(f * 4.1) * 0.09;
            if (s === 0) g.moveTo(px(col), py(row)); else g.lineTo(px(col), py(row));
          }
          g.stroke();
        });
      }

      // Trodden ground: the depot apron, both start cells and every cell a
      // robot stood on. These are what make a cell legible without ruling a
      // lattice over the fuel, and they come from the episode rather than
      // from a decorator's guess about where the crew went.
      var scuffs = [DEPOT];
      world.robot_start_cells.forEach(function (cell) { scuffs.push(cell); });
      for (var st = 0; st < STEPS; st++) {
        for (var sr = 0; sr < ROBOTS; sr++) {
          var fields = payload.robots[st][sr];
          scuffs.push([fields[0], fields[1]]);
        }
      }
      scuffs.forEach(function (cell) {
        for (var k = 0; k < 14; k++) {
          var a2 = rnd() * 6.28, rr = rnd() * 0.44;
          var sx2 = px(cell[1] + Math.cos(a2) * rr), sy2 = py(cell[0] + Math.sin(a2) * rr);
          g.globalAlpha = 0.07 + rnd() * 0.17;
          g.fillStyle = mode === "albedo" ? EARTH[1 + (k % 4)] : (mode === "height" ? "#666666" : "#EAEAEA");
          g.beginPath();
          g.ellipse(sx2, sy2, PPU * (0.09 + rnd() * 0.18), PPU * (0.06 + rnd() * 0.13),
            rnd() * 3.14, 0, Math.PI * 2);
          g.fill();
        }
      });
      g.globalAlpha = 1;
      return cv;
    }

    var terrainAlbedo = new THREE.CanvasTexture(paintTerrain(2048, "albedo"));
    terrainAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    terrainAlbedo.encoding = THREE.sRGBEncoding;
    var terrainNormal = normalMapFrom(renderer, paintTerrain(1024, "height"), 3.8);
    var terrainRough = new THREE.CanvasTexture(paintTerrain(512, "rough"));
    terrainRough.anisotropy = 4;

    var terrainGeo = new THREE.PlaneGeometry(TERRAIN, TERRAIN, 220, 220);
    (function displace() {
      var pos = terrainGeo.attributes.position;
      for (var i = 0; i < pos.count; i++) {
        // The plane is built in its own XY and rotated flat afterwards, so its
        // y is the world z. Convert both back to grid space before sampling.
        var col = pos.getX(i) + HC;
        var row = -pos.getY(i) + HR;
        var y = groundY(row, col);
        var rim = Math.max(Math.abs(pos.getX(i)), Math.abs(pos.getY(i))) / (TERRAIN / 2);
        y -= smooth(0.90, 1.0, rim) * 1.4;
        pos.setZ(i, y);
      }
      terrainGeo.computeVertexNormals();
    })();

    var terrain = new THREE.Mesh(terrainGeo, new THREE.MeshStandardMaterial({
      map: terrainAlbedo, normalMap: terrainNormal,
      normalScale: new THREE.Vector2(1.8, 1.8),
      roughnessMap: terrainRough, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.48
    }));
    terrain.rotation.x = -Math.PI / 2;
    terrain.receiveShadow = true;
    scene.add(terrain);

    /* A ring of hills beyond the rim. They hide where the mesh stops and give
       the stand somewhere to be, sitting deep enough in the haze to stay a
       silhouette rather than a second lit surface. */
    (function hills() {
      var rnd = mulberry(7301);
      for (var i = 0; i < 40; i++) {
        var a = (i / 40) * Math.PI * 2 + rnd() * 0.16;
        var d = TERRAIN * 0.56 + rnd() * TERRAIN * 0.56;
        var rad = TERRAIN * 0.125 + rnd() * TERRAIN * 0.25;
        var hill = new THREE.Mesh(new THREE.SphereGeometry(rad, 9, 6),
          new THREE.MeshStandardMaterial({
            color: new THREE.Color().setHSL(0.10 + rnd() * 0.06, 0.10 + rnd() * 0.07, 0.15 + rnd() * 0.07),
            roughness: 1.0, metalness: 0.0, flatShading: true
          }));
        hill.position.set(Math.cos(a) * d, -rad * (0.70 + rnd() * 0.22), Math.sin(a) * d);
        hill.scale.y = 0.30 + rnd() * 0.22;
        hill.rotation.y = rnd() * 6.28;
        scene.add(hill);
      }
    })();

    /* ------------------------------------------------------------- props */
    function place(obj, row, col, lift) {
      obj.position.set(wx(col), groundY(row, col) + (lift || 0), wz(row));
    }

    /* The dark patch a canopy puts on the ground. A shadow map alone leaves a
       prop looking pasted on, because it misses the ambient the prop occludes.
       Offset in the depth test rather than lifted, so it never fights the
       ground it lies on -- two large near-coplanar surfaces stripe under a
       16-bit depth buffer. */
    function contactBlob(radius, alpha) {
      var m = new THREE.Mesh(new THREE.PlaneGeometry(radius * 2, radius * 2),
        new THREE.MeshBasicMaterial({
          map: poolTex, color: 0x0E0A06, transparent: true, opacity: alpha, depthWrite: false,
          polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4
        }));
      m.rotation.x = -Math.PI / 2;
      m.position.y = 0.03;
      return m;
    }

    /* Trees, tied to the cell they stand on so a canopy chars when that cell
       goes. Placed by rejection rather than by a hand-written list: they must
       keep off the obstacles, off the depot, off every cell the fire reaches
       and off every cell a robot stands on, because a tree planted on the
       crew's route would be a decoration contradicting the episode. */
    var swayers = [];
    var cellTrees = {};
    (function placeTrees() {
      var busy = {};
      world.obstacle_cells.forEach(function (c) { busy[c[0] + ":" + c[1]] = true; });
      busy[DEPOT[0] + ":" + DEPOT[1]] = true;
      for (var t = 0; t < STEPS; t++) {
        for (var c = 0; c < CELLS; c++) {
          if (FIRE[t][c] !== UNBURNT) busy[Math.floor(c / COLS) + ":" + (c % COLS)] = true;
        }
        for (var r = 0; r < ROBOTS; r++) {
          var f = payload.robots[t][r];
          busy[f[0] + ":" + f[1]] = true;
        }
      }
      var free = [];
      for (var row = 0; row < ROWS; row++) {
        for (var col = 0; col < COLS; col++) {
          if (!busy[row + ":" + col]) free.push([row, col]);
        }
      }
      var rnd = mulberry(4711);
      var wanted = Math.min(free.length, Math.round(CELLS * 0.26));
      for (var pick = free.length - 1; pick > 0; pick--) {
        var swap = Math.floor(rnd() * (pick + 1));
        var keep = free[pick]; free[pick] = free[swap]; free[swap] = keep;
      }
      free.slice(0, wanted).forEach(function (cell, ti) {
        var row = cell[0] + (rnd() - 0.5) * 0.22, col = cell[1] + (rnd() - 0.5) * 0.22;
        var tree = buildTree(cell[0] * 31 + cell[1] * 17 + 7, ti % 4);
        place(tree, row, col, 0);
        tree.rotation.y = rnd() * 6.28;
        var sc = 0.70 + rnd() * 0.30;
        // Never a uniform scale: a canopy squashed or drawn out in one axis is
        // most of what stops a stand reading as one tree stamped twenty times.
        tree.scale.set(sc * (0.85 + rnd() * 0.35), sc * (0.85 + rnd() * 0.45), sc * (0.85 + rnd() * 0.35));
        scene.add(tree);
        swayers.push(tree.userData);
        var key = cell[0] + ":" + cell[1];
        (cellTrees[key] = cellTrees[key] || []).push(tree);

        // Broken shade, not one soft disc, thrown well to one side by the low sun.
        for (var sb = 0; sb < 3; sb++) {
          var sblob = contactBlob((0.30 + rnd() * 0.26) * sc, 0.30 + rnd() * 0.16);
          place(sblob, row + 0.55 + (rnd() - 0.5) * 0.7, col + 1.05 + (rnd() - 0.5) * 0.7, 0.03);
          scene.add(sblob);
        }
        // Saplings at the foot, so a tree is a place and not an object
        // standing on a lawn.
        for (var k = 0; k < 2; k++) {
          var sa = rnd() * 6.28, sd = 0.32 + rnd() * 0.3;
          var sap = buildTree(cell[0] * 91 + k * 13, 3);
          place(sap, row + Math.sin(sa) * sd, col + Math.cos(sa) * sd, 0);
          var ss = 0.26 + rnd() * 0.22;
          sap.scale.set(ss, ss, ss);
          sap.rotation.y = rnd() * 6.28;
          scene.add(sap);
          cellTrees[key].push(sap);
        }
      });
    })();

    /* Ground cover. Instanced, because close range wants thousands of items
       and thousands of draw calls would not run. Clumped rather than
       sprinkled, and denser on and near the grid than out in the valley,
       because the near cells are where a viewer's eye goes to decide whether
       the ground is real. */
    (function scatter() {
      var rnd = mulberry(20260919);
      function bladeClump() {
        var v = [], n = 6;
        for (var b = 0; b < n; b++) {
          var a = (b / n) * 6.28 + rnd() * 0.6;
          var lean = 0.024 + rnd() * 0.046;
          var hgt = 0.058 + rnd() * 0.062;
          var tx = Math.cos(a) * lean, tz = Math.sin(a) * lean;
          var px2 = Math.cos(a + 1.57) * 0.010, pz2 = Math.sin(a + 1.57) * 0.010;
          v.push(-px2, 0, -pz2, px2, 0, pz2, tx, hgt, tz);
          v.push(px2, 0, pz2, -px2, 0, -pz2, tx, hgt, tz);
        }
        var geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
        geo.computeVertexNormals();
        return geo;
      }
      function instanced(geo, mat, count) {
        var im = new THREE.InstancedMesh(geo, mat, count);
        im.castShadow = true;
        im.receiveShadow = true;
        scene.add(im);
        return im;
      }
      var scale = clamp(CELLS / 100, 0.3, 2.0);
      var TUFTS = Math.round(6400 * scale), STONES = Math.round(520 * scale);
      var HEADS = Math.round(700 * scale), LITTER = Math.round(1800 * scale);
      var DEAD = Math.round(120 * scale);
      var white = function (rough, metal, flat, side) {
        return new THREE.MeshStandardMaterial({
          color: 0xFFFFFF, roughness: rough, metalness: metal,
          flatShading: !!flat, side: side || THREE.FrontSide
        });
      };
      var tufts = instanced(bladeClump(), white(0.96, 0.0, false, THREE.DoubleSide), TUFTS);
      var stones = instanced(new THREE.DodecahedronGeometry(0.085, 0), white(0.9, 0.04, true), STONES);
      var heads = instanced(new THREE.IcosahedronGeometry(0.024, 0), white(0.82, 0.0), HEADS);
      var litter = instanced(new THREE.PlaneGeometry(0.08, 0.05), white(1.0, 0.0, false, THREE.DoubleSide), LITTER);
      var dead = instanced(new THREE.CylinderGeometry(0.018, 0.026, 0.42, 5), white(0.98, 0.0, true), DEAD);

      var clumps = [];
      for (var cc = 0; cc < 340; cc++) {
        if (cc % 3) clumps.push([lerp(-2.5, ROWS + 1.5, rnd()), lerp(-2.5, COLS + 1.5, rnd()), 0.28 + rnd() * 0.95]);
        else clumps.push([lerp(RY0 + 2, RY1 - 2, rnd()), lerp(CX0 + 2, CX1 - 2, rnd()), 0.35 + rnd() * 1.6]);
      }
      function sample(nearClump) {
        if (nearClump && rnd() < 0.74) {
          var c = clumps[Math.floor(rnd() * clumps.length)];
          var a = rnd() * 6.28, d = Math.pow(rnd(), 0.6) * c[2];
          return [c[0] + Math.sin(a) * d, c[1] + Math.cos(a) * d];
        }
        return [lerp(RY0 + 1, RY1 - 1, rnd()), lerp(CX0 + 1, CX1 - 1, rnd())];
      }

      var tuftCols = [0x5A5432, 0x66603A, 0x726B42, 0x807850, 0x8E855C, 0x9C9268, 0xA89C74];
      var stoneCols = [0x5E564A, 0x6A6256, 0x746A5E, 0x4E463C];
      var headCols = [0x9A9070, 0x8C8260, 0x7E7458, 0xA69C7C];
      var litterCols = [0x6E5E3C, 0x7C6A44, 0x8A7A52, 0x5E5034, 0x8E8258];
      var deadCols = [0x5E5240, 0x6A5C46, 0x4E4434];
      var dummy = new THREE.Object3D();
      var n = { t: 0, s: 0, h: 0, l: 0, d: 0 }, guard = 0;

      while ((n.t < TUFTS || n.s < STONES || n.h < HEADS || n.l < LITTER || n.d < DEAD)
             && guard++ < 260000) {
        var want = n.t < TUFTS ? "t" : (n.h < HEADS ? "h" : (n.l < LITTER ? "l" : (n.d < DEAD ? "d" : "s")));
        var pt = sample(want !== "s");
        var row = pt[0], col = pt[1];
        if (row < RY0 + 0.5 || row > RY1 - 0.5 || col < CX0 + 0.5 || col > CX1 - 0.5) continue;
        var onGrid = row > -0.9 && row < ROWS - 0.1 && col > -0.9 && col < COLS - 0.1;
        // Obstacle cells are rock, so nothing grows on them.
        if (obstacleAt[Math.round(row) + ":" + Math.round(col)]) continue;
        if (!onGrid && rnd() > 0.45) continue;

        dummy.position.set(wx(col), groundY(row, col), wz(row));
        dummy.rotation.set(0, rnd() * 6.28, 0);
        var target, palette, index;
        if (want === "t") {
          var ts = 0.66 + rnd() * 0.8;
          dummy.scale.set(ts, ts * (0.6 + rnd() * 0.85), ts);
          target = tufts; palette = tuftCols; index = n.t++;
        } else if (want === "h") {
          dummy.position.y += 0.12 + rnd() * 0.09;
          var hs = 0.5 + rnd() * 0.6;
          dummy.scale.set(hs, hs * 2.2, hs);
          target = heads; palette = headCols; index = n.h++;
        } else if (want === "l") {
          dummy.rotation.set(-Math.PI / 2 + (rnd() - 0.5) * 0.5, 0, rnd() * 6.28);
          var ls = 0.6 + rnd() * 1.3;
          dummy.scale.set(ls, ls, ls);
          dummy.position.y += 0.012;
          target = litter; palette = litterCols; index = n.l++;
        } else if (want === "d") {
          dummy.rotation.set(Math.PI / 2 + (rnd() - 0.5) * 0.4, rnd() * 6.28, (rnd() - 0.5) * 0.5);
          var ds = 0.7 + rnd() * 1.0;
          dummy.scale.set(ds, ds, ds);
          dummy.position.y += 0.025;
          target = dead; palette = deadCols; index = n.d++;
        } else {
          dummy.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
          var ss2 = 0.45 + rnd() * 1.1;
          dummy.scale.set(ss2, ss2 * (0.5 + rnd() * 0.4), ss2);
          dummy.position.y -= 0.02;
          target = stones; palette = stoneCols; index = n.s++;
        }
        dummy.updateMatrix();
        target.setMatrixAt(index, dummy.matrix);
        target.setColorAt(index,
          new THREE.Color(palette[Math.floor(rnd() * palette.length)]).convertSRGBToLinear());
      }
      [[tufts, n.t], [stones, n.s], [heads, n.h], [litter, n.l], [dead, n.d]].forEach(function (p) {
        p[0].count = p[1];
        p[0].instanceMatrix.needsUpdate = true;
        if (p[0].instanceColor) p[0].instanceColor.needsUpdate = true;
      });
    })();

    /* The obstacle cells as broken rock. A robot cannot enter one and a fire
       cannot cross one, so they are the only shelter on the grid, and drawing
       them as anything a fire could cross would misstate the world. */
    (function outcrop() {
      var rnd = mulberry(3311);
      world.obstacle_cells.forEach(function (cell) {
        for (var k = 0; k < 7; k++) {
          var rock = new THREE.Mesh(new THREE.DodecahedronGeometry(0.16 + rnd() * 0.24, 0),
            new THREE.MeshStandardMaterial({
              color: new THREE.Color().setHSL(0.08, 0.05 + rnd() * 0.05, 0.22 + rnd() * 0.14),
              roughness: 0.92, metalness: 0.05, flatShading: true
            }));
          place(rock, cell[0] + (rnd() - 0.5) * 0.72, cell[1] + (rnd() - 0.5) * 0.72, 0.04 + rnd() * 0.16);
          rock.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
          rock.scale.set(1, 0.6 + rnd() * 0.5, 1);
          rock.castShadow = true;
          rock.receiveShadow = true;
          scene.add(rock);
        }
        var blob = contactBlob(0.7, 0.34);
        place(blob, cell[0] + 0.24, cell[1] + 0.42, 0.03);
        scene.add(blob);
      });
    })();

    /* The depot. In the model it is one thing: step into it and your tank is
       full again. Here it is a water tender on the access track, weathered
       like everything else, with the only saturated colour reserved for the
       markers that have to read at distance -- the beacon, and the crew. */
    var depotBeacon = null;
    (function buildDepot() {
      var g = new THREE.Group();
      place(g, DEPOT[0] - 0.12, DEPOT[1] - 0.30, 0);
      g.rotation.y = -0.22;
      scene.add(g);

      var bodyMat = new THREE.MeshStandardMaterial({ color: 0x6E4A32, roughness: 0.72, metalness: 0.18 });
      var tankMat = new THREE.MeshStandardMaterial({ color: 0x8A8478, roughness: 0.46, metalness: 0.55 });
      var cabMat = new THREE.MeshStandardMaterial({ color: 0x7A5238, roughness: 0.66, metalness: 0.22 });
      var glassMat = new THREE.MeshStandardMaterial({
        color: 0x1E2428, roughness: 0.12, metalness: 0.4, envMapIntensity: 1.6
      });

      var chassis = new THREE.Mesh(new THREE.BoxGeometry(1.46, 0.16, 0.62), bodyMat);
      chassis.position.set(0, 0.30, 0);
      chassis.castShadow = true; chassis.receiveShadow = true;
      g.add(chassis);
      var tank = new THREE.Mesh(new THREE.CylinderGeometry(0.30, 0.30, 0.84, 18), tankMat);
      tank.rotation.z = Math.PI / 2;
      tank.position.set(-0.22, 0.56, 0);
      tank.castShadow = true; tank.receiveShadow = true;
      g.add(tank);
      // Ribs, so the tank is a fabricated thing and not a smooth capsule.
      [-0.52, -0.22, 0.08].forEach(function (x) {
        var rib = new THREE.Mesh(new THREE.TorusGeometry(0.302, 0.018, 7, 20), tankMat);
        rib.rotation.y = Math.PI / 2;
        rib.position.set(x, 0.56, 0);
        rib.castShadow = true;
        g.add(rib);
      });
      var cab = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.44, 0.58), cabMat);
      cab.position.set(0.52, 0.56, 0);
      cab.castShadow = true; cab.receiveShadow = true;
      g.add(cab);
      var windscreen = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.24, 0.50), glassMat);
      windscreen.position.set(0.745, 0.62, 0);
      g.add(windscreen);
      [[0.52, 0.30], [-0.30, 0.30], [-0.62, 0.30]].forEach(function (p) {
        [-0.33, 0.33].forEach(function (z) {
          var wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.19, 0.14, 14), rubberMat);
          wheel.rotation.x = Math.PI / 2;
          wheel.position.set(p[0], 0.19, z);
          wheel.castShadow = true;
          g.add(wheel);
          var hub = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.155, 10), steelMat);
          hub.rotation.x = Math.PI / 2;
          hub.position.set(p[0], 0.19, z);
          g.add(hub);
        });
      });
      // The beacon: small, saturated, and the only thing on the truck allowed
      // to be. Above 1.0 on purpose, so the bright pass finds it.
      var beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.06, 0.07, 12),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(2.4, 1.0, 0.25) }));
      beacon.position.set(0.52, 0.82, 0);
      g.add(beacon);
      depotBeacon = new THREE.PointLight(0xFFA030, 1, 4.0, 2);
      depotBeacon.power = 26;
      depotBeacon.position.copy(beacon.position);
      g.add(depotBeacon);
      var coil = new THREE.Mesh(new THREE.TorusGeometry(0.15, 0.035, 8, 20),
        new THREE.MeshStandardMaterial({ color: 0x4A3C2C, roughness: 0.9 }));
      coil.rotation.x = Math.PI / 2;
      coil.position.set(-0.62, 0.46, 0.0);
      coil.castShadow = true;
      g.add(coil);
      var apron = contactBlob(1.5, 0.52);
      apron.position.set(0.24, 0.03, 0.30);
      g.add(apron);
    })();

    /* --------------------------------------------------------- cell decals
       One soft-edged patch per cell, tinted and roughened by that cell's
       category: char where the fire has been, a dark damp sheen where a hose
       has landed, a scorched ring under a cell that is alight. They are how
       the cells stay legible without a lattice ruled over the ground, and
       they are lit like everything else rather than glowing. */
    var decals = [];
    (function buildDecals() {
      var rnd = mulberry(881);
      for (var row = 0; row < ROWS; row++) {
        for (var col = 0; col < COLS; col++) {
          if (obstacleAt[row + ":" + col]) { decals.push(null); continue; }
          var mat = new THREE.MeshStandardMaterial({
            color: 0x1A1512, map: maskTex[Math.floor(rnd() * 4)],
            alphaMap: maskTex[Math.floor(rnd() * 4)],
            transparent: true, opacity: 0, depthWrite: false, roughness: 0.95, metalness: 0.0,
            polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3
          });
          var m = new THREE.Mesh(new THREE.PlaneGeometry(1.12, 1.12), mat);
          m.rotation.x = -Math.PI / 2;
          m.rotation.z = rnd() * 6.28;
          m.position.set(wx(col), groundY(row, col) + 0.022, wz(row));
          m.visible = false;
          scene.add(m);
          decals.push(m);
        }
      }
    })();

    /* -------------------------------------------------------------- fire */
    function flameMaterial(seed, gain) {
      return new THREE.ShaderMaterial({
        uniforms: {
          noiseTex: { value: noiseTex }, time: { value: 0 }, intensity: { value: 1 },
          seed: { value: seed }, lean: { value: 0 }, gain: { value: gain }
        },
        vertexShader: FLAME_VERT, fragmentShader: FLAME_FRAG,
        transparent: true, blending: THREE.AdditiveBlending,
        depthWrite: false, side: THREE.DoubleSide
      });
    }
    function smokeMaterial(seed) {
      return new THREE.ShaderMaterial({
        uniforms: {
          noiseTex: { value: noiseTex }, time: { value: 0 }, intensity: { value: 1 },
          seed: { value: seed }, lean: { value: 0 }, litAmount: { value: 1 }
        },
        vertexShader: FLAME_VERT, fragmentShader: SMOKE_FRAG,
        transparent: true, depthWrite: false, side: THREE.DoubleSide
      });
    }

    // A rig is built only for a cell that is alight at some point in this
    // episode, so an empty grid costs nothing.
    var fireRigs = [];
    var rigByCell = {};
    (function buildFires() {
      var seen = {};
      var cells = [];
      for (var t = 0; t < STEPS; t++) {
        for (var i = 0; i < CELLS; i++) {
          var cat = FIRE[t][i];
          if ((cat === SMOLDERING || cat === BURNING || cat === BURNT) && !seen[i]) {
            seen[i] = true;
            cells.push([Math.floor(i / COLS), i % COLS]);
          }
        }
      }
      cells.forEach(function (cell, idx) {
        var rnd = mulberry(1700 + idx * 53);
        var g = new THREE.Group();
        place(g, cell[0], cell[1], 0);
        scene.add(g);
        var flames = [], smokes = [];
        // Three flame sheets at different widths and speeds. One sheet is a
        // cardboard cut-out; three crossing each other read as a volume.
        for (var f = 0; f < 3; f++) {
          var mesh = new THREE.Mesh(
            new THREE.PlaneGeometry([0.98, 0.76, 1.22][f], [1.35, 1.05, 0.85][f]),
            flameMaterial(rnd() * 10, [1.95, 1.45, 1.15][f]));
          mesh.position.set((rnd() - 0.5) * 0.28, 0, (rnd() - 0.5) * 0.28);
          g.add(mesh);
          flames.push(mesh);
        }
        var smoke = new THREE.Mesh(new THREE.PlaneGeometry(4.6, 5.4), smokeMaterial(rnd() * 10));
        smoke.position.set((rnd() - 0.5) * 0.2, 0, (rnd() - 0.5) * 0.2);
        g.add(smoke);
        smokes.push(smoke);
        // Steam. A cell knocked down by a hose does not simply stop: it
        // gutters, and what comes off it for the rest of the step is white and
        // cool, not dirty and lit. Its own sheet, driven by the tween below.
        var steamMat = smokeMaterial(rnd() * 10 + 2.2);
        steamMat.uniforms.litAmount.value = 0.0;
        var steam = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 2.8), steamMat);
        steam.visible = false;
        g.add(steam);
        // The scorched ring the fire itself lights, under the flame.
        var glow = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 1.7),
          new THREE.MeshBasicMaterial({
            map: poolTex, color: new THREE.Color(0.85, 0.26, 0.06), transparent: true,
            opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
          }));
        glow.rotation.x = -Math.PI / 2;
        glow.position.y = 0.026;
        g.add(glow);

        var rig = { cell: cell, index: cell[0] * COLS + cell[1], group: g, flames: flames,
                    smokes: smokes, steam: steam, glow: glow, phase: rnd() * 6.28 };
        fireRigs.push(rig);
        rigByCell[cell[0] + ":" + cell[1]] = rig;
      });
    })();

    /* Fire is a light. Six shadowless sources with real lumens and
       inverse-square falloff, reassigned each frame to the strongest alight
       cells and flickering independently. A burning cell that does not light
       the ground, the smoke and the crew beside it is the single loudest tell
       that a fire is a decal. */
    var fireLights = [];
    for (var fl = 0; fl < 6; fl++) {
      var fireLight = new THREE.PointLight(0xFF7B28, 1, 5.6, 2);
      fireLight.power = 0;
      fireLight.visible = false;
      scene.add(fireLight);
      fireLights.push(fireLight);
    }

    /* Embers. One pool for the whole fire: each particle is reseeded at a
       random alight cell when it dies, rises on its own buoyancy and is
       carried downwind -- so the ember tracks are a second reading of the same
       hidden wind. Additive with per-particle colour, so fading to black IS
       fading out. */
    var EMBER_N = 230;
    var emberPos = new Float32Array(EMBER_N * 3);
    var emberCol = new Float32Array(EMBER_N * 3);
    var emberState = [];
    var embers;
    (function buildEmbers() {
      var rnd = mulberry(5511);
      for (var i = 0; i < EMBER_N; i++) {
        emberState.push({ life: rnd(), span: 1.4 + rnd() * 2.2, x: 0, y: -50, z: 0,
                          vy: 0, jx: 0, jz: 0, hot: rnd() });
        emberPos[i * 3 + 1] = -50;
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(emberPos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(emberCol, 3));
      embers = new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.052, map: partTex, vertexColors: true, transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
      }));
      // The bounding sphere is computed once, from a buffer in which every
      // particle is parked below the world, so three culls this object for the
      // whole episode and not one ember is ever drawn. Positions are rewritten
      // every frame; there is nothing for a bound to describe.
      embers.frustumCulled = false;
      scene.add(embers);
    })();

    /* ------------------------------------------------------------- crew */
    var shared = { steel: steelMat, pool: poolTex };
    var units = [];
    var unitLabels = [];
    for (var u = 0; u < ROBOTS; u++) {
      var unit = buildFirefighter(CREW_COLORS[u % CREW_COLORS.length], shared);
      unit.userData.sootBase = unit.userData.soot.map(function (m) { return m.color.clone(); });
      scene.add(unit);
      units.push(unit);
    }

    /* The sensing footprint: the Chebyshev square of radius rho around each
       live robot, which is exactly the window the observation reports. Drawn
       as an outline on the ground rather than a lit tile, because it is a
       window on the world and not a thing in it. */
    var footprints = units.map(function (ignored, k) {
      var side = SENSE * 2 + 1;
      var line = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.PlaneGeometry(side, side)),
        new THREE.LineBasicMaterial({
          color: CREW_COLORS[k % CREW_COLORS.length].helmet,
          transparent: true, opacity: 0.32, depthWrite: false
        }));
      line.rotation.x = -Math.PI / 2;
      scene.add(line);
      return line;
    });

    /* A sprite over each engine carrying its tag, health and tank. The
       environment reports all three exactly -- they are not part of the hidden
       state -- so they belong on the unit rather than in the belief. */
    function makeLabel() {
      var cv = document.createElement("canvas");
      cv.width = 320; cv.height = 96;
      var tex = new THREE.CanvasTexture(cv);
      tex.encoding = THREE.sRGBEncoding;
      var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, transparent: true, depthWrite: false, depthTest: false
      }));
      sprite.userData = { cv: cv, tex: tex, last: "" };
      return sprite;
    }
    function setLabel(sprite, text, colour) {
      if (sprite.userData.last === text) return;
      sprite.userData.last = text;
      var g = sprite.userData.cv.getContext("2d");
      g.clearRect(0, 0, 320, 96);
      g.font = "600 34px ui-monospace, Menlo, monospace";
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.lineWidth = 9;
      g.strokeStyle = "rgba(8,6,4,0.88)";
      g.strokeText(text, 160, 50);
      g.fillStyle = colour;
      g.fillText(text, 160, 50);
      sprite.userData.tex.needsUpdate = true;
    }
    units.forEach(function (unit, k) {
      var sprite = makeLabel();
      // Staggered heights: two engines standing on adjacent cells would
      // otherwise stack their labels on top of each other.
      sprite.position.set(0, 1.06 - (k % 2) * 0.24, 0);
      unit.add(sprite);
      unitLabels.push(sprite);
    });

    /* ------------------------------------------------------------ water */
    var JET_PER_ROBOT = 210;
    var JET_N = JET_PER_ROBOT * ROBOTS;
    var jetPos = new Float32Array(JET_N * 3);
    var jetCol = new Float32Array(JET_N * 3);
    var jetState = [];
    var jets;
    (function buildJets() {
      var rnd = mulberry(7717);
      for (var i = 0; i < JET_N; i++) {
        jetState.push({
          life: 9, span: 1, robot: Math.floor(i / JET_PER_ROBOT),
          vx: 0, vy: 0, vz: 0, x: 0, y: -50, z: 0,
          // A fifth of the drops break off the stream early and fall short,
          // which is what turns a solid arc into a jet with spray around it.
          breaks: rnd() < 0.22, target: null, hot: 0
        });
        jetPos[i * 3 + 1] = -50;
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(jetPos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(jetCol, 3));
      jets = new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.105, map: partTex, vertexColors: true, transparent: true,
        depthWrite: false, sizeAttenuation: true
      }));
      jets.frustumCulled = false;              // as with the embers, above
      scene.add(jets);
    })();

    /* The solid part of the jet. Water leaves a nozzle as a coherent stream
       and only breaks into droplets downstream, so the first half of the arc
       is a tapering tube swept along the ballistic path and the particles
       above take over from there. Scattered specks alone read as a sneeze.

       Allocated once and rewritten each frame: rebuilding a BufferGeometry per
       frame is exactly the per-frame work that shows up as a hitch. */
    var STREAM_RINGS = 18, STREAM_SEG = 9;
    var streams = units.map(function () {
      var pos = new Float32Array(STREAM_RINGS * STREAM_SEG * 3);
      var idx = [];
      for (var r = 0; r < STREAM_RINGS - 1; r++) {
        for (var s = 0; s < STREAM_SEG; s++) {
          var a = r * STREAM_SEG + s, b = r * STREAM_SEG + (s + 1) % STREAM_SEG;
          var c = (r + 1) * STREAM_SEG + (s + 1) % STREAM_SEG, d = (r + 1) * STREAM_SEG + s;
          idx.push(a, b, c, a, c, d);
        }
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setIndex(idx);
      // Basic and above 1.0: sunlit water is brighter than the exposure this
      // camera had to adopt for the fire, and the bright pass should find it.
      var mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: new THREE.Color(1.55, 1.95, 2.35), transparent: true, opacity: 0.80,
        depthWrite: false, side: THREE.DoubleSide
      }));
      mesh.frustumCulled = false;
      mesh.visible = false;
      scene.add(mesh);
      return { mesh: mesh, pos: pos };
    });

    /* Where the water lands: a short white puff, so the plus-shaped footprint
       the model sprays becomes five visible impacts rather than an assertion. */
    var splashes = [];
    for (var sp = 0; sp < 16; sp++) {
      var splashMat = smokeMaterial(sp * 0.71 + 3.3);
      splashMat.uniforms.litAmount.value = 0.0;
      var splashMesh = new THREE.Mesh(new THREE.PlaneGeometry(1.20, 1.35), splashMat);
      splashMesh.visible = false;
      scene.add(splashMesh);
      splashes.push({ mesh: splashMesh, mat: splashMat, born: -99, cell: null });
    }

    /* Ground that is being wetted right now, before the step resolves and the
       cell's own decal takes over. Water darkens soil the moment it lands, and
       without this the hose visibly does nothing until the step ticks. */
    var damps = [];
    (function buildDamp() {
      var rnd = mulberry(4242);
      for (var i = 0; i < 6 * ROBOTS; i++) {
        var m = new THREE.Mesh(new THREE.PlaneGeometry(1.06, 1.06),
          new THREE.MeshStandardMaterial({
            color: 0x22262A, map: maskTex[i % 4], alphaMap: maskTex[(i + 1) % 4],
            transparent: true, opacity: 0, depthWrite: false, roughness: 0.16, metalness: 0.0,
            polygonOffset: true, polygonOffsetFactor: -5, polygonOffsetUnits: -5
          }));
        m.material.color.convertSRGBToLinear();
        m.rotation.x = -Math.PI / 2;
        m.rotation.z = rnd() * 6.28;
        m.visible = false;
        scene.add(m);
        damps.push(m);
      }
    })();

    /* The true wind, off by default. It is the answer, so it is an overlay a
       reader asks for rather than part of the picture: an arrow over the depot
       apron, pointing the way the wind blows TOWARDS, which is what
       WindDirection means. */
    var trueWindArrow = new THREE.Group();
    (function buildTrueWind() {
      var mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(2.2, 0.5, 0.35) });
      var shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.9, 8), mat);
      shaft.rotation.z = -Math.PI / 2;
      trueWindArrow.add(shaft);
      var head = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.30, 10), mat);
      head.rotation.z = -Math.PI / 2;
      head.position.x = 0.58;
      trueWindArrow.add(head);
      place(trueWindArrow, DEPOT[0] + 0.6, DEPOT[1] + 1.9, 1.35);
      trueWindArrow.visible = false;
      scene.add(trueWindArrow);
    })();

    /* An optional lattice over the grid, off unless someone asks for it. Cells
       matter here -- every hidden variable is one -- but a glowing grid ruled
       over the ground is the loudest "game board" signal a frame can carry, so
       the cells are carried instead by the char, the wet patches, the trodden
       ground and the footprint outlines. */
    var cellGrid;
    (function buildGrid() {
      var verts = [];
      function run(fixedIsRow, fixed) {
        var prev = null;
        for (var i = 0; i <= 60; i++) {
          var v = lerp(-0.5, (fixedIsRow ? COLS : ROWS) - 0.5, i / 60);
          var row = fixedIsRow ? fixed : v, col = fixedIsRow ? v : fixed;
          var pt = [wx(col), groundY(row, col) + 0.035, wz(row)];
          if (prev) verts.push(prev[0], prev[1], prev[2], pt[0], pt[1], pt[2]);
          prev = pt;
        }
      }
      for (var row = 0; row <= ROWS; row++) run(true, row - 0.5);
      for (var col = 0; col <= COLS; col++) run(false, col - 0.5);
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      cellGrid = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
        color: 0xEFE6D8, transparent: true, opacity: 0.12, depthWrite: false
      }));
      cellGrid.visible = false;
      scene.add(cellGrid);
    })();

    /* Every PBR surface above was given an sRGB hex, which is how anyone reads
       a colour, but the shader needs linear. The flame, the smoke, the embers
       and the beacon are skipped on purpose by core's own pass: their values
       are already above 1.0 and are radiance, not paint. The leaf materials
       carry their own base and char colours, which the pass does not know
       about, so they are converted here. */
    core.linearize();
    scene.traverse(function (obj) {
      var mats = obj.material ? (Array.isArray(obj.material) ? obj.material : [obj.material]) : [];
      mats.forEach(function (m) {
        if (!m || !m.userData || !m.userData.base || m.userData.__ffConverted) return;
        m.userData.__ffConverted = true;
        m.userData.base.convertSRGBToLinear();
        m.userData.char.convertSRGBToLinear();
        m.color.copy(m.userData.base);
      });
    });
    // The soot bases were captured before core's pass, so recapture them after
    // it; otherwise the first frame would un-linearize the coats.
    units.forEach(function (unit) {
      unit.userData.sootBase = unit.userData.soot.map(function (m) { return m.color.clone(); });
    });

    /* --------------------------------------------------------- playback */
    var panel = buildPanel(world, ROBOTS);
    var showTrueWind = false, smokeOn = true, senseOn = true;
    panel.toggles.truewind.addEventListener("change", function () {
      showTrueWind = panel.toggles.truewind.checked;
      trueWindArrow.visible = showTrueWind;
      roseKey = "";                       // the mark changed, so repaint once
    });
    panel.toggles.smoke.addEventListener("change", function () {
      smokeOn = panel.toggles.smoke.checked;
    });
    panel.toggles.sense.addEventListener("change", function () {
      senseOn = panel.toggles.sense.checked;
    });
    panel.toggles.lattice.addEventListener("change", function () {
      cellGrid.visible = panel.toggles.lattice.checked;
    });

    var reduceMotion = global.matchMedia
      ? global.matchMedia("(prefers-reduced-motion: reduce)").matches : false;

    /* The step on screen is the one being executed, so it is the FLOOR of the
       clock and not the nearest. Rounding makes the readouts change half a
       step before the crew arrives, which reads as a stutter. */
    function stepOf(t) { return clamp(Math.floor(t), 0, STEPS - 1); }

    /* Easing between cells. A pure smoothstep has zero derivative at BOTH
       ends, so every figure comes to a dead stop on every step boundary and
       sets off again. Keeping 45% of the motion linear softens the
       turn-around without ever bringing the speed to zero. */
    function ease(f) { return lerp(f, smooth(0, 1, f), 0.55); }
    function cellAt(r, t) {
      var i0 = Math.floor(clamp(t, 0, STEPS - 1));
      var i1 = Math.min(i0 + 1, STEPS - 1);
      var f = ease(clamp(t - i0, 0, 1));
      var a = robotAt(i0, r), b = robotAt(i1, r);
      return { row: lerp(a[0], b[0], f), col: lerp(a[1], b[1], f) };
    }

    /* `headings[i]` is a compass bearing in grid space: atan2(dCol, dRow), so
       a bearing of 0 is south (+row, +z) and PI/2 is east (+col, +x). The
       figures are modelled facing their own local +x, and a group rotated by
       `rotation.y = bearing` aims its local +z, not its +x. The quarter turn
       is the correction, applied in exactly one place. */
    var MODEL_FRONT_OFFSET = -Math.PI / 2;
    function applyHeading(unit, bearing) { unit.rotation.y = bearing + MODEL_FRONT_OFFSET; }

    /* `swing` is the near leg's hip angle in radians: positive is forward. The
       knee only ever folds backwards, and only on the leg behind the body,
       which is the recovery half of a real stride. */
    function setGait(unit, swing) {
      var legs = unit.userData.legs, knees = unit.userData.knees;
      for (var L = 0; L < 2; L++) {
        var s = L === 0 ? swing : -swing;
        legs[L].rotation.z = s;
        knees[L].rotation.z = -clamp(-s, 0, 1) * 1.35;
      }
      // The arms hold the line, so they do not swing freely: the whole upper
      // assembly rocks a little against the stride, which is what a person
      // carrying a charged hose actually does.
      if (unit.userData.upper) unit.userData.upper.rotation.z = -swing * 0.085;
    }

    var headings = [], travelHeading = [], lastCell = [];
    for (var hi = 0; hi < ROBOTS; hi++) {
      headings.push(Math.PI / 2);
      travelHeading.push(Math.PI / 2);
      lastCell.push([robotAt(0, hi)[0], robotAt(0, hi)[1]]);
    }

    /* How much flame a category carries. The MODEL is binary -- a cell is
       alight or it is not, and a hose flips it in one step -- and none of
       these numbers touch it. They are the renderer's reading of the same five
       categories, so a step's worth of change can be drawn as a change rather
       than as a switch. */
    function fireLevel(cat) {
      return cat === BURNING ? 1.0 : (cat === SMOLDERING ? 0.42 : (cat === BURNT ? 0.13 : 0.0));
    }
    function charOf(cat) {
      return cat === BURNT ? 1.0 : (cat === BURNING ? 0.75 : (cat === SMOLDERING ? 0.35 : 0.0));
    }

    /* ---------------------------------------------------- the hose sweep
       The model sprays a plus -- the robot's own cell and its four neighbours
       -- and that coverage set is not negotiable. But water leaves a hose
       ALONG the hose, so emitting it towards all five cells at once puts water
       out of the firefighter's back and sides with no relation to where the
       nozzle points. The resolution is to spread the footprint over TIME
       rather than over direction: one stream, always along the nozzle's own
       axis, swept across the five cells during the step. Every cell still gets
       water inside the step, the coverage the model specifies is untouched,
       and at any instant there is exactly one jet going exactly where the line
       is aimed. */
    var sweepCache = [];
    for (var sc = 0; sc < ROBOTS; sc++) sweepCache.push({ step: -1, order: [], cum: [] });

    function buildSweep(r, step) {
      var cache = sweepCache[r];
      cache.step = step;
      cache.order = [];
      cache.cum = [];
      var own = robotAt(step, r);
      var ring = [];
      for (var d = 0; d < DIR.length; d++) {
        var rr = own[0] + DIR[d][0], cc = own[1] + DIR[d][1];
        if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS) ring.push([rr, cc]);
      }
      // Sorted by bearing, so consecutive aims are adjacent in angle and the
      // line traverses instead of snapping from one side to the other.
      ring.sort(function (a, b) {
        return Math.atan2(a[1] - own[1], a[0] - own[0]) - Math.atan2(b[1] - own[1], b[0] - own[0]);
      });
      cache.order.push([own[0], own[1]]);
      for (var i = 0; i < ring.length; i++) cache.order.push(ring[i]);

      var weights = [], sum = 0;
      for (var k = 0; k < cache.order.length; k++) {
        // Dwell longer on the cell that most needs it.
        var level = fireLevel(FIRE[step][cache.order[k][0] * COLS + cache.order[k][1]]);
        weights.push(1 + 2.4 * level);
        sum += 1 + 2.4 * level;
      }
      var acc = 0;
      for (var k2 = 0; k2 < weights.length; k2++) { cache.cum.push(acc / sum); acc += weights[k2]; }
    }

    /* The aim at this instant: dwelling on one cell for most of its slice,
       then traversing to the next. Fractional (row, col), so the stream is
       continuous rather than stepping between cell centres. */
    function sweepAim(r, step, frac) {
      var cache = sweepCache[r];
      if (cache.step !== step) buildSweep(r, step);
      var n = cache.order.length;
      if (!n) return null;
      var i = 0;
      while (i < n - 1 && frac >= cache.cum[i + 1]) i++;
      var hi2 = (i === n - 1) ? 1 : cache.cum[i + 1];
      var span = hi2 - cache.cum[i];
      var u = span > 1e-6 ? (frac - cache.cum[i]) / span : 0;
      var a = cache.order[i], b = cache.order[Math.min(i + 1, n - 1)];
      var blend = smooth(0.62, 1.0, u);
      return { row: lerp(a[0], b[0], blend), col: lerp(a[1], b[1], blend) };
    }

    /* Point the line at the aim. The nozzle's local yaw and pitch are solved
       from the SAME launch velocity the core and the droplets use, so the
       barrel cannot point one way while the water goes another -- which is
       precisely how an all-directions spray arises. */
    var aimTip = new THREE.Vector3();
    function solveAim(unit, aim, ox, oy, oz) {
      var ax = wx(aim.col), az = wz(aim.row);
      var ay = groundY(Math.round(aim.row), Math.round(aim.col)) + 0.06;
      var vx = (ax - ox) / JET_T;
      var vy = (ay - oy) / JET_T + 0.5 * GRAV * JET_T;
      var vz = (az - oz) / JET_T;
      var nozzle = unit.userData.nozzle;
      // World yaw of the barrel is the unit's yaw plus the nozzle's own, and a
      // Y-rotation by a takes local +x to (cos a, 0, -sin a).
      nozzle.rotation.y = Math.atan2(-vz, vx) - unit.rotation.y;
      nozzle.rotation.z = Math.atan2(vy, Math.hypot(vx, vz)) - unit.userData.upper.rotation.z;
    }
    function aimNozzle(unit, aim) {
      // Two passes. The tip's position depends on the angle being solved for,
      // so the first uses the shoulder as an estimate and the second re-solves
      // from where the tip actually ended up. One pass leaves the barrel about
      // seven degrees off the water it is throwing.
      solveAim(unit, aim, unit.position.x, unit.position.y + 0.345, unit.position.z);
      unit.updateMatrixWorld(true);
      unit.userData.nozzleTip.getWorldPosition(aimTip);
      solveAim(unit, aim, aimTip.x, aimTip.y, aimTip.z);
      unit.updateMatrixWorld(true);
    }

    /* Write one jet core into its pre-allocated buffer. The rings follow the
       ballistic path for the first 58% of the flight and taper to nothing, so
       the stream does not end at a flat disc -- it thins until the droplets
       are all that is left, which is what breaking up looks like. */
    var sTan = new THREE.Vector3(), sU = new THREE.Vector3();
    var sV = new THREE.Vector3(), sP = new THREE.Vector3();
    function writeStream(stream, origin, vel, phase) {
      var pos = stream.pos;
      for (var r = 0; r < STREAM_RINGS; r++) {
        var f = r / (STREAM_RINGS - 1);
        var tt = f * JET_T * 0.58;
        sP.set(origin.x + vel.x * tt,
               origin.y + vel.y * tt - 0.5 * GRAV * tt * tt,
               origin.z + vel.z * tt);
        sTan.set(vel.x, vel.y - GRAV * tt, vel.z).normalize();
        sU.set(-sTan.z, 0, sTan.x);
        if (sU.lengthSq() < 1e-8) sU.set(1, 0, 0);
        sU.normalize();
        sV.crossVectors(sTan, sU).normalize();
        var rad = (0.0195 * Math.pow(1 - f, 0.72) + 0.0016)
                * (1 + 0.16 * Math.sin(f * 9.0 - phase * 7.0));
        for (var s = 0; s < STREAM_SEG; s++) {
          var ang = s / STREAM_SEG * Math.PI * 2;
          var i3 = (r * STREAM_SEG + s) * 3;
          pos[i3] = sP.x + (Math.cos(ang) * sU.x + Math.sin(ang) * sV.x) * rad;
          pos[i3 + 1] = sP.y + (Math.cos(ang) * sU.y + Math.sin(ang) * sV.y) * rad;
          pos[i3 + 2] = sP.z + (Math.cos(ang) * sU.z + Math.sin(ang) * sV.z) * rad;
        }
      }
      stream.mesh.geometry.attributes.position.needsUpdate = true;
    }

    // One launch per engine per frame: origin, velocity and the cell being
    // hit. Everything wet on screen is derived from these.
    var jetLaunch = units.map(function () {
      return { on: false, o: new THREE.Vector3(), v: new THREE.Vector3(), cell: null, hot: 0 };
    });
    var sprayAim = units.map(function () { return null; });
    var splashCursor = 0;

    var decalColA = new THREE.Color(), decalColB = new THREE.Color();
    var tmpV = new THREE.Vector3();
    var roseKey = "";
    // What the shared HUD's belief line says. Written when the rose is
    // repainted, so the two always report the same posterior.
    var beliefLabel = "—";
    var crewFollow = { x: 0, z: 0, heading: 0, spread: 1 };

    /**
     * Advance the world to continuous step index t.
     *
     * @returns {Object} HUD fields for the player to display.
     */
    function update(t, dt, elapsed, playing) {
      var step = stepOf(t);
      var nextStep = Math.min(step + 1, STEPS - 1);
      var map = FIRE[step], mapNext = FIRE[nextStep];
      // One eased fraction through the step drives every tween below, so the
      // ground, the canopies and the flames all change together.
      var frac = clamp(t - step, 0, 1);
      var fe = ease(frac);
      var wind = WINDS[step];
      var windIndex = wind[0] * STRENGTHS + wind[1];
      // The hidden wind, as a unit vector in grid space. It drives the flame
      // lean, the smoke, the embers and the canopies -- and nothing else on
      // this page reveals it, which is the point.
      var windX = DIR[wind[0]][1], windZ = DIR[wind[0]][0];
      var windMag = wind[1] ? 1.0 : 0.58;

      // --- ground decals, tweened between this step's category and the next
      for (var i = 0; i < CELLS; i++) {
        var decal = decals[i];
        if (!decal) continue;
        var sa = DECAL[map[i]], sb = DECAL[mapNext[i]];
        if (!sa && !sb) { decal.visible = false; continue; }
        decal.visible = true;
        // An unburnt cell has no patch at all, so it borrows its neighbour in
        // time's colour and fades in or out of it rather than flashing.
        decalColA.setHex((sa || sb).color);
        decalColB.setHex((sb || sa).color);
        decal.material.color.copy(decalColA).lerp(decalColB, fe).convertSRGBToLinear();
        decal.material.opacity = lerp(sa ? sa.opacity : 0, sb ? sb.opacity : 0, fe);
        decal.material.roughness = lerp(sa ? sa.rough : 0.95, sb ? sb.rough : 0.95, fe);
      }

      // --- trees char as their own cell goes, over the step rather than on it
      Object.keys(cellTrees).forEach(function (key) {
        var parts = key.split(":");
        var index = (+parts[0]) * COLS + (+parts[1]);
        var charAmt = lerp(charOf(map[index]), charOf(mapNext[index]), fe);
        cellTrees[key].forEach(function (tree) {
          if (Math.abs((tree.userData.charAmt || 0) - charAmt) < 0.004) return;
          tree.userData.charAmt = charAmt;
          tree.userData.leafMats.forEach(function (m) {
            m.color.copy(m.userData.base).lerp(m.userData.char, charAmt);
          });
        });
      });

      // --- fire rigs, continuous across the step
      var lit = [];
      fireRigs.forEach(function (rig) {
        var cat = map[rig.index], catNext = mapNext[rig.index];
        var l0 = fireLevel(cat), l1 = fireLevel(catNext);

        /* Three kinds of change, each with its own curve, because they do not
           look alike. Catching starts slow and accelerates, so a new cell
           reads as embers and a tongue of flame before it is a fire. Being
           knocked down falls fast and early, which is what a hose does.
           Burning out is gradual, and then the remnant smoulders and dies. */
        var level;
        if (l1 > l0) level = lerp(l0, l1, smooth(0.06, 0.98, frac) * smooth(0.06, 0.98, frac));
        else if (l1 < l0) level = lerp(l0, l1, smooth(0.04, 0.72, frac));
        else level = l0;
        if (cat === BURNT || catNext === BURNT) {
          var age = t - burntAt[rig.index];
          if (age >= 0) level = Math.min(level, 0.13 * clamp(1 - age / 3.0, 0, 1));
        }
        // Guttering: a flame being put out does not fall smoothly, it stutters.
        var dying = l1 < l0 && catNext === WET;
        if (dying) level *= 0.72 + 0.28 * Math.abs(Math.sin(elapsed * 11.0 + rig.phase));

        // Steam comes off a cell for the rest of the step after water lands.
        var steamAmt = dying ? smooth(0.05, 0.45, frac) * (1 - smooth(0.55, 1.0, frac) * 0.55) : 0;
        rig.steam.visible = steamAmt > 0.02 && smokeOn;
        if (rig.steam.visible) {
          rig.steam.rotation.y = Math.atan2(core.camera.position.x - rig.group.position.x,
                                            core.camera.position.z - rig.group.position.z);
          rig.steam.position.y = 0.9 + frac * 0.9;
          rig.steam.scale.setScalar(0.7 + frac * 0.8);
          rig.steam.material.uniforms.time.value = elapsed * 1.4 + rig.phase;
          rig.steam.material.uniforms.intensity.value = steamAmt;
          rig.steam.material.uniforms.lean.value = windMag * 0.9;
        }

        rig.group.visible = level > 0.015 || rig.steam.visible;
        if (level <= 0.015) {
          rig.flames.forEach(function (m) { m.visible = false; });
          rig.smokes.forEach(function (m) { m.visible = false; });
          rig.glow.material.opacity = 0;
          return;
        }
        var flick = 0.80 + Math.sin(elapsed * 7.3 + rig.phase) * 0.12
                         + Math.sin(elapsed * 2.9 + rig.phase * 2.1) * 0.08;
        // Yaw-only billboarding: a flame column stays upright, it does not tip
        // towards the camera.
        var yaw = Math.atan2(core.camera.position.x - rig.group.position.x,
                             core.camera.position.z - rig.group.position.z);
        var lean = windMag * (0.48 + 0.16 * level);
        // The billboard's own +x in world terms decides the sign of the skew,
        // so the flame leans the same way from every camera angle.
        var alongX = Math.cos(yaw) * windX + (-Math.sin(yaw)) * windZ;

        rig.flames.forEach(function (mesh, fi) {
          mesh.visible = true;
          mesh.rotation.y = yaw;
          mesh.position.y = 0.02 + [0.68, 0.53, 0.44][fi] * (0.6 + 0.4 * level);
          mesh.scale.set(1, (0.55 + 0.75 * level) * (0.92 + 0.16 * flick), 1);
          var u = mesh.material.uniforms;
          u.time.value = elapsed + rig.phase;
          u.intensity.value = level * flick;
          u.lean.value = lean * alongX;
        });
        rig.smokes.forEach(function (mesh, si) {
          mesh.visible = smokeOn;
          mesh.rotation.y = yaw;
          mesh.position.y = 1.85 + si * 0.95;
          mesh.scale.set(1, 0.7 + 0.5 * level, 1);
          var u = mesh.material.uniforms;
          u.time.value = elapsed * (0.8 + si * 0.25) + rig.phase;
          u.intensity.value = level;
          // Smoke drifts further than flame leans: it has longer to be carried.
          u.lean.value = lean * alongX * 0.52;
          u.litAmount.value = level;
        });
        rig.glow.material.opacity = (0.13 + 0.20 * level) * flick;
        rig.glow.scale.setScalar(0.8 + 0.4 * level);
        lit.push({ rig: rig, intensity: level, flick: flick });
      });

      // --- fire as a light source
      lit.sort(function (a, b) { return b.intensity - a.intensity; });
      for (var L = 0; L < fireLights.length; L++) {
        var slot = lit[L];
        if (!slot) { fireLights[L].visible = false; fireLights[L].power = 0; continue; }
        fireLights[L].visible = true;
        fireLights[L].position.set(slot.rig.group.position.x,
                                   slot.rig.group.position.y + 0.55,
                                   slot.rig.group.position.z);
        // Real lumens with inverse-square decay, flickering on the same phase
        // as the flame it belongs to, so the light and the picture agree.
        fireLights[L].power = 178 * slot.intensity * slot.flick;
        fireLights[L].color.setRGB(1.0, 0.44 + 0.08 * slot.flick, 0.14);
      }

      // --- embers, carried downwind: the wind written across the sky
      for (var e = 0; e < EMBER_N; e++) {
        var p = emberState[e];
        p.life += dt;
        if (p.life > p.span || p.y < -40) {
          if (!lit.length || reduceMotion) { emberPos[e * 3 + 1] = -50; continue; }
          var src = lit[Math.floor(Math.random() * lit.length)].rig;
          p.x = src.group.position.x + (Math.random() - 0.5) * 0.7;
          p.z = src.group.position.z + (Math.random() - 0.5) * 0.7;
          p.y = src.group.position.y + 0.15 + Math.random() * 0.4;
          p.vy = 0.9 + Math.random() * 1.5;
          p.jx = (Math.random() - 0.5) * 0.5;
          p.jz = (Math.random() - 0.5) * 0.5;
          p.hot = 0.6 + Math.random() * 0.4;
          p.life = 0;
          p.span = 1.2 + Math.random() * 2.4;
        }
        var emberAge = p.life / p.span;
        p.vy = Math.max(0.12, p.vy - dt * 0.55);
        p.y += p.vy * dt;
        p.x += (windX * windMag * 1.35 + p.jx) * dt;
        p.z += (windZ * windMag * 1.35 + p.jz) * dt;
        emberPos[e * 3] = p.x;
        emberPos[e * 3 + 1] = p.y;
        emberPos[e * 3 + 2] = p.z;
        // Cools as it climbs: white-hot, then orange, then a dull red.
        var fade = (1 - emberAge) * (1 - emberAge);
        var twinkle = 0.65 + 0.35 * Math.sin(elapsed * 14 + e);
        emberCol[e * 3] = 1.7 * fade * p.hot * twinkle;
        emberCol[e * 3 + 1] = 0.46 * fade * fade * p.hot * twinkle;
        emberCol[e * 3 + 2] = 0.07 * fade * fade * fade * twinkle;
      }
      embers.geometry.attributes.position.needsUpdate = true;
      embers.geometry.attributes.color.needsUpdate = true;

      // --- the crew
      var midX = 0, midZ = 0, live = 0;
      for (var r = 0; r < ROBOTS; r++) {
        var unit = units[r];
        var cell = cellAt(r, t);
        var y = groundY(cell.row, cell.col);
        unit.position.set(wx(cell.col), y, wz(cell.row));
        var fields = robotAt(step, r);
        var aim = isSpraying(step, r) ? sweepAim(r, step, frac) : null;
        sprayAim[r] = aim;

        var dRow = cell.row - lastCell[r][0], dCol = cell.col - lastCell[r][1];
        if (Math.abs(dRow) + Math.abs(dCol) > 1e-4) {
          var want = Math.atan2(dCol, dRow);
          var diff = ((want - travelHeading[r] + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          travelHeading[r] += diff * clamp(dt * 9, 0, 1);
        }
        /* The body follows the line while spraying. A hose that swings
           independently of a fixed torso is half of what makes the water look
           detached from the figure holding it. */
        var face = aim ? Math.atan2(aim.col - cell.col, aim.row - cell.row) : travelHeading[r];
        if (!aim || Math.abs(aim.row - cell.row) + Math.abs(aim.col - cell.col) > 1e-3) {
          var fd = ((face - headings[r] + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          headings[r] += fd * clamp(dt * (aim ? 5.0 : 9.0), 0, 1);
        }
        applyHeading(unit, headings[r]);
        lastCell[r] = [cell.row, cell.col];

        var here = robotAt(step, r), next = robotAt(Math.min(step + 1, STEPS - 1), r);
        var moving = (Math.abs(here[0] - next[0]) + Math.abs(here[1] - next[1])) > 0;
        var swing = moving && !reduceMotion ? Math.sin(t * Math.PI * 4) * 0.55 : 0;
        setGait(unit, swing);
        unit.position.y = y + (moving && !reduceMotion ? Math.abs(Math.sin(t * Math.PI * 4)) * 0.018 : 0);

        if (aim) {
          // Braced against the nozzle reaction, and aimed from the same vector
          // the water is launched along.
          unit.userData.upper.rotation.z = -0.06;
          aimNozzle(unit, aim);
        } else {
          unit.userData.nozzle.rotation.y = 0;
          unit.userData.nozzle.rotation.z =
            lerp(unit.userData.nozzle.rotation.z, 0.10, clamp(dt * 6, 0, 1));
        }

        // Soot. A crew that has stood beside a fire for ten steps is not the
        // colour it started; health lost darkens the coat further.
        var soot = clamp((MAX_HEALTH - fields[3]) / MAX_HEALTH * 0.6 + step / STEPS * 0.22, 0, 0.8);
        unit.userData.soot.forEach(function (m, mi) {
          m.color.copy(unit.userData.sootBase[mi]).multiplyScalar(1 - soot * 0.55);
        });

        setLabel(unitLabels[r],
          "E" + (r + 1) + "  " + fields[3] + "/" + MAX_HEALTH + "hp  " + fields[2] + "/" + MAX_TANK,
          fields[3] < MAX_HEALTH ? "#FFC8A0" : "#F2E8D8");
        unitLabels[r].getWorldPosition(tmpV);
        var lk = clamp(core.camera.position.distanceTo(tmpV) * 0.030, 0.26, 0.52);
        unitLabels[r].scale.set(lk * 3.34, lk * 1.0, 1);

        // A disabled robot sees nothing, which is the model's own rule, so its
        // window is not drawn.
        footprints[r].visible = senseOn && fields[3] > 0;
        footprints[r].position.set(wx(cell.col), y + 0.040, wz(cell.row));

        if (fields[3] > 0) {
          midX += unit.position.x; midZ += unit.position.z; live++;
        }
      }

      // Where the crew camera looks: the live engines' midpoint, and how far
      // apart they are, so two engines working opposite flanks both stay in
      // frame instead of one filling it.
      if (live) {
        crewFollow.x = midX / live;
        crewFollow.z = midZ / live;
        var spread = 0;
        for (var a1 = 0; a1 < ROBOTS; a1++) {
          spread = Math.max(spread, Math.hypot(units[a1].position.x - crewFollow.x,
                                               units[a1].position.z - crewFollow.z));
        }
        crewFollow.spread = spread;
        crewFollow.heading = travelHeading[0];
      }

      /* --- water. One jet per engine, leaving the nozzle along the nozzle's
         own axis. The origin, the core and every droplet derive from a single
         launch velocity computed here, so there is no second spread anywhere
         that could send water in a direction the barrel is not pointing. The
         five cells of the model's footprint are covered by sweeping this one
         jet across them over the step, not by firing at all of them at once. */
      for (var rb = 0; rb < ROBOTS; rb++) {
        var launch = jetLaunch[rb];
        var aimR = sprayAim[rb];
        launch.on = !!aimR && !reduceMotion;
        if (!launch.on) { streams[rb].mesh.visible = false; continue; }
        // The real nozzle tip, after the aim above has turned it.
        units[rb].userData.nozzleTip.getWorldPosition(launch.o);
        var ax = wx(aimR.col), az = wz(aimR.row);
        var ay = groundY(Math.round(aimR.row), Math.round(aimR.col)) + 0.05;
        launch.v.set((ax - launch.o.x) / JET_T,
                     (ay - launch.o.y) / JET_T + 0.5 * GRAV * JET_T,
                     (az - launch.o.z) / JET_T);
        launch.cell = [Math.round(aimR.row), Math.round(aimR.col)];
        var landing = map[launch.cell[0] * COLS + launch.cell[1]];
        launch.hot = landing === BURNING ? 1.0 : (landing === SMOLDERING ? 0.5 : 0.0);
        writeStream(streams[rb], launch.o, launch.v, elapsed + rb);
        streams[rb].mesh.visible = true;
      }

      var splashWanted = [];
      for (var j = 0; j < JET_N; j++) {
        var drop = jetState[j];
        var source = jetLaunch[drop.robot];
        drop.life += dt;
        if (drop.life > drop.span) {
          // It landed. Mark the aim cell for a splash, then re-launch.
          if (drop.target && drop.life < drop.span + dt * 2) splashWanted.push(drop.target);
          if (!source.on) { jetPos[j * 3 + 1] = -50; drop.target = null; continue; }
          drop.x = source.o.x; drop.y = source.o.y; drop.z = source.o.z;
          drop.target = source.cell;
          drop.hot = source.hot;
          /* Dispersion is a narrow cone about the launch vector, not a target
             of its own: a drop may leave the nozzle a few degrees off the
             stream, and that is all. A per-droplet target is what produces a
             spray in every direction at once. */
          var speed = source.v.length();
          var cone = drop.breaks ? 0.085 : 0.028;
          drop.vx = source.v.x + (Math.random() - 0.5) * speed * cone;
          drop.vy = source.v.y + (Math.random() - 0.5) * speed * cone * 0.7;
          drop.vz = source.v.z + (Math.random() - 0.5) * speed * cone;
          drop.span = drop.breaks ? JET_T * (0.45 + Math.random() * 0.30) : JET_T;
          /* Start the drop part-way down the arc, where the core has thinned.
             Launching every drop at the nozzle puts a cloud of specks round
             the barrel and leaves nothing where the water actually is. */
          var t0 = JET_T * (0.34 + Math.random() * 0.10);
          drop.x += drop.vx * t0;
          drop.y += drop.vy * t0 - 0.5 * GRAV * t0 * t0;
          drop.z += drop.vz * t0;
          drop.vy -= GRAV * t0;
          drop.life = t0;
        }
        drop.vy -= GRAV * dt;
        drop.x += drop.vx * dt; drop.y += drop.vy * dt; drop.z += drop.vz * dt;
        jetPos[j * 3] = drop.x; jetPos[j * 3 + 1] = drop.y; jetPos[j * 3 + 2] = drop.z;
        // Bright at the nozzle where the stream is solid, thinning as it
        // breaks. Water in late sun is brighter than 1.0, so it survives the
        // exposure the fire forced on this camera.
        var head = clamp(1 - drop.life / Math.max(drop.span, 1e-3), 0, 1);
        var bright = (drop.breaks ? 1.25 : 2.45) * (0.50 + 0.50 * head);
        jetCol[j * 3] = bright * (0.72 + 0.80 * drop.hot);
        jetCol[j * 3 + 1] = bright * (0.88 + 0.16 * drop.hot);
        jetCol[j * 3 + 2] = bright * 1.06;
      }
      jets.geometry.attributes.position.needsUpdate = true;
      jets.geometry.attributes.color.needsUpdate = true;

      for (var sw = 0; sw < splashWanted.length; sw++) {
        var slot2 = splashes[splashCursor % splashes.length];
        splashCursor++;
        slot2.cell = splashWanted[sw];
        slot2.born = elapsed;
      }
      splashes.forEach(function (splash) {
        var splashAge = (elapsed - splash.born) / 0.85;
        var on = splash.cell && splashAge < 1 && smokeOn && !reduceMotion;
        splash.mesh.visible = !!on;
        if (!on) return;
        splash.mesh.position.set(wx(splash.cell[1]),
          groundY(splash.cell[0], splash.cell[1]) + 0.30 + splashAge * 0.42,
          wz(splash.cell[0]));
        splash.mesh.rotation.y = Math.atan2(core.camera.position.x - splash.mesh.position.x,
                                            core.camera.position.z - splash.mesh.position.z);
        splash.mesh.scale.setScalar(0.45 + splashAge * 0.75);
        splash.mat.uniforms.time.value = elapsed * 1.6;
        splash.mat.uniforms.intensity.value = 1 - splashAge;
        splash.mat.uniforms.lean.value = windMag * 0.7;
      });

      /* Ground darkening. Each covered cell goes dark as the sweep reaches it,
         in the order the line visits them, so by the end of the step the whole
         plus is wet -- which is the coverage the model applies all at once. */
      var dampUsed = 0;
      for (var rb2 = 0; rb2 < ROBOTS; rb2++) {
        if (!jetLaunch[rb2].on) continue;
        var order = sweepCache[rb2].order, cums = sweepCache[rb2].cum;
        for (var ti = 0; ti < order.length && dampUsed < damps.length; ti++) {
          var dc = order[ti];
          if (obstacleAt[dc[0] + ":" + dc[1]]) continue;
          var damp = damps[dampUsed++];
          damp.visible = true;
          damp.position.set(wx(dc[1]), groundY(dc[0], dc[1]) + 0.026, wz(dc[0]));
          damp.material.opacity = 0.88 * smooth(cums[ti], cums[ti] + 0.14, frac);
        }
      }
      for (; dampUsed < damps.length; dampUsed++) damps[dampUsed].visible = false;

      // --- the depot beacon, and the canopies leaning downwind on the same
      //     hidden value the fire spreads under
      if (!reduceMotion) {
        depotBeacon.power = 26 * (0.6 + 0.4 * Math.abs(Math.sin(elapsed * 2.2)));
        var gust = 0.55 + 0.45 * Math.sin(elapsed * 0.52);
        for (var sy = 0; sy < swayers.length; sy++) {
          var sway = swayers[sy];
          var amp = (0.018 + 0.026 * gust) * (0.5 + windMag);
          sway.canopy.rotation.z = windX * amp * 1.6 + Math.sin(elapsed * 1.25 + sway.phase) * amp * 0.5;
          sway.canopy.rotation.x = -windZ * amp * 1.6 + Math.cos(elapsed * 0.95 + sway.phase * 1.7) * amp * 0.4;
        }
      }
      trueWindArrow.rotation.y = Math.atan2(windX, windZ) - Math.PI / 2;
      dome.position.copy(core.camera.position);
      sun.target.position.set(0, 0, 0);
      sun.target.updateMatrixWorld();

      // --- the belief. Repainted only when it actually changes: the belief
      //     moves once per step, and rewriting sixteen SVG paths every frame
      //     is a layout pass per frame for a picture that changes a few dozen
      //     times in a whole episode.
      var key = step + ":" + (showTrueWind ? 1 : 0);
      if (key !== roseKey) {
        roseKey = key;
        var marginal = windMarginal(payload.beliefs[step], LAYOUT, STRENGTHS);
        if (marginal.mass) {
          var summary = drawRose(panel, marginal.mass, world.wind_labels, windIndex, showTrueWind);
          beliefLabel = (world.wind_labels[summary.best] || summary.best) +
            " " + marginal.mass[summary.best].toFixed(2) + ", " + summary.bits.toFixed(2) + " bits";
          if (marginal.written < marginal.particles) {
            beliefLabel += " (heaviest " + marginal.written + " of " + marginal.particles + ")";
          }
        } else {
          clearRose(panel, marginal.why);
          beliefLabel = marginal.why;
        }
        for (var cr = 0; cr < ROBOTS; cr++) {
          var crewFields = robotAt(step, cr);
          var act = actionAt(step, cr);
          var actName = act === null || act === undefined ? "—" : world.action_names[act];
          if (act === SUPPRESS && crewFields[2] <= 0) actName = "suppress (dry tank)";
          if (crewFields[3] <= 0) actName = "disabled";
          panel.crew[cr].textContent = actName + " · (" + crewFields[0] + ", " +
            crewFields[1] + ") · " + crewFields[3] + "/" + MAX_HEALTH + " hp · " +
            crewFields[2] + "/" + MAX_TANK + " tank";
        }
      }
      void playing;

      var envelope = trace.steps[step] || {};
      var lead = robotAt(step, 0);
      return {
        follow: crewFollow,
        step: step,
        action: describeActions(step),
        // The shared HUD has one position line, so it carries the lead engine;
        // the panel below the viewport carries every engine in full.
        x: lead[1],
        y: lead[0],
        reward: envelope.reward,
        ret: RETURN[step],
        belief: beliefLabel
      };
    }

    /* The HUD's one action line, for a joint action that is really N of them.
       The joint integer is what the planner chose; its digits are what the
       engines did, so both are shown rather than one standing in for the
       other. */
    function describeActions(step) {
      var acts = payload.robot_actions[step];
      if (acts === null || acts === undefined) return "—";
      var names = acts.map(function (a) { return world.action_names[a] || String(a); });
      var joint = (trace.steps[step] || {}).action;
      return names.join(" + ") + (joint === null || joint === undefined ? "" : " [" + joint + "]");
    }

    /* ------------------------------------------------------- crew camera
       The core rig's chase mode frames ONE agent at a fixed distance behind
       its heading. With two engines that is a camera pointed at one of them
       while the other works off screen, and with the engines on opposite
       flanks of a front it is a camera pointed at nothing in particular.

       So this module supplies its own, and does it without touching the rig
       every other environment shares: it wraps `createCameraRig` for the one
       call the player is about to make, puts the original straight back, and
       delegates every mode but `chase` to it. The rig, its three other modes,
       its drag and its wheel are untouched. Reported as a deviation. */
    (function installCrewChase() {
      var original = V.createCameraRig;
      V.createCameraRig = function (rigCore, canvas, config) {
        // One page builds one rig, so the wrapper is spent the moment it runs.
        V.createCameraRig = original;
        var rig = original(rigCore, canvas, config);
        var baseUpdate = rig.update;
        var pos = new THREE.Vector3(), look = new THREE.Vector3();
        var started = false;
        rig.update = function (dt, follow) {
          if (rig.state.mode !== "chase" || !follow) {
            started = false;
            return baseUpdate(dt, follow);
          }
          // Far enough back to hold both engines, and never closer than a
          // single-engine chase would be.
          var back = 3.4 + follow.spread * 1.35;
          var height = 1.9 + follow.spread * 0.5;
          var dirX = Math.sin(follow.heading), dirZ = Math.cos(follow.heading);
          if (!started) {
            started = true;
            pos.copy(rigCore.camera.position);
            look.set(follow.x, 0.4, follow.z);
          }
          pos.lerp(new THREE.Vector3(follow.x - dirX * back, height, follow.z - dirZ * back),
                   clamp(dt * 3.0, 0, 1));
          look.lerp(new THREE.Vector3(follow.x + dirX * 0.9, 0.45, follow.z + dirZ * 0.9),
                    clamp(dt * 4.0, 0, 1));
          rigCore.camera.position.copy(pos);
          rigCore.camera.lookAt(look);
        };
        return rig;
      };
    })();

    return {
      steps: STEPS,

      /* Framing scales with the world, because a trace decides how big the
         board is. The constant term is the margin for the props, which do not
         scale with the grid: a tender is the same size on any board, so a
         small board needs proportionally more headroom, not less. */
      camera: {
        board: [-SPAN * 0.58, SPAN * 0.74 + 1.1, SPAN * 1.14 + 1.3],
        top: [0.01, SPAN * 1.32 + 2.0, 0.02]
      },

      update: update
    };
  }

  V.scenes["multiagent_firefighting.v1"] = {
    build: build,
    // The sun carries real intensity and every fire real lumens, so the camera
    // stops down the way a real one would. This one number sets the whole
    // mood; it is tuned, not a knob to remove.
    exposure: 0.315
  };
})(window);
