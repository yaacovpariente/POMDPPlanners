/* SPDX-License-Identifier: MIT
 *
 * Chicheck Invaders scene module.
 *
 * Builds the world from a trace's `payload.world` block and moves it from the
 * trace's recorded state vectors, readings and beliefs. Nothing here is
 * invented: there is no fallback episode, no hand-placed chicken and no
 * synthetic belief. If the player hands this module no trace, it draws nothing
 * and says so.
 *
 * Three things about this environment decide how it is drawn.
 *
 *  1. The playfield is VERTICAL. A column is a place along the ship's rail and
 *     a row is an altitude, so the board is the XY plane and a dive goes down
 *     the screen. Cells are 1 x 1 in scene units, which is what keeps the drawn
 *     camera cone at the slope the observation model actually tests.
 *  2. The gun is HITSCAN. A shot resolves inside the step that fired it, so
 *     there is no projectile to follow and no lead to draw: the beam from the
 *     muzzle to the row it struck IS the event, flashed within that one step. A
 *     shot into an empty column runs the full height and rings nothing.
 *  3. The hidden variable is a set of OCCUPIED CELLS, not a position. So the
 *     belief is a lattice with one box per cell rather than a cloud: the
 *     weighted chance that at least one live chicken stands there, computed
 *     from the run's own particles at the run's own weights.
 *
 * The camera cone and the radar ring are built from the environment's two
 * predicates — |dx| <= slope * dy and dx^2 + dy^2 <= rho^2 — with the slope and
 * the radius read out of the trace, so a drawn footprint cannot drift from the
 * one a reading was drawn against.
 *
 * Out here a chicken's shadow has nothing to land on, so the birds cast none.
 * That is a deliberate omission: shadow casters on the flock cost real frame
 * time under software GL and changed not one pixel.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  function smooth(t) { return t * t * (3 - 2 * t); }

  var COLORS = {
    patrol: 0xF6C854,
    dive: 0xF06054
  };

  // Cap on drawn belief particles per step. A trace carries at most a few
  // hundred, and the lattice is a weighted sum, so this is a guard rather than
  // a budget.
  var MAX_BELIEF_PARTICLES = 4000;

  var CELL = 1.0;
  var DECK_Y = 1.25;      // altitude of row 0, the ship's own row
  var BELIEF_Z = -5.4;    // the belief lattice sits behind the sky

  /* ------------------------------------------------------------- geometry
     Ported from the Tiger viewer, which is where this repository settled the
     "how do you build an animal" question: a chain of spine samples with a
     superellipse section swept along it, stitched into one tube. One surface,
     so a bird shades as one creature rather than as a bag of spheres. */
  function sweepGeometry(samples, radial, upRefIn, capStart, capEnd) {
    var M = samples.length, N = radial;
    var pos = [], uv = [], idx = [];
    var upRef = (upRefIn || new THREE.Vector3(0, 1, 0)).clone().normalize();

    var tangents = [];
    var i, j;
    for (i = 0; i < M; i++) {
      var a = samples[Math.max(0, i - 1)].p;
      var b = samples[Math.min(M - 1, i + 1)].p;
      var t = new THREE.Vector3().subVectors(b, a);
      if (t.lengthSq() < 1e-12) t.set(0, 0, 1);
      tangents.push(t.normalize());
    }
    var lens = [0];
    for (i = 1; i < M; i++) lens.push(lens[i - 1] + samples[i].p.distanceTo(samples[i - 1].p));
    var total = lens[M - 1] || 1;

    var right = new THREE.Vector3(), up = new THREE.Vector3(), v3 = new THREE.Vector3();
    for (i = 0; i < M; i++) {
      var sm = samples[i], T = tangents[i];
      right.crossVectors(upRef, T);
      if (right.lengthSq() < 1e-8) right.set(1, 0, 0);
      right.normalize();
      up.crossVectors(T, right).normalize();
      var e = sm.exp || 2.0, k = 2 / e;
      for (j = 0; j <= N; j++) {
        var ang = (j / N) * Math.PI * 2;
        var c = Math.cos(ang), sn = Math.sin(ang);
        var cc = (c < 0 ? -1 : 1) * Math.pow(Math.abs(c), k);
        var ss = (sn < 0 ? -1 : 1) * Math.pow(Math.abs(sn), k);
        var h = sn >= 0 ? sm.hhTop : sm.hhBot;
        v3.copy(sm.p).addScaledVector(right, sm.hw * cc).addScaledVector(up, h * ss);
        pos.push(v3.x, v3.y, v3.z);
        uv.push(j / N, lens[i] / total);
      }
    }
    for (i = 0; i < M - 1; i++) {
      for (j = 0; j < N; j++) {
        var a2 = i * (N + 1) + j, b2 = a2 + 1, c2 = a2 + (N + 1), d2 = c2 + 1;
        idx.push(a2, b2, d2, a2, d2, c2);
      }
    }
    function cap(ringStart, T2, sgn) {
      var sm2 = samples[sgn > 0 ? M - 1 : 0];
      var centre = sm2.p.clone().addScaledVector(T2, sgn * Math.max(sm2.hw, sm2.hhTop) * 0.55);
      var ci = pos.length / 3;
      pos.push(centre.x, centre.y, centre.z);
      uv.push(0.5, sgn > 0 ? 1 : 0);
      for (var q = 0; q < N; q++) {
        // Wound so the fan faces out; reversed, a cap is a dark crater.
        if (sgn > 0) idx.push(ringStart + q, ringStart + q + 1, ci);
        else idx.push(ringStart + q, ci, ringStart + q + 1);
      }
    }
    if (capEnd) cap((M - 1) * (N + 1), tangents[M - 1], 1);
    if (capStart) cap(0, tangents[0], -1);

    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    /* The ring is duplicated at j = 0 and j = N so the UV can wrap, which
       leaves the seam's two halves with different normals and a visible crease
       down the flank. Averaging them closes it. */
    var nrm = geo.attributes.normal.array;
    for (i = 0; i < M; i++) {
      var ia = (i * (N + 1)) * 3, ib = (i * (N + 1) + N) * 3;
      var nx = nrm[ia] + nrm[ib], ny = nrm[ia + 1] + nrm[ib + 1], nz = nrm[ia + 2] + nrm[ib + 2];
      var len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      nrm[ia] = nrm[ib] = nx / len;
      nrm[ia + 1] = nrm[ib + 1] = ny / len;
      nrm[ia + 2] = nrm[ib + 2] = nz / len;
    }
    geo.attributes.normal.needsUpdate = true;
    return geo;
  }

  // Chain rows along local +X: [x, centre y, half width (z), half up, half down].
  function chainX(rows, z, exp) {
    return rows.map(function (r) {
      return {
        p: new THREE.Vector3(r[0], r[1], z || 0),
        hw: r[2], hhTop: r[3], hhBot: r[4], exp: exp || 2.0
      };
    });
  }
  // Chain rows along local +Y: [y, centre x, half width (x), half front, half back].
  function chainY(rows, z, exp) {
    return rows.map(function (r) {
      return {
        p: new THREE.Vector3(r[1], r[0], z || 0),
        hw: r[2], hhTop: r[3], hhBot: r[4], exp: exp || 2.0
      };
    });
  }
  function chainFree(rows, exp) {
    return rows.map(function (r) {
      return {
        p: new THREE.Vector3(r[0], r[1], r[2]),
        hw: r[3], hhTop: r[4] === undefined ? r[3] : r[4],
        hhBot: r[5] === undefined ? (r[4] === undefined ? r[3] : r[4]) : r[5],
        exp: exp || 2.0
      };
    });
  }

  /* ----------------------------------------------------------- canvas art */

  /* Two dust banks far behind the playfield. They give the black something to
     be black against; the dark lanes matter as much as the lit dust, because
     without them a nebula is a smudge. */
  function nebulaCanvas(seed, r1, g1, b1, r2, g2, b2) {
    var s = 512;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#000"; g.fillRect(0, 0, s, s);
    g.globalCompositeOperation = "lighter";
    for (var i = 0; i < 150; i++) {
      var x = rnd() * s, y = rnd() * s, rad = 30 + rnd() * 170;
      var mix = rnd();
      var cr = Math.round(r1 + (r2 - r1) * mix);
      var cg = Math.round(g1 + (g2 - g1) * mix);
      var cb = Math.round(b1 + (b2 - b1) * mix);
      var grd = g.createRadialGradient(x, y, 0, x, y, rad);
      grd.addColorStop(0, "rgba(" + cr + "," + cg + "," + cb + ",0.085)");
      grd.addColorStop(1, "rgba(" + cr + "," + cg + "," + cb + ",0)");
      g.fillStyle = grd;
      g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fill();
    }
    g.globalCompositeOperation = "destination-out";
    for (var k = 0; k < 40; k++) {
      var lx = rnd() * s, ly = rnd() * s, lr = 20 + rnd() * 90;
      var lg = g.createRadialGradient(lx, ly, 0, lx, ly, lr);
      lg.addColorStop(0, "rgba(0,0,0,0.55)");
      lg.addColorStop(1, "rgba(0,0,0,0)");
      g.fillStyle = lg;
      g.beginPath(); g.arc(lx, ly, lr, 0, Math.PI * 2); g.fill();
    }
    return new THREE.CanvasTexture(cv);
  }

  /* A gas giant, banded and sheared along its latitudes, with one storm —
     a featureless banded ball reads as a beach ball. */
  function planetCanvas() {
    var w = 1024, h = 512;
    var cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    var g = cv.getContext("2d");
    var rnd = mulberry(30219);
    var bands = [
      [0.00, "#6B5A44"], [0.10, "#8A7454"], [0.18, "#5C4C39"], [0.27, "#9C8460"],
      [0.36, "#7A6449"], [0.44, "#AE9068"], [0.52, "#6E5A42"], [0.61, "#93794F"],
      [0.70, "#5A4A36"], [0.79, "#856E4E"], [0.88, "#5F513C"], [1.00, "#4A3F30"]
    ];
    var grd = g.createLinearGradient(0, 0, 0, h);
    bands.forEach(function (b) { grd.addColorStop(b[0], b[1]); });
    g.fillStyle = grd; g.fillRect(0, 0, w, h);
    for (var i = 0; i < 1800; i++) {
      var y = rnd() * h, x = rnd() * w;
      var len = 20 + rnd() * 190, thick = 1 + rnd() * 7;
      g.fillStyle = rnd() > 0.5 ? "rgba(226,206,174,0.045)" : "rgba(38,30,22,0.075)";
      g.beginPath(); g.ellipse(x, y, len, thick, 0, 0, Math.PI * 2); g.fill();
    }
    var sx = w * 0.62, sy = h * 0.61;
    for (var s = 0; s < 7; s++) {
      g.fillStyle = "rgba(174,96,64," + (0.16 - s * 0.018).toFixed(3) + ")";
      g.beginPath(); g.ellipse(sx, sy, 92 - s * 8, 40 - s * 3.6, 0.1, 0, Math.PI * 2); g.fill();
    }
    var tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }

  /* The ship's skin: panel seams, rivet lines, repainted bays, exhaust scorch
     and squadron decals. Desaturated, with the one saturated element — the
     hazard chevrons — kept small. */
  function hullCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(77123);

    g.fillStyle = "#7C8189"; g.fillRect(0, 0, s, s);

    // No two panels ever came off the line the same shade.
    [[7, 15], [19, 10], [64, 6]].forEach(function (oct) {
      var cells = oct[0], strength = oct[1];
      var small = document.createElement("canvas");
      small.width = small.height = cells;
      var sg = small.getContext("2d");
      for (var a = 0; a < cells; a++) {
        for (var b = 0; b < cells; b++) {
          var v = (rnd() - 0.5) * 2 * strength;
          sg.fillStyle = "rgba(" + (v > 0 ? "224,228,236," : "18,20,26,") +
            (Math.abs(v) / (v > 0 ? 120 : 70)).toFixed(3) + ")";
          sg.fillRect(a, b, 1, 1);
        }
      }
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
      g.drawImage(small, 0, 0, s, s);
    });

    var xs = [], ys = [];
    for (var x = 0; x < s; x += 70 + rnd() * 90) xs.push(Math.round(x));
    for (var y = 0; y < s; y += 60 + rnd() * 110) ys.push(Math.round(y));
    g.lineWidth = 2;
    xs.forEach(function (px) {
      g.strokeStyle = "rgba(24,27,33,0.55)";
      g.beginPath(); g.moveTo(px, 0); g.lineTo(px, s); g.stroke();
      g.strokeStyle = "rgba(216,222,232,0.13)";
      g.beginPath(); g.moveTo(px + 2, 0); g.lineTo(px + 2, s); g.stroke();
      for (var ry = 6; ry < s; ry += 26) {
        g.fillStyle = "rgba(20,22,28,0.42)";
        g.beginPath(); g.arc(px - 7, ry, 1.7, 0, Math.PI * 2); g.fill();
      }
    });
    ys.forEach(function (py) {
      g.strokeStyle = "rgba(24,27,33,0.45)";
      g.beginPath(); g.moveTo(0, py); g.lineTo(s, py); g.stroke();
      g.strokeStyle = "rgba(216,222,232,0.10)";
      g.beginPath(); g.moveTo(0, py + 2); g.lineTo(s, py + 2); g.stroke();
    });

    for (var p = 0; p < 16; p++) {
      var bx = xs[Math.floor(rnd() * xs.length)] || 0;
      var by = ys[Math.floor(rnd() * ys.length)] || 0;
      g.fillStyle = rnd() > 0.5 ? "rgba(122,128,138,0.30)" : "rgba(92,88,80,0.26)";
      g.fillRect(bx, by, 60 + rnd() * 120, 50 + rnd() * 90);
    }

    var burn = g.createLinearGradient(0, s, 0, s * 0.55);
    burn.addColorStop(0, "rgba(16,14,14,0.62)");
    burn.addColorStop(1, "rgba(16,14,14,0)");
    g.fillStyle = burn; g.fillRect(0, s * 0.55, s, s * 0.45);
    for (var st = 0; st < 260; st++) {
      var sx2 = rnd() * s, sy2 = s * (0.55 + rnd() * 0.45), sl = 8 + rnd() * 70;
      g.strokeStyle = "rgba(10,9,9,0.12)";
      g.lineWidth = 1 + rnd() * 3;
      g.beginPath(); g.moveTo(sx2, sy2); g.lineTo(sx2 + (rnd() - 0.5) * 6, sy2 + sl); g.stroke();
    }
    for (var d = 0; d < 260; d++) {
      var dx2 = rnd() * s, dy2 = rnd() * s, dr = 1 + rnd() * 4;
      g.fillStyle = "rgba(12,13,17,0.38)";
      g.beginPath(); g.arc(dx2, dy2, dr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(228,232,240,0.10)";
      g.beginPath(); g.arc(dx2 - dr * 0.3, dy2 - dr * 0.3, dr * 0.7, 0, Math.PI * 2); g.fill();
    }

    g.save();
    g.translate(s * 0.30, s * 0.30);
    for (var ch = 0; ch < 5; ch++) {
      g.fillStyle = "rgba(196,132,38,0.78)";
      g.beginPath();
      g.moveTo(ch * 26, 0); g.lineTo(ch * 26 + 13, 22); g.lineTo(ch * 26 + 26, 0);
      g.lineTo(ch * 26 + 18, 0); g.lineTo(ch * 26 + 13, 10); g.lineTo(ch * 26 + 8, 0);
      g.closePath(); g.fill();
    }
    g.restore();
    g.fillStyle = "rgba(226,232,242,0.52)";
    g.font = "bold 62px monospace";
    g.fillText("CI-07", s * 0.56, s * 0.24);
    g.font = "bold 22px monospace";
    g.fillStyle = "rgba(226,232,242,0.30)";
    g.fillText("NO STEP", s * 0.12, s * 0.68);
    g.fillText("RESCUE", s * 0.70, s * 0.47);
    g.strokeStyle = "rgba(226,232,242,0.22)";
    g.lineWidth = 3;
    g.strokeRect(s * 0.685, s * 0.42, 150, 40);
    return cv;
  }

  /**
   * Build the Chicheck Invaders world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind chicheck_invaders.v1.
   * @returns {Object} The scene module the player drives.
   */
  // The world, the flock, the sensors, the belief and the playback all live in
  // one closure because they all read the same trace; splitting them would mean
  // threading a dozen values through as arguments.
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var SL = payload.layout.state;
    var OL = payload.layout.observation;
    var scene = core.scene;
    var renderer = core.renderer;

    var COLS = world.num_columns;
    var ROWS = world.num_rows;
    var NCH = world.num_chickens;
    var SLOPE = world.camera_slope;
    var RADIUS = world.radar_radius;

    function cx(col) { return (col - (COLS - 1) / 2) * CELL; }
    function cy(row) { return DECK_Y + row * CELL; }

    var reduceMotion = global.matchMedia
      ? global.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;

    /* ------------------------------------------------------- the episode
       A state is the raw float vector the run held; the layout that indexes it
       travels in the trace, so this decoder and the belief's decoder are the
       same one. A belief particle IS a state, which is why there is only one. */
    function decodeState(vector) {
      var chickens = [];
      for (var k = 0; k < NCH; k++) {
        var base = SL.ship_width + k * SL.chicken_width;
        chickens.push({
          c: vector[base + SL.chicken_column],
          r: vector[base + SL.chicken_row],
          dir: vector[base + SL.chicken_direction],
          diving: vector[base + SL.chicken_mode] === SL.mode_dive,
          alive: vector[base + SL.chicken_alive] > 0
        });
      }
      return {
        step: vector[SL.step],
        ship: vector[SL.ship_column],
        cooldown: vector[SL.cooldown],
        shipHit: vector[SL.ship_hit] > 0,
        chickens: chickens
      };
    }

    var FRAMES = payload.states.map(decodeState);
    var N_STEPS = FRAMES.length;

    /* The successor of each recorded step. For every step but the last that is
       simply the next recorded state; for the last it is `next_states`, which
       is the only place the true final state lives when the episode stopped on
       its step budget and no terminal bookkeeping record was written. */
    var NEXTS = [];
    for (var ni = 0; ni < N_STEPS; ni++) {
      if (ni + 1 < N_STEPS) NEXTS.push(FRAMES[ni + 1]);
      else if (payload.next_states[ni]) NEXTS.push(decodeState(payload.next_states[ni]));
      else NEXTS.push(FRAMES[ni]);
    }

    /* The observation drawn on step i describes the SUCCESSOR of step i, so the
       reading a frame showing state i can honestly light up is record i-1's —
       which is also the one the belief on this frame has already absorbed. */
    function readingFor(index) {
      var raw = index > 0 ? payload.observations[index - 1] : null;
      if (!raw || raw.length === undefined) return null;
      var slots = [];
      for (var k = 0; k < NCH; k++) {
        var base = OL.ship_width + k * OL.chicken_width;
        slots.push({
          camera: raw[base + OL.camera_reported] > 0,
          cameraOffset: raw[base + OL.camera_offset],
          radar: raw[base + OL.radar_reported] > 0,
          radarRows: raw[base + OL.radar_rows],
          radarDrop: raw[base + OL.radar_drop]
        });
      }
      return { shipColumn: raw[OL.ship_column], chickens: slots };
    }
    var READINGS = [];
    for (var ri = 0; ri < N_STEPS; ri++) READINGS.push(readingFor(ri));

    /* A chicken that drives past row 0 outside the ship's column pulls up: back
       to patrolling on the top row, same column. In the trace that is a row
       jumping from 1 to the top in one step, and interpolating it would draw a
       climb the environment never made. Detected once, up front. */
    var PULLUP = [], DIVESTART = [];
    for (var si = 0; si < N_STEPS; si++) {
      var pull = [], fresh = [];
      for (var k2 = 0; k2 < NCH; k2++) {
        var cur = FRAMES[si].chickens[k2];
        var nxt = NEXTS[si].chickens[k2];
        var prev = si > 0 ? FRAMES[si - 1].chickens[k2] : null;
        pull.push(!!(cur.alive && nxt.alive && nxt.r > cur.r + 1));
        // The first step of a dive, so the wind-up plays once when the bird
        // commits rather than at the top of every step it spends stooping.
        fresh.push(!!(cur.diving && (!prev || !prev.diving || !prev.alive)));
      }
      PULLUP.push(pull);
      DIVESTART.push(fresh);
    }

    // camera_sees / radar_sees, ported so the drawn footprints cannot drift
    // from the predicates the observation model tests against.
    function cameraSees(dx, dy) { return Math.abs(dx) <= SLOPE * dy; }
    function radarSees(dx, dy) { return dx * dx + dy * dy <= RADIUS * RADIUS; }

    /* ------------------------------------------------------------- scene */
    scene.background = new THREE.Color(0x020306).convertSRGBToLinear();

    /* Space, so the light budget is unusual: one hard key from a distant star
       with no atmosphere to soften it, a cool bounce off the planet's day side,
       and almost nothing else. Everything not facing the star goes nearly
       black, and the only things that fill that in are practicals — the ship's
       engines and each chicken's own pack. */
    scene.add(new THREE.HemisphereLight(0x2A3550, 0x0A0A12, 1.9));

    var key = new THREE.DirectionalLight(0xFFF2DC, 54.0);
    key.position.set(-25.8, 11.4, 18.2);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    /* Tight orthographic frustum: the only useful shadow out here is the ship
       shadowing itself, which is what makes the hull panels and the wing roots
       read. There is no ground, and nothing a chicken could cast onto. */
    var SH = 1.6;
    key.shadow.camera.left = -SH; key.shadow.camera.right = SH;
    key.shadow.camera.top = SH; key.shadow.camera.bottom = -SH;
    key.shadow.camera.near = 30; key.shadow.camera.far = 52;
    key.shadow.bias = -0.0006;
    key.shadow.normalBias = 0.035;   // not a big negative bias: that is acne
    key.target.position.set(0, DECK_Y, 0);
    scene.add(key); scene.add(key.target);

    // Planet-shine: cool, wide and weak. It keeps the shadow side from being a
    // silhouette without pretending there is a fill light in vacuum.
    var planetBounce = new THREE.DirectionalLight(0x6E92D8, 8.0);
    planetBounce.position.set(16, -10, -22);
    scene.add(planetBounce);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);
    var glowTex = V.radialTexture(0.9, 0.3);
    var starTex = V.radialTexture(1.0, 0.28);

    /* ------------------------------------------------------- the starfield
       Three layers at different distances, varied in brightness AND colour
       temperature. A uniform sprinkle of identical white dots is the clearest
       tell of a fake sky: real stars vary over orders of magnitude. */
    function starLayer(count, radius, size, seed, dim) {
      var rnd = mulberry(seed);
      var pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
      var c = new THREE.Color();
      for (var i = 0; i < count; i++) {
        var u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, r = Math.sqrt(1 - u * u);
        pos[i * 3] = Math.cos(th) * r * radius;
        pos[i * 3 + 1] = u * radius;
        pos[i * 3 + 2] = Math.sin(th) * r * radius;
        var mag = Math.pow(rnd(), 3.2);
        var t = rnd();
        if (t < 0.12) c.setRGB(0.62, 0.72, 1.00);        // hot blue-white
        else if (t < 0.55) c.setRGB(1.00, 0.98, 0.94);   // white
        else if (t < 0.84) c.setRGB(1.00, 0.90, 0.72);   // yellow
        else c.setRGB(1.00, 0.72, 0.50);                 // cool orange
        var b = (0.12 + mag * 2.6) * dim;
        col[i * 3] = c.r * b; col[i * 3 + 1] = c.g * b; col[i * 3 + 2] = c.b * b;
      }
      var g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(col, 3));
      /* sizeAttenuation OFF. A star is a point source at effectively infinite
         range, so its angular size does not depend on which shell of the fake
         sky it sits on — and with attenuation on, a dot 200 units out shrinks
         below a pixel and the whole sky vanishes at this exposure. */
      var points = new THREE.Points(g, new THREE.PointsMaterial({
        size: size, map: starTex, vertexColors: true, transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: false
      }));
      points.frustumCulled = false;
      scene.add(points);
    }
    starLayer(2000, 120, 3.4, 8123, 3.2);
    starLayer(2800, 200, 2.4, 4471, 2.0);
    starLayer(1400, 300, 1.7, 9907, 1.3);

    [[nebulaCanvas(551, 90, 40, 140, 40, 90, 190), -62, 26, -108, 150, 0.30],
     [nebulaCanvas(772, 170, 60, 60, 190, 120, 60), 78, -6, -126, 175, 0.20]
    ].forEach(function (nb) {
      var m = new THREE.Mesh(new THREE.PlaneGeometry(nb[4], nb[4] * 0.72),
        new THREE.MeshBasicMaterial({
          map: nb[0], transparent: true, opacity: nb[5],
          blending: THREE.AdditiveBlending, depthWrite: false
        }));
      m.position.set(nb[1], nb[2], nb[3]);
      m.frustumCulled = false;
      scene.add(m);
    });

    /* A gas giant off to one side, lit by the same key as everything else, so
       it carries a real terminator rather than a painted crescent. It is here
       for scale: without something enormous in frame, a ship and a few birds in
       a void have no size at all. */
    var PLANET_R = 26, PLANET_POS = new THREE.Vector3(-44, -30, -96);
    var planet = new THREE.Mesh(
      new THREE.SphereGeometry(PLANET_R, 96, 64),
      new THREE.MeshStandardMaterial({
        map: planetCanvas(), color: 0x6A6257, roughness: 0.98, metalness: 0.0,
        envMapIntensity: 0.03
      })
    );
    planet.position.copy(PLANET_POS);
    planet.rotation.z = 0.18;
    scene.add(planet);

    /* The atmosphere as a real limb: a back-faced shell whose opacity is a
       fresnel term times how much of that point faces the star, so the glow is
       brightest on the lit edge and dies through the terminator. A uniform ring
       would light the night side too. */
    var atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(PLANET_R * 1.035, 96, 64),
      new THREE.ShaderMaterial({
        uniforms: {
          glow: { value: new THREE.Color(0.30, 0.52, 1.05) },
          sunDir: { value: key.position.clone().normalize() }
        },
        vertexShader: [
          "varying vec3 vN; varying vec3 vW;",
          "void main() {",
          "  vN = normalize(mat3(modelMatrix) * normal);",
          "  vec4 wp = modelMatrix * vec4(position, 1.0);",
          "  vW = wp.xyz;",
          "  gl_Position = projectionMatrix * viewMatrix * wp;",
          "}"
        ].join("\n"),
        fragmentShader: [
          "uniform vec3 glow; uniform vec3 sunDir;",
          "varying vec3 vN; varying vec3 vW;",
          "void main() {",
          "  vec3 V = normalize(cameraPosition - vW);",
          "  float rim = pow(1.0 - abs(dot(normalize(vN), V)), 3.2);",
          "  float lit = clamp(dot(normalize(vN), normalize(sunDir)) * 0.5 + 0.5, 0.0, 1.0);",
          "  gl_FragColor = vec4(glow * rim * pow(lit, 2.0) * 1.9, 1.0);",
          "}"
        ].join("\n"),
        transparent: true, blending: THREE.AdditiveBlending,
        depthWrite: false, side: THREE.BackSide
      })
    );
    atmosphere.position.copy(PLANET_POS);
    scene.add(atmosphere);

    var moon = new THREE.Mesh(new THREE.SphereGeometry(2.1, 40, 28),
      new THREE.MeshStandardMaterial({ color: 0x5A564E, roughness: 1.0, metalness: 0.0 }));
    moon.position.set(30, 16, -78);
    scene.add(moon);

    /* The environment map: the same sky, so the ship's metal mirrors stars and
       nebula instead of a grey void. PBR metal with nothing to reflect reads as
       plastic, and out here there is no ground bounce to save it. This is built
       here rather than taken from the core's night sky, whose horizon gradient
       belongs to a scene that has a ground. */
    (function buildEnvironment() {
      var cv = document.createElement("canvas");
      cv.width = 256; cv.height = 128;
      var g = cv.getContext("2d");
      g.fillStyle = "#03040A"; g.fillRect(0, 0, 256, 128);
      var rnd = mulberry(1201);
      for (var n = 0; n < 60; n++) {
        var x = rnd() * 256, y = 50 + rnd() * 70, r = 12 + rnd() * 46;
        var grd = g.createRadialGradient(x, y, 0, x, y, r);
        var warm = rnd() > 0.5;
        grd.addColorStop(0, warm ? "rgba(90,40,70,0.16)" : "rgba(40,60,120,0.16)");
        grd.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = grd; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
      }
      for (var i = 0; i < 420; i++) {
        g.fillStyle = "rgba(215,225,245," + (0.25 + rnd() * 0.75).toFixed(2) + ")";
        g.fillRect(rnd() * 256, rnd() * 128, 1, 1);
      }
      // The star itself, so a highlight has somewhere to come from.
      var sg = g.createRadialGradient(28, 40, 0, 28, 40, 26);
      sg.addColorStop(0, "rgba(255,250,235,1)");
      sg.addColorStop(1, "rgba(255,240,210,0)");
      g.fillStyle = sg; g.beginPath(); g.arc(28, 40, 26, 0, Math.PI * 2); g.fill();

      var tex = new THREE.CanvasTexture(cv);
      tex.mapping = THREE.EquirectangularReflectionMapping;
      tex.encoding = THREE.sRGBEncoding;
      var pmrem = new THREE.PMREMGenerator(renderer);
      pmrem.compileEquirectangularShader();
      scene.environment = pmrem.fromEquirectangular(tex).texture;
      pmrem.dispose(); tex.dispose();
    })();

    /* --------------------------------------------------------- the ship
       A single-seat interceptor, nose up the playfield. Nose-up is not styling:
       the gun is hitscan straight up the ship's own column, so the one
       direction the craft ever shoots is scene +Y, and a fighter points where it
       shoots. Hence the cannon in the nose, the engines at the tail firing down,
       and the canopy facing the camera. */
    var hullCv = hullCanvas();
    var hullNormal = V.normalMapFrom(renderer, hullCv, 2.0);
    var hullTex = new THREE.CanvasTexture(hullCv);
    hullTex.encoding = THREE.sRGBEncoding;
    hullTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    hullTex.wrapS = hullTex.wrapT = THREE.RepeatWrapping;
    hullNormal.wrapS = hullNormal.wrapT = THREE.RepeatWrapping;
    // The sweep's UVs run 0..1 over a whole piece, so one tile stretched the
    // panel grid over the entire fuselage. Repeat it down to plate size.
    hullTex.repeat.set(3, 3);
    hullNormal.repeat.set(3, 3);

    var hullMat = new THREE.MeshStandardMaterial({
      map: hullTex, normalMap: hullNormal,
      normalScale: new THREE.Vector2(0.9, 0.9),
      color: 0x9AA0A8, roughness: 0.52, metalness: 0.72, envMapIntensity: 1.0
    });
    var hullDark = new THREE.MeshStandardMaterial({
      map: hullTex, normalMap: hullNormal,
      color: 0x4A5056, roughness: 0.58, metalness: 0.70, envMapIntensity: 0.9
    });
    var gunMetal = new THREE.MeshStandardMaterial({
      color: 0x3A4045, roughness: 0.34, metalness: 0.94, envMapIntensity: 1.2
    });
    var trimMat = new THREE.MeshStandardMaterial({
      color: 0xC4842A, roughness: 0.66, metalness: 0.28, envMapIntensity: 0.8
    });

    var ship = new THREE.Group();
    ship.scale.setScalar(0.88);        // the whole craft lives inside one cell
    scene.add(ship);
    var hullG = new THREE.Group();
    ship.add(hullG);

    // Fuselage, tail to nose along +Y. The belly is shallower than the spine,
    // so the craft has a top and a bottom.
    var fuselage = new THREE.Mesh(sweepGeometry(chainY([
      [-0.40, 0, 0.055, 0.045, 0.040],
      [-0.32, 0, 0.135, 0.105, 0.085],
      [-0.20, 0, 0.180, 0.135, 0.100],
      [-0.06, 0, 0.190, 0.140, 0.100],
      [0.08, 0, 0.170, 0.125, 0.090],
      [0.20, 0, 0.130, 0.098, 0.072],
      [0.30, 0, 0.092, 0.070, 0.052],
      [0.38, 0, 0.055, 0.044, 0.034],
      [0.44, 0, 0.026, 0.022, 0.018]
    ], 0, 2.7), 22, new THREE.Vector3(0, 0, 1), true, true), hullMat);
    fuselage.castShadow = true; fuselage.receiveShadow = true;
    hullG.add(fuselage);

    // Wings: swept surfaces running outboard and aft, thin in Z, so they read
    // as aerofoil-section plates rather than boxes.
    [1, -1].forEach(function (sgn) {
      var wing = new THREE.Mesh(sweepGeometry(chainFree([
        [sgn * 0.14, -0.02, 0, 0.115, 0.042, 0.034],
        [sgn * 0.30, -0.06, 0, 0.130, 0.034, 0.026],
        [sgn * 0.46, -0.13, 0, 0.115, 0.024, 0.019],
        [sgn * 0.58, -0.22, 0, 0.082, 0.016, 0.013],
        [sgn * 0.64, -0.30, 0, 0.040, 0.010, 0.008]
      ], 2.4), 16, new THREE.Vector3(0, 0, 1), true, true), hullMat);
      wing.castShadow = true; wing.receiveShadow = true;
      hullG.add(wing);

      var pod = new THREE.Mesh(new THREE.CylinderGeometry(0.030, 0.034, 0.15, 12), gunMetal);
      pod.position.set(sgn * 0.62, -0.27, 0);
      pod.rotation.z = -sgn * 0.42;
      pod.castShadow = true;
      hullG.add(pod);
      // The running lights are the only full-saturation ship colour.
      var navLight = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 8),
        new THREE.MeshBasicMaterial({
          color: sgn > 0 ? new THREE.Color(0.5, 2.2, 3.4) : new THREE.Color(3.2, 0.6, 0.5)
        }));
      navLight.position.set(sgn * 0.645, -0.32, 0);
      hullG.add(navLight);

      var chev = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.022, 0.05), trimMat);
      chev.position.set(sgn * 0.34, 0.01, 0.028);
      chev.rotation.z = -sgn * 0.30;
      hullG.add(chev);
    });

    // Canopy: transmissive glass over a dark interior, lit from inside so you
    // can see there is a place for a pilot.
    var cockpitWell = new THREE.Mesh(new THREE.SphereGeometry(0.098, 18, 12),
      new THREE.MeshStandardMaterial({ color: 0x14181E, roughness: 0.9, metalness: 0.2 }));
    cockpitWell.scale.set(1.0, 1.45, 0.8);
    cockpitWell.position.set(0, 0.06, 0.09);
    hullG.add(cockpitWell);
    var seat = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.105, 0.035),
      new THREE.MeshStandardMaterial({ color: 0x2C3138, roughness: 0.85, metalness: 0.15 }));
    seat.position.set(0, 0.02, 0.085);
    hullG.add(seat);
    var instrument = new THREE.Mesh(new THREE.PlaneGeometry(0.085, 0.045),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.18, 1.15, 0.95), transparent: true, opacity: 0.9
      }));
    instrument.position.set(0, 0.125, 0.115);
    instrument.rotation.x = -0.75;
    hullG.add(instrument);
    var cockpitLight = new THREE.PointLight(0x62E8D2, 1, 0.9, 2);
    cockpitLight.power = 2.2;
    cockpitLight.position.set(0, 0.08, 0.13);
    hullG.add(cockpitLight);

    var canopy = new THREE.Mesh(sweepGeometry(chainY([
      [-0.075, 0, 0.060, 0.075, 0.005],
      [0.000, 0, 0.086, 0.112, 0.005],
      [0.075, 0, 0.078, 0.104, 0.005],
      [0.140, 0, 0.050, 0.066, 0.005],
      [0.180, 0, 0.022, 0.030, 0.005]
    ], 0.06, 2.3), 20, new THREE.Vector3(0, 0, 1), true, true),
      new THREE.MeshStandardMaterial({
        color: 0x93BBD2, roughness: 0.07, metalness: 0.24,
        transparent: true, opacity: 0.30, envMapIntensity: 1.6,
        side: THREE.DoubleSide
      }));
    canopy.position.y = 0.05;
    canopy.castShadow = true;
    hullG.add(canopy);
    // Canopy ribs: what tells you it is glazing and not a blob.
    [-0.075, 0.06, 0.155].forEach(function (fy) {
      var rib = new THREE.Mesh(new THREE.TorusGeometry(0.072, 0.007, 6, 18, Math.PI), gunMetal);
      rib.position.set(0, 0.05 + fy, 0.06);
      rib.rotation.y = Math.PI / 2;
      rib.rotation.x = Math.PI / 2;
      hullG.add(rib);
    });

    // Dorsal spine and tail fins, so the silhouette is not a lozenge.
    var spine = new THREE.Mesh(sweepGeometry(chainY([
      [-0.30, 0, 0.040, 0.030, 0.010],
      [-0.16, 0, 0.052, 0.062, 0.010],
      [0.00, 0, 0.044, 0.050, 0.010],
      [0.14, 0, 0.028, 0.030, 0.010]
    ], -0.12, 2.6), 14, new THREE.Vector3(0, 0, 1), true, true), hullDark);
    spine.castShadow = true;
    hullG.add(spine);
    [1, -1].forEach(function (sgn) {
      var fin = new THREE.Mesh(sweepGeometry(chainFree([
        [sgn * 0.10, -0.28, -0.04, 0.050, 0.020, 0.016],
        [sgn * 0.16, -0.34, -0.14, 0.038, 0.014, 0.011],
        [sgn * 0.19, -0.40, -0.24, 0.020, 0.008, 0.006]
      ], 2.2), 12, new THREE.Vector3(0, 1, 0), true, true), hullDark);
      fin.castShadow = true;
      hullG.add(fin);
    });

    // The cannon, in the nose: the ship's local +Y is the one direction the
    // shot ever travels.
    var gun = new THREE.Group();
    gun.position.set(0, 0.30, 0.0);
    hullG.add(gun);
    var gunHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.070, 0.12, 12), hullDark);
    gunHousing.position.y = 0.02; gunHousing.castShadow = true; gun.add(gunHousing);
    var barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.032, 0.22, 14), gunMetal);
    barrel.position.y = 0.16; barrel.castShadow = true; gun.add(barrel);
    var brake = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.030, 0.06, 14), gunMetal);
    brake.position.y = 0.28; brake.castShadow = true; gun.add(brake);
    [0, 1, 2].forEach(function (i) {
      var ring = new THREE.Mesh(new THREE.TorusGeometry(0.031, 0.006, 5, 14), gunMetal);
      ring.position.y = 0.09 + i * 0.045; ring.rotation.x = Math.PI / 2; gun.add(ring);
    });
    // Coolant vanes: heat has to go somewhere and in vacuum it only radiates.
    [1, -1].forEach(function (sgn) {
      var vane = new THREE.Mesh(new THREE.BoxGeometry(0.010, 0.10, 0.055), hullDark);
      vane.position.set(sgn * 0.062, 0.04, 0);
      vane.castShadow = true;
      gun.add(vane);
    });

    /* Local Y of the muzzle inside hullG. It has to stay under one cell: a kill
       on row 1 draws a beam from here to cy(1), and a muzzle poking into row 1
       would leave nothing to draw. */
    var MUZZLE_Y = 0.30 + 0.31;

    var muzzleFlash = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: 0xDFFFF0, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    muzzleFlash.scale.set(0.7, 0.7, 1);
    muzzleFlash.position.y = MUZZLE_Y + 0.04;
    hullG.add(muzzleFlash);
    var muzzleLight = new THREE.PointLight(0xAEFFD8, 0, 7, 2);
    muzzleLight.position.y = MUZZLE_Y;
    hullG.add(muzzleLight);

    /* The two sensors, as hardware on the hull. The camera pod looks up the
       playfield into its own cone; the radar dish spins. Saturated colour is at
       the lens only, and each brightens on a step its sensor actually
       reported — which is the trace's reading, not a decoration. */
    var camPod = new THREE.Group();
    camPod.position.set(-0.20, 0.13, 0.10);
    hullG.add(camPod);
    var podBody = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.046, 0.085, 12), hullDark);
    podBody.castShadow = true; camPod.add(podBody);
    var podLens = new THREE.Mesh(new THREE.CylinderGeometry(0.030, 0.036, 0.026, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.30, 0.66, 1.55) }));
    podLens.position.y = 0.054; camPod.add(podLens);
    var podHood = new THREE.Mesh(
      new THREE.CylinderGeometry(0.043, 0.038, 0.032, 12, 1, true), gunMetal);
    podHood.position.y = 0.062; camPod.add(podHood);

    var dish = new THREE.Group();
    dish.position.set(0.20, 0.10, -0.10);
    hullG.add(dish);
    var dishMast = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.016, 0.085, 8), gunMetal);
    dishMast.position.y = 0.042; dish.add(dishMast);
    var dishHead = new THREE.Group();
    dishHead.position.y = 0.10;
    dish.add(dishHead);
    var cup = new THREE.Mesh(
      new THREE.SphereGeometry(0.055, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2.4),
      new THREE.MeshStandardMaterial({
        color: 0x6A7178, roughness: 0.34, metalness: 0.88,
        envMapIntensity: 1.2, side: THREE.DoubleSide
      }));
    cup.rotation.x = Math.PI * 0.60;
    dishHead.add(cup);
    var feed = new THREE.Mesh(new THREE.SphereGeometry(0.011, 8, 6),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.35, 1.5, 1.0) }));
    feed.position.set(0, 0.018, 0.034); dishHead.add(feed);

    // Main engines: two nozzles at the tail, firing -Y, carrying a real light,
    // so the ship lights its own underside.
    var thrusters = [];
    [-0.155, 0.155].forEach(function (dx) {
      var housing = new THREE.Mesh(new THREE.CylinderGeometry(0.072, 0.086, 0.16, 16), hullDark);
      housing.position.set(dx, -0.30, 0); housing.castShadow = true; hullG.add(housing);
      var bell = new THREE.Mesh(
        new THREE.CylinderGeometry(0.086, 0.062, 0.10, 16, 1, true), gunMetal);
      bell.position.set(dx, -0.41, 0); bell.castShadow = true; hullG.add(bell);
      var ring2 = new THREE.Mesh(new THREE.TorusGeometry(0.078, 0.010, 6, 18), gunMetal);
      ring2.position.set(dx, -0.36, 0); ring2.rotation.x = Math.PI / 2; hullG.add(ring2);

      var plume = new THREE.Mesh(new THREE.ConeGeometry(0.058, 0.34, 14),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(1.1, 2.4, 3.6), transparent: true,
          opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false
        }));
      plume.position.set(dx, -0.60, 0); plume.rotation.x = Math.PI; hullG.add(plume);
      var wash = new THREE.Sprite(new THREE.SpriteMaterial({
        map: poolTex, color: 0x64C8FF, transparent: true, opacity: 0.35,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      wash.scale.set(0.34, 0.60, 1);
      wash.position.set(dx, -0.62, 0);
      hullG.add(wash);

      var light = new THREE.PointLight(0x58C4FF, 1, 3.4, 2);
      light.power = 42; light.position.set(dx, -0.52, 0); hullG.add(light);
      thrusters.push({ core: plume, light: light, wash: wash });
    });

    /* Manoeuvring thrusters, venting on the side that pushes the ship the way
       it is going. A craft that slides sideways with no reaction mass leaving
       it is exactly what reads as a sprite on a board. */
    var rcs = [];
    [[0.20, 0.20, 1], [-0.20, 0.20, -1], [0.22, -0.16, 1], [-0.22, -0.16, -1]]
      .forEach(function (R) {
        var nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.020, 0.040, 8), gunMetal);
        nozzle.position.set(R[0], R[1], 0);
        nozzle.rotation.z = -R[2] * Math.PI / 2;
        hullG.add(nozzle);
        var puff = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTex, color: 0xCFE6FF, transparent: true, opacity: 0,
          blending: THREE.AdditiveBlending, depthWrite: false
        }));
        puff.scale.set(0.16, 0.16, 1);
        puff.position.set(R[0] + R[2] * 0.05, R[1], 0);
        hullG.add(puff);
        rcs.push({ puff: puff, side: R[2] });
      });

    /* ---------------------------------------------------------- the flock
       These are the invaders, and a chicken silhouette is unmistakable, so the
       build spends its budget on getting that silhouette right: a deep rounded
       body, a real neck carrying a distinct head, a beak, a comb and a wattle,
       a fanned tail, wings. Body and neck are ONE swept surface, so a bird
       shades as one creature instead of a stack of primitives.

       They are in space, so they wear the genre's own answer: a bubble helmet
       and a thruster pack. That is not only a joke — the pack is how a chicken
       dives, and both pieces put a lit element on every bird, which is what
       makes them readable against a black sky under one hard key.

       Local +X is the nose. The body yaws to put +X on the direction of travel
       and pitches nose-down for a dive. */
    var featherMat = new THREE.MeshStandardMaterial({
      color: 0x8E8474, roughness: 0.92, metalness: 0.0, envMapIntensity: 0.10
    });
    var featherDark = new THREE.MeshStandardMaterial({
      color: 0x5E564A, roughness: 0.94, metalness: 0.0, envMapIntensity: 0.09
    });
    var combMat = new THREE.MeshStandardMaterial({
      color: 0xC0342A, roughness: 0.62, metalness: 0.0, envMapIntensity: 0.22
    });
    var beakMat = new THREE.MeshStandardMaterial({
      color: 0xD9A03C, roughness: 0.55, metalness: 0.05, envMapIntensity: 0.25
    });
    var suitMat = new THREE.MeshStandardMaterial({
      color: 0x55606E, roughness: 0.48, metalness: 0.62, envMapIntensity: 0.9
    });
    var glassMat = new THREE.MeshStandardMaterial({
      color: 0x9FBACC, roughness: 0.08, metalness: 0.12,
      transparent: true, opacity: 0.20, envMapIntensity: 1.2, side: THREE.DoubleSide
    });

    function buildChicken(seed) {
      var rnd = mulberry(seed);
      var root = new THREE.Group();
      scene.add(root);
      /* Every bird owns its materials. Shared, the fade applied to the one the
         shot took reached all of them at once. */
      var feather = featherMat.clone(), featherD = featherDark.clone();
      var comb = combMat.clone(), beakM = beakMat.clone(), suit = suitMat.clone();
      var glass = glassMat.clone();
      // A slightly different plumage each: no flock is a set of identical birds.
      var tint = 0.86 + rnd() * 0.28;
      feather.color.multiplyScalar(tint);
      featherD.color.multiplyScalar(tint);
      var fadeMats = [feather, featherD, comb, beakM, suit, glass];

      /* `body` carries the yaw and the dive pitch. Everything visible hangs off
         `rollG` inside it, so a bank into the attack line rotates about the
         bird's own nose axis and leaves the world forward vector alone. */
      var body = new THREE.Group();
      root.add(body);
      var rollG = new THREE.Group();
      body.add(rollG);

      /* The bird: tail at -X through the breast, then up the neck to the base
         of the skull, as ONE chain. The kink at the shoulder is what makes a
         chicken a chicken — a deep body with the neck rising steeply out of the
         front of it, not a head stuck on a sphere. */
      var torso = new THREE.Mesh(sweepGeometry(chainX([
        [-0.150, -0.012, 0.052, 0.055, 0.048],   // tail root
        [-0.105, -0.020, 0.088, 0.088, 0.082],
        [-0.050, -0.022, 0.108, 0.104, 0.098],   // deepest point, the belly
        [0.005, -0.012, 0.106, 0.104, 0.092],
        [0.052, 0.008, 0.092, 0.096, 0.074],     // breast
        [0.082, 0.046, 0.066, 0.070, 0.052],     // shoulder, neck begins
        [0.094, 0.090, 0.044, 0.046, 0.040],
        [0.100, 0.130, 0.038, 0.040, 0.036],     // neck
        [0.108, 0.166, 0.044, 0.046, 0.042],
        [0.122, 0.196, 0.055, 0.056, 0.052],     // skull
        [0.150, 0.208, 0.052, 0.052, 0.050],
        [0.178, 0.204, 0.038, 0.038, 0.038]      // face
      ], 0, 2.15), 22, new THREE.Vector3(0, 1, 0), true, true), feather);
      // No castShadow anywhere on a bird: in open space there is nothing for a
      // chicken's shadow to land on, so a caster here is pure cost.
      rollG.add(torso);

      // Beak: two wedges meeting on a line, upper a little longer than lower.
      var beakUpper = new THREE.Mesh(sweepGeometry(chainX([
        [0.168, 0.206, 0.030, 0.020, 0.002],
        [0.205, 0.201, 0.020, 0.013, 0.001],
        [0.232, 0.194, 0.007, 0.005, 0.001]
      ], 0, 2.0), 10, new THREE.Vector3(0, 1, 0), true, true), beakM);
      rollG.add(beakUpper);
      var beakLower = new THREE.Mesh(sweepGeometry(chainX([
        [0.168, 0.194, 0.024, 0.002, 0.014],
        [0.199, 0.191, 0.015, 0.001, 0.009],
        [0.220, 0.188, 0.006, 0.001, 0.003]
      ], 0, 2.0), 10, new THREE.Vector3(0, 1, 0), true, true), beakM);
      rollG.add(beakLower);

      // Comb: overlapping plates with descending height, which reads as a
      // serrated comb from any angle.
      var combG = new THREE.Group();
      rollG.add(combG);
      [[0.108, 0.030], [0.128, 0.040], [0.148, 0.036], [0.166, 0.024]].forEach(function (cb) {
        var blade = new THREE.Mesh(new THREE.SphereGeometry(cb[1], 10, 8), comb);
        blade.scale.set(0.55, 1.0, 0.30);
        blade.position.set(cb[0], 0.234 + cb[1] * 0.45, 0);
        combG.add(blade);
      });
      // Wattles: small, but their absence is felt.
      [0.038, -0.038].forEach(function (wz) {
        var wattle = new THREE.Mesh(new THREE.SphereGeometry(0.024, 10, 8), comb);
        wattle.scale.set(0.60, 1.25, 0.45);
        wattle.position.set(0.158, 0.164, wz * 0.5);
        rollG.add(wattle);
      });

      // Eyes, on the sides of the skull the way a bird's are.
      var eyes = [];
      [1, -1].forEach(function (sgn) {
        var eye = new THREE.Mesh(new THREE.SphereGeometry(0.0165, 10, 8),
          new THREE.MeshBasicMaterial({
            color: new THREE.Color(0.06, 0.05, 0.05), transparent: true
          }));
        eye.position.set(0.152, 0.212, sgn * 0.046);
        rollG.add(eye);
        var spark = new THREE.Mesh(new THREE.SphereGeometry(0.0062, 6, 5),
          new THREE.MeshBasicMaterial({
            color: new THREE.Color(2.4, 2.4, 2.6), transparent: true
          }));
        spark.position.set(0.163, 0.218, sgn * 0.043);
        rollG.add(spark);
        eyes.push(eye, spark);
      });

      // Tail: a fan of flight feathers. Fanned, not a single blade — the fan is
      // half of a chicken's read.
      var tailG = new THREE.Group();
      tailG.position.set(-0.150, -0.005, 0);
      rollG.add(tailG);
      [-2, -1, 0, 1, 2].forEach(function (k) {
        var plate = new THREE.Mesh(sweepGeometry(chainX([
          [0.00, 0.000, 0.026, 0.010, 0.008],
          [-0.055, 0.045, 0.030, 0.008, 0.006],
          [-0.105, 0.098, 0.022, 0.005, 0.004],
          [-0.135, 0.142, 0.010, 0.003, 0.002]
        ], 0, 2.0), 10, new THREE.Vector3(0, 1, 0), true, true), k === 0 ? featherD : feather);
        plate.rotation.x = k * 0.30;
        plate.rotation.y = k * 0.10;
        tailG.add(plate);
      });

      // Wings: one swept surface each, hinged at the shoulder so they can beat
      // while patrolling and sweep back into a dive.
      var wings = [];
      [1, -1].forEach(function (sgn) {
        var hinge = new THREE.Group();
        hinge.position.set(-0.010, 0.030, sgn * 0.070);
        rollG.add(hinge);
        var plate = new THREE.Mesh(sweepGeometry(chainFree([
          [0.030, 0.000, sgn * 0.010, 0.056, 0.026, 0.020],
          [-0.010, -0.012, sgn * 0.070, 0.068, 0.020, 0.016],
          [-0.060, -0.026, sgn * 0.120, 0.058, 0.013, 0.011],
          [-0.105, -0.042, sgn * 0.150, 0.034, 0.008, 0.007],
          [-0.130, -0.054, sgn * 0.163, 0.014, 0.004, 0.003]
        ], 2.2), 14, new THREE.Vector3(0, 1, 0), true, true), featherD);
        hinge.add(plate);
        wings.push({ hinge: hinge, side: sgn });
      });

      // Legs, tucked. A flying chicken still has them, and their absence is one
      // of the things that makes a bird read as a toy.
      [1, -1].forEach(function (sgn) {
        var leg = new THREE.Mesh(sweepGeometry(chainFree([
          [-0.020, -0.090, sgn * 0.042, 0.016],
          [0.010, -0.120, sgn * 0.048, 0.012],
          [0.052, -0.126, sgn * 0.050, 0.009]
        ], 2.0), 8, new THREE.Vector3(0, 1, 0), true, true), beakM);
        rollG.add(leg);
        [-0.012, 0.012].forEach(function (tz) {
          var toe = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.003, 0.030, 5), beakM);
          toe.position.set(0.068, -0.126, sgn * 0.050 + tz);
          toe.rotation.z = Math.PI / 2;
          rollG.add(toe);
        });
      });

      // The suit: a collar ring at the base of the neck and a bubble helmet.
      var collar = new THREE.Mesh(new THREE.TorusGeometry(0.052, 0.014, 8, 18), suit);
      collar.position.set(0.096, 0.112, 0);
      collar.rotation.x = Math.PI / 2;
      collar.rotation.z = -0.22;
      rollG.add(collar);
      var helmet = new THREE.Mesh(new THREE.SphereGeometry(0.108, 20, 16), glass);
      helmet.position.set(0.142, 0.203, 0);
      rollG.add(helmet);
      // A specular band across the visor, so the glass reads as glass.
      var visorGlint = new THREE.Mesh(new THREE.TorusGeometry(0.104, 0.005, 6, 24, 1.5), suit);
      visorGlint.position.copy(helmet.position);
      visorGlint.rotation.set(0.5, 0.6, 0.9);
      rollG.add(visorGlint);

      // Thruster pack: how a chicken dives, and the bird's own practical light.
      var pack = new THREE.Mesh(new THREE.BoxGeometry(0.085, 0.100, 0.115), suit);
      pack.position.set(-0.075, 0.048, 0);
      rollG.add(pack);
      var packTank = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.095, 10), suit);
      packTank.position.set(-0.098, 0.052, 0.045);
      packTank.rotation.x = Math.PI / 2; packTank.rotation.z = 0.2;
      rollG.add(packTank);
      var packTank2 = packTank.clone();
      packTank2.position.z = -0.045;
      rollG.add(packTank2);

      var thrusts = [];
      [0.040, -0.040].forEach(function (tz) {
        var nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.020, 0.030, 8), suit);
        nozzle.position.set(-0.075, 0.104, tz);
        rollG.add(nozzle);
        var flame = new THREE.Mesh(new THREE.ConeGeometry(0.020, 0.13, 10),
          new THREE.MeshBasicMaterial({
            color: new THREE.Color(3.0, 0.9, 0.45), transparent: true,
            opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
          }));
        flame.position.set(-0.075, 0.180, tz);
        rollG.add(flame);
        thrusts.push(flame);
      });

      /* The mode marker: a beacon on the pack, amber patrolling and red diving.
         Full-saturation colour lives here and at the eyes and nowhere else —
         the plumage stays desaturated so the marker carries the state. */
      var marker = new THREE.Mesh(new THREE.SphereGeometry(0.026, 10, 8),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(3.0, 2.3, 0.7), transparent: true
        }));
      marker.position.set(-0.075, 0.104, 0);
      rollG.add(marker);
      var markerLight = new THREE.PointLight(COLORS.patrol, 1, 2.1, 2);
      markerLight.power = 11;
      markerLight.position.set(-0.05, 0.08, 0);
      rollG.add(markerLight);
      // A helmet lamp, so the bird's own face is lit even on the shadow side.
      var helmetLamp = new THREE.PointLight(0xBFE0FF, 1, 1.3, 2);
      helmetLamp.power = 3.2;
      helmetLamp.position.set(0.17, 0.21, 0);
      rollG.add(helmetLamp);

      var halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: COLORS.patrol, transparent: true, opacity: 0.22,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      halo.scale.set(1.15, 1.15, 1);
      root.add(halo);

      /* Two reach pips, one per sensor, lit when that sensor's own predicate
         says this bird's cell is inside its reach. They are what ties the drawn
         footprints to `camera_sees` and `radar_sees` rather than leaving the
         cone and the ring as shapes that merely look right: a bird sitting
         inside the drawn cone with its camera pip dark is a visible
         contradiction, which is the point. */
      var pips = [-1, 1].map(function (sgn) {
        var pip = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTex,
          color: sgn < 0 ? new THREE.Color(0.30, 0.66, 1.80) : new THREE.Color(0.30, 1.50, 0.95),
          transparent: true, opacity: 0,
          blending: THREE.AdditiveBlending, depthWrite: false
        }));
        pip.scale.set(0.13, 0.13, 1);
        pip.position.set(sgn * 0.11, 0.40, 0);
        root.add(pip);
        return pip;
      });

      // Debris puff for whichever bird the shot takes.
      var puffCount = 30;
      var puffGeo = new THREE.BufferGeometry();
      puffGeo.setAttribute(
        "position", new THREE.BufferAttribute(new Float32Array(puffCount * 3), 3));
      var puffDir = [];
      for (var pi = 0; pi < puffCount; pi++) {
        var th = rnd() * Math.PI * 2, ph = Math.acos(2 * rnd() - 1);
        puffDir.push([
          Math.sin(ph) * Math.cos(th), Math.sin(ph) * Math.sin(th) * 0.8, Math.cos(ph)
        ]);
      }
      var puff = new THREE.Points(puffGeo, new THREE.PointsMaterial({
        size: 0.075, map: partTex, color: new THREE.Color(2.6, 1.9, 1.1), transparent: true,
        opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
      }));
      scene.add(puff);

      fadeMats.forEach(function (m) { m.transparent = true; });
      return {
        root: root, body: body, roll: rollG, wings: wings, tail: tailG, fadeMats: fadeMats,
        eyes: eyes, marker: marker, markerLight: markerLight, helmetLamp: helmetLamp,
        thrusts: thrusts, halo: halo, pips: pips,
        puff: puff, puffGeo: puffGeo, puffDir: puffDir, puffCount: puffCount,
        bob: rnd(), beatRate: 1.40 + rnd() * 0.95, glassMat: glass
      };
    }
    var CHICKS = [];
    for (var ci = 0; ci < NCH; ci++) CHICKS.push(buildChicken(700 + ci * 91));

    /* ------------------------------------------------------ sensor footprints
       Drawn from the same two predicates the observation model tests: the cone
       is |dx| <= slope * dy, the radar is dx^2 + dy^2 <= rho^2. Because a cell
       is 1 x 1 in scene units, the cone's screen slope IS the parameter and the
       ring's screen radius IS the parameter. */
    var sensorGroup = new THREE.Group();
    scene.add(sensorGroup);

    var CONE_LEN = (ROWS + 0.6) * CELL;
    var coneGeo = new THREE.BufferGeometry();
    coneGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,
      -SLOPE * CONE_LEN, CONE_LEN, 0,
      SLOPE * CONE_LEN, CONE_LEN, 0
    ]), 3));
    sensorGroup.add(new THREE.Mesh(coneGeo, new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.20, 0.36, 0.78), transparent: true, opacity: 0.052,
      blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
    })));
    [-1, 1].forEach(function (s) {
      var len = Math.hypot(SLOPE * CONE_LEN, CONE_LEN);
      var tube = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, len, 6),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(0.55, 1.05, 2.3), transparent: true,
          opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false
        }));
      // Rotate the cylinder's +Y onto the edge direction (s*SLOPE, 1) in XY.
      tube.rotation.z = -s * Math.atan2(SLOPE, 1);
      tube.position.set(s * SLOPE * CONE_LEN / 2, CONE_LEN / 2, 0);
      sensorGroup.add(tube);
    });

    var radarRing = new THREE.Mesh(
      new THREE.TorusGeometry(RADIUS * CELL, 0.022, 8, 128),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.42, 1.6, 1.1), transparent: true,
        opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false
      })
    );
    sensorGroup.add(radarRing);
    var radarSweep = new THREE.Mesh(
      new THREE.CircleGeometry(RADIUS * CELL, 72, 0, 0.30),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.30, 1.1, 0.75), transparent: true,
        opacity: 0.045, blending: THREE.AdditiveBlending, depthWrite: false,
        side: THREE.DoubleSide
      })
    );
    sensorGroup.add(radarSweep);

    /* ------------------------------------------------------- the belief plane
       The hidden variable is a set of cells, so the belief is one box per cell,
       never a blob. Height and glow are the weighted chance that at least one
       live chicken stands there, summed from the run's own particles at the
       run's own weights. Empty cells keep their frame, so a dark cell reads as
       "ruled out" rather than "not drawn". */
    var beliefGroup = new THREE.Group();
    beliefGroup.position.z = BELIEF_Z;
    scene.add(beliefGroup);

    var BEL_HOT = new THREE.Color(0.42, 1.55, 1.30);
    var beliefCells = [];
    var frameMat = new THREE.LineBasicMaterial({
      color: 0x3E9E92, transparent: true, opacity: 0.30
    });
    for (var br = 0; br < ROWS; br++) {
      for (var bc = 0; bc < COLS; bc++) {
        var gx = cx(bc), gy = cy(br);
        var box = new THREE.Mesh(new THREE.BoxGeometry(CELL * 0.66, CELL * 0.66, 0.09),
          new THREE.MeshBasicMaterial({
            color: BEL_HOT.clone(), transparent: true, opacity: 0,
            blending: THREE.AdditiveBlending, depthWrite: false
          }));
        box.position.set(gx, gy, 0);
        beliefGroup.add(box);
        var cellFrame = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.PlaneGeometry(CELL * 0.92, CELL * 0.92)), frameMat);
        cellFrame.position.set(gx, gy, -0.05);
        beliefGroup.add(cellFrame);
        beliefCells.push({ box: box, col: bc, row: br });
      }
    }
    // An outer frame, so the lattice reads as one instrument rather than a
    // drift of cubes: the plate the belief is defined over.
    var plate = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(COLS * CELL + 0.30, ROWS * CELL + 0.30)),
      new THREE.LineBasicMaterial({ color: 0x4FB8A8, transparent: true, opacity: 0.52 }));
    plate.position.set(0, cy((ROWS - 1) / 2), -0.06);
    beliefGroup.add(plate);

    // A ghost of the rail at the foot of the lattice, so the plane reads as the
    // same world seen from the ship's side rather than a floating chart.
    var belRail = new THREE.Mesh(new THREE.BoxGeometry(COLS * CELL + 0.6, 0.035, 0.035),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.25, 0.85, 0.75), transparent: true,
        opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false
      }));
    belRail.position.set(0, cy(0) - CELL * 0.56, 0);
    beliefGroup.add(belRail);
    // A marker under the ship's own column: the column a shot would search.
    var belColumn = new THREE.Mesh(
      new THREE.BoxGeometry(CELL * 0.92, (ROWS - 1) * CELL + 0.8, 0.02),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.9, 1.5, 1.0), transparent: true,
        opacity: 0.05, blending: THREE.AdditiveBlending, depthWrite: false
      }));
    belColumn.position.y = cy(0) + ((ROWS - 1) * CELL) / 2 - 0.1;
    beliefGroup.add(belColumn);

    /* Column guides in the true plane. The columns are a real thing here — a
       shot searches one of them — so they get a mark that fades out with height
       rather than a full grid that would compete with the flock. */
    for (var cg = 0; cg < COLS; cg++) {
      var guideGeo = new THREE.BufferGeometry();
      guideGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
        cx(cg), cy(0) - 0.45, 0.42,
        cx(cg), cy(0) - 0.45 + ROWS * CELL, 0.42
      ]), 3));
      guideGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array([
        0.26, 0.34, 0.42, 0.0, 0.0, 0.0
      ]), 3));
      scene.add(new THREE.Line(guideGeo, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.55,
        blending: THREE.AdditiveBlending, depthWrite: false
      })));
    }

    /* --------------------------------------------------------------- the shot
       Hitscan. There is no projectile to follow and no lead to draw, so the
       beam is the whole event: muzzle to the row it struck, flashed inside the
       step that fired it. A shot into an empty column runs the full height and
       rings nothing. */
    var beam = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 10, 1, true),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(1.0, 3.0, 1.9), transparent: true,
        opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false,
        side: THREE.DoubleSide
      }));
    scene.add(beam);
    var beamCore = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 8, 1, true),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(2.6, 3.4, 3.0), transparent: true,
        opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
      }));
    scene.add(beamCore);
    var impact = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: 0xCFFFE4, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    impact.scale.set(1.0, 1.0, 1);
    scene.add(impact);
    var killRing = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.018, 8, 40),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(1.2, 3.0, 2.0), transparent: true,
        opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
      }));
    scene.add(killRing);

    /* ------------------------------------------------------------------ dust */
    var MOTES = 260;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(31337);
    for (var mi = 0; mi < MOTES; mi++) {
      motePos[mi * 3] = (moteSeed() - 0.5) * (COLS * CELL + 8);
      motePos[mi * 3 + 1] = 0.05 + moteSeed() * (ROWS * CELL + 2);
      motePos[mi * 3 + 2] = (moteSeed() - 0.5) * 9;
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    scene.add(new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.024, map: partTex, color: 0xB0A186, transparent: true, opacity: 0.20,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    })));

    core.linearize();

    /* ------------------------------------------------------------- the belief
       Occupancy, not an expected count: one particle contributes its whole
       weight to a cell once, however many of its chickens stand there. Adding a
       weight per chicken would render a different quantity, and two particles
       agreeing that one cell holds two chickens would paint it as certain. This
       is the same projection the GIF's belief panel draws, so the two agree. */
    var occupancy = new Float64Array(ROWS * COLS);
    var occupiedByParticle = new Uint8Array(ROWS * COLS);

    function beliefGrid(belief) {
      occupancy.fill(0);
      if (!belief || belief.kind !== "particles") return null;
      var particles = belief.particles, weights = belief.weights;
      var count = Math.min(particles.length, MAX_BELIEF_PARTICLES);
      if (!count || !Array.isArray(particles[0])) return null;
      var uniform = 1 / particles.length;
      for (var p = 0; p < count; p++) {
        var vector = particles[p];
        var weight = belief.weighted === false ? uniform : weights[p];
        occupiedByParticle.fill(0);
        for (var k = 0; k < NCH; k++) {
          var base = SL.ship_width + k * SL.chicken_width;
          if (!(vector[base + SL.chicken_alive] > 0)) continue;
          var col = Math.round(vector[base + SL.chicken_column]);
          var row = Math.round(vector[base + SL.chicken_row]);
          if (col < 0 || col >= COLS || row < 0 || row >= ROWS) continue;
          occupiedByParticle[row * COLS + col] = 1;
        }
        for (var cell = 0; cell < occupancy.length; cell++) {
          if (occupiedByParticle[cell]) occupancy[cell] += weight;
        }
      }
      return occupancy;
    }

    function drawBelief(index) {
      var belief = payload.beliefs[index];
      if (!belief) {
        for (var z = 0; z < beliefCells.length; z++) beliefCells[z].box.material.opacity = 0;
        return "—";
      }
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner. It
        // is not one episode's belief, and merging its members would show a
        // lattice that was never anyone's.
        for (var y = 0; y < beliefCells.length; y++) beliefCells[y].box.material.opacity = 0;
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }
      var grid = beliefGrid(belief);
      if (!grid) {
        for (var w = 0; w < beliefCells.length; w++) beliefCells[w].box.material.opacity = 0;
        // Named rather than silently blank, so the gap is diagnosable. A
        // Gaussian over a flock of integer cells is not a belief this
        // environment produces; if one ever arrives, it is a bug, not a look.
        return "not drawn (" + (belief.belief_class || belief.kind) + ")";
      }
      for (var i = 0; i < beliefCells.length; i++) {
        var ref = beliefCells[i];
        var pr = grid[ref.row * COLS + ref.col];
        ref.box.scale.set(0.35 + pr * 0.65, 0.35 + pr * 0.65, (0.06 + pr * 0.55) / 0.09);
        ref.box.material.opacity = pr <= 0.002 ? 0 : (0.20 + 0.80 * pr);
      }
      var label = belief.num_particles +
        (belief.weighted === false ? " uniform particles" : " particles");
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " drawn)";
      }
      return label + ", as cell occupancy";
    }

    /* ---------------------------------------------------------- the motion
       A stoop is not a fall. Out here a chicken cannot fall at all — there is
       nothing to fall towards — so a dive is the bird driving itself down its
       own column under thrust, and it builds speed the way a thrusted descent
       does rather than coasting at the constant rate an interpolation gives.

       `fresh` is the first step of a dive, which gets a short wind-up: the bird
       tenses and barely moves for a fifth of the step, then goes. A dive
       already under way skips that, so a three-step dive does not pulse once
       per step. Neither curve eases out at the end, because nothing about a
       stoop slows down before it arrives. */
    function diveEase(f, fresh) {
      if (!fresh) return f * f * (1.30 - 0.30 * f);
      var w = 0.20;
      if (f <= w) { var u = f / w; return 0.05 * u * u; }
      var g = (f - w) / (1 - w);
      return 0.05 + 0.95 * g * g * (1.32 - 0.32 * g);
    }

    /* Where a chicken is at fractional time t, handling the two discontinuities
       the environment actually produces: a kill (the bird is gone in the NEXT
       state, so it dies part-way through this step) and a pull-up (it drives
       past row 0 and re-enters at the top row).

       `atk` is how far into its attack the bird is, 0 at the moment it commits
       and 1 once it is fully over. The pitch, the roll, the wing tuck and the
       thrust are all driven off it, so the styling lives inside the dive pose
       and never changes WHETHER a dive is happening. */
    function chickenAt(slot, i0, f) {
      var a = FRAMES[i0].chickens[slot];
      if (!a.alive) return null;
      var b = NEXTS[i0].chickens[slot];
      var fresh = DIVESTART[i0][slot];

      if (a.alive && !b.alive) {
        // Killed by this step's shot. The shot resolves at the top of the step,
        // and a bird already in its stoop is fully committed when it is hit.
        return {
          col: a.c, row: a.r, diving: a.diving, dir: a.dir,
          atk: a.diving ? 1 : 0,
          alpha: 1 - smooth(clamp((f - 0.12) / 0.26, 0, 1)),
          dying: f > 0.12, killT: clamp((f - 0.12) / 0.5, 0, 1)
        };
      }
      if (PULLUP[i0][slot]) {
        /* Drives past row 0 and re-enters on the top row: two half-animations
           with a gap, not one interpolation, because it is not a climb. */
        if (f < 0.55) {
          var ff = f / 0.55;
          return {
            col: a.c, row: lerp(a.r, -0.9, diveEase(ff, true)),
            diving: ff > 0.10 ? true : a.diving, dir: a.dir,
            atk: clamp((ff - 0.10) / 0.28, 0, 1),
            alpha: 1 - smooth(clamp((ff - 0.72) / 0.28, 0, 1)), dying: false, killT: 0
          };
        }
        var gf = (f - 0.55) / 0.45;
        return {
          col: b.c, row: lerp(ROWS - 1 + 1.0, b.r, smooth(gf)), diving: false, dir: b.dir,
          atk: 0, alpha: smooth(clamp(gf / 0.45, 0, 1)), dying: false, killT: 0
        };
      }
      /* The dive coin is flipped inside the step, after the shot and before the
         flock moves, so a bird that switches this step also drops this step:
         the state at the top says patrol and the state at the bottom says dive.
         That is the moment the attack begins. The bird holds its patrol until
         it commits, so the frame at f = 0 still matches the state. */
      if (!a.diving && b.diving && b.alive && b.r < a.r) {
        var cf = clamp((f - 0.10) / 0.90, 0, 1);
        return {
          col: a.c, row: lerp(a.r, b.r, diveEase(cf, true)),
          diving: f > 0.10, dir: a.dir,
          atk: clamp((f - 0.10) / 0.30, 0, 1),
          alpha: 1, dying: false, killT: 0
        };
      }
      var descending = a.diving && b.r < a.r;
      return {
        col: lerp(a.c, b.c, smooth(f)),
        row: lerp(a.r, b.r, descending ? diveEase(f, fresh) : smooth(f)),
        diving: a.diving, dir: a.dir,
        atk: a.diving ? (fresh ? clamp((f - 0.06) / 0.26, 0, 1) : 1) : 0,
        alpha: 1, dying: false, killT: 0
      };
    }

    function sampleAt(t) {
      var i0 = Math.floor(clamp(t, 0, N_STEPS - 1));
      var f = clamp(t - i0, 0, 1);
      return {
        index: i0, frac: f,
        shipX: lerp(FRAMES[i0].ship, NEXTS[i0].ship, smooth(f))
      };
    }

    /* ------------------------------------------------------------- camera
       This scene owns its camera, and that is the one thing it does not take
       from the core. The core rig's Board and Top modes place a camera high
       above the origin and aim it at the ground plane, which is right for every
       environment whose board is flat — and wrong for this one, whose board is
       a vertical wall: from up there a vertical playfield is a foreshortened
       smear, and Top is an edge-on line. So the modes are rebuilt here, against
       the same four buttons, and applied by wrapping the core's own render call
       so the rig's camera write is the one that gets overwritten rather than
       the other way round. See the report: the core rig needs a way for a scene
       to supply its own modes, and no scene may edit it. */
    var camState = {
      mode: "board",
      orbit: { theta: 0.22, phi: 0.30, dist: Math.max(12, COLS * 1.6 + ROWS * 0.6) }
    };
    // The prototype's framing was tuned on an 8-wide, 7-high board; a board of
    // another size needs the same margins, not the same distances.
    var SPAN = Math.max(COLS, ROWS + 1) / 8;
    var camPos = new THREE.Vector3(1.9 * SPAN, 4.6 * SPAN, 12.9 * SPAN);
    var camLook = new THREE.Vector3(0, cy((ROWS - 1) * 0.47), -1.1);

    function boardCamera() {
      var k = (core.camera.aspect < 1 ? 1.55 : 1.0) * SPAN;
      return new THREE.Vector3(1.9 * k, 4.6 * k, 12.9 * k);
    }
    function flatCamera() {
      var k = (core.camera.aspect < 1 ? 1.55 : 1.0) * SPAN;
      return new THREE.Vector3(0.0, cy((ROWS - 1) / 2), 14.2 * k);
    }

    var canvas = renderer.domElement;
    var dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener("pointerdown", function (e) {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      // Grabbing the scene is a request to orbit it, which is also what the
      // core rig does with a pointer-down; the two stay in step.
      camState.mode = "orbit";
    });
    canvas.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      camState.orbit.theta -= (e.clientX - lastX) * 0.006;
      camState.orbit.phi = clamp(camState.orbit.phi + (e.clientY - lastY) * 0.005, -0.35, 1.30);
      lastX = e.clientX; lastY = e.clientY;
    });
    canvas.addEventListener("pointerup", function () { dragging = false; });
    canvas.addEventListener("wheel", function (e) {
      if (camState.mode !== "orbit") return;
      camState.orbit.dist = clamp(camState.orbit.dist + e.deltaY * 0.012, 6, 40);
    }, { passive: true });
    Array.prototype.slice.call(document.querySelectorAll("[data-cam]")).forEach(function (b) {
      b.addEventListener("click", function () { camState.mode = b.dataset.cam; });
    });

    var lastDt = 0.016, lastShipX = 0;
    function applyCamera() {
      var dt = lastDt;
      var sx = cx(lastShipX);
      if (camState.mode === "board") {
        camPos.lerp(boardCamera(), clamp(dt * 3, 0, 1));
        camLook.lerp(new THREE.Vector3(0, cy((ROWS - 1) * 0.47), -1.1), clamp(dt * 3, 0, 1));
      } else if (camState.mode === "overhead") {
        camPos.lerp(flatCamera(), clamp(dt * 3, 0, 1));
        camLook.lerp(new THREE.Vector3(0, cy((ROWS - 1) / 2), -0.9), clamp(dt * 3, 0, 1));
      } else if (camState.mode === "chase") {
        /* Sighted up the barrel, from far enough back that the whole column
           fits. Sitting on the muzzle put the frame entirely in empty sky. */
        camPos.lerp(
          new THREE.Vector3(sx, DECK_Y + 0.15, 7.6 * SPAN), clamp(dt * 3.4, 0, 1));
        camLook.lerp(new THREE.Vector3(sx, cy(ROWS * 0.47), -0.5), clamp(dt * 4.2, 0, 1));
      } else {
        var o = camState.orbit;
        camPos.lerp(new THREE.Vector3(
          Math.sin(o.theta) * Math.cos(o.phi) * o.dist,
          cy((ROWS - 1) * 0.4) + Math.sin(o.phi) * o.dist,
          Math.cos(o.theta) * Math.cos(o.phi) * o.dist
        ), clamp(dt * 4, 0, 1));
        camLook.lerp(
          new THREE.Vector3(0, cy((ROWS - 1) * 0.4), -0.9), clamp(dt * 4, 0, 1));
      }
      core.camera.position.copy(camPos);
      core.camera.lookAt(camLook);
    }

    var renderInner = core.renderComposite;
    core.renderComposite = function (elapsed) {
      applyCamera();
      renderInner(elapsed);
    };

    /* ------------------------------------------------------------- the HUD */
    var ACTION_LABELS = ["hold", "move left", "move right", "fire"];
    var running = [], total = 0;
    for (var rr = 0; rr < trace.steps.length; rr++) {
      var reward = trace.steps[rr].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, rr);
      }
      running.push(total);
    }

    return {
      steps: N_STEPS,

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      // One frame of a whole world: ship, flock, sensors, shot, belief.
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var index = sample.index, f = sample.frac;
        var state = FRAMES[index], next = NEXTS[index];
        var sx = cx(sample.shipX);
        lastDt = dt;
        lastShipX = sample.shipX;

        ship.position.set(sx, DECK_Y, 0);
        hullG.position.y = reduceMotion ? 0 : Math.sin(elapsed * 2.6 + 0.4) * 0.018;
        hullG.rotation.z = reduceMotion ? 0 : Math.sin(elapsed * 1.7) * 0.012;

        /* Engines flare on the step the ship is actually translating, which is
           the only time a real craft would be burning: a constant plume on a
           ship holding station is decoration. */
        var moving = next.ship !== state.ship ? 1 : 0;
        var burn = 1 + moving * 1.5 * Math.sin(Math.PI * f);
        var pulse = (0.72 + (reduceMotion ? 0 : Math.sin(elapsed * 13) * 0.09)) * burn;
        for (var ti = 0; ti < thrusters.length; ti++) {
          thrusters[ti].core.material.opacity = clamp(0.45 * pulse, 0, 1);
          thrusters[ti].core.scale.set(1, clamp(0.7 + burn * 0.5, 0.7, 2.2), 1);
          thrusters[ti].wash.material.opacity = clamp(0.22 * pulse, 0, 0.8);
          thrusters[ti].light.power = 42 * pulse;
        }
        var dir = Math.sign(next.ship - state.ship);
        var puffStrength = moving ? Math.sin(Math.PI * clamp(f / 0.6, 0, 1)) : 0;
        for (var rj = 0; rj < rcs.length; rj++) {
          rcs[rj].puff.material.opacity = (rcs[rj].side === -dir) ? 0.85 * puffStrength : 0;
          rcs[rj].puff.scale.setScalar(0.13 + 0.10 * puffStrength);
        }

        camPod.rotation.x = -0.34;          // tilted up into its own cone
        if (!reduceMotion) dishHead.rotation.y = elapsed * 1.9;

        /* Both footprints are anchored on the ship's cell centre, which is
           exactly where the environment measures dx and dy from. */
        sensorGroup.position.set(sx, cy(0), 0);
        radarSweep.rotation.z = reduceMotion ? 0 : (elapsed * 0.9) % (Math.PI * 2);

        /* The sensor hardware brightens on a step its sensor actually reported
           something — read out of the trace's own observation, not invented. */
        var reading = READINGS[index];
        var sawCamera = false, sawRadar = false;
        if (reading) {
          for (var q = 0; q < NCH; q++) {
            if (reading.chickens[q].camera) sawCamera = true;
            if (reading.chickens[q].radar) sawRadar = true;
          }
        }
        podLens.material.color.setRGB(0.30, 0.66, sawCamera ? 2.60 : 1.20);
        feed.material.color.setRGB(0.35, sawRadar ? 2.6 : 1.1, sawRadar ? 1.7 : 0.8);

        /* The key is a star at infinity, so it does not move with the ship. It
           only needs its shadow frustum kept over the action, which is cheaper
           than a spot and gives the harder edge. */
        key.target.position.set(sx, DECK_Y, 0);
        key.target.updateMatrixWorld();
        key.position.set(sx - 25.8, DECK_Y + 11.4, 18.2);

        var live = 0, lowest = -1;
        for (var lk = 0; lk < NCH; lk++) {
          if (!state.chickens[lk].alive) continue;
          live++;
          if (lowest < 0 || state.chickens[lk].r < lowest) lowest = state.chickens[lk].r;
        }

        // Chickens.
        for (var k = 0; k < NCH; k++) {
          var ch = CHICKS[k];
          var pose = chickenAt(k, index, f);
          if (!pose) {
            ch.root.visible = false;
            ch.puff.material.opacity = Math.max(0, ch.puff.material.opacity - dt * 1.6);
            continue;
          }
          ch.root.visible = true;
          ch.root.position.set(cx(pose.col), cy(pose.row), 0);

          /* Reach, asked of the predicates themselves, on the integer cell the
             observation model measures from: dx is the chicken's column less
             the ship's, dy is its row, because the ship sits on row 0. */
          var here = state.chickens[k];
          var dx = here.c - state.ship, dy = here.r;
          ch.pips[0].material.opacity = cameraSees(dx, dy) ? 0.85 * pose.alpha : 0;
          ch.pips[1].material.opacity = radarSees(dx, dy) ? 0.85 * pose.alpha : 0;

          /* Facing. A patrolling bird walks along its `direction` field: +1 is
             toward the higher column, which is scene +X. A diving bird has no
             sideways heading, so it keeps its last yaw and pitches nose-down. */
          var faceRight = pose.dir >= 0;
          ch.body.rotation.y = faceRight ? 0 : Math.PI;
          var mode = pose.diving;
          var atk = pose.atk || 0;

          /* The wing-beat. Downstroke and upstroke are NOT the same length: the
             downstroke is the power stroke and takes about a third of the
             cycle, the recovery the rest. An even sine makes a flap read as a
             metronome rather than as a bird. Each bird carries its own phase
             AND its own rate, because a flock beating in lockstep is the
             strongest tell that they are copies of one object. */
          var cyc = reduceMotion ? 0.25 : (elapsed * ch.beatRate + ch.bob) % 1;
          var DOWN = 0.34;
          var beat;
          if (cyc < DOWN) beat = -Math.cos(Math.PI * (cyc / DOWN));
          else beat = Math.cos(Math.PI * ((cyc - DOWN) / (1 - DOWN)));

          /* Pitch. A patrolling bird sits level and bobs against its own beat;
             a diving one drops the nose onto its attack line and holds it. The
             dive pitch deepens as it commits, which is the difference between a
             bird that is descending and a bird that is going at something. */
          ch.body.rotation.z = mode ? lerp(-1.12, -1.45, atk) : beat * 0.07;
          // Bank into the line about the bird's own nose axis.
          ch.roll.rotation.x = mode ? (faceRight ? 1 : -1) * 0.42 * atk : 0;
          // On `body`, never on `root`: root is the cell anchor and has to land
          // exactly on cy(row).
          ch.body.position.y = mode ? 0 : (reduceMotion ? 0 : beat * 0.028);

          ch.marker.material.color.setRGB(3.0, mode ? 0.75 : 2.3, mode ? 0.62 : 0.7);
          ch.markerLight.color.setHex(mode ? COLORS.dive : COLORS.patrol);
          ch.halo.material.color.setHex(mode ? COLORS.dive : COLORS.patrol);
          ch.halo.material.opacity = (mode ? 0.22 : 0.11) * pose.alpha;
          ch.halo.scale.setScalar(mode ? 1.25 : 0.95);
          ch.helmetLamp.power = 3.2 * pose.alpha;

          /* The pack is what makes a dive an attack rather than a drop: a
             chicken in vacuum has nothing to fall towards, so every metre it
             gains it has to push itself down for. Full burn once committed. */
          var thrustPulse = reduceMotion
            ? 1 : 0.82 + Math.sin(elapsed * 19 + ch.bob * 6.28) * 0.18;
          var burnT = mode ? (0.35 + 0.65 * atk) : 0;
          for (var tj = 0; tj < ch.thrusts.length; tj++) {
            ch.thrusts[tj].material.opacity = clamp(1.05 * thrustPulse * burnT, 0, 1) * pose.alpha;
            ch.thrusts[tj].scale.set(
              0.9 + 0.5 * burnT, (0.7 + 1.5 * burnT) * thrustPulse, 0.9 + 0.5 * burnT);
          }
          ch.markerLight.power = (mode ? 11 + 22 * atk : 11) * pose.alpha;

          /* Wings. Patrolling, they beat: the stroke sweeps a little forward as
             it comes down and the span folds in on the recovery, which is what
             stops the flap reading as a hinge. Diving, they tuck hard against
             the body — a bird folds to stoop and spreads to manoeuvre, and a
             stooping bird with its wings out looks like it is being dropped. */
          for (var wi = 0; wi < ch.wings.length; wi++) {
            var wg = ch.wings[wi];
            wg.hinge.rotation.x = wg.side * lerp(beat * 0.62, -1.45, mode ? atk : 0);
            wg.hinge.rotation.y =
              wg.side * lerp(0.26 * Math.max(0, beat), -0.30, mode ? atk : 0);
            wg.hinge.rotation.z = mode ? 0.62 * atk : 0.0;
            var fold = mode ? lerp(1, 0.52, atk) : (1 - 0.20 * Math.max(0, -beat));
            wg.hinge.scale.set(1, 1, fold);
          }
          // The tail fans wide to manoeuvre and clamps shut into the stoop.
          var tf = mode ? atk : 0;
          ch.tail.scale.set(1, lerp(1.0, 0.58, tf), lerp(1.0, 0.45, tf));
          ch.tail.rotation.z = lerp(0, -0.42, tf);

          /* Fade out the bird this step's shot took. Only the structural
             materials are faded; the emitters carry their own opacity. */
          for (var fm = 0; fm < ch.fadeMats.length; fm++) {
            ch.fadeMats[fm].opacity =
              (ch.fadeMats[fm] === ch.glassMat ? 0.20 : 1) * pose.alpha;
          }
          ch.marker.material.opacity = pose.alpha;
          for (var ei = 0; ei < ch.eyes.length; ei++) ch.eyes[ei].material.opacity = pose.alpha;
          if (pose.dying) {
            var pa = ch.puffGeo.attributes.position.array;
            var spread = pose.killT * 1.5;
            for (var pj = 0; pj < ch.puffCount; pj++) {
              pa[pj * 3] = cx(pose.col) + ch.puffDir[pj][0] * spread;
              pa[pj * 3 + 1] =
                cy(pose.row) + ch.puffDir[pj][1] * spread - pose.killT * pose.killT * 0.7;
              pa[pj * 3 + 2] = ch.puffDir[pj][2] * spread * 0.6;
            }
            ch.puffGeo.attributes.position.needsUpdate = true;
            ch.puff.material.opacity = 0.9 * (1 - pose.killT);
          } else {
            ch.puff.material.opacity = Math.max(0, ch.puff.material.opacity - dt * 1.6);
          }
        }

        /* The shot, drawn only inside the step that discharged the gun: muzzle
           to the row it struck, or the full height on a miss. */
        var shot = payload.shots[index] || { fired: false, target_slot: -1, target_row: -1 };
        var SHOT_LIFE = 0.62;
        var life = clamp(f / SHOT_LIFE, 0, 1);
        // Held near full for the first third of its life, then dropped: a
        // linear fade made the most important event on the page a faint smear.
        var fade = 1 - smooth(clamp((life - 0.30) / 0.70, 0, 1));
        var hit = shot.target_slot >= 0;
        if (shot.fired && f < SHOT_LIFE) {
          var y0 = DECK_Y + MUZZLE_Y * ship.scale.y;
          var y1 = cy(hit ? shot.target_row : ROWS - 1);
          var len = Math.max(0.05, y1 - y0);
          beam.scale.set(0.090, len, 0.090);
          beam.position.set(sx, y0 + len / 2, 0);
          beam.material.opacity = 0.80 * fade;
          beamCore.scale.set(0.030, len, 0.030);
          beamCore.position.copy(beam.position);
          beamCore.material.opacity = 0.95 * fade;
          beam.visible = beamCore.visible = true;

          impact.position.set(sx, y1, 0);
          impact.material.opacity = (hit ? 0.95 : 0.35) * fade;
          impact.scale.setScalar((hit ? 1.7 : 0.9) * (0.6 + life * 0.7));
          impact.visible = true;

          killRing.position.set(sx, y1, 0);
          killRing.material.opacity = hit ? 0.8 * fade : 0;
          killRing.scale.setScalar(0.7 + life * 0.9);
          killRing.visible = hit;

          muzzleFlash.material.opacity = 0.95 * fade;
          muzzleFlash.scale.setScalar(0.5 + life * 0.5);
          muzzleLight.power = 900 * fade;
        } else {
          beam.visible = beamCore.visible = impact.visible = killRing.visible = false;
          muzzleFlash.material.opacity = 0;
          muzzleLight.power = 0;
        }

        var beliefLabel = drawBelief(index);
        belColumn.position.x = cx(state.ship);

        if (!reduceMotion && playing) {
          for (var mj = 0; mj < MOTES; mj++) {
            motePos[mj * 3] += Math.sin(elapsed * 0.25 + mj) * 0.0013;
            motePos[mj * 3 + 1] += 0.0016;
            if (motePos[mj * 3 + 1] > ROWS * CELL + 2.2) motePos[mj * 3 + 1] = 0.05;
          }
          moteGeo.attributes.position.needsUpdate = true;
        }

        var step = trace.steps[index] || {};
        var action = step.action;
        return {
          // `follow` is unused: this scene drives its own camera, for the
          // reason set out where the modes are built.
          follow: null,
          step: index,
          action: action === null || action === undefined
            ? "—"
            : (ACTION_LABELS[action] || String(action)) +
              (action === 3 && !shot.fired ? " (cooldown: acts as hold)" : ""),
          // There is no (x, y) agent position in this world. The two numbers
          // that matter are where the ship is and how far down the nearest live
          // chicken has come, so those are what the two slots carry.
          x: state.ship,
          y: lowest < 0 ? 0 : lowest,
          reward: step.reward,
          ret: running[index],
          belief: beliefLabel + " · " + live + (live === 1 ? " chicken live" : " chickens live")
        };
      }
    };
  }

  V.scenes["chicheck_invaders.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down here. Tuned for
    // this scene's one hard key and its practicals; it is not a knob to remove.
    exposure: 0.125
  };
})(window);
