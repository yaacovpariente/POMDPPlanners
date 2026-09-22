/* SPDX-License-Identifier: MIT
 *
 * Battleship scene module.
 *
 * Builds the bay from a trace's `payload.world` block and plays it from the
 * trace's recorded probes, readings and beliefs. Nothing here is invented:
 * there is no fallback episode, no hand-placed fleet and no synthetic belief.
 *
 * Three registers, kept apart on purpose, because in this environment mixing
 * them would be a lie rather than a mess:
 *
 *  - Truth is the fleet, its damage and its sinking, *in* the water.
 *  - Evidence is the hit and miss marks and the reticle, *on* the water.
 *  - Belief is the per-cell posterior, in columns well *above* the water.
 *
 * The agent is told one bit per shot — hit or miss. It is never told which
 * ship it hit, and never told that a ship has gone down. So the truth layer
 * never feeds the belief columns or the agent-knowledge readout, which counts
 * cells and never ships, and the caption under the render says so.
 *
 * The environment stores no ship identities at all: `FleetLayoutTable`
 * enumerates flat occupancy bitmaps, so even the ground truth does not know
 * which cells form which hull. Splitting the occupied cells into ships is this
 * viewer's choice and is not in general unique; the caption reports how many
 * fleets fit the same occupancy, so nobody reads the drawn one as the answer.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* The accents are the GIF renderer's own constants, so the viewer reads as
     the same task: _AGENT_VIEW_COLORS[1..2], _PROBE_RING_COLOR, and the two
     upper stops of the belief colormap from battleship_visualizer.py. */
  var C = {
    lamp:   0xFFE4B4,
    hit:    0xBF4B36,
    miss:   0xC5E1E9,
    probe:  0xE7A624,
    belief: 0x77BCB3,
    deep:   0x125650
  };

  // Cells are two world units across, which puts a five-square board at ten
  // units and keeps the props (masts, turrets, hulls) at a believable size
  // against it.
  var CELL = 2.0;

  // Lumens. Service lamps on masts, not a stadium rig: the middle of the board
  // has to fall off or the water stops reading as night water.
  var LAMP_LUMENS = 2200;

  // When, within a step, the round lands. Everything the agent learns flips at
  // this one instant — the mark, the readout, the next aim and the posterior —
  // so the page can never show a belief that has run ahead of its evidence.
  var RESOLVE = 0.34;

  // How high the round lobs, in world units. Guns lob; they do not shoot flat.
  var ARC_H = 5.6;

  // Where the belief columns start, above the tallest masthead. With solid
  // ships in the water the two registers would otherwise interpenetrate, and
  // the ships are the subject, so the belief moves rather than the fleet.
  var BELIEF_BASE = 2.05;
  var BELIEF_MAX = 1.15;

  /* ------------------------------------------------------------- the fleet
     The environment discards ship identity, so the split has to be recovered
     here — and it is not always unique. Every partition of the occupied cells
     into straight runs of the fleet's lengths is enumerated; the first is
     drawn and the count is reported, so the page can say plainly that it is
     showing one of several fleets that produce this board. */
  function splitFleet(occupancy, shipLengths, board) {
    var lengths = shipLengths.slice().sort(function (a, b) { return b - a; });
    var occupied = [];
    for (var i = 0; i < occupancy.length; i++) if (occupancy[i]) occupied.push(i);

    // Every straight run of each needed length, as arrays of cell indices.
    var runsFor = {};
    lengths.forEach(function (len) {
      if (runsFor[len]) return;
      var runs = [];
      for (var r = 0; r < board; r++) {
        for (var c = 0; c + len <= board; c++) {
          var h = [];
          for (var k = 0; k < len; k++) h.push(r * board + c + k);
          runs.push(h);
        }
      }
      // A one-cell ship has no orientation; listing it twice would double
      // every count this function reports.
      if (len > 1) {
        for (var c2 = 0; c2 < board; c2++) {
          for (var r2 = 0; r2 + len <= board; r2++) {
            var v = [];
            for (var k2 = 0; k2 < len; k2++) v.push((r2 + k2) * board + c2);
            runs.push(v);
          }
        }
      }
      runsFor[len] = runs;
    });

    var free = {};
    occupied.forEach(function (cell) { free[cell] = true; });

    var first = null;
    var solutions = 0;
    var SOLUTION_CAP = 500;   // enough to say "more than one", cheap to reach

    function place(index, chosen, minRun) {
      if (solutions >= SOLUTION_CAP) return;
      if (index === lengths.length) {
        // Every ship placed; the split is a real one only if it used up
        // exactly the occupied cells and left none over.
        for (var cell in free) if (free[cell]) return;
        solutions++;
        if (!first) first = chosen.map(function (s) { return { len: s.length, cells: s.slice() }; });
        return;
      }
      var len = lengths[index];
      var runs = runsFor[len];
      // Ships of equal length are interchangeable, so a split is counted once
      // by requiring their run indices to increase — the same argument the
      // environment's own enumeration makes.
      var sameAsPrevious = index > 0 && lengths[index - 1] === len;
      for (var ri = sameAsPrevious ? minRun : 0; ri < runs.length; ri++) {
        var run = runs[ri];
        var ok = true;
        for (var q = 0; q < run.length && ok; q++) if (!free[run[q]]) ok = false;
        if (!ok) continue;
        for (var s = 0; s < run.length; s++) free[run[s]] = false;
        chosen.push(run);
        place(index + 1, chosen, ri + 1);
        chosen.pop();
        for (var u = 0; u < run.length; u++) free[run[u]] = true;
      }
    }
    place(0, [], 0);

    return { ships: first, solutions: solutions, capped: solutions >= SOLUTION_CAP };
  }

  /* The sea is drawn in two stages. The swell is painted first and the normal
     map derived from it, so every wave catches the dock lamps from the side
     the lamp is actually on. The grid is painted afterwards onto colour only:
     it marks the board and must not emboss the water. */
  function seaCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(20260919);

    g.fillStyle = "#0A1A22";
    g.fillRect(0, 0, s, s);

    // Broad swell: overlapping soft bands, so the surface has long structure
    // and not just noise.
    for (var i = 0; i < 260; i++) {
      var h = 6 + rnd() * 46;
      g.fillStyle = "rgba(" + (16 + rnd() * 26 | 0) + "," + (44 + rnd() * 34 | 0) + "," +
        (54 + rnd() * 34 | 0) + ",0.30)";
      g.beginPath();
      g.ellipse(rnd() * s, rnd() * s, 70 + rnd() * 260, h, (rnd() - 0.5) * 0.35, 0, Math.PI * 2);
      g.fill();
    }
    // Ripples: a lit crest and a dark trough, which the normal map turns into
    // real relief under the lamps.
    for (var p = 0; p < 2600; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 2 + rnd() * 7;
      g.strokeStyle = "rgba(6,14,18,0.5)";
      g.lineWidth = 1.6;
      g.beginPath(); g.ellipse(px, py + pr * 0.4, pr * 2.2, pr * 0.5, 0, 0, Math.PI); g.stroke();
      g.strokeStyle = "rgba(150,196,206,0.26)";
      g.beginPath(); g.ellipse(px, py, pr * 2.0, pr * 0.45, 0, Math.PI, Math.PI * 2); g.stroke();
    }
    for (var j = 0; j < 900; j++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(198,232,240,0.05)" : "rgba(0,0,0,0.22)";
      g.fillRect(rnd() * s, rnd() * s, 3, 2);
    }
    return cv;
  }

  function paintGrid(cv, board) {
    var s = cv.width;
    var g = cv.getContext("2d");
    g.strokeStyle = "rgba(160,220,225,0.10)";
    g.lineWidth = 3;
    for (var k = 0; k <= board; k++) {
      var p = (k / board) * s;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
      g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
    }
  }

  /* The plan view of a hull: a point at the bow, full beam amidships, a
     rounded stern. Extruded upward it gives a real silhouette, which a box
     with a cone stuck on the front does not. */
  function hullShape(halfBeam, halfLen) {
    var sh = new THREE.Shape();
    sh.moveTo(0, halfLen);
    sh.bezierCurveTo(halfBeam * 0.72, halfLen * 0.66, halfBeam, halfLen * 0.22, halfBeam, -halfLen * 0.38);
    sh.lineTo(halfBeam * 0.84, -halfLen * 0.87);
    sh.quadraticCurveTo(halfBeam * 0.72, -halfLen, 0, -halfLen);
    sh.quadraticCurveTo(-halfBeam * 0.72, -halfLen, -halfBeam * 0.84, -halfLen * 0.87);
    sh.lineTo(-halfBeam, -halfLen * 0.38);
    sh.bezierCurveTo(-halfBeam, halfLen * 0.22, -halfBeam * 0.72, halfLen * 0.66, 0, halfLen);
    return sh;
  }

  /* Extrude a plan shape into a slab between two heights. ExtrudeGeometry
     pushes along +Z, so the mesh is tipped a quarter turn to stand the
     extrusion up the Y axis and lay the length along +Z, bow forward. */
  function hullSlab(shape, yBottom, yTop, material) {
    var mesh = new THREE.Mesh(
      new THREE.ExtrudeGeometry(shape, { depth: yTop - yBottom, bevelEnabled: false, curveSegments: 14 }),
      material
    );
    mesh.rotation.x = Math.PI / 2;
    mesh.position.y = yTop;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  function smoothstep(t) { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); }

  /* The caption and the three toggles. The page chrome is shared by every
     environment, so this strip is built here: the note that the fleet is
     unobservable is not decoration, it is the one thing a viewer has to know
     before reading anything else on the screen. */
  function buildOverlay(root) {
    var box = document.createElement("div");
    box.style.cssText = [
      "position:absolute", "top:10px", "right:10px", "max-width:min(46ch, calc(100% - 20px))",
      "padding:8px 10px", "border-radius:6px", "background:rgba(8,7,6,.58)",
      "color:#f2ece2", "font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace",
      "display:flex", "flex-direction:column", "gap:4px", "text-align:right"
    ].join(";");

    var note = document.createElement("span");
    note.textContent = "Observer view — hulls, damage and sinkings are not observed by the agent";
    note.style.color = "#c9c0b4";
    box.appendChild(note);

    var known = document.createElement("span");
    box.appendChild(known);

    var split = document.createElement("span");
    split.style.color = "#c9c0b4";
    box.appendChild(split);

    var row = document.createElement("div");
    row.style.cssText = "display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap";
    box.appendChild(row);

    var toggles = {};
    [["belief", "Belief"], ["truth", "True fleet"], ["labels", "%"]].forEach(function (pair) {
      var label = document.createElement("label");
      label.style.cssText = "display:inline-flex;align-items:center;gap:4px;cursor:pointer";
      var input = document.createElement("input");
      input.type = "checkbox";
      input.checked = true;
      label.appendChild(input);
      label.appendChild(document.createTextNode(pair[1]));
      row.appendChild(label);
      toggles[pair[0]] = input;
    });

    root.appendChild(box);
    return { known: known, split: split, toggles: toggles };
  }

  /**
   * Build the Battleship world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind battleship.v1.
   * @returns {Object} The scene module the player drives.
   */
  // The bay is one world with a dozen parts, and splitting it into private
  // helpers would only move the same closures somewhere else.
  /* eslint-disable-next-line complexity */
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;

    var BOARD = Math.max(1, world.board_size | 0);
    var NCELL = BOARD * BOARD;
    var SPAN = BOARD * CELL;
    var HALF = SPAN / 2;
    var SHIP_CELLS = world.num_ship_cells | 0;
    var N_STEPS = trace.steps.length;

    var reduceMotion = global.matchMedia
      ? global.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;

    // Cell index -> scene position. Row 0 sits at the far side, which matches
    // the GIF's row axis running top to bottom.
    function cx(col) { return (col - (BOARD - 1) / 2) * CELL; }
    function cz(row) { return (row - (BOARD - 1) / 2) * CELL; }
    function rowOf(i) { return Math.floor(i / BOARD); }
    function colOf(i) { return i % BOARD; }

    var TRUTH = payload.occupancy || [];

    /* ------------------------------------------------------------- frames
       One record per recorded step, read straight off the envelope and the
       payload. The final record carries a board but no action: that is the
       terminal bookkeeping step, as StepData writes it. */
    var frames = [];
    var running = 0;
    for (var f = 0; f < N_STEPS; f++) {
      var step = trace.steps[f] || {};
      var action = step.action === null || step.action === undefined ? null : (step.action | 0);
      var observation = payload.observations ? payload.observations[f] : null;
      var hit = action === null
        ? null
        : (observation === null || observation === undefined
          ? !!TRUTH[action]
          : Number(observation) === 1);
      var probed = (payload.probed && payload.probed[f]) || [];
      var known = 0;
      for (var k = 0; k < NCELL; k++) if (probed[k] && TRUTH[k]) known++;
      frames.push({
        action: action,
        hit: hit,
        reward: step.reward === undefined ? null : step.reward,
        ret: running,
        knownHits: known,
        support: payload.support_sizes ? payload.support_sizes[f] : null
      });
      if (step.reward !== null && step.reward !== undefined) {
        running += step.reward * Math.pow(trace.discount_factor, f);
      }
    }

    /* ------------------------------------------------------------- belief
       The exporter writes the belief's exact per-cell posterior when the run
       held a BattleshipBelief, because that marginal is computed over the
       whole consistent-layout set. When it did not — a generic particle
       filter, say — the marginal is projected here from the particle cloud
       core serialised, and the readout says so: a sampled marginal can show a
       cell as certain on the luck of the draw, and this scene treats certainty
       as meaningful. */
    function particleMarginal(belief) {
      if (!belief || belief.kind !== "particles") return null;
      var items = belief.particles || [];
      if (!items.length || !Array.isArray(items[0]) || items[0].length < NCELL) return null;
      var out = new Array(NCELL);
      for (var c = 0; c < NCELL; c++) out[c] = 0;
      var total = 0;
      for (var p = 0; p < items.length; p++) {
        var w = belief.weighted === false ? 1 / items.length : (belief.weights[p] || 0);
        total += w;
        for (var c2 = 0; c2 < NCELL; c2++) out[c2] += w * (items[p][c2] > 0.5 ? 1 : 0);
      }
      if (total <= 0) return null;
      for (var c3 = 0; c3 < NCELL; c3++) out[c3] /= total;
      return out;
    }

    var exactMarginals = payload.occupancy_marginals || [];
    var marginals = [];
    var anyExact = false, anySampled = false, anyMissing = false;
    for (var mi = 0; mi < N_STEPS; mi++) {
      var exact = exactMarginals[mi];
      if (exact && exact.length === NCELL) { marginals.push(exact); anyExact = true; continue; }
      var sampled = particleMarginal((payload.beliefs || [])[mi]);
      if (sampled) { marginals.push(sampled); anySampled = true; continue; }
      marginals.push(null);
      anyMissing = true;
    }
    var beliefSource = anyExact
      ? (anySampled || anyMissing ? "mixed" : "exact posterior")
      : (anySampled ? "sampled from the belief's particles" : "not recorded");

    /* ----------------------------------------------------------- timeline
       Everything in the world is a pure function of the continuous step index
       t, so scrubbing backwards is exact rather than approximately undone. A
       probe fired on step i lands at i + RESOLVE, which is the same instant
       the mark, the readout and the posterior move. */
    function resolveT(i) { return i + RESOLVE; }
    var cellEvent = new Array(NCELL);
    for (var ce = 0; ce < NCELL; ce++) cellEvent[ce] = null;
    for (var fi = 0; fi < N_STEPS; fi++) {
      var fa = frames[fi].action;
      if (fa === null || fa < 0 || fa >= NCELL || cellEvent[fa]) continue;
      cellEvent[fa] = { t: resolveT(fi), hit: !!frames[fi].hit };
    }

    /* -------------------------------------------------------------- scene */
    scene.background = new THREE.Color(0x04070A).convertSRGBToLinear();
    // Sea haze. Enough that the far edge of the bay sits behind atmosphere,
    // little enough that the board stays readable from the raised view.
    scene.fog = new THREE.FogExp2(0x05090D, 0.020);
    scene.fog.color.convertSRGBToLinear();

    // A night scene lit by dock lamps: the sky dome is deliberately weak.
    scene.add(new THREE.HemisphereLight(0x22384A, 0x050A0C, 0.95));
    var moon = new THREE.DirectionalLight(0x93AECC, 0.42);
    moon.position.set(-6, 9, -4);
    scene.add(moon);
    core.buildNightEnvironment([
      [0.0, "#081020"], [0.45, "#101A26"], [0.55, "#22261E"], [1.0, "#04080A"]
    ]);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);
    var glowTex = V.radialTexture(0.9, 0.3);

    var sCanvas = seaCanvas();
    var seaNormal = V.normalMapFrom(renderer, sCanvas, 2.1);
    paintGrid(sCanvas, BOARD);
    var seaAlbedo = new THREE.CanvasTexture(sCanvas);
    seaAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    seaAlbedo.encoding = THREE.sRGBEncoding;   // colour data, not linear data

    var water = new THREE.Mesh(
      new THREE.PlaneGeometry(SPAN, SPAN),
      new THREE.MeshStandardMaterial({
        map: seaAlbedo, normalMap: seaNormal,
        normalScale: new THREE.Vector2(1.1, 1.1),
        // Water is smooth, so the lamps leave long specular streaks instead of
        // a broad diffuse wash. That streak is most of what reads as "wet".
        roughness: 0.22, metalness: 0.08, envMapIntensity: 0.9
      })
    );
    water.rotation.x = -Math.PI / 2;
    water.receiveShadow = true;
    scene.add(water);

    /* Open water carrying on past the board. Without it the board is a lit
       slab floating in a void, which is the biggest tell that a night render
       is a render. It sits well below the board's surface, not a few
       centimetres: two large coplanar planes stripe each other in a 16-bit
       depth buffer. */
    var openSeaNormal = seaNormal.clone();
    openSeaNormal.needsUpdate = true;
    openSeaNormal.wrapS = openSeaNormal.wrapT = THREE.RepeatWrapping;
    openSeaNormal.repeat.set(9, 9);
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(90, 90),
      new THREE.MeshStandardMaterial({
        normalMap: openSeaNormal, normalScale: new THREE.Vector2(0.9, 0.9),
        color: 0x060F14, roughness: 0.25, metalness: 0.1, envMapIntensity: 0.55
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.85;
    scene.add(outer);

    // The pontoon the board is framed by, so the drop to open water reads as a
    // structure rather than a seam.
    var frameMat = new THREE.MeshStandardMaterial({ color: 0x16242B, roughness: 0.72, metalness: 0.35 });
    var hullBelow = new THREE.Mesh(
      new THREE.BoxGeometry(SPAN + 0.9, 0.8, SPAN + 0.9),
      new THREE.MeshStandardMaterial({ color: 0x0A1418, roughness: 0.9, metalness: 0.1 })
    );
    hullBelow.position.y = -0.42;
    hullBelow.receiveShadow = true;
    scene.add(hullBelow);

    var kerbLen = SPAN + 0.44;
    [[0, -HALF - 0.22], [0, HALF + 0.22]].forEach(function (o) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(kerbLen, 0.22, 0.44), frameMat);
      m.position.set(o[0], 0.05, o[1]); m.receiveShadow = true; m.castShadow = true; scene.add(m);
    });
    [[-HALF - 0.22, 0], [HALF + 0.22, 0]].forEach(function (o) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.22, kerbLen), frameMat);
      m.position.set(o[0], 0.05, o[1]); m.receiveShadow = true; m.castShadow = true; scene.add(m);
    });

    /* Dock lamps at the four corners of the pontoon: a mast, a warm lens and a
       real light in lumens. Emitter colours deliberately exceed 1.0 — the
       scene buffer is half-float, so a filament sits above white, which is
       what lets the bright pass bloom only the things that are truly hot. */
    var lampLights = [];
    var lensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(6.2, 5.3, 4.1) });
    var mastMat = new THREE.MeshStandardMaterial({ color: 0x5E6A70, roughness: 0.38, metalness: 0.9 });
    var footMat = new THREE.MeshStandardMaterial({ color: 0x2C3A41, roughness: 0.6, metalness: 0.7 });
    var mastGeo = new THREE.CylinderGeometry(0.07, 0.1, 2.6, 12);
    var footGeo = new THREE.CylinderGeometry(0.26, 0.34, 0.16, 16);
    var lensGeo = new THREE.SphereGeometry(0.17, 16, 12);
    var hoodGeo = new THREE.ConeGeometry(0.34, 0.26, 16, 1, true);

    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(function (s) {
      var x = s[0] * (HALF + 0.55), z = s[1] * (HALF + 0.55);
      var foot = new THREE.Mesh(footGeo, footMat);
      foot.position.set(x, 0.14, z);
      foot.castShadow = true; foot.receiveShadow = true;
      scene.add(foot);
      var mast = new THREE.Mesh(mastGeo, mastMat);
      mast.position.set(x, 1.45, z); mast.castShadow = true;
      scene.add(mast);
      var lens = new THREE.Mesh(lensGeo, lensMat);
      lens.position.set(x, 2.72, z);
      scene.add(lens);
      var hood = new THREE.Mesh(hoodGeo, mastMat);
      hood.position.set(x, 2.94, z); hood.castShadow = true;
      scene.add(hood);
      var light = new THREE.PointLight(C.lamp, 1, 17, 2);
      light.power = LAMP_LUMENS;
      light.position.set(x, 2.66, z);
      scene.add(light);
      lampLights.push({ light: light, x: x, z: z });
    });

    /* One shadow-casting lamp, not four. It parks on whichever dock lamp is
       nearest the current target and aims at it, and that lamp's own light is
       dimmed by the same amount, so the bay stays evenly lit while whatever
       stands there always throws a shadow pointing the right way. Narrow cone
       plus normalBias, which is the fix for acne that a big negative bias is
       not. */
    var shadowLight = new THREE.SpotLight(C.lamp, 1, 26, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    // 1024, not 2048. The caster moves with the target, so the map is redrawn
    // every frame over a scene of a couple of hundred meshes; at this range
    // the extra resolution is not visible and the cost is.
    shadowLight.shadow.mapSize.set(1024, 1024);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.5;
    shadowLight.shadow.camera.far = 26;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, 2.7, 0);
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    /* -------------------------------------------------------- ground truth
       Real vessels, solid and lit, sitting in the bay. That the agent cannot
       see them is carried by the caption, by the toggle and by a readout that
       counts cells and never ships — not by making the ships faint, which an
       earlier pass tried and which cost the page its subject.

       Each cell of a hull carries its own state, because the observation is
       per cell: an intact cell keeps its deck plating, a hit one is torn open
       and burns, and the ship goes down only when the last of *its* cells is
       gone. */
    var truthGroup = new THREE.Group();
    scene.add(truthGroup);

    var hullMat = new THREE.MeshStandardMaterial({ color: 0x93A1AB, roughness: 0.58, metalness: 0.72, envMapIntensity: 1.0 });
    var deckMat = new THREE.MeshStandardMaterial({ color: 0x5E6A73, roughness: 0.78, metalness: 0.35, envMapIntensity: 0.7 });
    var bootMat = new THREE.MeshStandardMaterial({ color: 0x14100E, roughness: 0.9, metalness: 0.2 });
    var belowMat = new THREE.MeshStandardMaterial({ color: 0x5A2A23, roughness: 0.85, metalness: 0.25 });
    var superMat = new THREE.MeshStandardMaterial({ color: 0xA2B0BA, roughness: 0.5, metalness: 0.7, envMapIntensity: 1.0 });
    var steelMat = new THREE.MeshStandardMaterial({ color: 0x4D5860, roughness: 0.45, metalness: 0.85 });
    var plateMat = new THREE.MeshStandardMaterial({ color: 0x77848D, roughness: 0.72, metalness: 0.5, envMapIntensity: 0.8 });
    var windowMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.5, 1.25, 0.72) });
    var wreckMat = new THREE.MeshStandardMaterial({ color: 0x100D0B, roughness: 0.96, metalness: 0.25 });
    var scorchMat = new THREE.MeshStandardMaterial({ color: 0x17120F, roughness: 1.0, metalness: 0.1 });

    var fleet = splitFleet(TRUTH, world.ship_lengths || [], BOARD);
    var ships = [];
    var wreckRnd = mulberry(5150);

    (fleet.ships || []).forEach(function (ship) {
      var firstCell = ship.cells[0], lastCell = ship.cells[ship.cells.length - 1];
      var pFirst = new THREE.Vector3(cx(colOf(firstCell)), 0, cz(rowOf(firstCell)));
      var pLast = new THREE.Vector3(cx(colOf(lastCell)), 0, cz(rowOf(lastCell)));
      var centre = pFirst.clone().add(pLast).multiplyScalar(0.5);
      var bowDir = ship.cells.length > 1
        ? pLast.clone().sub(pFirst).normalize()
        : new THREE.Vector3(0, 0, 1);

      /* The hull is modelled along local +Z with its bow at +Z and the group
         is aimed with lookAt, which points local +Z at the target. No angle is
         guessed and no axis swapped by hand, which is how a hull ends up
         ninety degrees out in a way no still frame shows. */
      var anchor = new THREE.Group();
      anchor.position.copy(centre);
      truthGroup.add(anchor);
      anchor.lookAt(centre.clone().add(bowDir));

      // The sinking happens in the body, inside the aimed anchor, so a list
      // and a bow-down pitch are local rotations that cannot fight the aiming.
      var body = new THREE.Group();
      anchor.add(body);

      var halfLen = (ship.len * CELL - 0.42) / 2;
      var beam = CELL - 0.78;
      var halfBeam = beam / 2;

      var plan = hullShape(halfBeam, halfLen);
      var planBelow = hullShape(halfBeam * 0.93, halfLen * 0.985);
      var planDeck = hullShape(halfBeam * 0.88, halfLen * 0.965);

      // Below the waterline, the boot topping at it, the freeboard above it
      // and the deck on top. The waterline is what makes a hull sit *in* the
      // sea rather than on it.
      body.add(hullSlab(planBelow, -0.34, 0.0, belowMat));
      body.add(hullSlab(plan, 0.0, 0.07, bootMat));
      body.add(hullSlab(plan, 0.07, 0.46, hullMat));
      body.add(hullSlab(planDeck, 0.46, 0.52, deckMat));

      // Sheer strake: a raised rail along the deck edge, so the deck has an
      // edge to catch the lamps instead of ending in a flat cut.
      var rail = new THREE.Mesh(
        new THREE.ExtrudeGeometry(planDeck, { depth: 0.09, bevelEnabled: false, curveSegments: 14 }),
        steelMat
      );
      rail.rotation.x = Math.PI / 2;
      rail.position.y = 0.61;
      body.add(rail);
      var railInner = new THREE.Mesh(
        new THREE.ExtrudeGeometry(hullShape(halfBeam * 0.76, halfLen * 0.93), { depth: 0.12, bevelEnabled: false, curveSegments: 14 }),
        deckMat
      );
      railInner.rotation.x = Math.PI / 2;
      railInner.position.y = 0.62;
      body.add(railInner);

      // Superstructure amidships: bridge, lit windows, funnel, mast.
      var bridge = new THREE.Mesh(new THREE.BoxGeometry(beam * 0.56, 0.3, halfLen * 0.42), superMat);
      bridge.position.set(0, 0.67, -halfLen * 0.05);
      bridge.castShadow = true;
      body.add(bridge);
      var bridgeTop = new THREE.Mesh(new THREE.BoxGeometry(beam * 0.4, 0.2, halfLen * 0.24), superMat);
      bridgeTop.position.set(0, 0.92, -halfLen * 0.02);
      bridgeTop.castShadow = true;
      body.add(bridgeTop);
      var glass = new THREE.Mesh(new THREE.BoxGeometry(beam * 0.42, 0.07, halfLen * 0.26), windowMat);
      glass.position.set(0, 0.95, halfLen * 0.04);
      body.add(glass);

      var funnel = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.14, 0.38, 12), steelMat);
      funnel.position.set(0, 1.0, -halfLen * 0.3);
      funnel.castShadow = true;
      body.add(funnel);
      body.add((function () {
        var cap = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.05, 12), bootMat);
        cap.position.set(0, 1.2, -halfLen * 0.3);
        return cap;
      })());

      var shipMast = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.03, 0.8, 8), steelMat);
      shipMast.position.set(0, 1.35, halfLen * 0.02);
      body.add(shipMast);
      var yard = new THREE.Mesh(new THREE.BoxGeometry(beam * 0.5, 0.025, 0.025), steelMat);
      yard.position.set(0, 1.55, halfLen * 0.02);
      body.add(yard);
      var mastLight = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 8),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(2.4, 2.1, 1.4) }));
      mastLight.position.set(0, 1.76, halfLen * 0.02);
      body.add(mastLight);

      // Main battery: a turret forward, and aft as well when there is room.
      function turretAt(z, facing) {
        var g = new THREE.Group();
        g.position.set(0, 0.6, z);
        g.add(new THREE.Mesh(new THREE.CylinderGeometry(beam * 0.24, beam * 0.27, 0.12, 14), steelMat));
        var box = new THREE.Mesh(new THREE.BoxGeometry(beam * 0.42, 0.17, beam * 0.5), superMat);
        box.position.y = 0.13;
        box.castShadow = true;
        g.add(box);
        [-0.07, 0.07].forEach(function (ox) {
          var b = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.032, beam * 0.62, 8), steelMat);
          b.rotation.x = Math.PI / 2;
          b.position.set(ox, 0.15, facing * beam * 0.45);
          g.add(b);
        });
        return g;
      }
      body.add(turretAt(halfLen * 0.58, 1));
      if (ship.len >= 3) body.add(turretAt(-halfLen * 0.62, -1));

      // Deck lights: a ship at night is lit, and this is what stops a grey
      // hull disappearing in the middle of a dark board.
      var deckLight = new THREE.PointLight(0xFFE0B0, 1, 4.2, 2);
      deckLight.power = 155;
      deckLight.position.set(0, 1.0, 0);
      body.add(deckLight);

      var record = {
        cells: ship.cells.slice(), anchor: anchor, body: body,
        centre: centre, bowDir: bowDir, segments: [], sinkT: Infinity,
        deckLight: deckLight
      };

      ship.cells.forEach(function (cellIdx) {
        // Where this cell sits along the hull, measured rather than indexed,
        // so it is right whichever way round the ship was built.
        var worldPos = new THREE.Vector3(cx(colOf(cellIdx)), 0, cz(rowOf(cellIdx)));
        var dz = worldPos.clone().sub(centre).dot(bowDir);

        var seg = new THREE.Group();
        seg.position.set(0, 0, dz);
        body.add(seg);

        /* Intact: this cell's own panel of deck plating, raised a little and
           ribbed, so the parts of a hull are countable before anything has
           happened to them as well as after. */
        var intact = new THREE.Group();
        var panel = new THREE.Mesh(new THREE.BoxGeometry(beam * 0.66, 0.07, CELL - 0.62), plateMat);
        panel.position.y = 0.56;
        panel.castShadow = true; panel.receiveShadow = true;
        intact.add(panel);
        for (var rb = -1; rb <= 1; rb++) {
          var rib = new THREE.Mesh(new THREE.BoxGeometry(beam * 0.7, 0.035, 0.05), steelMat);
          rib.position.set(0, 0.60, rb * (CELL - 0.62) * 0.33);
          intact.add(rib);
        }
        seg.add(intact);

        /* Wrecked: the plating is gone. A scorched ring, a black hole punched
           through it, torn shards around the lip, an ember in the bottom and a
           flame standing out of it. Deterministic jitter, so the damage does
           not crawl between frames. */
        var wreck = new THREE.Group();
        wreck.visible = false;
        var scorch = new THREE.Mesh(new THREE.BoxGeometry(beam * 0.84, 0.02, CELL - 0.42), scorchMat);
        scorch.position.y = 0.535;
        wreck.add(scorch);
        var hole = new THREE.Mesh(
          new THREE.BoxGeometry(beam * 0.5, 0.42, Math.max(0.2, CELL - 1.0)),
          new THREE.MeshBasicMaterial({ color: 0x000000 })
        );
        hole.position.y = 0.32;
        wreck.add(hole);
        for (var w = 0; w < 7; w++) {
          var shard = new THREE.Mesh(
            new THREE.BoxGeometry(0.08 + wreckRnd() * 0.26, 0.03 + wreckRnd() * 0.09, 0.08 + wreckRnd() * 0.28),
            wreckMat
          );
          shard.position.set(
            (wreckRnd() - 0.5) * beam * 0.95,
            0.5 + wreckRnd() * 0.22,
            (wreckRnd() - 0.5) * (CELL - 0.7)
          );
          shard.rotation.set(wreckRnd() * 1.5, wreckRnd() * 3.1, wreckRnd() * 1.5);
          wreck.add(shard);
        }
        var ember = new THREE.Mesh(
          new THREE.SphereGeometry(0.17, 10, 8),
          new THREE.MeshBasicMaterial({ color: new THREE.Color(3.4, 0.95, 0.24) })
        );
        ember.position.y = 0.42;
        wreck.add(ember);
        var flame = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTex, color: new THREE.Color(2.8, 1.15, 0.36), transparent: true,
          opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false
        }));
        flame.scale.set(1.1, 1.35, 1);
        flame.position.y = 0.85;
        wreck.add(flame);
        seg.add(wreck);

        record.segments.push({ cell: cellIdx, intact: intact, wreck: wreck, ember: ember, flame: flame });
      });

      // The ship goes down when the last of its own cells is hit — not when
      // the fleet is finished. The environment has no per-ship sunk state and
      // pays nothing for a sinking; this is the viewer saying what the water
      // would do, in the truth layer where it belongs.
      var lastT = 0;
      record.cells.forEach(function (c) {
        var ev = cellEvent[c];
        lastT = ev && ev.hit ? Math.max(lastT, ev.t) : Infinity;
      });
      record.sinkT = lastT;

      /* What is left on the surface afterwards: a slick, floating debris and
         steam. It lives in world space, not on the body, because it stays
         where the ship was. */
      var after = new THREE.Group();
      after.visible = false;
      truthGroup.add(after);
      var slick = new THREE.Sprite(new THREE.SpriteMaterial({
        map: poolTex, color: 0x120E0A, transparent: true, opacity: 0.0, depthWrite: false
      }));
      slick.scale.set(ship.len * CELL * 1.05, ship.len * CELL * 1.05, 1);
      slick.position.set(centre.x, 0.07, centre.z);
      after.add(slick);
      var debris = [];
      for (var d = 0; d < 9; d++) {
        var piece = new THREE.Mesh(
          new THREE.BoxGeometry(0.1 + wreckRnd() * 0.26, 0.05, 0.1 + wreckRnd() * 0.3),
          new THREE.MeshStandardMaterial({ color: 0x323C42, roughness: 0.8, metalness: 0.45 })
        );
        var off = (wreckRnd() - 0.5) * ship.len * CELL * 0.85;
        var side = (wreckRnd() - 0.5) * 1.2;
        piece.position.set(
          centre.x + bowDir.x * off - bowDir.z * side, 0.05,
          centre.z + bowDir.z * off + bowDir.x * side
        );
        piece.rotation.y = wreckRnd() * 3.1;
        after.add(piece);
        debris.push(piece);
      }
      record.after = after;
      record.slick = slick;
      record.debris = debris;

      ships.push(record);
    });

    /* A board whose occupancy no fleet of these lengths explains — a trace
       from a configuration this splitter does not understand — gets its ship
       cells drawn as plain blocks rather than as invented hulls. */
    if (!fleet.ships) {
      for (var bare = 0; bare < NCELL; bare++) {
        if (!TRUTH[bare]) continue;
        var block = new THREE.Mesh(
          new THREE.BoxGeometry(CELL - 0.7, 0.5, CELL - 0.7),
          new THREE.MeshStandardMaterial({ color: 0x35576A, roughness: 0.6, metalness: 0.6 })
        );
        block.position.set(cx(colOf(bare)), 0.25, cz(rowOf(bare)));
        block.castShadow = true; block.receiveShadow = true;
        truthGroup.add(block);
      }
    }

    /* --------------------------------------------------------- belief field
       One column per cell, standing well above the water. Height and
       brightness are the posterior chance that cell holds a ship; the label
       repeats the number, because nobody should have to estimate a probability
       off a brightness.

       Nothing in this section knows about hulls, damage or sinking. The agent
       is told one bit per shot, so this is the only place its knowledge
       lives. */
    var beliefCells = [];
    var labelsVisible = true;
    var beliefOn = true;

    function makeLabel() {
      var cv = document.createElement("canvas");
      cv.width = 128; cv.height = 64;
      var tex = new THREE.CanvasTexture(cv);
      var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, transparent: true, depthWrite: false, depthTest: false
      }));
      sprite.scale.set(0.98, 0.49, 1);
      sprite.renderOrder = 10;
      return { canvas: cv, tex: tex, sprite: sprite, last: -1 };
    }

    function drawLabel(entry, p) {
      var pct = Math.round(p * 100);
      if (entry.last === pct) return;
      entry.last = pct;
      var g = entry.canvas.getContext("2d");
      g.clearRect(0, 0, 128, 64);
      g.font = "600 40px ui-monospace, SFMono-Regular, Menlo, monospace";
      g.textAlign = "center";
      g.textBaseline = "middle";
      // Certainty is white; everything between is the belief colormap's teal.
      var hot = pct >= 100 || pct === 0;
      g.fillStyle = "rgba(0,0,0,0.55)";
      g.fillText(pct + "%", 65, 34);
      g.fillStyle = hot ? "#FFFFFF" : "#C8F0E6";
      g.fillText(pct + "%", 64, 32);
      entry.tex.needsUpdate = true;
    }

    var colGeo = new THREE.CylinderGeometry(0.25, 0.29, 1, 18, 1, true);
    var capGeo = new THREE.CircleGeometry(0.25, 18);
    var tileGeo = new THREE.PlaneGeometry(CELL - 0.24, CELL - 0.24);

    for (var ci = 0; ci < NCELL; ci++) {
      var bx = cx(colOf(ci)), bz = cz(rowOf(ci));

      // The lit tile: the same number, read off the board itself.
      var tile = new THREE.Mesh(tileGeo, new THREE.MeshBasicMaterial({
        color: C.belief, transparent: true, opacity: 0.0,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      tile.rotation.x = -Math.PI / 2;
      tile.position.set(bx, 0.05, bz);
      scene.add(tile);

      var col = new THREE.Mesh(colGeo, new THREE.MeshBasicMaterial({
        color: C.belief, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      }));
      col.position.set(bx, BELIEF_BASE, bz);
      scene.add(col);

      var cap = new THREE.Mesh(capGeo, new THREE.MeshBasicMaterial({
        color: C.belief, transparent: true, opacity: 0.8,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      }));
      cap.rotation.x = -Math.PI / 2;
      scene.add(cap);

      var label = makeLabel();
      scene.add(label.sprite);

      beliefCells.push({ tile: tile, col: col, cap: cap, label: label, x: bx, z: bz });
    }

    /* ------------------------------------------------------- probe outcomes
       What the agent actually knows: a pale ring where a probe found water, a
       burning mark where it found a ship. Both sit on the surface, between the
       ships below and the belief above. Nothing here says which ship was hit
       or whether one went down, because the observation is one bit. */
    var marks = [];
    var ringGeo = new THREE.RingGeometry(0.62, 0.74, 44);
    for (var mk = 0; mk < NCELL; mk++) {
      var mx = cx(colOf(mk)), mz = cz(rowOf(mk));
      var group = new THREE.Group();
      group.visible = false;
      scene.add(group);

      var disc = new THREE.Sprite(new THREE.SpriteMaterial({
        map: poolTex, color: C.miss, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      disc.scale.set(CELL * 0.8, CELL * 0.8, 1);
      disc.position.set(mx, 0.09, mz);

      var ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
        color: C.miss, transparent: true, opacity: 0.55,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(mx, 0.12, mz);

      group.add(disc);
      group.add(ring);

      marks.push({ group: group, disc: disc, ring: ring, x: mx, z: mz });
    }

    /* A pool of fire lights, not one per cell. A real light per board cell
       puts thirty-odd point lights in the scene, and three.js evaluates every
       one of them for every fragment of every standard material: on a five by
       five board that alone took the viewer from sixty frames a second to
       about one. The pool is parked on the most recently hit cells, which is
       where a viewer is looking, and the rest of the burning cells keep their
       ember and flame sprites — those are emissive geometry and cost nothing. */
    var FIRE_LIGHTS = 4;
    var fireLights = [];
    for (var fl2 = 0; fl2 < FIRE_LIGHTS; fl2++) {
      var fire = new THREE.PointLight(C.hit, 1, 5.5, 2);
      fire.power = 0;
      fire.position.set(0, 0.45, 0);
      scene.add(fire);
      fireLights.push(fire);
    }

    // The reticle over the cell being fired on now. Amber, the GIF's own probe
    // colour, and deliberately the only amber thing in the scene.
    var reticle = new THREE.Group();
    scene.add(reticle);
    var reticleRing = new THREE.Mesh(
      new THREE.RingGeometry(CELL * 0.40, CELL * 0.46, 56),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(2.6, 1.5, 0.28), transparent: true, opacity: 0.95,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    reticleRing.rotation.x = -Math.PI / 2;
    reticleRing.position.y = 0.16;
    reticle.add(reticleRing);
    for (var tk = 0; tk < 4; tk++) {
      var tickAngle = tk * Math.PI / 2;
      var tick = new THREE.Mesh(
        new THREE.BoxGeometry(0.34, 0.012, 0.055),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(2.6, 1.5, 0.28) })
      );
      tick.position.set(Math.cos(tickAngle) * CELL * 0.52, 0.16, Math.sin(tickAngle) * CELL * 0.52);
      tick.rotation.y = -tickAngle;
      reticle.add(tick);
    }

    /* ---------------------------------------------------------- the battery
       The agent is gunnery, not reconnaissance: an action is a round fired at
       a cell. The battery stands off the near corner, elevates onto the target
       and fires; the round arcs in and lands. */
    var GUN = new THREE.Vector3(-(HALF + 1.2), 0, HALF + 3.9);

    var deckPad = new THREE.Mesh(
      new THREE.BoxGeometry(3.0, 0.5, 3.0),
      new THREE.MeshStandardMaterial({ color: 0x141F25, roughness: 0.85, metalness: 0.25 })
    );
    deckPad.position.set(GUN.x, -0.1, GUN.z);
    deckPad.castShadow = true; deckPad.receiveShadow = true;
    scene.add(deckPad);

    var barbette = new THREE.Mesh(
      new THREE.CylinderGeometry(0.62, 0.74, 0.5, 20),
      new THREE.MeshStandardMaterial({ color: 0x27343B, roughness: 0.55, metalness: 0.7 })
    );
    barbette.position.set(GUN.x, 0.35, GUN.z);
    barbette.castShadow = true;
    scene.add(barbette);

    // The turret is modelled facing local +Z, so aiming is one lookAt at the
    // target and the barrels cannot come out mirrored.
    var turret = new THREE.Group();
    turret.position.set(GUN.x, 0.72, GUN.z);
    scene.add(turret);

    var turretMat = new THREE.MeshStandardMaterial({ color: 0x33434B, roughness: 0.45, metalness: 0.8 });
    var houseGeo = new THREE.BoxGeometry(1.0, 0.52, 1.15);
    var house = new THREE.Mesh(houseGeo, turretMat);
    house.castShadow = true;
    turret.add(house);
    turret.add(new THREE.LineSegments(new THREE.EdgesGeometry(houseGeo),
      new THREE.LineBasicMaterial({ color: 0x8FACBB, transparent: true, opacity: 0.4 })));

    var muzzles = [];
    [-0.22, 0.22].forEach(function (ox) {
      var barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.095, 2.3, 12), turretMat);
      barrel.rotation.x = Math.PI / 2;          // the cylinder's +Y becomes +Z
      barrel.position.set(ox, 0.06, 1.05);
      turret.add(barrel);
      var m = new THREE.Object3D();
      m.position.set(ox, 0.06, 2.25);
      turret.add(m);
      muzzles.push(m);
    });

    var flashSprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: new THREE.Color(3.4, 2.2, 0.9), transparent: true,
      opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
    }));
    // No light goes with the muzzle flash. Its sprite is emissive well above
    // 1.0, so the bright pass blooms it into the water on its own, and a real
    // light for two frames of every step is not worth what three.js charges
    // every fragment of every standard material for having it in the scene.
    flashSprite.scale.set(2.0, 2.0, 1);
    scene.add(flashSprite);

    // The round in flight, and the trail behind it, so the arc itself is
    // visible and not just the point it has reached.
    var shell = new THREE.Mesh(
      THREE.CapsuleGeometry ? new THREE.CapsuleGeometry(0.07, 0.2, 4, 8) : new THREE.SphereGeometry(0.11, 10, 8),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(2.2, 1.4, 0.7) })
    );
    shell.visible = false;
    scene.add(shell);
    var tracer = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: new THREE.Color(2.4, 1.3, 0.4), transparent: true,
      opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
    }));
    tracer.scale.set(1.15, 1.15, 1);
    scene.add(tracer);
    var TRAIL = 12;
    var trail = [];
    for (var tr = 0; tr < TRAIL; tr++) {
      var puff = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: new THREE.Color(1.5, 0.95, 0.45), transparent: true,
        opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
      }));
      scene.add(puff);
      trail.push(puff);
    }

    /* The impact. One plume and one flash between them: only the most recent
       round is ever showing, and both are driven straight off t, so they
       replay correctly when the scrubber is dragged backwards. */
    var plume = new THREE.Mesh(
      new THREE.CylinderGeometry(0.26, 0.62, 3.0, 18, 1, true),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: new THREE.Color(1.3, 1.9, 2.1), transparent: true,
        opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    plume.visible = false;
    scene.add(plume);

    var blast = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: new THREE.Color(3.0, 1.3, 0.4), transparent: true,
      opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false
    }));
    blast.visible = false;
    scene.add(blast);
    var blastLight = new THREE.PointLight(0xFFA347, 1, 12, 2);
    blastLight.power = 0;
    scene.add(blastLight);

    /* ----------------------------------------------------------------- air
       Spray hanging over the water, smoke off every burning cell, and steam
       where a ship went under. */
    var MOTES = 300;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(1357);
    for (var mo = 0; mo < MOTES; mo++) {
      motePos[mo * 3] = (moteSeed() - 0.5) * (SPAN + 3);
      motePos[mo * 3 + 1] = 0.1 + moteSeed() * 3.2;
      motePos[mo * 3 + 2] = (moteSeed() - 0.5) * (SPAN + 3);
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    scene.add(new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.04, map: partTex, color: 0xCBE4EC, transparent: true, opacity: 0.32,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    })));

    // Water and debris thrown up by a round as it lands.
    var SPRAY = 140;
    var sprayPos = new Float32Array(SPRAY * 3);
    var sprayVel = new Float32Array(SPRAY * 3);
    var sprayLife = new Float32Array(SPRAY);
    for (var sp = 0; sp < SPRAY; sp++) sprayPos[sp * 3 + 1] = -5;
    var sprayGeo = new THREE.BufferGeometry();
    sprayGeo.setAttribute("position", new THREE.BufferAttribute(sprayPos, 3));
    var spray = new THREE.Points(sprayGeo, new THREE.PointsMaterial({
      size: 0.16, map: partTex, color: 0xDCF2F8, transparent: true, opacity: 0.55,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    }));
    scene.add(spray);
    var sprayCursor = 0;
    var burstRnd = mulberry(8080);

    function burst(x, z, hot) {
      spray.material.color.setHex(hot ? 0xFFC089 : 0xDCF2F8);
      for (var k = 0; k < 34; k++) {
        var idx = sprayCursor++ % SPRAY;
        var a = burstRnd() * Math.PI * 2, r = burstRnd() * 0.45;
        sprayPos[idx * 3] = x + Math.cos(a) * r;
        sprayPos[idx * 3 + 1] = 0.1;
        sprayPos[idx * 3 + 2] = z + Math.sin(a) * r;
        sprayVel[idx * 3] = Math.cos(a) * (0.6 + burstRnd() * 1.2);
        sprayVel[idx * 3 + 1] = (hot ? 2.2 : 1.7) + burstRnd() * 2.1;
        sprayVel[idx * 3 + 2] = Math.sin(a) * (0.6 + burstRnd() * 1.2);
        sprayLife[idx] = 1;
      }
    }

    // Smoke. One plume per burning cell, anchored where the cell is, because
    // it stays over the water after the hull under it has gone.
    var SMOKE_PER = 9;
    var SMOKE = Math.max(1, SHIP_CELLS) * SMOKE_PER;
    var smokePos = new Float32Array(SMOKE * 3);
    for (var sk = 0; sk < SMOKE; sk++) smokePos[sk * 3 + 1] = -5;
    var smokeGeo = new THREE.BufferGeometry();
    smokeGeo.setAttribute("position", new THREE.BufferAttribute(smokePos, 3));
    scene.add(new THREE.Points(smokeGeo, new THREE.PointsMaterial({
      size: 0.85, map: partTex, color: 0x2A2723, transparent: true, opacity: 0.5,
      blending: THREE.NormalBlending, depthWrite: false, sizeAttenuation: true
    })));

    // Steam boiling off the water over a wreck.
    var STEAM = 60;
    var steamPos = new Float32Array(STEAM * 3);
    for (var st = 0; st < STEAM; st++) steamPos[st * 3 + 1] = -5;
    var steamGeo = new THREE.BufferGeometry();
    steamGeo.setAttribute("position", new THREE.BufferAttribute(steamPos, 3));
    var steam = new THREE.Points(steamGeo, new THREE.PointsMaterial({
      size: 0.5, map: partTex, color: 0x9FB6BE, transparent: true, opacity: 0.32,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    }));
    scene.add(steam);

    core.linearize();

    var overlay = buildOverlay(document.getElementById("viewer"));
    overlay.toggles.belief.addEventListener("change", function () {
      beliefOn = overlay.toggles.belief.checked;
    });
    overlay.toggles.truth.addEventListener("change", function () {
      truthGroup.visible = overlay.toggles.truth.checked;
    });
    overlay.toggles.labels.addEventListener("change", function () {
      labelsVisible = overlay.toggles.labels.checked;
    });
    overlay.split.textContent = fleet.ships
      ? (fleet.solutions === 1
        ? "one fleet fits this occupancy; the state stores no ship identities"
        : (fleet.capped ? "500+" : fleet.solutions) +
          " fleets fit this occupancy — one is drawn")
      : "no fleet of these lengths explains this board; ship cells drawn as blocks";

    /* Labels are laid out in screen space, not hoped about in world space.
       Every number is projected, sized in pixels and tested against the ones
       already placed; a loser is dropped rather than drawn on top of a winner.
       Its column still carries the value, so nothing is lost but the digits.
       This is what makes "no two numbers overlap" a property of the page and
       not a property of one camera angle. */
    var viewSize = new THREE.Vector2();
    var projTmp = new THREE.Vector3();
    var placedBoxes = [];

    function placeLabels(queue) {
      renderer.getSize(viewSize);
      var H = viewSize.y || 1;
      var tanHalfFov = Math.tan(core.camera.fov * Math.PI / 360);
      core.camera.updateMatrixWorld();

      var live = [];
      for (var i = 0; i < queue.length; i++) {
        var q = queue[i];
        if (!labelsVisible || !beliefOn || q.alpha <= 0.02) {
          q.entry.label.sprite.material.opacity = 0;
          continue;
        }
        projTmp.copy(q.entry.label.sprite.position);
        var dist = core.camera.position.distanceTo(projTmp);
        projTmp.project(core.camera);
        if (projTmp.z > 1) { q.entry.label.sprite.material.opacity = 0; continue; }
        // World units to pixels at this depth, for a sprite of this world size.
        var perUnit = H / (2 * dist * tanHalfFov);
        q.h = q.entry.label.sprite.scale.y * perUnit;
        q.w = q.entry.label.sprite.scale.x * perUnit;
        q.sx = (projTmp.x * 0.5 + 0.5) * viewSize.x;
        q.sy = (-projTmp.y * 0.5 + 0.5) * viewSize.y;
        q.dist = dist;
        live.push(q);
      }

      // Most informative first, nearer wins a tie: the label someone is most
      // likely to be reading should not be the one that gets dropped.
      live.sort(function (a, b) { return (b.score - a.score) || (a.dist - b.dist); });

      placedBoxes.length = 0;
      var pad = 3;
      for (var j = 0; j < live.length; j++) {
        var L = live[j];
        var x0 = L.sx - L.w / 2 - pad, x1 = L.sx + L.w / 2 + pad;
        var y0 = L.sy - L.h / 2 - pad, y1 = L.sy + L.h / 2 + pad;
        var blocked = false;
        for (var b2 = 0; b2 < placedBoxes.length; b2++) {
          var P = placedBoxes[b2];
          if (x0 < P[2] && x1 > P[0] && y0 < P[3] && y1 > P[1]) { blocked = true; break; }
        }
        if (blocked) {
          L.entry.label.sprite.material.opacity = 0;
        } else {
          placedBoxes.push([x0, y0, x1, y1]);
          // Small and far reads as noise, so it fades rather than crowding.
          var small = clamp((L.h - 9) / 7, 0, 1);
          L.entry.label.sprite.material.opacity = L.alpha * (0.25 + 0.75 * small);
        }
      }
    }

    var aimPoint = new THREE.Vector3();
    var shellFrom = new THREE.Vector3();
    var shellTo = new THREE.Vector3();
    var vTmp = new THREE.Vector3();
    var vTmp2 = new THREE.Vector3();
    var lastResolved = -1;

    /* One sample of the episode at continuous step index t.

       The convention is the GIF's: frame i shows the board the agent chose
       from, and frame i's shot resolves partway to frame i + 1. Everything the
       agent knows flips at that one instant — the mark, the readout, the next
       aim and the posterior move together. Letting the belief update while the
       readout still showed the old step is the one thing in this scene that
       could be read as a lie. */
    function sampleAt(t) {
      var i0 = Math.floor(clamp(t, 0, N_STEPS - 1));
      var i1 = Math.min(i0 + 1, N_STEPS - 1);
      var frac = clamp(t - i0, 0, 1);
      var resolved = frac >= RESOLVE ? 1 : 0;
      return {
        i0: i0, i1: i1, frac: frac,
        step: i0 + resolved,
        // The posterior snaps in as the reading comes back and finishes
        // exactly when the round lands.
        reveal: smoothstep(clamp((frac - 0.16) / (RESOLVE - 0.16), 0, 1))
      };
    }

    // Where the battery points, and how far the round has flown, at time t.
    function gunneryAt(t) {
      var i = Math.floor(clamp(t, 0, N_STEPS - 1));
      var frac = clamp(t - i, 0, 1);
      var here = frames[i];
      // After the round lands the battery is already training on the next
      // cell: that is the target the reticle marks too.
      var aimFrame = frac >= RESOLVE ? frames[Math.min(i + 1, N_STEPS - 1)] : here;
      var firing = here.action !== null && frac >= 0.06 && frac < RESOLVE;
      return {
        aimCell: aimFrame.action,
        shotCell: here.action,
        // 0 at the muzzle, 1 at the target.
        flight: firing ? clamp((frac - 0.06) / (RESOLVE - 0.06), 0, 1) : -1,
        muzzleAge: here.action === null ? 99 : (frac - 0.06) / 0.09
      };
    }

    function updateBattery(gun, aimCell) {
      aimPoint.set(cx(colOf(aimCell)), 2.6, cz(rowOf(aimCell)));
      // lookAt points the turret's local +Z at the aim point. The raised
      // point is the barrel's elevation: guns lob, they do not shoot flat.
      turret.lookAt(aimPoint);
      turret.updateMatrixWorld(true);
      muzzles[0].getWorldPosition(vTmp);
      muzzles[1].getWorldPosition(vTmp2);
      shellFrom.copy(vTmp).add(vTmp2).multiplyScalar(0.5);

      var mFlash = gun.shotCell === null ? 0 : clamp(1 - Math.abs(gun.muzzleAge), 0, 1);
      flashSprite.material.opacity = mFlash * 0.95;
      flashSprite.position.copy(shellFrom);
      flashSprite.scale.setScalar(1.2 + mFlash * 1.6);

      if (gun.flight < 0) {
        shell.visible = false;
        tracer.material.opacity = 0;
        for (var tq = 0; tq < TRAIL; tq++) trail[tq].material.opacity = 0;
        return;
      }

      shellTo.set(cx(colOf(gun.shotCell)), 0.12, cz(rowOf(gun.shotCell)));
      var u = gun.flight;
      shell.visible = true;
      var arc = ARC_H * 4 * u * (1 - u);
      shell.position.set(
        lerp(shellFrom.x, shellTo.x, u),
        lerp(shellFrom.y, shellTo.y, u) + arc,
        lerp(shellFrom.z, shellTo.z, u)
      );
      // Nose along the flight path: sample the arc a moment ahead, look at it.
      var u2 = Math.min(1, u + 0.02);
      var arc2 = ARC_H * 4 * u2 * (1 - u2);
      vTmp.set(
        lerp(shellFrom.x, shellTo.x, u2),
        lerp(shellFrom.y, shellTo.y, u2) + arc2,
        lerp(shellFrom.z, shellTo.z, u2)
      );
      shell.lookAt(vTmp);
      shell.rotateX(Math.PI / 2);      // the capsule's long axis is +Y
      tracer.position.copy(shell.position);
      tracer.material.opacity = 0.85;
      for (var tp = 0; tp < TRAIL; tp++) {
        var ut = u - (tp + 1) * 0.032;
        if (ut <= 0) { trail[tp].material.opacity = 0; continue; }
        var arcT = ARC_H * 4 * ut * (1 - ut);
        trail[tp].position.set(
          lerp(shellFrom.x, shellTo.x, ut),
          lerp(shellFrom.y, shellTo.y, ut) + arcT,
          lerp(shellFrom.z, shellTo.z, ut)
        );
        trail[tp].material.opacity = 0.42 * (1 - tp / TRAIL);
        trail[tp].scale.setScalar(0.85 * (1 - tp / TRAIL) + 0.2);
      }
    }

    function updateImpact(t) {
      var impact = -1;
      for (var ii = N_STEPS - 1; ii >= 0; ii--) {
        if (frames[ii].action !== null && resolveT(ii) <= t) { impact = ii; break; }
      }
      if (impact < 0) {
        plume.visible = false;
        blast.visible = false;
        blastLight.power = 0;
      } else {
        var age = t - resolveT(impact);
        var ic = frames[impact].action;
        var ix = cx(colOf(ic)), iz = cz(rowOf(ic));
        var wasHit = !!frames[impact].hit;

        // A miss throws a column of water; a hit throws fire.
        var pl = clamp(1 - age / 0.75, 0, 1);
        plume.visible = !wasHit && pl > 0.01;
        if (plume.visible) {
          var grow = smoothstep(clamp(age / 0.22, 0, 1));
          plume.position.set(ix, 1.5 * grow, iz);
          plume.scale.set(0.5 + grow * 0.9, grow * 1.15, 0.5 + grow * 0.9);
          plume.material.opacity = 0.42 * pl * pl;
        }
        var bl = clamp(1 - age / 0.45, 0, 1);
        blast.visible = wasHit && bl > 0.01;
        if (blast.visible) {
          blast.position.set(ix, 0.65, iz);
          blast.scale.setScalar(1.4 + (1 - bl) * 3.4);
          blast.material.opacity = bl * bl;
        }
        blastLight.position.set(ix, 0.8, iz);
        blastLight.power = wasHit ? 5200 * bl * bl : 900 * pl * pl;
      }

      // One splash of spray per round, thrown as it lands.
      if (impact !== lastResolved) {
        var forward = impact > lastResolved;
        lastResolved = impact;
        if (forward && impact >= 0 && !reduceMotion) {
          burst(cx(colOf(frames[impact].action)), cz(rowOf(frames[impact].action)), !!frames[impact].hit);
        }
      }
    }

    /* What the agent has observed, and strictly that: hit or miss. Nothing
       here changes when a ship goes down, because nothing tells the agent. */
    function updateMarks(t, elapsed) {
      var burning = [];
      for (var m = 0; m < NCELL; m++) {
        var M = marks[m];
        var ev = cellEvent[m];
        var on = !!ev && t >= ev.t;
        M.group.visible = on;
        if (!on) continue;
        if (ev.hit) {
          M.disc.material.color.setRGB(2.4, 0.72, 0.42);
          M.ring.material.color.setRGB(2.6, 0.85, 0.5);
          M.disc.material.opacity = 0.6;
          M.ring.material.opacity = 0.85;
          burning.push({ mark: M, cell: m, lit: ev.t });
        } else {
          M.disc.material.color.setRGB(1.05, 1.5, 1.7);
          M.ring.material.color.setRGB(1.2, 1.75, 2.0);
          M.disc.material.opacity = 0.42;
          M.ring.material.opacity = 0.9;
        }
      }

      // The pool goes to the newest fires, which are the ones being looked at.
      burning.sort(function (a, b) { return b.lit - a.lit; });
      for (var fi2 = 0; fi2 < fireLights.length; fi2++) {
        var entry = burning[fi2];
        if (!entry) { fireLights[fi2].power = 0; continue; }
        fireLights[fi2].position.set(entry.mark.x, 0.45, entry.mark.z);
        var flicker = reduceMotion
          ? 1
          : 0.75 + Math.sin(elapsed * 11 + entry.cell) * 0.15 +
            Math.sin(elapsed * 27 + entry.cell * 3) * 0.1;
        fireLights[fi2].power = 130 * flicker;
      }
    }

    function updateTruth(t, elapsed) {
      var smokeCursor = 0;
      var steamCursor = 0;
      for (var si = 0; si < ships.length; si++) {
        var S = ships[si];

        // Each cell of the hull carries its own state, so a half-wrecked ship
        // reads as exactly which halves are gone.
        for (var sg = 0; sg < S.segments.length; sg++) {
          var seg = S.segments[sg];
          var sev = cellEvent[seg.cell];
          var wrecked = !!sev && sev.hit && t >= sev.t;
          seg.intact.visible = !wrecked;
          seg.wreck.visible = wrecked;
          if (!wrecked) continue;

          var grow2 = smoothstep(clamp((t - sev.t) / 0.3, 0, 1));
          var fl = reduceMotion ? 1 : 0.8 + Math.sin(elapsed * 9 + seg.cell) * 0.2;
          seg.flame.scale.set(0.75 * grow2 * fl, (0.95 + 0.35 * fl) * grow2, 1);
          seg.ember.scale.setScalar(0.7 + 0.3 * fl);

          // Fire is doused as the cell goes under.
          var sunk = clamp((t - S.sinkT) / 1.1, 0, 1);
          if (S.deckLight) S.deckLight.power = 155 * (1 - sunk);
          seg.flame.material.opacity = 0.9 * (1 - sunk);
          seg.ember.visible = sunk < 0.7;

          // Smoke rises from where the cell is, and outlives the hull.
          if (!reduceMotion) {
            var sx = cx(colOf(seg.cell)), sz = cz(rowOf(seg.cell));
            for (var q = 0; q < SMOKE_PER && smokeCursor < SMOKE; q++) {
              var ph = (elapsed * 0.22 + q / SMOKE_PER + seg.cell * 0.13) % 1;
              var idx3 = smokeCursor++ * 3;
              smokePos[idx3] = sx + Math.sin(elapsed * 0.5 + q) * ph * 1.4;
              smokePos[idx3 + 1] = 0.5 + ph * 3.4;
              smokePos[idx3 + 2] = sz + Math.cos(elapsed * 0.4 + q * 2) * ph * 1.1;
            }
          }
        }

        var p = clamp((t - S.sinkT) / 1.1, 0, 1);
        var e = smoothstep(p);
        S.body.position.y = -3.1 * e * e;
        S.body.rotation.z = 0.62 * e;          // lists to starboard
        S.body.rotation.x = 0.34 * e * e;      // and goes down by the bow
        S.body.visible = p < 1;

        S.after.visible = p > 0.18;
        if (!S.after.visible) continue;
        var af = smoothstep(clamp((p - 0.18) / 0.5, 0, 1));
        S.slick.material.opacity = 0.55 * af;
        S.slick.scale.setScalar((S.cells.length * CELL * 0.75) * (0.6 + af * 0.55));
        for (var db = 0; db < S.debris.length; db++) {
          S.debris[db].visible = af > 0.3;
          S.debris[db].position.y = 0.05 + (reduceMotion ? 0 : Math.sin(elapsed * 1.6 + db) * 0.02);
        }
        if (!reduceMotion) {
          var stAmt = clamp(1.6 - p, 0, 1);
          for (var sq = 0; sq < 20 && steamCursor < STEAM; sq++) {
            var sph = (elapsed * 0.35 + sq / 20 + si * 0.3) % 1;
            var i3 = steamCursor++ * 3;
            steamPos[i3] = S.centre.x + (sq % 5 - 2) * 0.32;
            steamPos[i3 + 1] = stAmt > 0.02 ? 0.1 + sph * 1.6 : -5;
            steamPos[i3 + 2] = S.centre.z + (Math.floor(sq / 5) - 2) * 0.32;
          }
          steam.material.opacity = 0.3 * stAmt;
        }
      }
      for (var sc = smokeCursor; sc < SMOKE; sc++) smokePos[sc * 3 + 1] = -5;
      for (var tc = steamCursor; tc < STEAM; tc++) steamPos[tc * 3 + 1] = -5;
      smokeGeo.attributes.position.needsUpdate = true;
      steamGeo.attributes.position.needsUpdate = true;
    }

    /* A column stands only where uncertainty is left. Once a cell has been
       fired on its marginal is 0 or 1 *by observation* rather than by
       inference, so a full-height column over it would redraw the shot already
       marked on the water — and stand in front of the burning hull that shot
       produced. The column collapses as the round lands and the flat mark
       takes over. A cell the belief has *deduced* is certain keeps its column,
       because that is inference and worth seeing. So the board quietens as the
       episode runs, from every cell to the handful still in doubt. */
    function updateBelief(t, sample) {
      var from = marginals[sample.i0];
      var to = marginals[sample.i1];
      var labelQueue = [];
      var inDoubt = 0;

      for (var bc = 0; bc < NCELL; bc++) {
        var E = beliefCells[bc];
        if (!from || !to) {
          E.col.visible = E.cap.visible = E.tile.visible = false;
          E.label.sprite.material.opacity = 0;
          continue;
        }
        var pb = lerp(from[bc], to[bc], sample.reveal);
        var bev = cellEvent[bc];
        var collapse = bev && t >= bev.t ? smoothstep(clamp((t - bev.t) / 0.22, 0, 1)) : 0;
        var live = 1 - collapse;

        var standing = live > 0.001;
        E.col.visible = E.cap.visible = E.tile.visible = standing && beliefOn;
        if (!standing) { E.label.sprite.material.opacity = 0; continue; }
        if (pb > 0.001 && pb < 0.999) inDoubt++;

        var hgt = Math.max(0.02, pb * BELIEF_MAX * live);
        E.col.scale.set(live, hgt, live);
        E.col.position.y = BELIEF_BASE + hgt / 2;
        E.cap.scale.set(live, live, live);
        E.cap.position.set(E.x, BELIEF_BASE + hgt, E.z);
        E.label.sprite.position.set(E.x, BELIEF_BASE + hgt + 0.36, E.z);
        drawLabel(E.label, pb);

        /* Brightness is the probability. The colour runs from the colormap's
           pale teal at the low end to a value well above 1.0 at certainty, so
           a cell the belief is sure about blooms and one it doubts does not. */
        var hot = 0.25 + pb * pb * 3.4;
        E.col.material.color.setRGB(0.28 * hot, 0.80 * hot, 0.74 * hot);
        E.col.material.opacity = (0.16 + pb * 0.5) * live;
        E.cap.material.color.copy(E.col.material.color);
        E.cap.material.opacity = (0.3 + pb * 0.65) * live;
        var tileHot = 0.4 + pb * 1.6;
        E.tile.material.color.setRGB(0.20 * tileHot, 0.62 * tileHot, 0.58 * tileHot);
        E.tile.material.opacity = (0.05 + pb * 0.30) * live;

        /* Certainty outranks vagueness when two numbers want the same patch of
           screen: a deduced 100% or 0% is the belief telling you something, a
           28% is the belief shrugging. */
        labelQueue.push({
          entry: E,
          alpha: clamp(0.35 + pb * 0.9, 0, 1) * live,
          score: pb < 0.02 ? 0.5 : pb
        });
      }
      placeLabels(labelQueue);
      return inDoubt;
    }

    return {
      steps: N_STEPS,

      /* Framing scales with the board, because a trace decides how big it is:
         the fixed distance that frames a five-square board leaves a ten-square
         one running off the canvas. The constant term is the margin for the
         props, which do not scale with the board — a lamp mast is the same
         height whatever the board, so a small board needs proportionally more
         headroom, not less.

         Both are further out than the standalone preview's were, because the
         shared rig aims at y = 0.3 while this scene's content runs from the
         hulls' waterline up to the belief columns at y ≈ 3.3. Aiming low with
         tall content pushes everything into the top of the frame, so the
         camera backs off instead of the rig being changed for one scene. */
      camera: {
        board: [0, SPAN * 1.38 + 1.8, SPAN * 1.30 + 1.8],
        top: [0, SPAN * 1.82 + 1.8, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed) {
        var sample = sampleAt(t);
        var gun = gunneryAt(t);

        // The terminal step has no action, so the battery keeps the last cell
        // it fired on rather than snapping to the origin.
        var aimCell = gun.aimCell === null ? gun.shotCell : gun.aimCell;
        if (aimCell === null) {
          for (var back = N_STEPS - 1; back >= 0 && aimCell === null; back--) {
            aimCell = frames[back].action;
          }
        }
        if (aimCell === null) aimCell = 0;

        updateBattery(gun, aimCell);
        updateImpact(t);
        updateMarks(t, elapsed);
        updateTruth(t, elapsed);

        reticle.visible = gun.aimCell !== null;
        if (reticle.visible) {
          reticle.position.set(cx(colOf(gun.aimCell)), 0, cz(rowOf(gun.aimCell)));
          var pulse = reduceMotion ? 1 : 0.78 + Math.sin(elapsed * 5.0) * 0.22;
          reticleRing.material.opacity = 0.55 + 0.4 * pulse;
          reticle.scale.setScalar(1 + (reduceMotion ? 0 : Math.sin(elapsed * 5.0) * 0.015));
        }

        var inDoubt = updateBelief(t, sample);

        /* Move the one shadow-casting lamp to the dock lamp nearest the target
           and take that much light back off that lamp's own point light, so
           the shadow follows the action without brightening one corner. */
        var tx = cx(colOf(aimCell)), tz = cz(rowOf(aimCell));
        var nearest = null, nd = Infinity;
        for (var bl2 = 0; bl2 < lampLights.length; bl2++) {
          var L = lampLights[bl2];
          L.light.power = LAMP_LUMENS;
          var d2 = (L.x - tx) * (L.x - tx) + (L.z - tz) * (L.z - tz);
          if (d2 < nd) { nd = d2; nearest = L; }
        }
        if (nearest) {
          shadowLight.position.set(nearest.x, 2.7, nearest.z);
          shadowLight.target.position.set(tx, 0.4, tz);
          shadowLight.target.updateMatrixWorld();
          var share = 0.58 * clamp(1 - (Math.sqrt(nd) - 3.0) / 11.0, 0.1, 1);
          shadowLight.power = LAMP_LUMENS * share;
          nearest.light.power = LAMP_LUMENS * (1 - share);
        }

        if (!reduceMotion) {
          for (var mi2 = 0; mi2 < MOTES; mi2++) {
            motePos[mi2 * 3] += Math.sin(elapsed * 0.3 + mi2) * 0.0015;
            motePos[mi2 * 3 + 1] += 0.0018;
            if (motePos[mi2 * 3 + 1] > 3.4) motePos[mi2 * 3 + 1] = 0.1;
          }
          moteGeo.attributes.position.needsUpdate = true;

          for (var sj = 0; sj < SPRAY; sj++) {
            if (sprayLife[sj] <= 0) continue;
            sprayLife[sj] -= dt * 1.5;
            sprayPos[sj * 3] += sprayVel[sj * 3] * dt;
            sprayPos[sj * 3 + 1] += sprayVel[sj * 3 + 1] * dt;
            sprayPos[sj * 3 + 2] += sprayVel[sj * 3 + 2] * dt;
            sprayVel[sj * 3 + 1] -= 4.2 * dt;
            if (sprayLife[sj] <= 0 || sprayPos[sj * 3 + 1] < 0.02) {
              sprayLife[sj] = 0;
              sprayPos[sj * 3 + 1] = -5;
            }
          }
          sprayGeo.attributes.position.needsUpdate = true;

          // The sea itself: scroll the swell, so the water is never a photo.
          seaNormal.offset.y = (elapsed * 0.006) % 1;
          seaNormal.offset.x = (elapsed * 0.003) % 1;
          openSeaNormal.offset.y = (elapsed * 0.010) % 1;
        }

        var frame = frames[sample.step] || frames[N_STEPS - 1];

        /* Cells, never ships. The agent is not told which ship it hit, nor
           that one has gone down, so this line cannot mention either. */
        overlay.known.textContent =
          frame.knownHits + " of " + SHIP_CELLS + " ship cells hit · " +
          (frame.support === null || frame.support === undefined
            ? inDoubt + " cells in doubt"
            : frame.support.toLocaleString("en-US") + " layouts consistent");

        /* The chase camera is the gunner's: the core rig places itself behind
           `follow` along its heading and looks ahead of it, so handing it the
           target cell and the line of fire puts the camera on the gun's side
           of the cell being fired on, looking across it. The rig is left
           alone; only what it follows is this scene's business. */
        var lineX = tx - GUN.x, lineZ = tz - GUN.z;
        var heading = Math.atan2(lineZ, lineX);

        return {
          follow: { x: tx, z: tz, heading: heading },
          step: sample.step,
          // The outcome belongs here because it is the whole observation: one
          // bit, hit or miss, and never which ship or whether one sank.
          action: frame.action === null
            ? "—"
            : "probe cell " + frame.action + (frame.hit ? " → HIT" : " → MISS"),
          // A board has rows and columns, not an x and a y, and the shared
          // player lets a scene say so.
          pos: "row " + rowOf(aimCell) + "  col " + colOf(aimCell),
          x: colOf(aimCell),
          y: rowOf(aimCell),
          reward: frame.reward,
          ret: frame.ret,
          belief: "P(ship) per cell, " + beliefSource
        };
      }
    };
  }

  V.scenes["battleship.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // scene's lamp power; it is not a knob to remove.
    exposure: 0.130
  };
})(window);
