/* SPDX-License-Identifier: MIT
 *
 * LaserTag scene module — discrete grid and continuous arena, one payload kind.
 *
 * Builds the arena from a trace's `payload.world` block and moves it from the
 * trace's recorded states, ranges and beliefs. Nothing here is invented: there
 * is no fallback episode, no hand-placed waypoint and no synthetic belief.
 *
 * The eight beams are the point of the page, and this module does not own
 * them. It never casts a ray. Every beam ends at
 *
 *     robot[i] + direction[d] * payload.laser_ranges[i][d]
 *
 * where all three come from the trace, and the directions are the
 * environment's own table. Between two steps a beam is interpolated from one
 * true beam to the next, so it is exact at every step boundary and is never
 * re-derived in between. That matters because the two copies have disagreed
 * before: the discrete GIF's ray walk ignores the opponent, while the
 * observation model stops at it. Drawing from the recorded ranges is how this
 * viewer cannot drift from the model the agent is actually given.
 *
 * The green beam is the truth the agent never sees; the amber bead on it sits
 * at the noisy range it actually received at that same state. The gap between
 * them is the whole difficulty of the environment, made visible.
 *
 * The two variants keep their own frames and their own direction tables. The
 * discrete table steps in whole cells as (drow, dcol); the continuous one is
 * unit (dx, dy). They are related, but the arenas they index are stored
 * differently, so each is drawn in the frame its own source uses and neither
 * is converted into the other.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* Every colour is one the LaserTag GIF renderer already uses
     (laser_tag_renderer.py, laser_tag_assets.py), so the viewer reads as the
     same world as the GIF. */
  var COLORS = {
    laser: 0x99CC99,   // LASER_COLOR
    robot: 0xD32F2F,
    opponent: 0x1976D2,
    robotBelief: 0xF6B3B3,   // ROBOT_BELIEF_COLOR
    oppBelief: 0xCEE8F0,   // OPPONENT_BELIEF_COLOR
    robotPath: 0xFFA0A0,
    oppPath: 0x90CAF9,
    lamp: 0xCFE3E8
  };

  // Lumens on the gantry. Real inverse-square falloff is what lets the corners
  // between lamps go dark on their own instead of being painted dark.
  var LAMP_LUMENS = 2500;

  // Height of the sensor drum's apertures, shared by the lenses and the beams.
  var BEAM_Y = 0.50;

  // Cap on drawn belief particles. A trace may carry several hundred; beyond
  // this it is draw cost with no extra information at this scale.
  var MAX_DRAWN_PARTICLES = 320;
  // Cap on drawn belief cells in the discrete variant.
  var MAX_DRAWN_CELLS = 34;

  var BEAM_COLOR = new THREE.Color(1.2, 2.6, 1.2);
  var BEAM_HIT_COLOR = new THREE.Color(3.4, 1.2, 0.5);
  var OPP_BELIEF_HOT = new THREE.Color(0.85, 2.05, 2.90);   // CEE8F0, at its hue
  var ROBOT_BELIEF_HOT = new THREE.Color(2.90, 1.05, 1.05); // F6B3B3, at its hue

  function smooth(e0, e1, x) {
    var t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  function hash2(x, z) {
    var v = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
    return v - Math.floor(v);
  }

  /* The floor is the environment's own steel: metal_texture's base colour with
     the same three octaves of blotch, the same fine grain and the same rivets.
     A normal map is derived from a blurred copy, so every scratch catches the
     overhead lamps from the side the lamp is actually on. */
  function groundCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(260202);

    g.fillStyle = "#20272B";                 // metal_texture base (32, 39, 43)
    g.fillRect(0, 0, s, s);

    /* metal_texture's (5, 12), (18, 9), (75, 5). Each octave is drawn tiny and
       scaled up with the browser's smoothing, which is what Pillow's BICUBIC
       resize does — painting the cells at full size gives hard squares, which
       is not what steel looks like. */
    [[5, 12], [18, 9], [75, 5]].forEach(function (oct) {
      var cells = oct[0], strength = oct[1];
      var small = document.createElement("canvas");
      small.width = small.height = cells;
      var sg = small.getContext("2d");
      for (var a = 0; a < cells; a++) {
        for (var b = 0; b < cells; b++) {
          var v = (rnd() - 0.5) * 2 * strength;
          sg.fillStyle = "rgba(" + (v > 0 ? "168,190,198," : "4,7,9,") +
            (Math.abs(v) / (v > 0 ? 130 : 70)).toFixed(3) + ")";
          sg.fillRect(a, b, 1, 1);
        }
      }
      g.imageSmoothingEnabled = true;
      g.imageSmoothingQuality = "high";
      g.drawImage(small, 0, 0, s, s);
    });

    // Brushed grain: fine horizontal strokes, the tell of rolled steel plate.
    for (var i = 0; i < 5200; i++) {
      var y = rnd() * s, len = 20 + rnd() * 190;
      g.strokeStyle = rnd() > 0.5 ? "rgba(196,214,220,0.018)" : "rgba(3,6,8,0.06)";
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(rnd() * s, y);
      g.lineTo(rnd() * s + len, y + (rnd() - 0.5) * 2);
      g.stroke();
    }
    // Rivets and scuffs, so the plate has something for the lamps to catch.
    for (var p = 0; p < 300; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.5 + rnd() * 3.5;
      g.fillStyle = "rgba(6,9,11,0.5)";
      g.beginPath(); g.arc(px + pr * 0.3, py + pr * 0.3, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(163,169,166,0.12)";  // rivet highlight from draw_wall
      g.beginPath(); g.arc(px, py, pr * 0.8, 0, Math.PI * 2); g.fill();
    }
    return cv;
  }

  // The one-unit cell grid, painted onto the colour only: it carries the
  // world's scale and must not emboss the plate.
  function paintGrid(cv, cellsA, cellsB) {
    var s = cv.width;
    var g = cv.getContext("2d");
    g.lineWidth = 2;
    g.strokeStyle = "rgba(122,140,146,0.16)";
    var k;
    for (k = 0; k <= cellsA; k++) {
      var p = (k / cellsA) * s;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
    }
    for (k = 0; k <= cellsB; k++) {
      var q = (k / cellsB) * s;
      g.beginPath(); g.moveTo(0, q); g.lineTo(s, q); g.stroke();
    }
  }

  /* One hazard's broken ground: a displaced polar mesh over Voronoi plates,
     each lifted and tilted, so plate boundaries come out as cliffs and the
     seams sink into fissures.

     It is a shallow pad sitting ON the floor rather than a pit cut into it,
     because the floor is one opaque plane and anything below y = 0 would be
     hidden by it. Nothing is allowed past HAZ_PAD_R, which is inside the
     environment's true radius: debris spilling over the red ring would
     misreport where the penalty applies.

     Rock albedo is matched to the steel floor's on purpose. A hazard costs
     reward; it does not blind the agent. In a lamp-lit scene a darker patch of
     ground reads as "the light does not reach here", which is a claim about
     the observation model and not about the penalty. So the ground inside a
     ring takes the same light as the ground outside it, and the danger is
     carried entirely by geometry and the rim. */
  var ROCK_LIT = new THREE.Color(0x393730).convertSRGBToLinear();
  var ROCK_DARK = new THREE.Color(0x100F0D).convertSRGBToLinear();
  var ROCK_SCORCH = new THREE.Color(0x3A2A22).convertSRGBToLinear();
  var HAZ_BASE = 0.035;        // clearance over the steel floor
  var HAZ_RINGS = 15, HAZ_SPOKES = 76;

  function hazardTerrain(seed, radius, rubbleMat, emberMat, shardGeos) {
    var padR = radius * 0.94;   // terrain and debris stay inside the true radius
    var rnd = mulberry(seed);
    var group = new THREE.Group();

    var plates = [];
    for (var pi = 0; pi < 11; pi++) {
      var pa = rnd() * Math.PI * 2, pr = Math.sqrt(rnd()) * padR * 0.88;
      plates.push({
        x: Math.cos(pa) * pr, z: Math.sin(pa) * pr,
        lift: (0.040 + rnd() * 0.092) * radius,
        tx: (rnd() - 0.5) * 0.34, tz: (rnd() - 0.5) * 0.34
      });
    }

    // Height, and "how deep in a fissure am I", from the two nearest plates.
    function sample(vx, vz) {
      var d1 = 1e9, d2 = 1e9, k = 0;
      for (var i = 0; i < plates.length; i++) {
        var dx = vx - plates[i].x, dz = vz - plates[i].z;
        var d = Math.sqrt(dx * dx + dz * dz);
        if (d < d1) { d2 = d1; d1 = d; k = i; } else if (d < d2) { d2 = d; }
      }
      var pl = plates[k];
      var h = pl.lift + pl.tx * (vx - pl.x) + pl.tz * (vz - pl.z);
      h += (hash2(vx * 9.3, vz * 9.3) - 0.5) * 0.042;   // grit
      h += (hash2(vx * 2.7, vz * 2.7) - 0.5) * 0.072;   // buckle
      // On a plate boundary the two distances agree: carve the seam down.
      var seam = clamp(1 - (d2 - d1) / 0.13, 0, 1);
      var fissure = Math.pow(seam, 1.7);
      h *= (1 - 0.94 * fissure);
      var r = Math.sqrt(vx * vx + vz * vz) / padR;
      h *= smooth(1.0, 0.80, r);                        // settle to the rim
      return { h: HAZ_BASE + Math.max(0, h), fissure: fissure, r: r };
    }

    var verts = [], cols = [], idx = [];
    var tmpC = new THREE.Color();
    function pushVertex(vx, vz) {
      var sm = sample(vx, vz);
      verts.push(vx, sm.h, vz);
      /* Rock, darkening only where the ground is actually split. The exponent
         is deliberately below 1: above 1, even a mild seam value drags the
         whole surface toward black, which reads as an unlit patch. */
      tmpC.copy(ROCK_DARK).lerp(ROCK_LIT, Math.pow(1 - sm.fissure, 0.5));
      var burn = smooth(0.95, 0.15, sm.r) * (0.25 + 0.4 * hash2(vx * 4.1, vz * 4.1));
      tmpC.lerp(ROCK_SCORCH, clamp(burn * (1 - sm.fissure * 0.6), 0, 0.30));
      cols.push(tmpC.r, tmpC.g, tmpC.b);
    }

    /* A single centre vertex with a triangle fan around it. Giving every spoke
       its own coincident centre vertex leaves a ring of degenerate slivers,
       which renders as a black pinwheel exactly where the eye goes. */
    pushVertex(0, 0);
    function vid(i, j) { return 1 + (i - 1) * HAZ_SPOKES + (j % HAZ_SPOKES); }
    for (var i = 1; i <= HAZ_RINGS; i++) {
      var base = padR * Math.pow(i / HAZ_RINGS, 0.88);
      for (var j = 0; j < HAZ_SPOKES; j++) {
        var rr = base, th = (j / HAZ_SPOKES) * Math.PI * 2;
        /* Jitter the lattice: a clean polar grid reads as a polar grid, its
           triangles lining up into radial streaks no broken ground has. The
           outer ring is left alone so it still meets the skirt exactly. */
        if (i < HAZ_RINGS) {
          rr += (hash2(i * 3.1 + j * 7.7, j * 2.3) - 0.5) * (padR / HAZ_RINGS) * 0.8;
          th += (hash2(j * 5.3, i * 11.1) - 0.5) * (Math.PI * 2 / HAZ_SPOKES) * 0.9;
          rr = clamp(rr, 0.02, padR - 0.01);
        }
        pushVertex(Math.cos(th) * rr, Math.sin(th) * rr);
      }
    }
    // Wound so the normals come out pointing up. The other order is back-face
    // culled from every camera that matters, which reads as "the terrain did
    // not render" rather than as a winding mistake.
    for (var f = 0; f < HAZ_SPOKES; f++) idx.push(0, vid(1, f + 1), vid(1, f));
    for (var a = 1; a < HAZ_RINGS; a++) {
      for (var b = 0; b < HAZ_SPOKES; b++) {
        idx.push(vid(a, b), vid(a + 1, b + 1), vid(a + 1, b));
        idx.push(vid(a, b), vid(a, b + 1), vid(a + 1, b + 1));
      }
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();

    var pad = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.86, metalness: 0.18, envMapIntensity: 0.22,
      flatShading: true,
      // Belt and braces next to the floor plane, per the 16-bit depth rule.
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2
    }));
    pad.receiveShadow = true;
    pad.castShadow = true;
    group.add(pad);

    // The broken edge of the pad. Vertical, so it cannot fight the floor.
    var skirt = new THREE.Mesh(
      new THREE.CylinderGeometry(padR, padR, HAZ_BASE + 0.012, HAZ_SPOKES, 1, true),
      new THREE.MeshStandardMaterial({ color: 0x24221D, roughness: 0.95, metalness: 0.06 })
    );
    skirt.position.y = (HAZ_BASE + 0.012) / 2 - 0.006;
    skirt.receiveShadow = true;
    group.add(skirt);

    /* Upthrust slabs and scree, rejection-placed so that a centre distance
       plus the piece's own bounding radius stays inside padR — nothing can
       reach the ring, let alone cross it. */
    function place(maxR, tries) {
      for (var t = 0; t < tries; t++) {
        var ang = rnd() * Math.PI * 2, rad = Math.sqrt(rnd()) * padR;
        if (rad + maxR <= padR) return { x: Math.cos(ang) * rad, z: Math.sin(ang) * rad };
      }
      return null;
    }
    var sI, spot, sc;
    for (sI = 0; sI < 11; sI++) {
      sc = (0.058 + rnd() * 0.070) * radius;
      spot = place(sc * 2.1, 30);
      if (!spot) continue;
      var shard = new THREE.Mesh(shardGeos[sI % shardGeos.length], rubbleMat);
      shard.scale.set(sc * (1.0 + rnd()), sc * (0.50 + rnd() * 0.55), sc * (1.0 + rnd()));
      shard.position.set(spot.x, sample(spot.x, spot.z).h + sc * 0.22, spot.z);
      shard.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
      shard.castShadow = true;
      group.add(shard);
    }
    for (var gI = 0; gI < 26; gI++) {
      var gs = (0.012 + rnd() * 0.021) * radius;
      var gspot = place(gs * 2, 30);
      if (!gspot) continue;
      var pebble = new THREE.Mesh(shardGeos[gI % shardGeos.length], rubbleMat);
      pebble.scale.setScalar(gs * (0.8 + rnd() * 0.9));
      pebble.position.set(gspot.x, sample(gspot.x, gspot.z).h + gs * 0.4, gspot.z);
      pebble.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
      group.add(pebble);
    }

    // Embers, only where the ground is actually split open. They give the zone
    // a reason to be dangerous that survives the night exposure.
    var lit = 0;
    for (var eI = 0; eI < 70 && lit < 5; eI++) {
      var ea = rnd() * Math.PI * 2, er = Math.sqrt(rnd()) * padR * 0.85;
      var ex = Math.cos(ea) * er, ez = Math.sin(ea) * er;
      var es = sample(ex, ez);
      if (es.fissure < 0.55) continue;
      var ember = new THREE.Mesh(new THREE.SphereGeometry(0.019 * radius, 7, 5), emberMat);
      ember.position.set(ex, es.h + 0.009, ez);
      group.add(ember);
      lit++;
    }
    return group;
  }

  /* The readout the shared HUD has no field for. Eight rows: the label, the
     true range, a bar, and the noisy reading the agent received. This is the
     observation, so it gets permanent space rather than a tooltip, and it is
     built by this module into the viewer element rather than by editing the
     page every environment shares. */
  var PANEL_CSS = [
    ".lt-scan{position:absolute;top:10px;right:10px;padding:8px 10px;border-radius:6px;",
    "background:rgba(8,7,6,.62);color:#f2ece2;",
    "font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;max-width:46%}",
    ".lt-scan b{display:block;font-size:10px;letter-spacing:.09em;text-transform:uppercase;",
    "color:#9BA7A6;font-weight:600;margin-bottom:4px}",
    ".lt-scan table{border-collapse:collapse;font-variant-numeric:tabular-nums}",
    ".lt-scan td{padding:1px 0}",
    ".lt-scan td.l{color:#9BA7A6;width:2.4em}",
    ".lt-scan td.t{color:#99CC99;width:3.4em;text-align:right;padding-right:7px}",
    ".lt-scan td.o{color:#FFD24A;width:3.4em;text-align:right;padding-left:7px}",
    ".lt-scan .bar{position:relative;width:64px;height:6px;background:rgba(153,204,153,.16)}",
    ".lt-scan .bar i{position:absolute;left:0;top:0;bottom:0;background:#99CC99}",
    ".lt-scan .bar u{position:absolute;top:-2px;width:2px;height:10px;background:#FFD24A}",
    ".lt-scan tr.hit td.l,.lt-scan tr.hit td.t{color:#FF8A6B}",
    ".lt-scan .foot{margin-top:5px;color:#9BA7A6}",
    ".lt-scan .keys{margin-top:6px;display:flex;flex-wrap:wrap;gap:2px 10px}",
    ".lt-scan .keys label{display:inline-flex;align-items:center;gap:4px;color:#9BA7A6;cursor:pointer}",
    ".lt-scan .keys input{accent-color:#99CC99;width:12px;height:12px;margin:0}"
  ].join("");

  function buildPanel(root, labels) {
    if (!document.getElementById("lt-scan-style")) {
      var style = document.createElement("style");
      style.id = "lt-scan-style";
      style.textContent = PANEL_CSS;
      document.head.appendChild(style);
    }
    var panel = document.createElement("div");
    panel.className = "lt-scan";
    panel.innerHTML = "<b>Observation — 8 laser ranges</b><table></table>" +
      '<div class="foot"></div><div class="keys"></div>';
    var table = panel.querySelector("table");
    var rows = [];
    for (var i = 0; i < labels.length; i++) {
      var tr = document.createElement("tr");
      tr.innerHTML = '<td class="l"></td><td class="t"></td>' +
        '<td><div class="bar"><i></i><u></u></div></td><td class="o"></td>';
      tr.children[0].textContent = labels[i];
      table.appendChild(tr);
      rows.push({
        tr: tr, truth: tr.children[1], obs: tr.children[3],
        fill: tr.children[2].querySelector("i"), tick: tr.children[2].querySelector("u")
      });
    }
    var keys = panel.querySelector(".keys");
    var toggles = {};
    [["beams", "Beams"], ["belief", "Belief"], ["trails", "Trails"], ["hazards", "Hazards"]]
      .forEach(function (entry) {
        var label = document.createElement("label");
        var box = document.createElement("input");
        box.type = "checkbox";
        box.checked = true;
        label.appendChild(box);
        label.appendChild(document.createTextNode(entry[1]));
        keys.appendChild(label);
        toggles[entry[0]] = box;
      });
    root.appendChild(panel);
    return { rows: rows, foot: panel.querySelector(".foot"), toggles: toggles };
  }

  /**
   * Build the LaserTag arena from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind laser_tag.v1.
   * @returns {Object} The scene module the player drives.
   */
  // Long because it is one world: floor, walls, hazards, lamps, two bodies,
  // eight beams, two belief views and two trails, each built once.
  // eslint-disable-next-line max-statements
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;
    var discrete = world.variant === "discrete";

    /* The two variants store their arena differently, so each gets the frame
       its own source uses and neither is converted into the other.

       Discrete: positions are cell indices 0..rows-1, so the plate spans
       rows x cols cells and a cell's centre is its index.
       Continuous: positions are real, so the arena spans 0..width by
       0..height and the plate is exactly that.

       Both then map axis A to scene x and axis B to scene z, with B negated so
       that a larger second coordinate is further from the camera. */
    var spanA, spanB, originA, originB;
    if (discrete) {
      spanA = Math.max(1, world.floor_shape[0]);
      spanB = Math.max(1, world.floor_shape[1]);
      originA = (spanA - 1) / 2;
      originB = (spanB - 1) / 2;
    } else {
      spanA = Math.max(1, world.grid_size[0]);
      spanB = Math.max(1, world.grid_size[1]);
      originA = spanA / 2;
      originB = spanB / 2;
    }
    function wx(a) { return a - originA; }
    function wz(b) { return originB - b; }

    var directions = world.laser_directions;
    var labels = world.beam_labels;
    var nBeams = directions.length;

    scene.background = new THREE.Color(0x05070A).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x070A0D, 0.020);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x1E2A34, 0x070605, 0.5));
    var moon = new THREE.DirectionalLight(0x8FA3C8, 0.3);
    moon.position.set(-7, 10, -5);
    scene.add(moon);
    core.buildNightEnvironment([
      [0.0, "#0A1018"], [0.45, "#131B22"], [0.55, "#1E2A2C"], [1.0, "#050606"]
    ]);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);
    var glowTex = V.radialTexture(0.9, 0.3);

    // Floor. The grit is painted first and the normal map derived from it, so
    // every scratch catches a lamp from the side the lamp is on. The cell grid
    // is painted afterwards, onto colour only.
    var gCanvas = groundCanvas();
    var groundNormal = V.normalMapFrom(renderer, gCanvas, 2.1);
    paintGrid(gCanvas, Math.round(spanA), Math.round(spanB));
    var groundAlbedo = new THREE.CanvasTexture(gCanvas);
    groundAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    groundAlbedo.encoding = THREE.sRGBEncoding;

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(spanA, spanB),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.9, 0.9),
        roughness: 0.80, metalness: 0.22, envMapIntensity: 0.22
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    /* Floor carrying on past the arena, dropped clear of the plate. Without it
       the plate is a lit slab floating in a void, which is the single biggest
       tell that a render is a render. */
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(90, 90),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.8, 0.8),
        color: 0x141819, roughness: 0.95, metalness: 0.2, envMapIntensity: 0.10
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.6;
    outer.receiveShadow = true;
    scene.add(outer);

    var plinth = new THREE.Mesh(
      new THREE.BoxGeometry(spanA + 0.8, 0.66, spanB + 0.8),
      new THREE.MeshStandardMaterial({ color: 0x121719, roughness: 0.75, metalness: 0.4 })
    );
    plinth.position.y = -0.48;
    plinth.castShadow = true;
    plinth.receiveShadow = true;
    scene.add(plinth);

    /* The arena boundary, which in both variants is a hard limit: the robot is
       held inside it, and every laser stops at it. */
    var halfA = spanA / 2, halfB = spanB / 2;
    var railMat = new THREE.MeshStandardMaterial({ color: 0x2A3338, roughness: 0.4, metalness: 0.86 });
    var railHi = new THREE.MeshStandardMaterial({ color: 0x424C50, roughness: 0.5, metalness: 0.55 });
    [-halfB - 0.14, halfB + 0.14].forEach(function (z) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(spanA + 0.56, 0.26, 0.28), railMat);
      m.position.set(0, 0.13, z); m.castShadow = true; m.receiveShadow = true; scene.add(m);
      var cap = new THREE.Mesh(new THREE.BoxGeometry(spanA + 0.56, 0.035, 0.30), railHi);
      cap.position.set(0, 0.27, z); scene.add(cap);
    });
    [-halfA - 0.14, halfA + 0.14].forEach(function (x) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.26, spanB + 0.56), railMat);
      m.position.set(x, 0.13, 0); m.castShadow = true; m.receiveShadow = true; scene.add(m);
      var cap = new THREE.Mesh(new THREE.BoxGeometry(0.30, 0.035, spanB + 0.56), railHi);
      cap.position.set(x, 0.27, 0); scene.add(cap);
    });

    /* Walls. Each block is exactly the footprint the environment stores: a
       discrete wall is one cell, a continuous wall is its AABB. The bevel and
       the corner rivets are draw_wall's own decoration, in 3D. */
    var wallBody = new THREE.MeshStandardMaterial({ color: 0x1C2327, roughness: 0.55, metalness: 0.55 });
    var wallTop = new THREE.MeshStandardMaterial({ color: 0x1E262A, roughness: 0.62, metalness: 0.4 });
    var wallEdge = new THREE.MeshStandardMaterial({ color: 0x0C1012, roughness: 0.6, metalness: 0.5 });
    var wallLip = new THREE.MeshStandardMaterial({ color: 0x333B3E, roughness: 0.58, metalness: 0.45 });
    var wallInner = new THREE.MeshStandardMaterial({ color: 0x242B2E, roughness: 0.68, metalness: 0.4 });
    var rivetMat = new THREE.MeshStandardMaterial({ color: 0xA3A9A6, roughness: 0.28, metalness: 0.95 });
    var WALL_H = 0.92;

    (world.walls || []).forEach(function (w) {
      var sx = discrete ? 1 : w[2] * 2;
      var sz = discrete ? 1 : w[3] * 2;
      var g = new THREE.Group();
      g.position.set(wx(w[0]), 0, wz(w[1]));
      scene.add(g);

      var base = new THREE.Mesh(new THREE.BoxGeometry(sx, 0.09, sz), wallEdge);
      base.position.y = 0.045; base.castShadow = true; base.receiveShadow = true; g.add(base);
      var body = new THREE.Mesh(new THREE.BoxGeometry(sx * 0.9, WALL_H - 0.14, sz * 0.9), wallBody);
      body.position.y = 0.09 + (WALL_H - 0.14) / 2;
      body.castShadow = true; body.receiveShadow = true; g.add(body);
      var cap = new THREE.Mesh(new THREE.BoxGeometry(sx * 0.98, 0.06, sz * 0.98), wallTop);
      cap.position.y = WALL_H - 0.02; cap.castShadow = true; cap.receiveShadow = true; g.add(cap);
      // The pale top-left / dark bottom-right pair draw_wall paints.
      var lip = new THREE.Mesh(new THREE.BoxGeometry(sx * 0.99, 0.022, sz * 0.99), wallLip);
      lip.position.y = WALL_H + 0.018; g.add(lip);
      var inner = new THREE.Mesh(new THREE.BoxGeometry(sx * 0.6, 0.012, sz * 0.6), wallInner);
      inner.position.y = WALL_H + 0.032; g.add(inner);
      [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(function (c) {
        var r = new THREE.Mesh(new THREE.SphereGeometry(0.033, 8, 6), rivetMat);
        r.position.set(c[0] * sx * 0.36, WALL_H + 0.03, c[1] * sz * 0.36);
        g.add(r);
      });
    });

    /* Hazards. The red ring is the authority: it sits on the environment's own
       dangerous_area_radius, which is the radius the penalty is scored
       against. At the shipped defaults a hazard is neither a wall nor a
       gamble — the robot may drive across and simply pays for the step — so
       the terrain inside is deliberately low. Terrain that read as impassable
       would be a lie about the dynamics. */
    var rubbleMat = new THREE.MeshStandardMaterial({
      color: 0x35322B, roughness: 0.92, metalness: 0.08, flatShading: true
    });
    var emberMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(2.6, 0.62, 0.16) });
    var shardGeos = [
      new THREE.TetrahedronGeometry(1, 0),
      new THREE.IcosahedronGeometry(1, 0),
      new THREE.DodecahedronGeometry(1, 0)
    ];
    var hazardGroup = new THREE.Group();
    scene.add(hazardGroup);
    var hazardRims = [];
    var hazardRadius = world.hazard_radius;
    (world.hazards || []).forEach(function (h) {
      var x = wx(h[0]), z = wz(h[1]);
      var terrain = hazardTerrain(
        70207 + h[0] * 131 + h[1] * 17, hazardRadius, rubbleMat, emberMat, shardGeos
      );
      terrain.position.set(x, 0, z);
      hazardGroup.add(terrain);

      var column = new THREE.Mesh(
        new THREE.CylinderGeometry(hazardRadius, hazardRadius, 0.9, 44, 1, true),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(1.6, 0.32, 0.26), transparent: true, opacity: 0.035,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      column.position.set(x, 0.45, z);
      hazardGroup.add(column);

      var rim = new THREE.Mesh(
        new THREE.RingGeometry(hazardRadius - 0.045, hazardRadius + 0.045, 64),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(2.6, 0.55, 0.42), transparent: true, opacity: 0.55,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      rim.rotation.x = -Math.PI / 2;
      rim.position.set(x, 0.03, z);
      hazardGroup.add(rim);
      hazardRims.push(rim);
    });

    /* Work lamps on a gantry, carrying real lumens. They are what makes the
       arena read as a lit place rather than a flat board: the light falls off,
       so the corners between lamps go dark on their own. Placed by fraction of
       the plate, so a larger arena is lit the same way rather than in one
       bright middle. */
    var lamps = [];
    var lampLens = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.4, 3.8, 4.0) });
    var lampShell = new THREE.MeshStandardMaterial({ color: 0x39434A, roughness: 0.4, metalness: 0.85 });
    [1 / 6, 1 / 2, 5 / 6].forEach(function (fa) {
      [1 / 4, 3 / 4].forEach(function (fb) {
        var x = (fa - 0.5) * spanA, z = -(fb - 0.5) * spanB, y = 3.25;
        var shade = new THREE.Mesh(new THREE.CylinderGeometry(0.30, 0.16, 0.22, 16, 1, true), lampShell);
        shade.position.set(x, y, z); shade.castShadow = true; scene.add(shade);
        var lens = new THREE.Mesh(new THREE.SphereGeometry(0.13, 14, 10), lampLens);
        lens.position.set(x, y - 0.08, z); scene.add(lens);
        var cage = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.018, 6, 16), lampShell);
        cage.position.set(x, y - 0.1, z); cage.rotation.x = Math.PI / 2; scene.add(cage);
        var stem = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.1, 8), lampShell);
        stem.position.set(x, y + 0.6, z); scene.add(stem);
        // Faint haze under the lens. The halo itself comes from the bloom pass
        // reading the lens's own brightness, so this stays very low.
        var shaft = new THREE.Mesh(
          new THREE.CylinderGeometry(0.2, 2.1, 3.2, 20, 1, true),
          new THREE.MeshBasicMaterial({
            map: poolTex, color: COLORS.lamp, transparent: true, opacity: 0.016,
            blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
          })
        );
        shaft.position.set(x, y - 1.6, z);
        scene.add(shaft);
        var light = new THREE.PointLight(COLORS.lamp, 1, 12, 2);
        light.power = LAMP_LUMENS;
        light.position.set(x, y - 0.12, z);
        scene.add(light);
        lamps.push({ light: light, x: x, z: z });
      });
    });

    /* One shadow-casting lamp, not six. It parks on whichever gantry lamp is
       nearest the robot and aims at it, taking that share of the lamp's
       output, so the arena stays evenly lit while the robot always throws a
       shadow that points the right way. Six shadow-casting point lights would
       be thirty-six shadow passes a frame; this is one. Narrow cone plus
       normalBias, which is the fix for acne that a big negative bias is not. */
    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 14, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.5;
    shadowLight.shadow.camera.far = 15;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, 3.1, 0);
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    /* Robot: a red tracked hunter. The sensor drum on top is the thing that
       matters — the eight beams leave it, at the eight directions the
       environment casts. */
    var robot = new THREE.Group();
    scene.add(robot);
    var hull = new THREE.Group();
    robot.add(hull);

    var redPaint = new THREE.MeshStandardMaterial({ color: COLORS.robot, roughness: 0.42, metalness: 0.45 });
    var redDark = new THREE.MeshStandardMaterial({ color: 0xB71C1C, roughness: 0.5, metalness: 0.45 });
    var darkMetal = new THREE.MeshStandardMaterial({ color: 0x24292C, roughness: 0.5, metalness: 0.8 });
    var trackMat = new THREE.MeshStandardMaterial({ color: 0x15181A, roughness: 0.9, metalness: 0.15 });

    var chassis = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.18, 0.46), redPaint);
    chassis.position.y = 0.20; chassis.castShadow = true; hull.add(chassis);
    var shoulder = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.14, 0.40), redDark);
    shoulder.position.y = 0.36; shoulder.castShadow = true; hull.add(shoulder);
    var drum = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 0.16, 20), darkMetal);
    drum.position.y = 0.50; drum.castShadow = true; hull.add(drum);
    var drumCap = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.025, 20),
      new THREE.MeshStandardMaterial({ color: 0x343C3F, roughness: 0.55, metalness: 0.5 }));
    drumCap.position.y = 0.585; hull.add(drumCap);

    /* The apertures are world-fixed, not body-fixed. Every ray leaves along a
       direction from the environment's table, which does not turn with the
       robot, so a lens bolted to the hull would drift off its own beam as the
       body turned. The ring hangs off `robot` and cancels the body rotation
       each frame, so a beam always leaves a lens. Its angles come from the
       trace's own direction table, not from a ring of eight assumed to be
       evenly spaced. */
    var lensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.9, 2.4, 0.9) });
    var lensRing = new THREE.Group();
    robot.add(lensRing);
    directions.forEach(function (d) {
      var len = Math.hypot(d[0], d[1]) || 1;
      var lens = new THREE.Mesh(new THREE.SphereGeometry(0.028, 8, 6), lensMat);
      lens.position.set((d[0] / len) * 0.165, BEAM_Y, -(d[1] / len) * 0.165);
      lensRing.add(lens);
    });

    var visor = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.055, 0.30),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(3.6, 0.85, 0.45) }));
    visor.position.set(0.30, 0.27, 0); hull.add(visor);
    var noseGuard = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.10, 0.34), darkMetal);
    noseGuard.position.set(0.30, 0.16, 0); noseGuard.castShadow = true; hull.add(noseGuard);
    var tailPlate = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.16, 0.34), darkMetal);
    tailPlate.position.set(-0.29, 0.22, 0); tailPlate.castShadow = true; hull.add(tailPlate);

    var robotWheels = [];
    [0.26, -0.26].forEach(function (z) {
      var t = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.17, 0.14), trackMat);
      t.position.set(0, 0.10, z); t.castShadow = true; robot.add(t);
      [-0.22, 0, 0.22].forEach(function (x) {
        var wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.095, 0.095, 0.15, 14), trackMat);
        wheel.position.set(x, 0.095, z); wheel.rotation.x = Math.PI / 2;
        wheel.castShadow = true; robot.add(wheel); robotWheels.push(wheel);
        var hub = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.155, 10),
          new THREE.MeshStandardMaterial({ color: 0x6E7477, roughness: 0.35, metalness: 0.92 }));
        hub.position.set(x, 0.095, z); hub.rotation.x = Math.PI / 2; robot.add(hub);
      });
    });

    var robotBlob = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.0),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.42, depthWrite: false
      }));
    robotBlob.rotation.x = -Math.PI / 2; robotBlob.position.y = 0.008; robot.add(robotBlob);
    var robotGlow = new THREE.PointLight(0xFF6A5A, 0.5, 2.4, 2);
    robotGlow.power = 22; robotGlow.position.y = 0.4; robot.add(robotGlow);

    // The tag radius the continuous variant actually tests against. The
    // discrete variant tags on the same cell and has no radius, so it gets no
    // ring rather than a ring that means nothing.
    var tagRing = null;
    if (!discrete && world.tag_radius) {
      tagRing = new THREE.Mesh(
        new THREE.RingGeometry(world.tag_radius - 0.03, world.tag_radius + 0.03, 56),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(3.0, 2.2, 0.7), transparent: true, opacity: 0.35,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      tagRing.rotation.x = -Math.PI / 2;
      tagRing.position.y = 0.02;
      robot.add(tagRing);
    }

    // Opponent: a blue evader on wheels, lower and lighter than the robot.
    var opponent = new THREE.Group();
    scene.add(opponent);
    var bluePaint = new THREE.MeshStandardMaterial({ color: COLORS.opponent, roughness: 0.38, metalness: 0.5 });
    var blueDark = new THREE.MeshStandardMaterial({ color: 0x0D47A1, roughness: 0.48, metalness: 0.5 });
    var oBody = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.15, 0.38), bluePaint);
    oBody.position.y = 0.17; oBody.castShadow = true; opponent.add(oBody);
    var oDome = new THREE.Mesh(
      new THREE.SphereGeometry(0.15, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2), blueDark);
    oDome.position.y = 0.245; oDome.castShadow = true; opponent.add(oDome);
    var oEye = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.05, 0.20),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.8, 2.3, 3.8) }));
    oEye.position.set(0.23, 0.20, 0); opponent.add(oEye);
    var oFin = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.14, 0.12), blueDark);
    oFin.position.set(-0.20, 0.30, 0); opponent.add(oFin);
    [[0.16, 0.20], [0.16, -0.20], [-0.16, 0.20], [-0.16, -0.20]].forEach(function (w) {
      var m = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.07, 14), trackMat);
      m.position.set(w[0], 0.085, w[1]); m.rotation.x = Math.PI / 2;
      m.castShadow = true; opponent.add(m);
    });
    var oBlob = new THREE.Mesh(new THREE.PlaneGeometry(0.95, 0.85),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.4, depthWrite: false
      }));
    oBlob.rotation.x = -Math.PI / 2; oBlob.position.y = 0.008; opponent.add(oBlob);
    var oGlow = new THREE.PointLight(0x4FA8FF, 0.4, 2.0, 2);
    oGlow.power = 14; oGlow.position.y = 0.3; opponent.add(oGlow);

    /* The beams. A beam is a thin additive tube ending in a flare where the
       range model says it lands, plus the amber bead at the noisy reading. */
    var beamGeo = new THREE.CylinderGeometry(1, 1, 1, 7, 1, true);
    var beams = [];
    var upVec = new THREE.Vector3(0, 1, 0);
    var beamVec = new THREE.Vector3();
    for (var bi = 0; bi < nBeams; bi++) {
      var tube = new THREE.Mesh(beamGeo, new THREE.MeshBasicMaterial({
        color: BEAM_COLOR.clone(), transparent: true, opacity: 0.6,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      }));
      scene.add(tube);
      var flare = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xBFEFBF, transparent: true, opacity: 0.85,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      flare.scale.set(0.34, 0.34, 1);
      scene.add(flare);
      var bead = new THREE.Mesh(new THREE.SphereGeometry(0.055, 10, 8),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(3.4, 2.4, 0.6) }));
      scene.add(bead);
      beams.push({ tube: tube, flare: flare, bead: bead });
    }

    /* Belief views.
       Continuous: a particle cloud, because the belief is a particle set over
       continuous positions.
       Discrete: one bar per cell, because the belief is a distribution over
       cells and a cloud would imply resolution it does not have.
       Both are drawn from the kind core serialised, never from the belief
       class that produced it. */
    function makeCloud(count, size) {
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
      geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
      geo.setDrawRange(0, 0);
      var pts = new THREE.Points(geo, new THREE.PointsMaterial({
        size: size, map: partTex, vertexColors: true, transparent: true, opacity: 1.0,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
      }));
      scene.add(pts);
      return pts;
    }
    var oppCloud = makeCloud(MAX_DRAWN_PARTICLES, 0.26);
    var robCloud = makeCloud(MAX_DRAWN_PARTICLES, 0.19);

    function makeBars(color, width) {
      var arr = [];
      for (var i = 0; i < MAX_DRAWN_CELLS; i++) {
        var m = new THREE.Mesh(new THREE.BoxGeometry(width, 1, width),
          new THREE.MeshBasicMaterial({
            color: color, transparent: true, opacity: 0.5,
            blending: THREE.AdditiveBlending, depthWrite: false
          }));
        m.visible = false;
        scene.add(m);
        arr.push(m);
      }
      return arr;
    }
    var oppBars = makeBars(OPP_BELIEF_HOT, 0.46), robBars = makeBars(ROBOT_BELIEF_HOT, 0.30);

    // Trails: the recorded paths, drawn up to the current step.
    function makeTrail(hex) {
      var max = payload.robots.length + 1;
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(max * 3), 3));
      geo.setDrawRange(0, 0);
      var line = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: new THREE.Color(hex).convertSRGBToLinear().multiplyScalar(2.8)
      }));
      scene.add(line);
      return line;
    }
    var robTrail = makeTrail(COLORS.robotPath), oppTrail = makeTrail(COLORS.oppPath);

    // Drifting dust, so the lamp shafts have something to sit in.
    var MOTES = 240;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(1357);
    for (var m0 = 0; m0 < MOTES; m0++) {
      motePos[m0 * 3] = (moteSeed() - 0.5) * (spanA + 2);
      motePos[m0 * 3 + 1] = 0.12 + moteSeed() * 2.6;
      motePos[m0 * 3 + 2] = (moteSeed() - 0.5) * (spanB + 2);
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    var motes = new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.022, map: partTex, color: 0x8FA3AA, transparent: true, opacity: 0.16,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    }));
    scene.add(motes);

    core.linearize();

    var panel = buildPanel(document.getElementById("viewer"), labels);

    // Running discounted return, on the same convention as History's.
    var running = [];
    var total = 0;
    for (var ri = 0; ri < trace.steps.length; ri++) {
      var reward = trace.steps[ri].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, ri);
      }
      running.push(total);
    }

    var heading = 0, oppHeading = 0, wheelSpin = 0;
    var steps = payload.robots.length;
    var longest = Math.max(spanA, spanB);

    function sampleAt(t) {
      var i0 = Math.floor(clamp(t, 0, steps - 1));
      var i1 = Math.min(i0 + 1, steps - 1);
      return { i0: i0, i1: i1, f: clamp(t - i0, 0, 1) };
    }

    /* A beam between two steps sweeps from one true beam to the next. It is
       exact at every step boundary, and no part of it is ever recast here: the
       two endpoints it interpolates are both `robot + direction * range` with
       all three read out of the trace. */
    function beamPoint(index, beam, ranges) {
      var r = ranges[index];
      if (!r || r.length <= beam) return null;
      var p = payload.robots[index], d = directions[beam];
      return [p[0] + d[0] * r[beam], p[1] + d[1] * r[beam]];
    }

    function placeBeam(beam, sample, hit, pulse) {
      var b = beams[beam];
      var a0 = beamPoint(sample.i0, beam, payload.laser_ranges);
      var a1 = beamPoint(sample.i1, beam, payload.laser_ranges);
      var p0 = payload.robots[sample.i0], p1 = payload.robots[sample.i1];
      if (!a0 || !a1) {
        b.tube.visible = false; b.flare.visible = false; b.bead.visible = false;
        return;
      }
      var sx = wx(lerp(p0[0], p1[0], sample.f)), sz = wz(lerp(p0[1], p1[1], sample.f));
      var tx = wx(lerp(a0[0], a1[0], sample.f)), tz = wz(lerp(a0[1], a1[1], sample.f));
      beamVec.set(tx - sx, 0, tz - sz);
      var len = Math.max(0.001, beamVec.length());
      b.tube.scale.set(0.028, len, 0.028);
      b.tube.position.set((sx + tx) / 2, BEAM_Y, (sz + tz) / 2);
      b.tube.quaternion.setFromUnitVectors(upVec, beamVec.normalize());
      b.tube.material.color.copy(hit ? BEAM_HIT_COLOR : BEAM_COLOR);
      b.tube.material.opacity = (hit ? 0.95 : 0.62) * pulse;
      b.tube.visible = true;

      b.flare.position.set(tx, BEAM_Y, tz);
      b.flare.material.color.setHex(hit ? 0xFFB07A : 0xBFEFBF);
      b.flare.scale.setScalar(hit ? 0.52 : 0.3);
      b.flare.visible = true;

      // The reading, in the same units along the same ray. Past the true hit
      // it hangs beyond the surface, which is what an over-reading range does.
      var o0 = beamPoint(sample.i0, beam, payload.observed_ranges);
      var o1 = beamPoint(sample.i1, beam, payload.observed_ranges);
      if (!o0 || !o1) {
        b.bead.visible = false;
        return;
      }
      b.bead.position.set(
        wx(lerp(o0[0], o1[0], sample.f)), BEAM_Y, wz(lerp(o0[1], o1[1], sample.f))
      );
      b.bead.visible = true;
    }

    function hideBelief() {
      oppCloud.geometry.setDrawRange(0, 0);
      robCloud.geometry.setDrawRange(0, 0);
      for (var i = 0; i < MAX_DRAWN_CELLS; i++) {
        oppBars[i].visible = false;
        robBars[i].visible = false;
      }
    }

    /* A LaserTag particle is the whole 5-vector state, so the same cloud
       carries two things: where the robot might be and where the opponent
       might be. They are drawn as two clouds in their own colours rather than
       one, because merging them would show a belief nobody holds. */
    function drawClouds(belief) {
      var count = Math.min(belief.particles.length, MAX_DRAWN_PARTICLES);
      var oppPos = oppCloud.geometry.attributes.position.array;
      var oppCol = oppCloud.geometry.attributes.color.array;
      var robPos = robCloud.geometry.attributes.position.array;
      var robCol = robCloud.geometry.attributes.color.array;
      var written = 0, p;
      // Shade by each particle's share of the heaviest, so the bright end of
      // the cloud is where the belief's mass actually is. An unweighted belief
      // is uniform by construction, so shading it would imply structure it
      // does not have and every particle is drawn equally bright.
      var peak = 0;
      if (belief.weighted !== false) {
        for (p = 0; p < count; p++) {
          if (belief.weights[p] > peak) peak = belief.weights[p];
        }
      }
      for (p = 0; p < count; p++) {
        var s = belief.particles[p];
        if (!Array.isArray(s) || s.length < 4) continue;
        var rel = belief.weighted === false ? 1
          : (peak > 0 ? belief.weights[p] / peak : 0);
        var shade = 0.28 + 0.72 * rel;
        oppPos[written * 3] = wx(s[2]);
        oppPos[written * 3 + 1] = 0.22 + (written % 4) * 0.015;
        oppPos[written * 3 + 2] = wz(s[3]);
        oppCol[written * 3] = OPP_BELIEF_HOT.r * shade;
        oppCol[written * 3 + 1] = OPP_BELIEF_HOT.g * shade;
        oppCol[written * 3 + 2] = OPP_BELIEF_HOT.b * shade;
        robPos[written * 3] = wx(s[0]);
        robPos[written * 3 + 1] = 0.20 + (written % 4) * 0.015;
        robPos[written * 3 + 2] = wz(s[1]);
        robCol[written * 3] = ROBOT_BELIEF_HOT.r * shade;
        robCol[written * 3 + 1] = ROBOT_BELIEF_HOT.g * shade;
        robCol[written * 3 + 2] = ROBOT_BELIEF_HOT.b * shade;
        written++;
      }
      oppCloud.geometry.attributes.position.needsUpdate = true;
      oppCloud.geometry.attributes.color.needsUpdate = true;
      robCloud.geometry.attributes.position.needsUpdate = true;
      robCloud.geometry.attributes.color.needsUpdate = true;
      oppCloud.geometry.setDrawRange(0, written);
      robCloud.geometry.setDrawRange(0, written);
      if (!written) {
        return belief.num_particles + " non-spatial particles";
      }
      var label = belief.num_particles +
        (belief.weighted === false ? " uniform particles" : " particles");
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " drawn)";
      }
      return label;
    }

    /* The discrete belief is a distribution over cells, so the particles are
       summed onto their cells and drawn as bars. Their weights are the
       belief's own, so a bar's height is probability and not particle count. */
    function cellBars(belief, index0, index1, bars) {
      var tally = {}, total = 0, i;
      for (i = 0; i < belief.particles.length; i++) {
        var s = belief.particles[i];
        if (!Array.isArray(s) || s.length < 4) continue;
        var key = Math.round(s[index0]) + "," + Math.round(s[index1]);
        var w = belief.weighted === false ? 1 : (belief.weights[i] || 0);
        tally[key] = (tally[key] || 0) + w;
        total += w;
      }
      var cells = Object.keys(tally).map(function (k) {
        var parts = k.split(",");
        return { a: +parts[0], b: +parts[1], p: tally[k] / (total || 1) };
      }).sort(function (x, y) { return y.p - x.p; }).slice(0, MAX_DRAWN_CELLS);
      var peak = cells.length ? cells[0].p : 1;
      for (i = 0; i < bars.length; i++) {
        var cell = cells[i];
        if (!cell) { bars[i].visible = false; continue; }
        var h = 0.08 + (cell.p / peak) * 0.85;
        bars[i].visible = true;
        bars[i].scale.set(1, h, 1);
        bars[i].position.set(wx(cell.a), h / 2, wz(cell.b));
        bars[i].material.opacity = 0.10 + 0.34 * (cell.p / peak);
      }
      return cells.length;
    }

    function drawBelief(index) {
      var belief = payload.beliefs[index];
      if (!belief) { hideBelief(); return "—"; }
      if (belief.kind === "particles") {
        if (discrete) {
          oppCloud.geometry.setDrawRange(0, 0);
          robCloud.geometry.setDrawRange(0, 0);
          var drawn = cellBars(belief, 2, 3, oppBars);
          cellBars(belief, 0, 1, robBars);
          return belief.num_particles + " particles over " + drawn + " cells";
        }
        for (var i = 0; i < MAX_DRAWN_CELLS; i++) {
          oppBars[i].visible = false;
          robBars[i].visible = false;
        }
        return drawClouds(belief);
      }
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner;
        // merging its members would show a cloud that was never anyone's.
        hideBelief();
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }
      // A belief kind this environment has no spatial reading for. Named, so
      // the gap is diagnosable, and drawn as nothing, so it is not invented.
      hideBelief();
      return "not drawn (" + (belief.belief_class || belief.kind) + ")";
    }

    function drawTrail(line, series, sample) {
      var pos = line.geometry.attributes.position.array;
      var n = 0, i;
      for (i = 0; i <= sample.i0; i++) {
        pos[n * 3] = wx(series[i][0]);
        pos[n * 3 + 1] = 0.06;
        pos[n * 3 + 2] = wz(series[i][1]);
        n++;
      }
      if (sample.f > 0) {
        pos[n * 3] = wx(lerp(series[sample.i0][0], series[sample.i1][0], sample.f));
        pos[n * 3 + 1] = 0.06;
        pos[n * 3 + 2] = wz(lerp(series[sample.i0][1], series[sample.i1][1], sample.f));
        n++;
      }
      line.geometry.attributes.position.needsUpdate = true;
      line.geometry.setDrawRange(0, n > 1 ? n : 0);
    }

    function formatAction(action) {
      if (action === null || action === undefined) return "—";
      if (Array.isArray(action)) {
        return "[" + action.map(function (v) { return Number(v).toFixed(2); }).join(", ") + "]";
      }
      if (discrete && typeof action === "number") {
        return ["North", "South", "East", "West", "Tag"][action] || String(action);
      }
      return String(action);
    }

    function updatePanel(index) {
      var truth = payload.laser_ranges[index] || [];
      var obs = payload.observed_ranges[index];
      var hit = payload.hit_opponent[index] || [];
      // The bar is scaled to the arena, so a full bar means a beam that ran
      // the length of the plate and a short one means something is close.
      var span = longest;
      for (var i = 0; i < panel.rows.length; i++) {
        var row = panel.rows[i];
        var t = truth.length > i ? truth[i] : null;
        var o = obs && obs.length > i ? obs[i] : null;
        row.truth.textContent = t === null ? "—" : (discrete ? t.toFixed(0) : t.toFixed(2));
        row.obs.textContent = o === null ? "—" : o.toFixed(2);
        row.fill.style.width = (t === null ? 0 : clamp(t / span, 0, 1) * 100).toFixed(1) + "%";
        row.tick.style.left = (o === null ? 0 : clamp(o / span, 0, 1) * 100).toFixed(1) + "%";
        row.tick.style.visibility = o === null ? "hidden" : "visible";
        row.tr.className = hit[i] ? "hit" : "";
      }
      var note;
      if (!truth.length) {
        note = "terminal state — no measurement";
      } else if (!obs) {
        // The reading beside a state is the one recorded on the step before
        // it, so the first state of an episode has none.
        note = "reading — none recorded at this state";
      } else {
        note = "green: true range · amber: reading, noise σ " +
          world.measurement_noise.toFixed(2);
      }
      panel.foot.textContent = note;
    }

    return {
      steps: steps,

      /* Framing scales with the arena, because a trace decides how big it is.
         The two constants were tuned on this environment's default 11 x 7
         plate, so a default arena is framed exactly as the approved design
         framed it and a larger one pulls back in proportion. */
      camera: (function () {
        var k = Math.max(spanA / 11, spanB / 7);
        return { board: [0, 9.4 * k, 8.6 * k], top: [0, 12.8 * k, 0.01] };
      })(),

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var r0 = payload.robots[sample.i0], r1 = payload.robots[sample.i1];
        var o0 = payload.opponents[sample.i0], o1 = payload.opponents[sample.i1];
        var ra = lerp(r0[0], r1[0], sample.f), rb = lerp(r0[1], r1[1], sample.f);
        var oa = lerp(o0[0], o1[0], sample.f), ob = lerp(o0[1], o1[1], sample.f);
        var px = wx(ra), pz = wz(rb);
        var qx = wx(oa), qz = wz(ob);

        // Heading from the motion between steps, so the bodies face where they
        // go. Scene-space travel angle: env (da, db) becomes scene (da, -db).
        var mva = r1[0] - r0[0], mvb = r1[1] - r0[1];
        if (Math.hypot(mva, mvb) > 1e-6) {
          var target = Math.atan2(-mvb, mva);
          var diff = ((target - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          heading += diff * clamp(dt * 6, 0, 1);
        }
        var oma = o1[0] - o0[0], omb = o1[1] - o0[1];
        if (Math.hypot(oma, omb) > 1e-6) {
          var otarget = Math.atan2(-omb, oma);
          var odiff = ((otarget - oppHeading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          oppHeading += odiff * clamp(dt * 5, 0, 1);
        }

        robot.position.set(px, 0, pz);
        robot.rotation.y = -heading;
        opponent.position.set(qx, 0, qz);
        opponent.rotation.y = -oppHeading;
        // The apertures are world-fixed, so cancel the body's rotation on the
        // ring. robot.rotation.y is -heading, so the ring carries +heading.
        lensRing.rotation.y = heading;

        var speed = playing ? 1 : 0;
        wheelSpin += dt * speed * 5;
        for (var wI = 0; wI < robotWheels.length; wI++) robotWheels[wI].rotation.y = wheelSpin;

        /* Park the one shadow-casting lamp on whichever gantry lamp is nearest
           the robot and take that much power off it, so the shadow follows
           without brightening a corner. */
        var nearest = null, nd = Infinity;
        for (var li = 0; li < lamps.length; li++) {
          lamps[li].light.power = LAMP_LUMENS;
          var d2 = (lamps[li].x - px) * (lamps[li].x - px) + (lamps[li].z - pz) * (lamps[li].z - pz);
          if (d2 < nd) { nd = d2; nearest = lamps[li]; }
        }
        if (nearest) {
          shadowLight.position.set(nearest.x, 3.13, nearest.z);
          shadowLight.target.position.set(px, 0.1, pz);
          shadowLight.target.updateMatrixWorld();
          var share = 0.66 * clamp(1 - (Math.sqrt(nd) - 1.2) / 5.5, 0.10, 1);
          shadowLight.power = LAMP_LUMENS * share;
          nearest.light.power = LAMP_LUMENS * (1 - share);
        }

        // The step whose measurement is on screen: the nearer of the two.
        var shown = sample.f > 0.5 ? sample.i1 : sample.i0;
        var hits = payload.hit_opponent[shown] || [];
        var pulse = 0.86 + Math.sin(elapsed * 9) * 0.14;
        var beamsOn = panel.toggles.beams.checked;
        for (var b = 0; b < nBeams; b++) {
          placeBeam(b, sample, !!hits[b], pulse);
          if (!beamsOn) {
            beams[b].tube.visible = false;
            beams[b].flare.visible = false;
            beams[b].bead.visible = false;
          }
        }

        var beliefLabel;
        if (panel.toggles.belief.checked) {
          beliefLabel = drawBelief(shown);
        } else {
          hideBelief();
          beliefLabel = "hidden";
        }

        drawTrail(robTrail, payload.robots, sample);
        drawTrail(oppTrail, payload.opponents, sample);
        robTrail.visible = panel.toggles.trails.checked;
        oppTrail.visible = panel.toggles.trails.checked;
        hazardGroup.visible = panel.toggles.hazards.checked;

        if (tagRing) {
          tagRing.material.opacity = 0.18 + 0.12 * (0.6 + Math.sin(elapsed * 6) * 0.4);
        }
        for (var mi = 0; mi < MOTES; mi++) {
          motePos[mi * 3] += Math.sin(elapsed * 0.3 + mi) * 0.0014;
          motePos[mi * 3 + 1] += 0.0019;
          if (motePos[mi * 3 + 1] > 2.9) motePos[mi * 3 + 1] = 0.12;
        }
        moteGeo.attributes.position.needsUpdate = true;
        var hp = 0.36 + Math.sin(elapsed * 2.0) * 0.16;
        for (var hi = 0; hi < hazardRims.length; hi++) hazardRims[hi].material.opacity = hp;

        updatePanel(shown);

        var step = trace.steps[shown] || {};
        return {
          follow: { x: px, z: pz, heading: heading },
          step: shown,
          action: formatAction(payload.actions[shown]),
          x: ra,
          y: rb,
          reward: step.reward,
          ret: running[shown],
          belief: beliefLabel
        };
      }
    };
  }

  V.scenes["laser_tag.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // arena's six 2500 lm lamps; it is not a knob to remove.
    exposure: 0.112
  };
})(window);
