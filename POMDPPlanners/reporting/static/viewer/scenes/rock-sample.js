/* SPDX-License-Identifier: MIT
 *
 * RockSample scene module.
 *
 * Builds the survey site from a trace's `payload.world` block and moves it from
 * the trace's recorded states, checks and beliefs. Nothing here is invented:
 * there is no fallback episode, no hand-placed waypoint and no synthetic
 * reading. If the player hands this module no trace, it draws nothing.
 *
 * The belief in RockSample is not spatial. The rover always knows where it is;
 * what it does not know is which rocks are worth sampling. So the belief is
 * drawn as one gauge per rock, floating over the rock it describes, each
 * showing that rock's marginal probability of being good. That marginal is
 * computed from the run's own particle cloud — the share of belief mass in
 * which that rock is good — and never from the true state, which is drawn
 * separately, on purpose, so the two can visibly disagree.
 *
 * A check is staged as the event it is: the mast turns to the rock, a beam
 * reaches it, a ring flashes the colour of the bit that came back, and the
 * gauge moves. Which rock, and what came back, are read from the trace.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* The 2D renderer's palette, from rock_sample_visualizer.py, so the viewer
     and the GIF are recognisably the same world. */
  var COLORS = {
    danger: 0xE83728,   // COLOR_DANGER
    exit: 0xF4C428,     // COLOR_EXIT
    lamp: 0xFFE4BC
  };

  // Lumens. Four service floodlights on masts, not a stadium: the far corner
  // of the pad has to fall off, or the night is only decorative.
  var MAST_LUMENS = 2100;

  // The circular hazard cut-outs and the basins that fill them are built as
  // regular polygons of this many sides, so their edges share vertices exactly
  // and the seam between pad and basin cannot open.
  var HAZARD_SEGMENTS = 80;

  // The rover moves for this fraction of a step and rests for the remainder.
  // The rest is where a check or a drill happens.
  var MOVE_FRACTION = 0.55;

  /* ------------------------------------------------------------- materials */

  /* The floor is painted in two passes. The grit goes down first and the normal
     map is derived from it, so every pebble catches a floodlight from the side
     the lamp is actually on. The grid is painted afterwards, onto colour only:
     the cell lines are survey marks and must not emboss the soil. The base
     tones are the orange soil of the environment's own terrain.png, darkened
     for a night exposure. */
  function groundCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(20260919);

    g.fillStyle = "#7A3C1E";
    g.fillRect(0, 0, s, s);

    for (var i = 0; i < 1200; i++) {
      var r = 10 + rnd() * 105;
      g.fillStyle = "rgba(" + (150 + rnd() * 70 | 0) + "," + (68 + rnd() * 40 | 0) + "," +
        (30 + rnd() * 26 | 0) + ",0.26)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
    }
    // Pebbles: a lit cap and a dark side, which the normal map turns to relief.
    for (var p = 0; p < 1700; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.5 + rnd() * 5.0;
      g.fillStyle = "rgba(48,20,10,0.55)";
      g.beginPath(); g.arc(px + pr * 0.35, py + pr * 0.35, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(228,160,110,0.32)";
      g.beginPath(); g.arc(px, py, pr * 0.85, 0, Math.PI * 2); g.fill();
    }
    for (var j = 0; j < 1400; j++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(255,214,170,0.05)" : "rgba(30,12,6,0.26)";
      g.fillRect(rnd() * s, rnd() * s, 3, 3);
    }
    return cv;
  }

  function paintGrid(cv, rows, cols) {
    var s = cv.width;
    var g = cv.getContext("2d");
    g.strokeStyle = "rgba(255,206,150,0.55)";   // COLOR_GRID, as in the GIF
    g.lineWidth = 3;
    var k;
    for (k = 0; k <= cols; k++) {
      var x = (k / cols) * s;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, s); g.stroke();
    }
    for (k = 0; k <= rows; k++) {
      var y = (k / rows) * s;
      g.beginPath(); g.moveTo(0, y); g.lineTo(s, y); g.stroke();
    }
  }

  /* ---------------------------------------------------------------- belief */

  /**
   * Marginal P(rock i is good) for every rock, from one serialised belief.
   *
   * The payload core writes carries whole particles — a RockSample particle is
   * ``[row, col, rock_0, ..., rock_n]`` — so each rock's marginal is the share
   * of belief mass in which that rock's slot is set. Computing it here rather
   * than writing it into the trace keeps one number for one thing: the gauge
   * is a view of the recorded belief, and cannot drift away from it.
   *
   * @param {Object} belief    One entry of payload.beliefs.
   * @param {number} numRocks  How many rocks the world has.
   * @returns {?Array<number>} One probability per rock, or null when this
   *   belief is not a drawable particle cloud.
   */
  function rockMarginals(belief, numRocks) {
    if (!belief || belief.kind !== "particles" || !numRocks) return null;
    var particles = belief.particles || [];
    if (!particles.length || !Array.isArray(particles[0])) return null;
    if (particles[0].length < 2 + numRocks) return null;

    var out = [];
    for (var k = 0; k < numRocks; k++) out.push(0);
    var total = 0;
    // An unweighted cloud is uniform by construction; shading it by its stored
    // weights would imply structure the belief does not have.
    var uniform = belief.weighted === false;
    for (var i = 0; i < particles.length; i++) {
      var w = uniform ? 1 : (belief.weights ? belief.weights[i] : 0);
      if (!(w > 0)) continue;
      total += w;
      for (var r = 0; r < numRocks; r++) {
        if (particles[i][2 + r] > 0.5) out[r] += w;
      }
    }
    if (!(total > 0)) return null;
    for (var n = 0; n < numRocks; n++) out[n] /= total;
    return out;
  }

  /** What to say in the HUD when a belief cannot be turned into gauges. */
  function beliefGap(belief) {
    if (!belief) return "—";
    if (belief.kind === "particle_batch") {
      // A batch is several beliefs held together for a vectorized planner; it
      // is not one episode's belief, and merging its members would show a
      // posterior that was never anyone's.
      return "batch of " + belief.batch_size + " beliefs, not drawn";
    }
    return "not recorded (" + (belief.belief_class || belief.kind) + ")";
  }

  /* ----------------------------------------------------------------- scene */

  /**
   * Build the RockSample world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload kind rock_sample.v1.
   * @returns {Object} The scene module the player drives.
   */
  // The scene is one world built in one pass; splitting it would only move the
  // same statements behind names nothing else calls.
  // eslint-disable-next-line max-statements
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;

    var ROWS = world.map_size[0], COLS = world.map_size[1];
    var ROCKS = world.rock_positions || [];
    var DANGER = world.dangerous_areas || [];
    var DANGER_R = world.dangerous_area_radius;
    var numRocks = ROCKS.length;

    // Cell (row, col) -> scene (x, z). Column grows east (+x), row grows south
    // (+z), which is the orientation the 2D renderer draws.
    var HR = (ROWS - 1) / 2, HC = (COLS - 1) / 2;
    function sx(col) { return col - HC; }
    function sz(row) { return row - HR; }

    /* The recorded path, with the exit resolved for drawing only. Leaving east
       off the last column ends the episode, and the environment records that as
       the (-1, -1) sentinel — a state with no cell. Drawing the rover at the
       sentinel would park it at the middle of the board; drawing it one cell
       east of where it drove off is where it actually went. The row is the row
       it left from, taken from the previous state, not guessed. */
    var path = [];
    var exited = [];
    for (var s = 0; s < payload.states.length; s++) {
      var cell = payload.states[s];
      if (cell[0] < 0 || cell[1] < 0) {
        var back = path.length ? path[path.length - 1] : { row: 0, col: COLS - 1 };
        path.push({ row: back.row, col: COLS });
        exited.push(true);
      } else {
        path.push({ row: cell[0], col: cell[1] });
        exited.push(false);
      }
    }

    // Per-step per-rock posteriors, and a label for the steps where the belief
    // is not a shape this scene can reduce to a gauge.
    var marginals = [], gaps = [];
    for (var b = 0; b < payload.beliefs.length; b++) {
      var m = rockMarginals(payload.beliefs[b], numRocks);
      marginals.push(m);
      gaps.push(m ? null : beliefGap(payload.beliefs[b]));
    }

    scene.background = new THREE.Color(0x070607).convertSRGBToLinear();
    // Thin dust haze: enough that the far mast sits behind some atmosphere,
    // little enough that the grid stays readable from the raised view.
    scene.fog = new THREE.FogExp2(0x0C0806, 0.030);
    scene.fog.color.convertSRGBToLinear();

    // Moonlight and skylight, deliberately weak: this is a night site lit by
    // four floodlights.
    scene.add(new THREE.HemisphereLight(0x2A2A3E, 0x120804, 0.26));
    var moon = new THREE.DirectionalLight(0x93A2C6, 0.17);
    moon.position.set(-5, 8, -4);
    scene.add(moon);
    core.buildNightEnvironment([
      [0.0, "#090C18"], [0.45, "#141420"], [0.55, "#2E1E10"], [1.0, "#070403"]
    ]);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);
    var glowTex = V.radialTexture(0.9, 0.3);

    var gCanvas = groundCanvas();
    var groundNormal = V.normalMapFrom(renderer, gCanvas, 2.0);
    paintGrid(gCanvas, ROWS, COLS);
    var groundAlbedo = new THREE.CanvasTexture(gCanvas);
    groundAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    groundAlbedo.encoding = THREE.sRGBEncoding;   // colour data, not linear data

    /* The pad is cut, not painted. A hazard is broken ground below the
       surface, so the pad and its foundation both carry a hole at the hazard's
       true radius and the broken terrain is dropped into it. */
    function padShape(halfCols, halfRows) {
      var sh = new THREE.Shape();
      sh.moveTo(-halfCols, -halfRows);
      sh.lineTo(halfCols, -halfRows);
      sh.lineTo(halfCols, halfRows);
      sh.lineTo(-halfCols, halfRows);
      sh.closePath();
      DANGER.forEach(function (d) {
        var hole = new THREE.Path();
        // Shape x is world x; shape y is world -z (the plane is rotated -90 deg).
        hole.absarc(sx(d[1]), -sz(d[0]), DANGER_R, 0, Math.PI * 2, true);
        sh.holes.push(hole);
      });
      return sh;
    }

    var floorGeo = new THREE.ShapeGeometry(padShape(COLS / 2, ROWS / 2), HAZARD_SEGMENTS);
    /* ShapeGeometry writes the shape's own coordinates as UVs, which would put
       the soil texture in world units. Rewrite them to the 0..1 a plane would
       have, so the grit and the painted grid land on the cells they describe. */
    (function fixPadUVs() {
      var pos = floorGeo.attributes.position;
      var uv = floorGeo.attributes.uv;
      for (var i = 0; i < pos.count; i++) {
        uv.setXY(i, (pos.getX(i) + COLS / 2) / COLS, (pos.getY(i) + ROWS / 2) / ROWS);
      }
      uv.needsUpdate = true;
    })();

    var floor = new THREE.Mesh(
      floorGeo,
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.9, 0.9),
        roughness: 0.95, metalness: 0.0, envMapIntensity: 0.3
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    /* Terrain carrying on past the survey pad. Without it the field is a lit
       slab floating in a void, which is the single biggest tell that a night
       render is a render. Its plane sits 0.55 below the pad, far enough that
       the 16-bit depth buffer never has to choose between them. */
    var outerAlbedo = groundAlbedo.clone();
    outerAlbedo.needsUpdate = true;
    outerAlbedo.wrapS = outerAlbedo.wrapT = THREE.RepeatWrapping;
    outerAlbedo.repeat.set(9, 9);
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(80, 80),
      new THREE.MeshStandardMaterial({
        map: outerAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.8, 0.8),
        color: 0x241408, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.1
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.55;
    outer.receiveShadow = true;
    scene.add(outer);

    // The side of the pad the survey grid is cut into, so the drop to the
    // surrounding ground reads as a structure rather than a seam.
    var plinth = new THREE.Mesh(
      new THREE.ExtrudeGeometry(padShape((COLS + 0.7) / 2, (ROWS + 0.7) / 2), {
        depth: 0.6, bevelEnabled: false, curveSegments: HAZARD_SEGMENTS
      }),
      new THREE.MeshStandardMaterial({ color: 0x2A1810, roughness: 0.88, metalness: 0.12 })
    );
    plinth.rotation.x = -Math.PI / 2;   // extrudes upward from its own base
    plinth.position.y = -0.74;          // top at -0.14
    // Deliberately casts nothing: it is buried under an opaque pad, and as an
    // occluder it only threw the spotlight's shadow across the hazard floor,
    // which sits below its top face.
    plinth.castShadow = false;
    plinth.receiveShadow = true;
    scene.add(plinth);

    // A low kerb on three sides. The east side is left open: that edge is the
    // exit, and a kerb there would read as a wall.
    var kerbMat = new THREE.MeshStandardMaterial({ color: 0x33221A, roughness: 0.9, metalness: 0.1 });
    [[0, -HR - 0.67], [0, HR + 0.67]].forEach(function (o) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(COLS + 0.34, 0.17, 0.34), kerbMat);
      m.position.set(o[0], 0.085, o[1]);
      m.receiveShadow = true; m.castShadow = true;
      scene.add(m);
    });
    (function westKerb() {
      var m = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.17, ROWS + 0.34), kerbMat);
      m.position.set(-HC - 0.67, 0.085, 0);
      m.receiveShadow = true; m.castShadow = true;
      scene.add(m);
    })();

    /* Scattered surface rock, deterministic and kept off the rock cells, the
       hazards and the driven line. Two jobs: the raised view gets parallax and
       a sense of ground, and the shadows the floodlights throw give the floor
       real depth. */
    var debrisGeos = [
      new THREE.DodecahedronGeometry(0.10, 0),
      new THREE.IcosahedronGeometry(0.09, 0),
      new THREE.TetrahedronGeometry(0.12, 0)
    ];
    var debrisMat = new THREE.MeshStandardMaterial({
      color: 0x6B4429, roughness: 0.96, metalness: 0.05, flatShading: true
    });
    var rockRnd = mulberry(90210);
    var placed = 0, guard = 0;
    while (placed < 46 && guard++ < 900) {
      var dr = rockRnd() * ROWS - 0.5, dc = rockRnd() * COLS - 0.5;
      var tooClose = false;
      for (var q = 0; q < ROCKS.length && !tooClose; q++) {
        if (Math.hypot(dr - ROCKS[q][0], dc - ROCKS[q][1]) < 0.75) tooClose = true;
      }
      // Each hazard has its own broken ground; pad debris would hang over the hole.
      for (var q3 = 0; q3 < DANGER.length && !tooClose; q3++) {
        if (Math.hypot(dr - DANGER[q3][0], dc - DANGER[q3][1]) < DANGER_R + 0.12) tooClose = true;
      }
      for (var q2 = 0; q2 < path.length && !tooClose; q2++) {
        if (Math.hypot(dr - path[q2].row, dc - path[q2].col) < 0.42) tooClose = true;
      }
      if (tooClose) continue;
      var deb = new THREE.Mesh(debrisGeos[placed % debrisGeos.length], debrisMat);
      var dsc = 0.4 + rockRnd() * 0.85;
      deb.scale.set(dsc, dsc * (0.5 + rockRnd() * 0.4), dsc);
      deb.position.set(sx(dc), 0.035 * dsc, sz(dr));
      deb.rotation.set(rockRnd() * 3, rockRnd() * 3, rockRnd() * 3);
      deb.castShadow = true; deb.receiveShadow = true;
      scene.add(deb);
      placed++;
    }

    /* ------------------------------------------------------- floodlights */
    var mastLights = [];
    var lensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(5.0, 4.3, 3.2) });
    var mastMat = new THREE.MeshStandardMaterial({ color: 0x7A6A50, roughness: 0.42, metalness: 0.85 });
    var baseMat = new THREE.MeshStandardMaterial({ color: 0x3A3026, roughness: 0.6, metalness: 0.7 });
    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(function (corner) {
      var x = corner[0] * (HC + 1.75), z = corner[1] * (HR + 1.75);

      var foot = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.26, 0.12, 16), baseMat);
      foot.position.set(x, -0.49, z);
      foot.castShadow = true; foot.receiveShadow = true;
      scene.add(foot);

      var post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, 2.6, 12), mastMat);
      post.position.set(x, 0.85, z);
      post.castShadow = true;
      scene.add(post);

      var head = new THREE.Mesh(new THREE.BoxGeometry(0.30, 0.16, 0.22), baseMat);
      head.position.set(x, 2.14, z);
      head.castShadow = true;
      scene.add(head);

      var lens = new THREE.Mesh(new THREE.SphereGeometry(0.095, 14, 10), lensMat);
      lens.position.set(x - corner[0] * 0.03, 2.07, z - corner[1] * 0.03);
      scene.add(lens);

      // Haze in the air right at the head. The glow itself comes from the
      // bloom pass reading the lens's own brightness, so this stays faint.
      var shaft = new THREE.Mesh(
        new THREE.CylinderGeometry(0.16, 1.8, 2.3, 18, 1, true),
        new THREE.MeshBasicMaterial({
          map: poolTex, color: COLORS.lamp, transparent: true, opacity: 0.022,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      shaft.position.set(x * 0.86, 1.0, z * 0.86);
      scene.add(shaft);

      var light = new THREE.PointLight(COLORS.lamp, 1, 16, 2);
      light.power = MAST_LUMENS;
      light.position.set(x, 2.08, z);
      scene.add(light);
      mastLights.push({ light: light, x: x, z: z });
    });

    /* One shadow-casting lamp, not four. It is parked on whichever mast is
       nearest the rover and aimed at it, and that mast's own light is dimmed by
       the same amount, so the site stays evenly lit while the rover always
       throws a shadow that points the right way. Narrow cone plus normalBias,
       which is the fix for acne that a big negative bias is not. */
    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 16, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.5;
    shadowLight.shadow.camera.far = 16;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, 2.08, 0);
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    /* ------------------------------------------------------ hazard zones
       A zone is broken ground, not a tinted disc. A tint says "scoring
       region"; shattered rock says "this will cost you", which is what the
       environment charges for. The red ring sits exactly on the radius the
       environment penalises and stays the authority on where that edge is.
       Nothing built here reaches past it. */

    /* The height of the broken ground at polar (r, theta) about a zone centre.
       Everything is forced to zero at the rim, so the basin meets the pad on
       the penalty radius and not a centimetre past it.

       The break is carried mostly UPWARD. A deep bowl turns its own floor away
       from four low corner masts and goes black, which would state that the
       lamps do not reach the zone — a claim about sensing, not about the
       penalty. So the buckle is shallow and the drama comes from plates levered
       up into the light and the shadows they throw across it. */
    var HAZARD_FLOOR = -0.22;
    var HAZARD_CRACKS = [0.6, 2.5, 4.3];
    function smooth(t) { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); }
    function hazardHeight(r, theta) {
      var t = clamp(r / DANGER_R, 0, 1);
      var h = -0.105 * (1 - t * t);                                 // a slumped floor
      // Broad sinusoids rather than smooth noise: the surface should break into
      // plates and facets, not roll like a dune.
      h += 0.090 * Math.sin(theta * 3.0 + 0.7) * (1 - t * 0.7);
      h += 0.055 * Math.sin(theta * 5.0 - 1.9) * (1 - t * 0.5);
      h += 0.045 * Math.sin(r * 11.0 + theta * 2.0);
      h += 0.030 * Math.sin(r * 19.0 - theta * 4.0);
      // Radial fissures, deep until they die out short of the rim. They fade at
      // the centre too, or all three would meet and punch a hole.
      var near = Math.min(1, r / 0.3);
      for (var k = 0; k < HAZARD_CRACKS.length; k++) {
        var dth = Math.abs(((theta - HAZARD_CRACKS[k] + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
        if (dth < 0.10) h -= 0.14 * (1 - dth / 0.10) * (1 - Math.pow(t, 5)) * near;
      }
      var edge = 1 - smooth((t - 0.84) / 0.16);
      return Math.max(h * edge, HAZARD_FLOOR);
    }

    /* The broken ground is the SAME ground, lit the same way. Darkening the
       floor inside a zone would say the floodlights do not reach it, and in a
       POMDP viewer that is a claim about the observation model: a reader primed
       by a sensor whose reliability decays with distance reads a dark patch as
       "the agent cannot see here". The hazard costs reward; it does not blind
       anyone. So the basin takes the pad's own soil, normal map and material
       constants, and the danger is carried entirely by geometry — fissures,
       upthrust plates, scree and the shadows they throw — plus the red rim.
       Any darkness inside a zone is cast, not painted.

       The one difference from the pad is that this copy of the grit is never
       grid-painted: the cell lines are survey marks, and the surface that
       carried them here is destroyed. */
    var hazCanvas = groundCanvas();
    var hazAlbedo = new THREE.CanvasTexture(hazCanvas);
    hazAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    hazAlbedo.encoding = THREE.sRGBEncoding;
    var hazMat = new THREE.MeshStandardMaterial({
      map: hazAlbedo, normalMap: groundNormal,
      normalScale: new THREE.Vector2(1.15, 1.15),
      roughness: 0.95, metalness: 0.0,
      envMapIntensity: 0.3,      // identical to the pad floor
      flatShading: true          // every plate catches the floodlights on its own
    });
    // The same stone as the pad's loose surface rock, for the same reason.
    var shardMat = new THREE.MeshStandardMaterial({
      color: 0x6B4429, roughness: 0.96, metalness: 0.05, flatShading: true
    });

    /* The basin surface: a polar mesh whose rim ring is the same polygon the
       pad's hole is, so the two share an edge exactly. */
    function buildBasin() {
      var RINGS = 14, SECT = HAZARD_SEGMENTS;
      var verts = [], uvs = [], idx = [];
      verts.push(0, hazardHeight(0, 0), 0);
      uvs.push(0.5, 0.5);
      for (var j = 1; j <= RINGS; j++) {
        var r = DANGER_R * (j / RINGS);
        for (var i = 0; i < SECT; i++) {
          var th = (i / SECT) * Math.PI * 2;
          var x = Math.cos(th) * r, z = Math.sin(th) * r;
          verts.push(x, hazardHeight(r, th), z);
          uvs.push(0.5 + x / (DANGER_R * 1.1), 0.5 + z / (DANGER_R * 1.1));
        }
      }
      function at(j, i) { return j === 0 ? 0 : 1 + (j - 1) * SECT + (i % SECT); }
      for (var i2 = 0; i2 < SECT; i2++) idx.push(at(0, 0), at(1, i2 + 1), at(1, i2));
      for (var j2 = 1; j2 < RINGS; j2++) {
        for (var i3 = 0; i3 < SECT; i3++) {
          idx.push(at(j2, i3), at(j2, i3 + 1), at(j2 + 1, i3 + 1));
          idx.push(at(j2, i3), at(j2 + 1, i3 + 1), at(j2 + 1, i3));
        }
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
      geo.setIndex(idx);
      geo.computeVertexNormals();
      return geo;
    }

    var hazardRims = [];
    DANGER.forEach(function (d) {
      var x = sx(d[1]), z = sz(d[0]);

      var basin = new THREE.Mesh(buildBasin(), hazMat);
      basin.position.set(x, 0, z);
      basin.receiveShadow = true;
      scene.add(basin);

      /* Loose scree and upthrust plates, seeded deterministically and placed on
         the basin surface. Everything is held inside 0.82 of the radius so that
         no shard, at any tilt, can reach past it and misstate where the cost
         applies. */
      var hz = mulberry(7360 + d[0] * 31 + d[1]);
      for (var n = 0; n < 34; n++) {
        var rr = (0.10 + hz() * 0.72) * DANGER_R, th2 = hz() * Math.PI * 2;
        var shard = new THREE.Mesh(
          n % 3 === 0
            ? new THREE.TetrahedronGeometry(0.055 + hz() * 0.05, 0)
            : new THREE.DodecahedronGeometry(0.04 + hz() * 0.045, 0),
          shardMat
        );
        var scl = 0.7 + hz() * 0.8;
        shard.scale.set(scl, scl * (0.5 + hz() * 0.7), scl);
        shard.position.set(
          x + Math.cos(th2) * rr, hazardHeight(rr, th2) + 0.022, z + Math.sin(th2) * rr
        );
        shard.rotation.set(hz() * 3, hz() * 3, hz() * 3);
        shard.castShadow = true; shard.receiveShadow = true;
        scene.add(shard);
      }
      // Upthrust plates: the ground levered up and snapped, one edge still
      // buried. They stand into the mast light, so the zone is broken by relief
      // and cast shadow rather than by being painted dark.
      for (var sl = 0; sl < 8; sl++) {
        var sr = (0.12 + hz() * 0.5) * DANGER_R, sth = hz() * Math.PI * 2;
        var slab = new THREE.Mesh(
          new THREE.BoxGeometry(0.30 + hz() * 0.24, 0.042, 0.22 + hz() * 0.18), hazMat
        );
        slab.position.set(
          x + Math.cos(sth) * sr,
          hazardHeight(sr, sth) + 0.085 + hz() * 0.05,
          z + Math.sin(sth) * sr
        );
        slab.rotation.set((hz() - 0.5) * 1.1, hz() * Math.PI * 2, 0.7 + hz() * 0.7);
        slab.castShadow = true; slab.receiveShadow = true;
        scene.add(slab);
      }

      var column = new THREE.Mesh(
        new THREE.CylinderGeometry(DANGER_R, DANGER_R, 1.0, 40, 1, true),
        new THREE.MeshBasicMaterial({
          color: COLORS.danger, transparent: true, opacity: 0.10,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      column.position.set(x, 0.5, z);
      scene.add(column);

      var rim = new THREE.Mesh(
        new THREE.RingGeometry(DANGER_R - 0.04, DANGER_R + 0.04, 60),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(3.4, 0.5, 0.35), transparent: true, opacity: 0.4,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      rim.rotation.x = -Math.PI / 2;
      rim.position.set(x, 0.028, z);
      scene.add(rim);
      hazardRims.push(rim);
    });

    /* ---------------------------------------------------------- east exit
       The whole east edge ends the episode, so it is drawn as a lit gate line
       with a pair of markers, not as a single goal pad. */
    var exitGate = new THREE.Mesh(
      new THREE.PlaneGeometry(ROWS, 0.9),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: COLORS.exit, transparent: true, opacity: 0.10,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    exitGate.rotation.y = -Math.PI / 2;
    exitGate.position.set(HC + 0.5, 0.45, 0);
    scene.add(exitGate);

    var exitStrip = new THREE.Mesh(
      new THREE.PlaneGeometry(0.1, ROWS),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(1.9, 1.5, 0.28) })
    );
    exitStrip.rotation.x = -Math.PI / 2;
    exitStrip.position.set(HC + 0.5, 0.035, 0);
    scene.add(exitStrip);

    [-1, 1].forEach(function (side) {
      var pylon = new THREE.Mesh(
        new THREE.CylinderGeometry(0.045, 0.06, 0.8, 10),
        new THREE.MeshStandardMaterial({ color: 0x6B5A2C, roughness: 0.5, metalness: 0.75 })
      );
      pylon.position.set(HC + 0.5, 0.4, side * (HR + 0.3));
      pylon.castShadow = true;
      scene.add(pylon);
      var cap = new THREE.Mesh(
        new THREE.SphereGeometry(0.06, 10, 8),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(3.2, 2.4, 0.5) })
      );
      cap.position.set(HC + 0.5, 0.82, side * (HR + 0.3));
      scene.add(cap);
    });

    /* The exit apron. Driving east off the last column ends the episode, so the
       rover's final drawn position is off the grid; without ground under it, it
       finishes the run hovering in the dark. The slab butts against the pad
       edge rather than overlapping it, because two coplanar surfaces a few
       millimetres apart stripe in a 16-bit depth buffer. It is placed on the
       row the episode actually left from. */
    var exitRow = path.length ? path[path.length - 1].row : 0;
    var apron = new THREE.Mesh(
      new THREE.BoxGeometry(1.6, 0.62, 1.7),
      new THREE.MeshStandardMaterial({ color: 0x4A2E1C, roughness: 0.95, metalness: 0.06 })
    );
    apron.position.set(HC + 0.5 + 0.8, -0.31, sz(exitRow));
    apron.receiveShadow = true; apron.castShadow = true;
    apron.visible = exited[exited.length - 1] === true;
    scene.add(apron);

    // The start cell, marked the way the 2D renderer marks it.
    var startPad = new THREE.Mesh(
      new THREE.RingGeometry(0.26, 0.34, 32),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(1.2, 1.2, 3.4), transparent: true,
        opacity: 0.75, side: THREE.DoubleSide
      })
    );
    startPad.rotation.x = -Math.PI / 2;
    startPad.position.set(sx(world.init_pos[1]), 0.02, sz(world.init_pos[0]));
    scene.add(startPad);

    /* --------------------------------------------------------------- rocks
       Each rock is a boulder cluster with an ore core, and above it a floating
       gauge showing the belief's current P(good) for that rock. The gauge is
       the belief; the boulder's truth marker is the true state, and the two are
       deliberately drawn apart so they can disagree. */
    function drawGauge(cv, id, p, truth, flash) {
      var g = cv.getContext("2d");
      g.clearRect(0, 0, 256, 128);
      g.fillStyle = "rgba(10,12,12,0.78)";
      g.strokeStyle = flash ? "#E88614" : "rgba(210,154,102,0.55)";
      g.lineWidth = flash ? 5 : 3;
      g.beginPath();
      if (g.roundRect) g.roundRect(8, 14, 240, 100, 12); else g.rect(8, 14, 240, 100);
      g.fill(); g.stroke();

      g.font = "600 34px system-ui, sans-serif";
      g.fillStyle = "#EDE5D6";
      g.textBaseline = "middle";
      g.fillText("R" + id, 24, 44);

      g.font = "500 30px ui-monospace, monospace";
      g.textAlign = "right";
      g.fillText(p === null ? "—" : p.toFixed(2), 234, 44);
      g.textAlign = "left";

      // The bar: green is the probability the rock is good, the rest is red.
      // A belief with no probability to show is left empty, not filled: a full
      // grey bar would read as a confident answer rather than as no answer.
      g.fillStyle = "rgba(248,72,72,0.45)";
      g.fillRect(24, 74, 208, 24);
      if (p !== null) {
        g.fillStyle = "#44A044";
        g.fillRect(24, 74, 208 * clamp(p, 0, 1), 24);
      }
      g.strokeStyle = "rgba(237,229,214,0.6)";
      g.lineWidth = 2;
      g.strokeRect(24, 74, 208, 24);
      // Half-way mark: the prior, and the line a greedy sampler would use.
      g.fillStyle = "rgba(237,229,214,0.7)";
      g.fillRect(24 + 104 - 1, 70, 2, 32);
      // The true quality, marked outside the bar so it cannot be mistaken for
      // it. The rover never sees this; the viewer does, which is the point.
      g.fillStyle = "#EDE5D6";
      g.fillRect(truth ? 230 : 22, 66, 4, 40);
    }

    var rockNodes = [];
    var oreGeos = [
      new THREE.DodecahedronGeometry(0.30, 0),
      new THREE.IcosahedronGeometry(0.26, 0),
      new THREE.DodecahedronGeometry(0.22, 0)
    ];
    var boulderMat = new THREE.MeshStandardMaterial({
      color: 0x4E4238, roughness: 0.82, metalness: 0.22, flatShading: true
    });
    ROCKS.forEach(function (rk, i) {
      var group = new THREE.Group();
      group.position.set(sx(rk[1]), 0, sz(rk[0]));
      group.scale.setScalar(1.22);
      scene.add(group);

      var main = new THREE.Mesh(oreGeos[i % oreGeos.length], boulderMat);
      main.position.y = 0.2;
      main.rotation.set(0.5 + i, 0.9 * i, 0.3);
      main.scale.set(1.0, 0.82, 1.0);
      main.castShadow = true; main.receiveShadow = true;
      group.add(main);

      var shardRnd = mulberry(4000 + i * 37);
      for (var k = 0; k < 4; k++) {
        var chip = new THREE.Mesh(
          new THREE.TetrahedronGeometry(0.07 + shardRnd() * 0.07, 0), boulderMat
        );
        var a = shardRnd() * Math.PI * 2, rr = 0.24 + shardRnd() * 0.16;
        chip.position.set(Math.cos(a) * rr, 0.04, Math.sin(a) * rr);
        chip.rotation.set(shardRnd() * 3, shardRnd() * 3, shardRnd() * 3);
        chip.castShadow = true; chip.receiveShadow = true;
        group.add(chip);
      }

      /* The ore core. Its colour tracks the belief, not the truth: red at
         P(good) = 0, green at 1, grey when the belief cannot be reduced to a
         probability at all. */
      var core3d = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.115, 0),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(0.9, 0.9, 0.9) })
      );
      core3d.position.y = 0.40;
      group.add(core3d);

      var coreGlow = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xFFFFFF, transparent: true, opacity: 0.55,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      coreGlow.scale.set(0.8, 0.8, 1);
      coreGlow.position.y = 0.40;
      group.add(coreGlow);

      var coreLight = new THREE.PointLight(0xFFFFFF, 1, 2.0, 2);
      coreLight.power = 22;
      coreLight.position.y = 0.42;
      group.add(coreLight);

      var stem = new THREE.Mesh(
        new THREE.CylinderGeometry(0.006, 0.006, 0.50, 6),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(1.6, 1.3, 0.85), transparent: true, opacity: 0.65
        })
      );
      stem.position.y = 0.70;
      group.add(stem);

      var cv = document.createElement("canvas");
      cv.width = 256; cv.height = 128;
      drawGauge(cv, i, null, false, false);
      var tex = new THREE.CanvasTexture(cv);
      tex.encoding = THREE.sRGBEncoding;
      var gauge = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, color: new THREE.Color(5.5, 5.5, 5.5),
        transparent: true, depthWrite: false, depthTest: true
      }));
      gauge.scale.set(0.86, 0.43, 1);
      gauge.position.y = 1.14;
      group.add(gauge);

      // The check ring: hidden until a recorded check targets this rock.
      var ring = new THREE.Mesh(
        new THREE.RingGeometry(0.3, 0.38, 40),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(3.0, 1.6, 0.36), transparent: true, opacity: 0,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.05;
      group.add(ring);

      rockNodes.push({
        group: group, core: core3d, coreGlow: coreGlow, coreLight: coreLight,
        gauge: gauge, canvas: cv, tex: tex, ring: ring,
        drawn: -1, drawnFlash: false, drawnTruth: null,
        x: sx(rk[1]), z: sz(rk[0])
      });
    });

    /* The height of whatever the rover is standing on: the broken ground
       inside a hazard, flat pad everywhere else. Sampling it fore/aft and
       left/right is what lets the hull sit on the terrain instead of on the
       idea of terrain. */
    function groundHeightAt(wxp, wzp) {
      for (var i = 0; i < DANGER.length; i++) {
        var ddx = wxp - sx(DANGER[i][1]), ddz = wzp - sz(DANGER[i][0]);
        var rr = Math.hypot(ddx, ddz);
        if (rr < DANGER_R) return hazardHeight(rr, Math.atan2(ddz, ddx));
      }
      return 0;
    }

    /* How far into a hazard a continuous position is: 0 outside, 1 well
       inside. It ramps across the boundary rather than switching on, so the
       ride roughens as the rover reaches the rim instead of snapping. */
    function zoneDepth(row, col) {
      var best = 0;
      for (var i = 0; i < DANGER.length; i++) {
        var d = Math.hypot(row - DANGER[i][0], col - DANGER[i][1]);
        best = Math.max(best, clamp((DANGER_R + 0.28 - d) / 0.45, 0, 1));
      }
      return best;
    }

    /* ---------------------------------------------------------------- rover */
    var rover = new THREE.Group();
    rover.scale.setScalar(1.3);
    scene.add(rover);

    // The hull rides on the wheels, so it can bob and pitch without moving them.
    var hull = new THREE.Group();
    rover.add(hull);

    var paintMat = new THREE.MeshStandardMaterial({
      color: 0x1B3ECB, roughness: 0.42, metalness: 0.42     // COLOR_ROBOT, lifted
    });
    var darkMetal = new THREE.MeshStandardMaterial({
      color: 0x2A2522, roughness: 0.55, metalness: 0.75
    });

    var chassis = new THREE.Mesh(new THREE.BoxGeometry(0.60, 0.16, 0.40), paintMat);
    chassis.position.y = 0.165;
    chassis.castShadow = true;
    hull.add(chassis);

    var deck = new THREE.Mesh(
      new THREE.BoxGeometry(0.38, 0.09, 0.32),
      new THREE.MeshStandardMaterial({ color: 0x27309B, roughness: 0.5, metalness: 0.45 })
    );
    deck.position.set(-0.04, 0.29, 0);
    deck.castShadow = true;
    hull.add(deck);

    // Solar deck, angled so it picks up a specular glint when a lamp passes.
    var panel = new THREE.Mesh(
      new THREE.BoxGeometry(0.29, 0.02, 0.29),
      new THREE.MeshStandardMaterial({ color: 0x17233C, roughness: 0.18, metalness: 0.65 })
    );
    panel.position.set(-0.10, 0.35, 0);
    panel.rotation.z = -0.06;
    panel.castShadow = true;
    hull.add(panel);

    var cabin = new THREE.Mesh(
      new THREE.BoxGeometry(0.19, 0.12, 0.25),
      new THREE.MeshStandardMaterial({
        color: 0x2B4E8E, roughness: 0.12, metalness: 0.3,
        emissive: 0x13203A, emissiveIntensity: 0.8
      })
    );
    cabin.position.set(0.06, 0.39, 0);
    cabin.castShadow = true;
    hull.add(cabin);

    /* The sampling drill under the belly: what the sample action uses. It drops
       and spins during a sample step, which is the visible difference between
       sampling and checking. */
    var drill = new THREE.Mesh(
      new THREE.CylinderGeometry(0.028, 0.012, 0.18, 8),
      new THREE.MeshStandardMaterial({ color: 0x8A8378, roughness: 0.35, metalness: 0.9 })
    );
    drill.position.set(-0.02, 0.05, 0);
    hull.add(drill);

    /* The mast and sensor head: the thing that does the checking. The head is
       its own group so it can turn to face the rock being checked. */
    var mastPivot = new THREE.Group();
    mastPivot.position.set(-0.15, 0.46, 0);
    hull.add(mastPivot);
    mastPivot.add(new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.24, 8), darkMetal));
    var sensorHead = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.07, 0.06), darkMetal);
    sensorHead.position.set(0.02, 0.16, 0);
    sensorHead.castShadow = true;
    mastPivot.add(sensorHead);
    var sensorEye = new THREE.Mesh(
      new THREE.SphereGeometry(0.019, 10, 10),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.6, 0.9, 1.3) })
    );
    sensorEye.position.set(0.07, 0.16, 0);
    mastPivot.add(sensorEye);
    var dish = new THREE.Mesh(
      new THREE.CylinderGeometry(0.075, 0.075, 0.012, 16),
      new THREE.MeshStandardMaterial({
        color: 0xC8BCA4, roughness: 0.3, metalness: 0.7, side: THREE.DoubleSide
      })
    );
    dish.position.set(0.055, 0.16, 0);
    dish.rotation.z = Math.PI / 2;
    mastPivot.add(dish);

    [0.13, -0.13].forEach(function (z) {
      var bar = new THREE.Mesh(new THREE.TorusGeometry(0.10, 0.011, 6, 12, Math.PI), darkMetal);
      bar.position.set(-0.04, 0.33, z);
      bar.rotation.y = Math.PI / 2;
      hull.add(bar);
    });

    var wheels = [];
    var wheelGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.095, 18);
    var wheelMat = new THREE.MeshStandardMaterial({ color: 0x171514, roughness: 0.88, metalness: 0.12 });
    var hubMat = new THREE.MeshStandardMaterial({ color: 0x6E5A3C, roughness: 0.4, metalness: 0.85 });
    [[0.19, 0.21], [0.19, -0.21], [-0.19, 0.21], [-0.19, -0.21]].forEach(function (w) {
      var m = new THREE.Mesh(wheelGeo, wheelMat);
      m.position.set(w[0], 0.11, w[1]);
      m.rotation.x = Math.PI / 2;
      m.castShadow = true;
      rover.add(m);
      wheels.push(m);
      // Treads, so the wheels visibly turn instead of just spinning smoothly.
      for (var t = 0; t < 8; t++) {
        var a = (t / 8) * Math.PI * 2;
        var tread = new THREE.Mesh(new THREE.BoxGeometry(0.033, 0.10, 0.024), wheelMat);
        tread.position.set(Math.cos(a) * 0.103, 0, Math.sin(a) * 0.103);
        tread.rotation.y = -a;
        m.add(tread);
      }
      m.add(new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.10, 10), hubMat));
    });

    var lampGeo = new THREE.SphereGeometry(0.04, 10, 10);
    var lampMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.4, 3.1, 2.6) });
    [[0.30, 0.12], [0.30, -0.12]].forEach(function (h) {
      var m = new THREE.Mesh(lampGeo, lampMat);
      m.position.set(h[0], 0.22, h[1]);
      hull.add(m);
      var flare = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xFFF1D6, transparent: true, opacity: 0.45,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      flare.scale.set(0.38, 0.38, 1);
      flare.position.set(h[0] + 0.01, 0.22, h[1]);
      hull.add(flare);
    });

    // Tail lamps, brighter when the rover is at rest in a cell.
    var tails = [];
    [[-0.31, 0.11], [-0.31, -0.11]].forEach(function (t) {
      var m = new THREE.Mesh(
        new THREE.BoxGeometry(0.02, 0.04, 0.065),
        new THREE.MeshBasicMaterial({ color: 0x8A1410 })
      );
      m.position.set(t[0], 0.20, t[1]);
      hull.add(m);
      tails.push(m);
    });

    // Contact shadow. The shadow map handles the cast shadow; this is the dark
    // patch under the hull that keeps the rover from looking pasted on.
    var blob = new THREE.Mesh(
      new THREE.PlaneGeometry(0.95, 0.78),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.42, depthWrite: false
      })
    );
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.008;
    rover.add(blob);

    // The headlights are what make the dark readable from the chase camera.
    var headlight = new THREE.SpotLight(0xFFF1D6, 1, 8, 0.5, 0.6, 2);
    headlight.power = 950;            // lumens, a pair of driving lamps
    headlight.castShadow = true;
    headlight.shadow.mapSize.set(1024, 1024);
    headlight.shadow.camera.near = 0.2;
    headlight.shadow.camera.far = 9;
    headlight.shadow.bias = -0.0008;
    headlight.shadow.normalBias = 0.03;
    headlight.position.set(0.3, 0.25, 0);
    var headTarget = new THREE.Object3D();
    headTarget.position.set(4, -0.18, 0);
    rover.add(headlight);
    rover.add(headTarget);
    headlight.target = headTarget;

    var beamCone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 1.0, 2.7, 20, 1, true),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0xFFE9BE, transparent: true, opacity: 0.026,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    beamCone.rotation.z = Math.PI / 2 + 0.06;
    beamCone.position.set(1.55, 0.21, 0);
    rover.add(beamCone);

    var roverGlow = new THREE.PointLight(0xFFB08A, 0.45, 2.0, 2);
    roverGlow.position.y = 0.3;
    rover.add(roverGlow);

    /* --------------------------------------------------- the sensor beam
       A check is an event, so it gets its own geometry: a lance from the
       sensor dish to the rock, a pulse travelling along it, and a ring at the
       rock that flashes the colour of the bit that came back. All of it is
       driven by the step's progress, so scrubbing shows it too. */
    var beamPos = new Float32Array(6);
    var beamMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(9.0, 4.4, 0.8), transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false
    });
    // A unit cylinder along +y, re-aimed each frame between the dish and the
    // rock. A one-pixel line is not an event; this reads from the board camera.
    var beamLine = new THREE.Mesh(
      new THREE.CylinderGeometry(0.022, 0.022, 1, 8, 1, true), beamMat
    );
    beamLine.geometry.translate(0, 0.5, 0);
    scene.add(beamLine);
    var beamA = new THREE.Vector3(), beamB = new THREE.Vector3(), beamDir = new THREE.Vector3();
    var beamUp = new THREE.Vector3(0, 1, 0);
    function aimBeam() {
      beamA.set(beamPos[0], beamPos[1], beamPos[2]);
      beamB.set(beamPos[3], beamPos[4], beamPos[5]);
      beamDir.subVectors(beamB, beamA);
      var len = Math.max(beamDir.length(), 1e-4);
      beamLine.position.copy(beamA);
      beamLine.quaternion.setFromUnitVectors(beamUp, beamDir.normalize());
      beamLine.scale.set(1, len, 1);
    }

    var PULSES = 26;
    var pulsePos = new Float32Array(PULSES * 3);
    var pulseGeo = new THREE.BufferGeometry();
    pulseGeo.setAttribute("position", new THREE.BufferAttribute(pulsePos, 3));
    var pulseMat = new THREE.PointsMaterial({
      size: 0.1, map: partTex, color: new THREE.Color(3.0, 1.6, 0.36),
      transparent: true, opacity: 0, blending: THREE.AdditiveBlending,
      depthWrite: false, sizeAttenuation: true
    });
    scene.add(new THREE.Points(pulseGeo, pulseMat));

    /* ------------------------------------------------------------------ dust
       Two systems. Motes hang in the floodlight beams and give the air some
       body; the wheel plume is kicked up behind the rover while it drives. */
    var MOTES = 240;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(1357);
    for (var mi0 = 0; mi0 < MOTES; mi0++) {
      motePos[mi0 * 3] = (moteSeed() - 0.5) * (COLS + 3);
      motePos[mi0 * 3 + 1] = 0.10 + moteSeed() * 2.0;
      motePos[mi0 * 3 + 2] = (moteSeed() - 0.5) * (ROWS + 3);
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    scene.add(new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.03, map: partTex, color: 0xFFD8AC, transparent: true, opacity: 0.36,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    })));

    var PLUME = 90;
    var plumePos = new Float32Array(PLUME * 3);
    var plumeLife = new Float32Array(PLUME);
    var plumeVel = new Float32Array(PLUME * 3);
    for (var pz = 0; pz < PLUME; pz++) plumePos[pz * 3 + 1] = -5;
    var plumeGeo = new THREE.BufferGeometry();
    plumeGeo.setAttribute("position", new THREE.BufferAttribute(plumePos, 3));
    scene.add(new THREE.Points(plumeGeo, new THREE.PointsMaterial({
      size: 0.11, map: partTex, color: 0xB3835C, transparent: true, opacity: 0.24,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    })));
    var plumeCursor = 0;
    var plumeRnd = mulberry(5150);

    /* ---------------------------------------------------------------- trail
       The driven path, in the 2D renderer's own path colour. It is the rover's
       history, which in this environment is known exactly. */
    var trailPos = new Float32Array((path.length + 1) * 3);
    var trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setDrawRange(0, 0);
    scene.add(new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      color: new THREE.Color(1.5, 1.5, 4.2), transparent: true, opacity: 0.9
    })));

    core.linearize();

    /* ------------------------------------------------------------ playback */
    var reduceMotion = global.matchMedia
      ? global.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;

    var running = [];
    var total = 0;
    for (var ri = 0; ri < trace.steps.length; ri++) {
      var reward = trace.steps[ri].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, ri);
      }
      running.push(total);
    }

    /* The rover moves between cells during the first part of a step and rests
       for the remainder, which is what makes a discrete grid episode read as
       motion rather than teleporting. Non-move steps keep it parked, and that
       pause is where the check and the drill happen. */
    function samplePose(t) {
      var i0 = Math.floor(clamp(t, 0, path.length - 1));
      var f = clamp(t - i0, 0, 1);
      var here = path[i0];
      var next = path[Math.min(i0 + 1, path.length - 1)];
      var base = smooth(clamp(f / MOVE_FRACTION, 0, 1));

      /* Lurching across broken ground: the rover catches and slips instead of
         gliding. The environment does not slow it down — the penalty is reward,
         and the step still takes one step — so this may not move a step
         boundary. The envelope sin(pi * base) is exactly zero at both ends, so
         every step still starts and ends on the cell the trace says, at the
         moment the trace says. It is a function of the step index only, never
         of wall-clock time, so scrubbing is repeatable.

         Roughness is read at the step's own midpoint rather than at the moving
         position, so the lurch cannot feed back into where the rover is. */
      var rough = zoneDepth((here.row + next.row) / 2, (here.col + next.col) / 2);
      var e = base;
      if (rough > 0 && !reduceMotion) {
        var env = Math.sin(base * Math.PI);
        e = clamp(base + rough * 0.15 * env * Math.sin(base * Math.PI * 3.0 + i0 * 2.39), 0, 1);
      }
      var row = lerp(here.row, next.row, e), col = lerp(here.col, next.col, e);
      return {
        index: i0, f: f, row: row, col: col, x: sx(col), z: sz(row),
        moving: (here.row !== next.row || here.col !== next.col) && f < MOVE_FRACTION,
        rough: zoneDepth(row, col)
      };
    }

    var heading = 0;
    var wheelSpin = 0;

    /** Move the belief gauges to the posterior this step ends on. */
    function updateGauges(index, mix, flashRock, elapsed) {
      var before = marginals[index];
      var after = marginals[Math.min(index + 1, marginals.length - 1)] || before;
      var truth = payload.rock_truth[index] || [];
      for (var k = 0; k < rockNodes.length; k++) {
        var node = rockNodes[k];
        var p = before ? (after ? lerp(before[k], after[k], mix) : before[k]) : null;
        var isTrue = !!truth[k];
        if (p === null || Math.abs(p - node.drawn) > 0.004 ||
            (flashRock === k) !== node.drawnFlash || isTrue !== node.drawnTruth) {
          drawGauge(node.canvas, k, p, isTrue, flashRock === k);
          node.tex.needsUpdate = true;
          node.drawn = p === null ? -1 : p;
          node.drawnFlash = flashRock === k;
          node.drawnTruth = isTrue;
        }
        // The ore core carries the same number as a colour: red at 0, green
        // at 1, and a flat grey when there is no number to carry.
        var cr = 0.5, cg = 0.5, cb = 0.5;
        if (p !== null) {
          cr = lerp(4.4, 0.5, p); cg = lerp(0.55, 3.6, p); cb = lerp(0.5, 0.8, p);
        }
        node.core.material.color.setRGB(cr, cg, cb);
        node.coreGlow.material.color.setRGB(cr * 0.5, cg * 0.5, cb * 0.5);
        node.coreGlow.material.opacity = p === null ? 0.08 : 0.4 + 0.18 * Math.sin(elapsed * 2 + k);
        node.coreLight.color.setRGB(
          clamp(cr / 4.4, 0, 1), clamp(cg / 4.4, 0, 1), clamp(cb / 4.4, 0, 1)
        );
        node.coreLight.power = p === null ? 2 : 46;
        node.gauge.position.y = 1.14 + (reduceMotion ? 0 : Math.sin(elapsed * 1.4 + k * 2) * 0.02);
      }
    }

    /** Stage one recorded check: mast, beam, pulses, and the outcome ring. */
    function stageCheck(check, pose, dt, elapsed) {
      var node = rockNodes[check.rock];
      if (!node) return;
      var px = pose.x, pz = pose.z;
      // The mast turns to the rock, the beam reaches out, the ring flashes.
      var ang = Math.atan2(node.z - pz, node.x - px);
      var local = heading - ang;                   // the hull is already at -heading
      var md = ((local - mastPivot.rotation.y + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
      mastPivot.rotation.y += md * clamp(dt * 9, 0, 1);

      var fire = clamp((pose.f - 0.12) / 0.18, 0, 1);        // beam extends
      var hold = 1 - smooth(clamp((pose.f - 0.72) / 0.22, 0, 1));
      var ex = lerp(px, node.x, fire), ez = lerp(pz, node.z, fire);
      beamPos[0] = px - Math.cos(heading) * 0.195;
      beamPos[1] = 0.81;
      beamPos[2] = pz - Math.sin(heading) * 0.195;
      beamPos[3] = ex; beamPos[4] = lerp(0.81, 0.52, fire); beamPos[5] = ez;
      aimBeam();
      beamMat.opacity = 0.85 * hold;

      for (var pi = 0; pi < PULSES; pi++) {
        var u = ((pi / PULSES + elapsed * 1.6) % 1) * fire;
        pulsePos[pi * 3] = lerp(beamPos[0], ex, u);
        pulsePos[pi * 3 + 1] = lerp(0.81, beamPos[4], u);
        pulsePos[pi * 3 + 2] = lerp(beamPos[2], ez, u);
      }
      pulseGeo.attributes.position.needsUpdate = true;
      pulseMat.opacity = 0.8 * hold;

      // The bit comes back at 45% of the step: the ring snaps to its colour.
      var good = check.observation === "good";
      var got = pose.f > 0.45;
      var ringP = clamp((pose.f - 0.45) / 0.45, 0, 1);
      node.ring.material.opacity = got ? 0.9 * (1 - ringP) : 0.35 + 0.25 * Math.sin(elapsed * 9);
      node.ring.scale.setScalar(got ? 1 + ringP * 1.5 : 1);
      if (got) node.ring.material.color.setRGB(good ? 0.5 : 4.2, good ? 3.4 : 0.6, 0.5);
      else node.ring.material.color.setRGB(3.0, 1.6, 0.36);
    }

    /** The rover's ride: attitude from the terrain under its own corners. */
    function rideTerrain(pose, dt, elapsed, moving) {
      var px = pose.x, pz = pose.z;
      var rough = reduceMotion ? 0 : pose.rough;
      var fwdX = Math.cos(heading), fwdZ = Math.sin(heading);
      var hF = groundHeightAt(px + fwdX * 0.26, pz + fwdZ * 0.26);
      var hB = groundHeightAt(px - fwdX * 0.26, pz - fwdZ * 0.26);
      var hL = groundHeightAt(px - fwdZ * 0.26, pz + fwdX * 0.26);
      var hR = groundHeightAt(px + fwdZ * 0.26, pz - fwdX * 0.26);
      var judder = rough * (Math.sin(elapsed * 23.0) * 0.6 + Math.sin(elapsed * 37.0 + 1.7) * 0.4);

      var pitch = (hB - hF) * 2.2 + judder * 0.10 * (0.4 + moving);
      var roll = (hR - hL) * 2.4 + Math.sin(elapsed * 19.0 + 2.2) * rough * 0.13 * (0.4 + moving);
      var heave = (hF + hB + hL + hR) * 0.25 + judder * 0.022;
      var snap = clamp(dt * (8 + rough * 26), 0, 1);      // stiffer, so it jolts
      hull.rotation.x = lerp(hull.rotation.x, moving ? 0.03 + pitch : pitch, snap);
      hull.rotation.z = lerp(hull.rotation.z, roll, snap);
      hull.position.y =
        (reduceMotion ? 0 : Math.sin(elapsed * 7.5) * 0.005 * (moving ? 1 : 0.2)) + heave;

      /* Wheel slip. Over rubble the wheels turn faster than the rover advances
         and they catch: the spin runs ahead of ground speed and stutters. The
         rover still arrives on time — only the wheels are lying. */
      var slip = 1 + rough * 2.0 + rough * Math.sin(elapsed * 27.0) * 0.9;
      wheelSpin += dt * 5.5 *
        (moving ? slip : rough * 0.8 * Math.max(0, Math.sin(elapsed * 13.0)));
      for (var w = 0; w < wheels.length; w++) {
        wheels[w].rotation.y = wheelSpin;
        wheels[w].position.y =
          0.11 + (reduceMotion ? 0 : Math.sin(elapsed * 29.0 + w * 1.9) * rough * 0.012);
      }
      return rough;
    }

    /** Dust: motes drift and wrap; the plume spawns behind the rear wheels. */
    function updateDust(pose, dt, elapsed, moving, rough) {
      if (reduceMotion) return;
      for (var mi = 0; mi < MOTES; mi++) {
        motePos[mi * 3] += Math.sin(elapsed * 0.3 + mi) * 0.0014;
        motePos[mi * 3 + 1] += 0.0020;
        if (motePos[mi * 3 + 1] > 2.2) motePos[mi * 3 + 1] = 0.10;
      }
      moteGeo.attributes.position.needsUpdate = true;

      // Slipping wheels throw grit whether or not the rover is making ground.
      var spawn = moving ? 2 + Math.round(rough * 4) : Math.round(rough * 3);
      var kick = 1 + rough * 1.9;
      for (var kk = 0; kk < spawn; kk++) {
        var idx = plumeCursor++ % PLUME;
        plumePos[idx * 3] =
          pose.x - Math.cos(heading) * 0.22 + (plumeRnd() - 0.5) * 0.12 * kick;
        plumePos[idx * 3 + 1] = 0.05;
        plumePos[idx * 3 + 2] =
          pose.z - Math.sin(heading) * 0.22 + (plumeRnd() - 0.5) * 0.12 * kick;
        plumeVel[idx * 3] = (plumeRnd() - 0.5) * 0.12 * kick;
        plumeVel[idx * 3 + 1] = (0.14 + plumeRnd() * 0.14) * kick;
        plumeVel[idx * 3 + 2] = (plumeRnd() - 0.5) * 0.12 * kick;
        plumeLife[idx] = 1;
      }
      for (var pj = 0; pj < PLUME; pj++) {
        if (plumeLife[pj] <= 0) continue;
        plumeLife[pj] -= dt * 1.1;
        plumePos[pj * 3] += plumeVel[pj * 3] * dt;
        plumePos[pj * 3 + 1] += plumeVel[pj * 3 + 1] * dt;
        plumePos[pj * 3 + 2] += plumeVel[pj * 3 + 2] * dt;
        plumeVel[pj * 3 + 1] *= 0.96;
        if (plumeLife[pj] <= 0) plumePos[pj * 3 + 1] = -5;
      }
      plumeGeo.attributes.position.needsUpdate = true;
    }

    function drawTrail(pose) {
      var count = 0;
      for (var i = 0; i <= pose.index; i++) {
        trailPos[count * 3] = sx(path[i].col);
        trailPos[count * 3 + 1] = 0.05;
        trailPos[count * 3 + 2] = sz(path[i].row);
        count++;
      }
      trailPos[count * 3] = pose.x;
      trailPos[count * 3 + 1] = 0.05;
      trailPos[count * 3 + 2] = pose.z;
      count++;
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.setDrawRange(0, count);
    }

    /** The one shadow caster follows the mast nearest the rover. */
    function placeShadowLamp(px, pz) {
      var nearest = null, nd = Infinity;
      for (var ml = 0; ml < mastLights.length; ml++) {
        var L = mastLights[ml];
        L.light.power = MAST_LUMENS;
        var d2 = (L.x - px) * (L.x - px) + (L.z - pz) * (L.z - pz);
        if (d2 < nd) { nd = d2; nearest = L; }
      }
      if (!nearest) return;
      shadowLight.position.set(nearest.x, 2.08, nearest.z);
      shadowLight.target.position.set(px, 0.1, pz);
      shadowLight.target.updateMatrixWorld();
      var share = 0.6 * clamp(1 - (Math.sqrt(nd) - 2.0) / 6.0, 0.1, 1);
      shadowLight.power = MAST_LUMENS * share;
      nearest.light.power = MAST_LUMENS * (1 - share);
    }

    /** The HUD's belief line: the same marginals the gauges show. */
    function beliefLine(index) {
      if (!marginals[index]) return gaps[index];
      if (!numRocks) return "no rocks";
      var parts = [];
      for (var k = 0; k < numRocks; k++) {
        parts.push("R" + k + " " + marginals[index][k].toFixed(2));
      }
      return "P(good) " + parts.join("  ");
    }

    return {
      steps: path.length,

      /* Framing scales with the world, because a trace decides how big the
         board is: a distance that frames a 5x5 pad leaves an 11x11 one running
         off the canvas. The constant term is the margin for the props, which do
         not scale with the grid — a floodlight mast is the same height on any
         board, so a small board needs proportionally more headroom. */
      camera: {
        board: [0, Math.max(ROWS, COLS) * 1.12 + 2.0, Math.max(ROWS, COLS) * 1.0 + 2.0],
        top: [0, Math.max(ROWS, COLS) * 1.65 + 2.35, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed) {
        var pose = samplePose(t);
        var px = pose.x, pz = pose.z;
        var step = trace.steps[pose.index] || {};
        var check = payload.checks[pose.index] || null;

        // Heading follows the direction of travel, so the body turns into the
        // corner rather than snapping between the four step directions, and a
        // step that does not move keeps the last heading rather than spinning.
        var next = path[Math.min(pose.index + 1, path.length - 1)];
        var here = path[pose.index];
        var dRow = next.row - here.row, dCol = next.col - here.col;
        if (dRow !== 0 || dCol !== 0) {
          // dRow is the scene z step and dCol the scene x step, so this is
          // atan2(dz, dx) in scene space despite the grid names.
          var target = Math.atan2(dRow, dCol);
          var diff = ((target - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          heading += diff * clamp(dt * 8, 0, 1);
        }
        rover.position.set(px, 0, pz);
        rover.rotation.y = -heading;

        var moving = pose.moving ? 1 : 0;
        var rough = rideTerrain(pose, dt, elapsed, moving);

        for (var tl = 0; tl < tails.length; tl++) {
          if (moving) tails[tl].material.color.setRGB(0.5, 0.06, 0.04);
          else tails[tl].material.color.setRGB(2.2, 0.2, 0.11);
        }

        // The drill drops and spins through the resting half of a sample step.
        if (step.action === 0) {
          var dz = smooth(clamp((pose.f - 0.2) / 0.3, 0, 1)) *
            (1 - smooth(clamp((pose.f - 0.75) / 0.25, 0, 1)));
          drill.position.y = 0.05 - dz * 0.06;
          drill.rotation.y += dt * 26 * dz;
        } else {
          drill.position.y = 0.05;
        }

        placeShadowLamp(px, pz);

        // Clear every ring first: a check only writes its own, and a stale ring
        // left on another rock reads as a second check that never happened.
        for (var rz = 0; rz < rockNodes.length; rz++) rockNodes[rz].ring.material.opacity = 0;
        if (check) {
          stageCheck(check, pose, dt, elapsed);
        } else {
          beamMat.opacity = 0;
          pulseMat.opacity = 0;
          // The mast drifts back to facing forward when nothing is checked.
          var back = ((0 - mastPivot.rotation.y + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          mastPivot.rotation.y += back * clamp(dt * 3, 0, 1);
        }

        // The gauge moves from this step's posterior to the next one's across
        // the step, so a belief update is something you watch happen rather
        // than a jump between frames.
        updateGauges(pose.index, smooth(clamp((pose.f - 0.45) / 0.3, 0, 1)),
          check ? check.rock : -1, elapsed);

        updateDust(pose, dt, elapsed, moving, rough);
        drawTrail(pose);

        if (!reduceMotion) {
          var pulse = 0.55 + Math.sin(elapsed * 2.1) * 0.18;
          for (var h = 0; h < hazardRims.length; h++) hazardRims[h].material.opacity = pulse;
          exitGate.material.opacity = 0.09 + Math.sin(elapsed * 1.3) * 0.02;
        }

        var name = world.action_names[step.action];
        return {
          follow: { x: px, z: pz, heading: heading },
          step: pose.index,
          action: step.action === null || step.action === undefined
            ? "—"
            : String(step.action) + (name ? " " + name : ""),
          // The HUD labels these x and y; on this board x is the column, which
          // grows east, and y is the row, which grows south.
          x: pose.col,
          y: pose.row,
          reward: step.reward,
          ret: running[pose.index],
          belief: beliefLine(pose.index)
        };
      }
    };
  }

  V.scenes["rock_sample.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for four
    // 2100 lm masts over a small pad; it is not a knob to remove.
    exposure: 0.105
  };
})(window);
