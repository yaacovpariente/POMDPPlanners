/* SPDX-License-Identifier: MIT
 *
 * T-Maze scene module.
 *
 * Builds the corridor from a trace's `payload.world` block and moves it from
 * the trace's recorded states, cue phases and beliefs. Nothing here is
 * invented: there is no fallback episode, no hand-placed waypoint and no
 * synthetic belief. If a field the scene needs is missing from the trace, the
 * scene says so rather than drawing something plausible.
 *
 * Two things are drawn differently here than in any other scene, and both are
 * forced by what the environment is:
 *
 *  1. The belief is TWO NUMBERS, not a cloud. The hidden state is one bit —
 *     which arm pays — and the agent's position is observable, so a particle
 *     cloud on the floor would be a lie about the shape of the uncertainty.
 *     It is drawn as one labelled column per arm, standing at the arm it is a
 *     belief about, filled to P(goal = that side). The probability is summed
 *     out of core's own particle payload, over the goal-side slot the exporter
 *     names — never re-derived from the true state.
 *  2. The cue is an object with a phase. It fires once, one cell above the
 *     start, and the next action consumes it; after that the reading exists
 *     only in the belief. The pylon is lit while the trace says the phase is
 *     "emitting" and dark afterwards, and the agent's lantern takes the cue's
 *     colour at that moment and keeps it. That carried light is the memory,
 *     and without the phase in the trace the page could not show it.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  // The renderer's own constants from maze_visualizer.py, so the viewer reads
  // as the same world as the GIF.
  var COLORS = {
    corridor: 0xF4F1E8,
    agent: 0xC74636,
    belief: 0xE7AE38,
    goal: 0x278365,
    cue: 0x1F77B4,
    lamp: 0xFFE9BE
  };

  // World units per maze cell, and the knee-high walls. A full-height maze
  // seen from a raised camera is a picture of a wall.
  var CELL = 2.2;
  var WALL_H = 0.92;
  var WALL_T = 0.22;

  function smoothstep(e0, e1, x) {
    var t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  /* The corridor floor is painted first and the normal map derived from it, so
     the screed catches a lamp from the side the lamp is actually on. */
  function floorCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(20260919);

    g.fillStyle = "#CFCABB";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 900; i++) {
      g.fillStyle = "rgba(" + (188 + rnd() * 40 | 0) + "," + (182 + rnd() * 38 | 0) + "," +
        (168 + rnd() * 36 | 0) + ",0.30)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, 12 + rnd() * 90, 0, Math.PI * 2); g.fill();
    }
    // Aggregate: a lit cap and a dark side, which the normal map turns into
    // real relief under a raking lamp.
    for (var p = 0; p < 1600; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.2 + rnd() * 3.4;
      g.fillStyle = "rgba(92,88,78,0.35)";
      g.beginPath(); g.arc(px + pr * 0.35, py + pr * 0.35, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(246,243,233,0.45)";
      g.beginPath(); g.arc(px, py, pr * 0.85, 0, Math.PI * 2); g.fill();
    }
    for (var j = 0; j < 900; j++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(255,255,255,0.08)" : "rgba(40,36,30,0.10)";
      g.fillRect(rnd() * s, rnd() * s, 3, 3);
    }
    return cv;
  }

  /* Text painted onto a canvas and hung in the scene. The arm labels and the
     belief numbers have to be readable from every camera, and a sprite is the
     only thing that stays readable at every angle. */
  function paintLabel(canvas, text, hex, px) {
    var g = canvas.getContext("2d");
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.font = "700 " + px + "px 'Chakra Petch', system-ui, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillStyle = hex;
    g.fillText(text, canvas.width / 2, canvas.height / 2);
  }

  function labelSprite(text, hex, px, glow) {
    var cv = document.createElement("canvas");
    cv.width = 256; cv.height = 128;
    paintLabel(cv, text, hex, px);
    var tex = new THREE.CanvasTexture(cv);
    var mat = new THREE.SpriteMaterial({
      map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
    });
    // Above 1.0 on purpose: these are emitters, so the bright pass finds them.
    mat.color.setScalar(glow || 2.0);
    var sprite = new THREE.Sprite(mat);
    sprite.userData.canvas = cv;
    sprite.userData.tex = tex;
    return sprite;
  }

  function repaintLabel(sprite, text, hex, px) {
    if (sprite.userData.text === text && sprite.userData.hex === hex) return;
    sprite.userData.text = text;
    sprite.userData.hex = hex;
    paintLabel(sprite.userData.canvas, text, hex, px);
    sprite.userData.tex.needsUpdate = true;
  }

  /* The cue's face: a chevron pointing at the arm the cue named. A cue that
     was never read leaves the board blank — there is nothing to point at. */
  function chevronTexture(side) {
    var s = 256;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    g.fillStyle = "#07121C";
    g.fillRect(0, 0, s, s);
    if (side === "left" || side === "right") {
      g.strokeStyle = "#8FD4FF";
      g.lineWidth = 26;
      g.lineCap = "round";
      g.lineJoin = "round";
      var sign = side === "left" ? -1 : 1;
      for (var k = 0; k < 2; k++) {
        var ox = s / 2 + sign * (k * 52 - 26);
        g.beginPath();
        g.moveTo(ox - sign * 34, s / 2 - 46);
        g.lineTo(ox + sign * 34, s / 2);
        g.lineTo(ox - sign * 34, s / 2 + 46);
        g.stroke();
      }
    }
    var tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }

  /**
   * P(goal = left) for one recorded belief, read out of core's own payload.
   *
   * The goal side is one slot of a state vector, so a particle belief answers
   * this exactly: sum the weights of the particles whose goal slot is the left
   * encoding. A belief shape that cannot answer it is reported by name and
   * drawn as nothing, rather than replaced by 0.5, which would look like a
   * planner that had learnt nothing.
   *
   * @param {Object} belief   One entry of payload.beliefs.
   * @param {number} slot     Index of the goal-side slot in a state vector.
   * @param {number} leftValue Encoding of the left goal side.
   * @returns {{p: (number|null), label: string}}
   */
  function goalSideProbability(belief, slot, leftValue) {
    if (!belief) return { p: null, label: "—" };

    if (belief.kind === "particles") {
      var total = 0, left = 0, usable = 0;
      for (var i = 0; i < belief.particles.length; i++) {
        var particle = belief.particles[i];
        if (!Array.isArray(particle) || particle.length <= slot) continue;
        var weight = belief.weights[i];
        total += weight;
        usable++;
        if (particle[slot] === leftValue) left += weight;
      }
      if (!usable || total <= 0) {
        // Particles this scene cannot read the goal side out of. Naming the
        // count is diagnosable; a number here would not be.
        return { p: null, label: belief.num_particles + " particles, no goal side" };
      }
      var p = left / total;
      var label = "P(left) " + p.toFixed(2) + " · P(right) " + (1 - p).toFixed(2);
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " of " + belief.num_particles + ")";
      }
      return { p: p, label: label };
    }

    if (belief.kind === "particle_batch") {
      // Several beliefs held together for a vectorized planner. Merging its
      // members would show a belief that was never anyone's.
      return { p: null, label: "batch of " + belief.batch_size + " beliefs, not drawn" };
    }

    // The goal side is a discrete label, so a Gaussian over it has no reading
    // to give; and an unsupported class is named so the gap is diagnosable.
    return { p: null, label: "not readable (" + (belief.belief_class || belief.kind) + ")" };
  }

  /**
   * Build the T-Maze world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind t_maze.v1.
   * @returns {Object} The scene module the player drives.
   */
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;

    var STEM = world.stem_length;
    var ARM = world.arm_length;
    var slots = world.state_slots;
    var GOAL_LEFT_VALUE = world.goal_left_value;

    var cells = world.cells;
    var cellKey = {};
    cells.forEach(function (c) { cellKey[c[0] + "," + c[1]] = true; });
    function valid(x, y) { return !!cellKey[x + "," + y]; }

    // Centre the T on the origin: the stem runs 0..STEM in y, so shift by half.
    // Every camera mode then works without knowing the maze's shape.
    var ZOFF = (STEM / 2) * CELL;
    function wx(x) { return x * CELL; }
    function wz(y) { return ZOFF - y * CELL; }

    scene.background = new THREE.Color(0x05070B).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x070A0F, 0.022);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x27374A, 0x08070A, 0.75));
    var moon = new THREE.DirectionalLight(0x8FA3C8, 0.5);
    moon.position.set(-7, 10, 5);
    scene.add(moon);
    core.buildNightEnvironment([
      [0.0, "#0A1020"], [0.45, "#121A26"], [0.55, "#1E2430"], [1.0, "#060608"]
    ]);

    /* A lamp over the junction. The arms have to be visible from the stem —
       the task is not "find the arms", it is "know which one pays" — but it is
       warm and neutral, so it says nothing about the goal side. */
    var junctionLamp = new THREE.PointLight(COLORS.lamp, 1, 9, 2);
    junctionLamp.power = 900;
    junctionLamp.position.set(0, 2.5, wz(STEM));
    scene.add(junctionLamp);

    var poolTex = V.radialTexture(0.85, 0.42);
    var glowTex = V.radialTexture(0.9, 0.3);

    // ── the corridor ────────────────────────────────────────────────────
    var fCanvas = floorCanvas();
    var floorNormal = V.normalMapFrom(renderer, fCanvas, 2.0);
    var floorAlbedo = new THREE.CanvasTexture(fCanvas);
    floorAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    floorAlbedo.encoding = THREE.sRGBEncoding;     // colour data, not linear data
    floorAlbedo.wrapS = floorAlbedo.wrapT = THREE.RepeatWrapping;
    floorNormal.wrapS = floorNormal.wrapT = THREE.RepeatWrapping;

    var slabMat = new THREE.MeshStandardMaterial({
      map: floorAlbedo, normalMap: floorNormal,
      normalScale: new THREE.Vector2(0.8, 0.8),
      color: COLORS.corridor, roughness: 0.88, metalness: 0.0, envMapIntensity: 0.3
    });

    /* One slab per valid cell rather than one shape, because the seams between
       them are what say the maze is discrete — which is what the GIF draws. */
    cells.forEach(function (c) {
      var slab = new THREE.Mesh(new THREE.BoxGeometry(CELL, 0.16, CELL), slabMat);
      slab.position.set(wx(c[0]), -0.08, wz(c[1]));
      slab.receiveShadow = true;
      slab.castShadow = true;
      scene.add(slab);
    });

    /* Walls close every cell edge with no valid neighbour behind it — the same
       rule the environment uses to reject a move, so what you see is what the
       agent may do. The south face of the start cell is left open so the board
       camera can see into the stem. */
    var wallMat = new THREE.MeshStandardMaterial({
      color: 0x2E3B49, roughness: 0.72, metalness: 0.18, envMapIntensity: 0.5
    });
    var capMat = new THREE.MeshStandardMaterial({
      color: 0x55697D, roughness: 0.45, metalness: 0.4, envMapIntensity: 0.9
    });
    var DIRS = [[0, 1], [0, -1], [-1, 0], [1, 0]];
    var startCell = world.start_cell;
    cells.forEach(function (c) {
      DIRS.forEach(function (d) {
        if (valid(c[0] + d[0], c[1] + d[1])) return;
        if (c[0] === startCell[0] && c[1] === startCell[1] && d[1] === -1) return;
        var along = d[0] !== 0;                                // wall runs in z
        var w = along ? WALL_T : CELL + WALL_T;
        var depth = along ? CELL + WALL_T : WALL_T;
        var px = wx(c[0]) + d[0] * (CELL / 2);
        var pz = wz(c[1]) - d[1] * (CELL / 2);
        var wall = new THREE.Mesh(new THREE.BoxGeometry(w, WALL_H, depth), wallMat);
        wall.position.set(px, WALL_H / 2, pz);
        wall.castShadow = true; wall.receiveShadow = true;
        scene.add(wall);
        // A brighter coping. It is the only part of a wall a raking lamp
        // really catches, and it draws the plan of the T from above.
        var cap = new THREE.Mesh(new THREE.BoxGeometry(w + 0.05, 0.05, depth + 0.05), capMat);
        cap.position.set(px, WALL_H + 0.025, pz);
        cap.castShadow = true;
        scene.add(cap);
      });
    });

    /* Ground carrying on past the maze. Without it the T is a lit slab
       floating in a void. Kept 0.6 below the slabs so no two large surfaces
       sit within a few cm of each other — the depth buffer here is 16-bit. */
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(90, 90),
      new THREE.MeshStandardMaterial({
        map: floorAlbedo, normalMap: floorNormal,
        normalScale: new THREE.Vector2(0.7, 0.7),
        color: 0x1A1F26, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.15
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.62;
    outer.receiveShadow = true;
    scene.add(outer);

    var rockRnd = mulberry(90210);
    var rockGeos = [
      new THREE.DodecahedronGeometry(0.3, 0),
      new THREE.IcosahedronGeometry(0.26, 0),
      new THREE.TetrahedronGeometry(0.34, 0)
    ];
    var rockMat = new THREE.MeshStandardMaterial({
      color: 0x3E4650, roughness: 0.95, metalness: 0.06, flatShading: true, envMapIntensity: 0.4
    });
    var placedRocks = 0, rockGuard = 0;
    while (placedRocks < 42 && rockGuard++ < 900) {
      var rxw = (rockRnd() - 0.5) * 26;
      var rzw = (rockRnd() - 0.5) * 24;
      var inside = false;
      for (var ci = 0; ci < cells.length && !inside; ci++) {
        if (Math.abs(rxw - wx(cells[ci][0])) < CELL * 0.9 &&
            Math.abs(rzw - wz(cells[ci][1])) < CELL * 0.9) inside = true;
      }
      if (inside) continue;
      var rock = new THREE.Mesh(rockGeos[placedRocks % rockGeos.length], rockMat);
      var rs = 0.5 + rockRnd() * 1.5;
      rock.scale.set(rs, rs * (0.4 + rockRnd() * 0.5), rs);
      rock.position.set(rxw, -0.5 + 0.06 * rs, rzw);
      rock.rotation.set(rockRnd() * 3, rockRnd() * 3, rockRnd() * 3);
      rock.castShadow = true; rock.receiveShadow = true;
      scene.add(rock);
      placedRocks++;
    }

    // ── the cue ─────────────────────────────────────────────────────────
    /* The whole task is this object: a lit pylon standing in the cue cell that
       fires once, on entry, and is consumed by the next action. It is stood
       against the wall rather than on the centre line, because the agent walks
       through this cell. */
    var cueGroup = new THREE.Group();
    cueGroup.position.set(wx(world.cue_cell[0]) - CELL * 0.33, 0, wz(world.cue_cell[1]));
    scene.add(cueGroup);

    var cueMetal = new THREE.MeshStandardMaterial({
      color: 0x3A4A5A, roughness: 0.38, metalness: 0.88
    });
    var cueFoot = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.32, 0.1, 18), cueMetal);
    cueFoot.position.y = 0.05;
    cueFoot.castShadow = true; cueFoot.receiveShadow = true;
    cueGroup.add(cueFoot);

    var cueMast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 1.5, 12), cueMetal);
    cueMast.position.y = 0.8;
    cueMast.castShadow = true;
    cueGroup.add(cueMast);

    // The lens. Its colour is driven every frame: far above 1.0 while
    // emitting, so the bright pass blooms it, and nearly black once consumed.
    var cueLensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0, 0, 0) });
    var cueLens = new THREE.Mesh(new THREE.SphereGeometry(0.17, 18, 14), cueLensMat);
    cueLens.position.y = 1.58;
    cueGroup.add(cueLens);

    var cueCage = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.025, 8, 20), cueMetal);
    cueCage.position.y = 1.58;
    cueCage.rotation.x = Math.PI / 2;
    cueCage.castShadow = true;
    cueGroup.add(cueCage);

    // The sign board carries the reading the episode actually received, not
    // the true goal side: the cue is allowed to lie, and drawing the truth
    // here would hide exactly the episodes where it did.
    var signMat = new THREE.MeshStandardMaterial({
      map: chevronTexture(payload.cue_reading),
      emissiveMap: chevronTexture(payload.cue_reading),
      emissive: new THREE.Color(0x2E9BE0),
      emissiveIntensity: 0.0,
      color: 0x0A141E, roughness: 0.4, metalness: 0.25
    });
    var signPivot = new THREE.Group();
    signPivot.rotation.y = -0.42;            // angled down the stem, at the agent
    cueGroup.add(signPivot);
    var sign = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.66, 0.06), signMat);
    sign.position.set(0, 1.12, 0.26);
    sign.castShadow = true;
    signPivot.add(sign);
    var signFrame = new THREE.Mesh(new THREE.BoxGeometry(0.73, 0.73, 0.05), cueMetal);
    signFrame.position.set(0, 1.12, 0.22);
    signPivot.add(signFrame);

    // Haze around the lens while it fires, so the cue reads as a light source
    // and not a bright decal.
    var cueHaze = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22, 2.2, 1.7, 22, 1, true),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: COLORS.cue, transparent: true, opacity: 0.0,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    cueHaze.position.y = 0.8;
    cueGroup.add(cueHaze);

    var cueLight = new THREE.PointLight(0x58B4F0, 1, 12, 2);
    cueLight.power = 0;
    cueLight.position.y = 1.55;
    cueGroup.add(cueLight);

    // A pilot lamp on the foot: blue while the cue is live, dull red once
    // spent, so a still frame still says which phase the episode is in.
    var pilotMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0, 0, 0) });
    var pilot = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), pilotMat);
    pilot.position.set(0.0, 0.16, 0.3);
    cueGroup.add(pilot);

    var cueLabel = labelSprite("CUE", "#8FD4FF", 46, 1.6);
    cueLabel.scale.set(1.15, 0.58, 1);
    cueLabel.position.set(-0.55, 2.1, 0.1);
    cueGroup.add(cueLabel);

    // ── start and junction ──────────────────────────────────────────────
    var startRing = new THREE.Mesh(
      new THREE.RingGeometry(0.5, 0.66, 40),
      new THREE.MeshBasicMaterial({
        color: 0xBFC4C9, transparent: true, opacity: 0.55, side: THREE.DoubleSide
      })
    );
    startRing.rotation.x = -Math.PI / 2;
    startRing.position.set(wx(startCell[0]), 0.012, wz(startCell[1]));
    scene.add(startRing);

    var startLabel = labelSprite("START", "#C8CDD3", 38, 1.2);
    startLabel.scale.set(1.5, 0.75, 1);
    startLabel.position.set(wx(startCell[0]), 0.5, wz(startCell[1]) + 0.75);
    scene.add(startLabel);

    // The junction: where the memory has to be cashed in. From the agent's
    // side it is the only landmark that is not identical corridor.
    var junctionMark = new THREE.Mesh(
      new THREE.RingGeometry(0.62, 0.74, 4),
      new THREE.MeshBasicMaterial({
        color: 0x6E86A0, transparent: true, opacity: 0.5, side: THREE.DoubleSide
      })
    );
    junctionMark.rotation.x = -Math.PI / 2;
    junctionMark.rotation.z = Math.PI / 4;
    junctionMark.position.set(wx(world.junction[0]), 0.012, wz(world.junction[1]));
    scene.add(junctionMark);

    // ── endpoints and the belief columns ────────────────────────────────
    /* Both arms get exactly the same pad: from the agent's point of view they
       are indistinguishable, and the scene has to say that. The green star on
       the paying one is observer furniture the agent has no access to. */
    var endpoints = [
      { side: "left", cell: world.left_endpoint },
      { side: "right", cell: world.right_endpoint }
    ];
    var beliefColumns = [];

    endpoints.forEach(function (ep) {
      var x = wx(ep.cell[0]), z = wz(ep.cell[1]);

      var pad = new THREE.Mesh(
        new THREE.CylinderGeometry(0.78, 0.78, 0.05, 36),
        new THREE.MeshStandardMaterial({
          color: 0x8E97A4, roughness: 0.55, metalness: 0.35, envMapIntensity: 0.8
        })
      );
      pad.position.set(x, 0.035, z);
      pad.receiveShadow = true;
      scene.add(pad);

      var padRim = new THREE.Mesh(
        new THREE.RingGeometry(0.78, 0.86, 44),
        new THREE.MeshBasicMaterial({
          color: 0x9FB2C6, transparent: true, opacity: 0.6, side: THREE.DoubleSide
        })
      );
      padRim.rotation.x = -Math.PI / 2;
      padRim.position.set(x, 0.07, z);
      scene.add(padRim);

      // A lamp over each endpoint, equally dim on both sides. It is what makes
      // the arms visible from the junction, and it gives away nothing. Offset
      // off the cell centre, because parked overhead its glare sat inside the
      // belief column and ate the one reading the column exists for.
      var armLight = new THREE.PointLight(COLORS.lamp, 1, 7.5, 2);
      armLight.power = 820;
      armLight.position.set(x + (ep.side === "left" ? -0.45 : 0.45), 2.1, z + 0.72);
      scene.add(armLight);

      /* The belief, as one column per hypothesis: a glass tube with a fill
         whose height is P(goal = this side), standing at the arm it is a
         belief about. */
      var tube = new THREE.Mesh(
        new THREE.CylinderGeometry(0.19, 0.19, 1.9, 20, 1, true),
        new THREE.MeshStandardMaterial({
          color: 0x93A3B4, roughness: 0.12, metalness: 0.1,
          transparent: true, opacity: 0.16, side: THREE.DoubleSide, envMapIntensity: 1.6
        })
      );
      tube.position.set(x, 1.05, z);
      scene.add(tube);

      var fillMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(COLORS.belief).convertSRGBToLinear().multiplyScalar(2.4),
        transparent: true, opacity: 0.92, blending: THREE.AdditiveBlending, depthWrite: false
      });
      var fill = new THREE.Mesh(new THREE.CylinderGeometry(0.155, 0.155, 1, 18), fillMat);
      fill.position.set(x, 0.1, z);
      scene.add(fill);

      var capRing = new THREE.Mesh(
        new THREE.TorusGeometry(0.2, 0.022, 8, 22),
        new THREE.MeshStandardMaterial({ color: 0x8A97A6, roughness: 0.3, metalness: 0.85 })
      );
      capRing.rotation.x = Math.PI / 2;
      capRing.position.set(x, 2.0, z);
      scene.add(capRing);

      var sideLabel = labelSprite(ep.side === "left" ? "L" : "R", "#FFD98A", 74, 1.7);
      sideLabel.scale.set(0.72, 0.72, 1);
      sideLabel.position.set(x, 2.42, z);
      scene.add(sideLabel);

      var numLabel = labelSprite("—", "#9DA8B4", 44, 1.5);
      numLabel.scale.set(1.3, 0.65, 1);
      numLabel.position.set(x + (ep.side === "left" ? -0.88 : 0.88), 1.75, z);
      scene.add(numLabel);

      beliefColumns.push({ side: ep.side, fill: fill, num: numLabel });

      if (ep.side === payload.goal_side) {
        var starShape = new THREE.Shape();
        for (var s = 0; s < 10; s++) {
          var ang = (Math.PI / 5) * s - Math.PI / 2;
          var rad = s % 2 === 0 ? 0.34 : 0.145;
          var sx = Math.cos(ang) * rad, sy = Math.sin(ang) * rad;
          if (s === 0) starShape.moveTo(sx, sy); else starShape.lineTo(sx, sy);
        }
        starShape.closePath();
        var star = new THREE.Mesh(
          new THREE.ExtrudeGeometry(starShape, {
            depth: 0.07, bevelEnabled: true, bevelSize: 0.015,
            bevelThickness: 0.015, bevelSegments: 1
          }),
          new THREE.MeshStandardMaterial({
            color: 0x1E7A56, roughness: 0.42, metalness: 0.3,
            emissive: 0x0B3A28, emissiveIntensity: 1.4
          })
        );
        star.rotation.x = -Math.PI / 2;
        star.position.set(x, 1.15, z + 0.66);
        scene.add(star);

        var goalGlow = new THREE.PointLight(COLORS.goal, 1, 3.4, 2);
        goalGlow.power = 26;
        goalGlow.position.set(x, 1.2, z + 0.66);
        scene.add(goalGlow);
      }
    });

    // ── the agent ───────────────────────────────────────────────────────
    var agent = new THREE.Group();
    scene.add(agent);
    var hull = new THREE.Group();
    agent.add(hull);

    var paintMat = new THREE.MeshStandardMaterial({
      color: COLORS.agent, roughness: 0.42, metalness: 0.4, envMapIntensity: 1.0
    });
    var darkMetal = new THREE.MeshStandardMaterial({
      color: 0x25211F, roughness: 0.5, metalness: 0.8
    });

    var chassis = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.38, 0.22, 22), paintMat);
    chassis.position.y = 0.22;
    chassis.castShadow = true;
    hull.add(chassis);

    var shoulder = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.31, 0.12, 22), darkMetal);
    shoulder.position.y = 0.36;
    shoulder.castShadow = true;
    hull.add(shoulder);

    /* The memory lantern. Before the cue it is dark; the moment the pylon
       fires the lantern takes the cue's colour and keeps it for the rest of
       the run, long after the pylon is dead. That carried light is the memory,
       made into an object you can see. */
    var lanternMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.06, 0.07, 0.09) });
    var lantern = new THREE.Mesh(new THREE.SphereGeometry(0.135, 18, 14), lanternMat);
    lantern.position.y = 0.55;
    hull.add(lantern);

    var lanternCage = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.018, 6, 18), darkMetal);
    lanternCage.rotation.x = Math.PI / 2;
    lanternCage.position.y = 0.55;
    hull.add(lanternCage);

    var lanternLight = new THREE.PointLight(0x58B4F0, 1, 4.2, 2);
    lanternLight.power = 0;
    lanternLight.position.y = 0.56;
    hull.add(lanternLight);

    var lanternFlare = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: 0x9AD8FF, transparent: true, opacity: 0.0,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    lanternFlare.scale.set(0.62, 0.62, 1);
    lanternFlare.position.y = 0.55;
    hull.add(lanternFlare);

    var visor = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.11, 0.34), darkMetal);
    visor.position.set(0.3, 0.3, 0);
    visor.castShadow = true;
    hull.add(visor);

    var headMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.4, 3.2, 2.7) });
    [[0.34, 0.1], [0.34, -0.1]].forEach(function (h) {
      var m = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 10), headMat);
      m.position.set(h[0], 0.3, h[1]);
      hull.add(m);
    });

    var wheels = [];
    var wheelGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.09, 18);
    var wheelMat = new THREE.MeshStandardMaterial({
      color: 0x141312, roughness: 0.9, metalness: 0.1
    });
    [[0.0, 0.3], [0.0, -0.3]].forEach(function (w) {
      var m = new THREE.Mesh(wheelGeo, wheelMat);
      m.position.set(w[0], 0.14, w[1]);
      m.rotation.x = Math.PI / 2;
      m.castShadow = true;
      agent.add(m);
      wheels.push(m);
      for (var t = 0; t < 8; t++) {
        var a = (t / 8) * Math.PI * 2;
        var tread = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.1, 0.028), wheelMat);
        tread.position.set(Math.cos(a) * 0.132, 0, Math.sin(a) * 0.132);
        tread.rotation.y = -a;
        m.add(tread);
      }
    });
    var castor = new THREE.Mesh(new THREE.SphereGeometry(0.075, 12, 10), darkMetal);
    castor.position.set(0.26, 0.075, 0);
    castor.castShadow = true;
    agent.add(castor);

    // Contact shadow. The shadow map handles the cast shadow; this is the dark
    // patch under the hull that keeps the agent from looking pasted on.
    var blob = new THREE.Mesh(
      new THREE.PlaneGeometry(1.1, 1.1),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.42, depthWrite: false
      })
    );
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.09;
    agent.add(blob);

    /* One shadow-casting light in the whole scene, carried by the agent. A
       shadow pass per lamp would be six passes a frame for no picture.
       normalBias, not a big negative bias, which is what acnes. */
    var headlight = new THREE.SpotLight(0xFFF1D6, 1, 11, 0.52, 0.55, 2);
    headlight.power = 620;
    headlight.castShadow = true;
    headlight.shadow.mapSize.set(2048, 2048);
    headlight.shadow.camera.near = 0.4;
    headlight.shadow.camera.far = 12;
    headlight.shadow.normalBias = 0.035;
    headlight.position.set(0.3, 0.32, 0);
    var headTarget = new THREE.Object3D();
    headTarget.position.set(5, -0.2, 0);
    agent.add(headlight);
    agent.add(headTarget);
    headlight.target = headTarget;

    var beamCone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.07, 1.25, 3.3, 20, 1, true),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0xFFE9BE, transparent: true, opacity: 0.03,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    beamCone.rotation.z = Math.PI / 2 + 0.05;
    beamCone.position.set(1.85, 0.3, 0);
    agent.add(beamCone);

    // Spill off the agent's own body: without it the chassis is a silhouette
    // in every frame, because the only lamp it carries points away from it.
    var agentGlow = new THREE.PointLight(0xFFC8A8, 1, 2.6, 2);
    agentGlow.power = 42;
    agentGlow.position.set(-0.1, 0.5, 0);
    agent.add(agentGlow);

    // ── dust ────────────────────────────────────────────────────────────
    var MOTES = 240;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(1357);
    for (var mi = 0; mi < MOTES; mi++) {
      motePos[mi * 3] = (moteSeed() - 0.5) * 8;
      motePos[mi * 3 + 1] = 0.1 + moteSeed() * 2.2;
      motePos[mi * 3 + 2] = (moteSeed() - 0.5) * 11;
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    scene.add(new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.03, map: V.radialTexture(0.95, 0.35), color: 0xFFE3BC, transparent: true,
      opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    })));

    // ── trail ───────────────────────────────────────────────────────────
    var states = payload.states;
    var trailPos = new Float32Array(states.length * 3);
    var trailCol = new Float32Array(states.length * 3);
    var trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3));
    trailGeo.setDrawRange(0, 0);
    scene.add(new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.9
    })));

    core.linearize();

    // ── playback ────────────────────────────────────────────────────────
    var phases = payload.cue_phases || [];
    // The one step of the episode the cue was live on, taken from the trace.
    // An episode where the agent never entered the cue cell has none, and the
    // pylon then stays dark for the whole replay — which is the truth about
    // that episode, and is worth seeing.
    var emittingStep = phases.indexOf("emitting");

    var running = [];
    var total = 0;
    for (var ri = 0; ri < trace.steps.length; ri++) {
      var reward = trace.steps[ri].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, ri);
      }
      running.push(total);
    }

    var heading = -Math.PI / 2;      // facing up the stem, towards -z
    var wheelSpin = 0;
    var lastX = null, lastZ = null;

    function sampleAt(t) {
      var n = states.length;
      var i0 = Math.floor(clamp(t, 0, n - 1));
      var i1 = Math.min(i0 + 1, n - 1);
      var f = clamp(t - i0, 0, 1);
      // Eased between cell centres: the environment's moves are discrete, so
      // the agent should visibly settle into a cell rather than glide through.
      var e = smoothstep(0.12, 0.92, f);
      var a = states[i0], b = states[i1];
      return {
        index: i0, frac: f,
        cellX: lerp(a[slots.x], b[slots.x], e),
        cellY: lerp(a[slots.y], b[slots.y], e)
      };
    }

    /* How hot the cue is at continuous step t. Dark until the agent is nearly
       in the cue cell, firing as it arrives, and snuffed out by the next
       action — the environment's CUE_EMITTING -> CUE_CONSUMED transition,
       read off the trace's own phases rather than from a fixed time. */
    function cueHeat(t) {
      if (emittingStep < 0) return 0;
      var rise = smoothstep(emittingStep - 0.45, emittingStep - 0.05, t);
      var fall = 1 - smoothstep(emittingStep + 0.1, emittingStep + 0.5, t);
      return clamp(Math.min(rise, fall), 0, 1);
    }
    function memoryStrength(t) {
      if (emittingStep < 0) return 0;
      return smoothstep(emittingStep - 0.18, emittingStep + 0.15, t);
    }

    function drawTrail(index) {
      var count = index + 1;
      for (var i = 0; i < count; i++) {
        trailPos[i * 3] = wx(states[i][slots.x]);
        trailPos[i * 3 + 1] = 0.1;
        trailPos[i * 3 + 2] = wz(states[i][slots.y]);
        var age = count > 1 ? i / (count - 1) : 1;
        trailCol[i * 3] = lerp(0.42, 0.95, age);
        trailCol[i * 3 + 1] = lerp(0.12, 0.30, age);
        trailCol[i * 3 + 2] = lerp(0.09, 0.22, age);
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
      trailGeo.setDrawRange(0, Math.max(0, count));
    }

    var CUE_TEXT = {
      unseen: "cue unseen",
      emitting: "cue firing",
      consumed: "cue spent, held in belief"
    };

    /* The maze is taller than it is wide, so the framing follows its longest
       span: a stem of ten cells and a stem of two need very different
       distances, and a fixed one leaves the short maze a smudge. The constant
       term is headroom for the props, which do not scale with the maze — a cue
       pylon is the same height however long the stem is. */
    var span = Math.max((STEM + 1) * CELL, (2 * ARM + 1) * CELL);

    return {
      steps: states.length,

      camera: {
        board: [0, span * 1.06 + 1.6, span * 1.21 + 1.8],
        top: [0, span * 1.48 + 2.0, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var px = wx(sample.cellX), pz = wz(sample.cellY);

        // Heading from actual motion, so the body turns into the junction
        // rather than snapping between the four action directions.
        if (lastX === null) { lastX = px; lastZ = pz; }
        var dx = px - lastX, dz = pz - lastZ;
        if (Math.abs(dx) + Math.abs(dz) > 1e-5) {
          var target = Math.atan2(dz, dx);
          var diff = ((target - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          heading += diff * clamp(dt * 7, 0, 1);
        }
        var moved = Math.hypot(dx, dz);
        lastX = px; lastZ = pz;

        agent.position.set(px, 0, pz);
        agent.rotation.y = -heading;
        hull.position.y = Math.sin(elapsed * 7.5) * 0.006 * (moved > 1e-4 ? 1 : 0.2);
        wheelSpin += moved * 6.5;
        for (var w = 0; w < wheels.length; w++) wheels[w].rotation.y = wheelSpin;

        // The cue, as a light.
        var heat = cueHeat(t);
        var flicker = 0.94 + Math.sin(elapsed * 31.0) * 0.06;
        cueLensMat.color.setRGB(0.5 * heat * flicker, 2.4 * heat * flicker, 5.2 * heat * flicker);
        cueLight.power = 2400 * heat * flicker;
        signMat.emissiveIntensity = 2.6 * heat;
        cueHaze.material.opacity = 0.045 * heat;

        var spent = emittingStep >= 0 && t > emittingStep + 0.5;
        if (heat > 0.02) pilotMat.color.setRGB(0.2, 1.1, 2.2);
        else if (spent) pilotMat.color.setRGB(0.55, 0.05, 0.04);
        else pilotMat.color.setRGB(0.05, 0.06, 0.08);
        repaintLabel(
          cueLabel,
          heat > 0.02 ? "CUE • LIVE" : (spent ? "CUE • SPENT" : "CUE"),
          heat > 0.02 ? "#BFE8FF" : "#6E8598",
          40
        );

        // The carried memory: the lantern takes the cue's colour and holds it.
        var mem = memoryStrength(t);
        lanternMat.color.setRGB(0.06 + 0.28 * mem, 0.07 + 0.85 * mem, 0.09 + 1.9 * mem);
        lanternLight.power = 90 * mem;
        lanternFlare.material.opacity = 0.2 * mem;

        // The belief, from the run's own particles.
        var reading = goalSideProbability(
          payload.beliefs[sample.index], slots.goal_side, GOAL_LEFT_VALUE
        );
        for (var b = 0; b < beliefColumns.length; b++) {
          var column = beliefColumns[b];
          if (reading.p === null) {
            // Nothing readable: the column is emptied rather than parked at a
            // half-full default, which would read as a belief of 0.5.
            column.fill.visible = false;
            repaintLabel(column.num, "—", "#9DA8B4", 44);
            continue;
          }
          column.fill.visible = true;
          var p = column.side === "left" ? reading.p : 1 - reading.p;
          var height = Math.max(0.02, p * 1.9);
          column.fill.scale.y = height;
          column.fill.position.y = 0.1 + height / 2;
          // Brighter at the top of the range, so a confident belief reads as
          // brighter and not only as taller.
          var hot = 1.05 + 1.15 * p;
          column.fill.material.color.setRGB(0.906 * hot, 0.682 * hot, 0.22 * hot);
          repaintLabel(column.num, p.toFixed(2), p > 0.5 ? "#FFD98A" : "#9DA8B4", 44);
        }

        for (var m = 0; m < MOTES; m++) {
          motePos[m * 3] += Math.sin(elapsed * 0.3 + m) * 0.0014;
          motePos[m * 3 + 1] += playing ? 0.002 : 0.0004;
          if (motePos[m * 3 + 1] > 2.4) motePos[m * 3 + 1] = 0.1;
        }
        moteGeo.attributes.position.needsUpdate = true;

        drawTrail(sample.index);

        var step = trace.steps[sample.index] || {};
        /* The phase readout is driven by the same two numbers as the pylon, so
           the words and the object can never disagree — a readout that said
           "firing" over a dark pylon would be read as a broken renderer. Away
           from the cue step both fall through to the trace's own phase for the
           step the agent is nearest. */
        var phaseText = heat > 0.02
          ? CUE_TEXT.emitting
          : (spent
            ? CUE_TEXT.consumed
            : CUE_TEXT[phases[Math.min(Math.round(t), phases.length - 1)]]);
        /* The observation shown is the one that BROUGHT the agent to this
           state — `observations[i]` is sampled from the successor of step i,
           so state i's reading is `observations[i - 1]`. Showing
           `observations[i]` instead would put the cue on the step after the
           one that read it, and the belief would appear to move a step early. */
        var observation = sample.index > 0 ? payload.observations[sample.index - 1] : null;
        return {
          follow: { x: px, z: pz, heading: heading },
          step: sample.index,
          action: step.action === null || step.action === undefined ? "—" : String(step.action),
          x: sample.cellX,
          y: sample.cellY,
          reward: step.reward,
          ret: running[sample.index],
          belief: reading.label + " · " + (phaseText || "cue —") +
            " · obs " + (observation === null || observation === undefined ? "—" : observation)
        };
      }
    };
  }

  V.scenes["t_maze.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // scene's lamp power; it is not a knob to remove.
    exposure: 0.105
  };
})(window);
