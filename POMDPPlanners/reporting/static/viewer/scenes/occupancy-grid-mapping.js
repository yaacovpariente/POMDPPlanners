/* SPDX-License-Identifier: MIT
 *
 * Occupancy-grid mapping scene module.
 *
 * Builds the room from a trace's `payload.world` block and moves it from the
 * trace's recorded poses, scans and maps. Nothing here is invented: there is no
 * fallback episode, no hand-placed wall and no synthetic map. If the player
 * hands this module no trace, it draws nothing and says so.
 *
 * This world has two map-shaped quantities and the whole viewer turns on
 * keeping them apart:
 *
 *   the robot's map   `payload.log_odds` — the inverse-sensor estimate the
 *                     robot carries and the reward is paid on. It is drawn in
 *                     place rather than on a floating chart: an unmapped cell
 *                     is a volume of murk you cannot see into, a believed-free
 *                     cell resolves into lit floor, a believed-occupied cell
 *                     rises as a block, and how far each has resolved is the
 *                     robot's certainty, |log-odds| / clamp.
 *   the hidden map    `payload.world.true_map` — what is really there. It is
 *                     the optional layer, never the default, and every view
 *                     that uses it says so.
 *
 * The belief is a cloud of whole maps, serialised by core as ordinary
 * particles. It is not a second map on the floor: every particle shares the
 * observation-derived log-odds map that is already drawn, and what the cloud
 * adds is how much the hypotheses still disagree about the hidden occupancy.
 * That is reported as a number, in the HUD, from the recorded weights — not
 * redrawn as a rival surface that would compete with the murk for the same
 * meaning.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  var COLORS = {
    occupied: 0xF0A63C,   /* COLOR_CHART_OCCUPIED, from the GIF's palette */
    freeBelief: 0x1D6E78, /* between COLOR_HOLO_MID and COLOR_CHART_FREE */
    beam: 0x7EF0D0,       /* COLOR_BEAM */
    robot: 0xFF5242,      /* COLOR_ROBOT */
    robotDark: 0x3A100E,  /* COLOR_ROBOT_DARK */
    robotLight: 0xFFE2C4  /* COLOR_ROBOT_LIGHT */
  };

  /* Radiance, not paint: above 1.0 so the bright pass catches them. Used by
     nothing else in the frame, so an error mark can never be mistaken for a
     lamp, a beam or a block. */
  var C_MISSED = new THREE.Color(4.6, 0.55, 2.4);    /* #FF3DA6 */
  var C_PHANTOM = new THREE.Color(1.7, 1.05, 4.8);   /* #9B7BFF */

  // Lumens. Service lamps on masts around the room: enough to read the floor
  // by, not enough to make the far half free of travel.
  var LAMP_LUMENS = 7000;

  var ACTION_LABEL = ["forward", "turn left", "turn right"];
  var HEADING_LABEL = ["N", "E", "S", "W"];

  var VIEW_NOTE = {
    map: "The robot’s own map. Unmapped cells are murk — you cannot see into them, because it cannot.",
    truth: "The room as it really is, hidden from the robot. The reference the map is trying to reach.",
    both: "The map with the truth over it. Pink marks a wall the robot missed; violet one it invented."
  };

  function groundCanvas(seed) {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#1B262D";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 900; i++) {
      g.fillStyle = "rgba(" + (34 + rnd() * 40 | 0) + "," + (48 + rnd() * 38 | 0) + "," +
        (58 + rnd() * 34 | 0) + ",0.30)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, 12 + rnd() * 90, 0, Math.PI * 2); g.fill();
    }
    // Chips of grit: a lit cap and a dark side, which the normal map turns
    // into relief once the lamps hit it.
    for (var p = 0; p < 1500; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.4 + rnd() * 4.2;
      g.fillStyle = "rgba(11,17,21,0.5)";
      g.beginPath(); g.arc(px + pr * 0.35, py + pr * 0.35, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(176,199,210,0.26)";
      g.beginPath(); g.arc(px, py, pr * 0.85, 0, Math.PI * 2); g.fill();
    }
    // Scuffs, so a poured floor does not read as a texture swatch.
    for (var k = 0; k < 90; k++) {
      g.strokeStyle = "rgba(150,176,188,0.05)";
      g.lineWidth = 1 + rnd() * 2;
      g.beginPath();
      var cx = rnd() * s, cy = rnd() * s;
      g.moveTo(cx, cy);
      for (var seg = 0; seg < 5; seg++) {
        cx += (rnd() - 0.5) * 90; cy += (rnd() - 0.5) * 90;
        g.lineTo(cx, cy);
      }
      g.stroke();
    }
    return cv;
  }

  function paintGrid(cv, cells) {
    var s = cv.width;
    var g = cv.getContext("2d");
    g.strokeStyle = "rgba(190,226,238,0.055)";
    g.lineWidth = 2;
    for (var k = 0; k <= cells; k++) {
      var p = (k / cells) * s;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
      g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
    }
  }

  /** Blobby cloud, so a cell of murk has structure instead of reading as a box. */
  function fogTexture() {
    var s = 256;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(4242);
    g.fillStyle = "#6E8391";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 160; i++) {
      var a = 0.05 + rnd() * 0.16;
      g.fillStyle = rnd() > 0.5 ? "rgba(214,232,240," + a + ")" : "rgba(28,42,52," + a + ")";
      g.beginPath();
      g.arc(rnd() * s, rnd() * s, 8 + rnd() * 52, 0, Math.PI * 2);
      g.fill();
    }
    var tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }

  /**
   * The three-way view control, injected rather than declared.
   *
   * The episode page's markup is shared by every environment, so a control only
   * this world needs is built here. It is appended to the transport bar and
   * borrows that bar's own button styling, so it looks like the camera buttons
   * beside it rather than a bolted-on widget.
   *
   * @param {Function} onChange  Called with "map", "truth" or "both".
   * @returns {Object} { setNote } — a live readout line under the bar.
   */
  function buildViewControl(onChange) {
    var root = document.getElementById("viewer");
    var bar = root ? root.querySelector(".viewer-bar") : null;
    var host = bar || root;
    if (!host) return { setNote: function () {} };

    var group = document.createElement("span");
    group.className = "cams";
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "What to show");

    var note = document.createElement("p");
    note.style.cssText = "margin:8px 0 0;font-size:13px;line-height:1.45;color:#9aa3ad";

    var buttons = {};
    [["map", "Robot’s map"], ["truth", "Ground truth"], ["both", "Both"]].forEach(
      function (entry) {
        var button = document.createElement("button");
        button.type = "button";
        button.textContent = entry[1];
        button.setAttribute("aria-pressed", entry[0] === "map" ? "true" : "false");
        button.addEventListener("click", function () { apply(entry[0]); });
        group.appendChild(button);
        buttons[entry[0]] = button;
      }
    );

    function apply(mode) {
      Object.keys(buttons).forEach(function (m) {
        buttons[m].setAttribute("aria-pressed", m === mode ? "true" : "false");
      });
      onChange(mode);
    }

    host.appendChild(group);
    var status = document.getElementById("viewer-status");
    if (status && status.parentNode) {
      status.parentNode.insertBefore(note, status.nextSibling);
    } else if (root && root.parentNode) {
      root.parentNode.insertBefore(note, root.nextSibling);
    }
    return {
      setNote: function (text) { note.textContent = text; }
    };
  }

  /**
   * Build the occupancy-grid mapping world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind
   *   occupancy_grid_mapping.v1.
   * @returns {Object} The scene module the player drives.
   */
  // The world is one room with several independent pieces of furniture, and
  // splitting it into builders would hide which of them share the palette.
  // eslint-disable-next-line max-statements
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;

    var ROWS = world.num_rows, COLS = world.num_cols;
    var CELLS = ROWS * COLS;
    var NB = world.num_beams, FOV = world.field_of_view_degrees;
    var MAXR = world.max_range_cells;
    var CLAMP = world.log_odds_clamp;
    var RESOLVED = world.resolved_log_odds;
    var LAYOUT = world.state_layout || {};

    // Row grows downwards and column to the right, the convention
    // occupancy_grid_sensor.py fixes, so row maps to +z and column to +x.
    var HR = (ROWS - 1) / 2, HC = (COLS - 1) / 2;
    function wx(col) { return col - HC; }
    function wz(row) { return row - HR; }

    /* Beam bearings, straight out of build_ray_templates: beams sit at bin
       centres of the fan, bearing is clockwise from north, and the unit
       direction of bearing theta is (-cos, +sin) in (row, col). */
    function bearingOf(beam) {
      return (-FOV / 2 + FOV * (beam + 0.5) / NB) * Math.PI / 180;
    }

    scene.background = new THREE.Color(0x0D1318).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x0B1116, 0.017);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x22384A, 0x0A0C0E, 0.55));
    var moon = new THREE.DirectionalLight(0x7D93B4, 0.55);
    moon.position.set(-7, 11, -5);
    scene.add(moon);
    core.buildNightEnvironment([
      [0.0, "#0A1118"], [0.45, "#131C24"], [0.55, "#1D2A30"], [1.0, "#05080A"]
    ]);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);
    var glowTex = V.radialTexture(0.9, 0.3);

    // Floor. The grit is painted first and the normal map derived from it, so
    // every chip catches a lamp from the side the lamp is on. The grid is
    // painted afterwards, onto colour only: it marks cell boundaries and must
    // not emboss the concrete.
    var gCanvas = groundCanvas(20260910);
    var groundNormal = V.normalMapFrom(renderer, gCanvas, 2.0);
    paintGrid(gCanvas, COLS);
    var groundAlbedo = new THREE.CanvasTexture(gCanvas);
    groundAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    groundAlbedo.encoding = THREE.sRGBEncoding;

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(COLS, ROWS),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.9, 0.9),
        roughness: 0.94, metalness: 0.0, envMapIntensity: 0.3
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    /* Ground carrying on past the room. Without it the room is a lit slab in a
       void, which is the single biggest tell that a night render is a render.
       Parked well below the floor so the two never fight the 16-bit depth
       buffer. */
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(90, 90),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.8, 0.8),
        color: 0x131C22, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.1
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.7;
    outer.receiveShadow = true;
    scene.add(outer);

    var plinth = new THREE.Mesh(
      new THREE.BoxGeometry(COLS + 0.06, 0.95, ROWS + 0.06),
      new THREE.MeshStandardMaterial({ color: 0x0C1216, roughness: 0.82, metalness: 0.18 })
    );
    plinth.position.y = -0.52;
    plinth.castShadow = true;
    plinth.receiveShadow = true;
    scene.add(plinth);

    /* ------------------------------------------- the world, as it is known
       The default view, and the point of the environment. Nothing in this
       block is drawn from the hidden map: every cell is driven by the robot's
       own log-odds, so what a reader can see is what the robot established.

       Three pieces per cell, one number each — certainty, |log-odds| / clamp:

         fog    an unknown cell is a volume of murk. Opacity is 1 - certainty,
                so a cell the robot half believes is half veiled. Lit grey-blue
                rather than black on purpose: black reads as ordinary shadow,
                and a reader who mistakes unknown space for shade has been told
                something false.
         block  a believed-occupied cell grows a solid block out of the floor,
                rising into place over the sightings that confirm it.
         wash   a believed-free cell resolves into lit floor with a cool tint,
                which is what separates "swept and empty" from "not looked at". */
    var knownGroup = new THREE.Group();
    scene.add(knownGroup);

    var fogTex = fogTexture();
    var fogRnd = mulberry(8123);
    var blockGeo = new THREE.BoxGeometry(0.96, 1, 0.96);
    var fogGeo = new THREE.BoxGeometry(1.0, 1.02, 1.0);
    var washGeo = new THREE.PlaneGeometry(0.97, 0.97);

    var cellBlocks = [], cellFog = [], cellWash = [];
    for (var ci = 0; ci < CELLS; ci++) {
      var cx = wx(ci % COLS), cz = wz(Math.floor(ci / COLS));

      var block = new THREE.Mesh(blockGeo, new THREE.MeshStandardMaterial({
        color: 0x2C3740, roughness: 0.72, metalness: 0.2,
        emissive: new THREE.Color(0, 0, 0), envMapIntensity: 0.55
      }));
      block.position.set(cx, 0, cz);
      block.scale.y = 0.001;
      block.castShadow = true;
      block.receiveShadow = true;
      block.visible = false;
      knownGroup.add(block);
      cellBlocks.push(block);

      var wash = new THREE.Mesh(washGeo, new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.45, 2.2, 2.5), transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      wash.rotation.x = -Math.PI / 2;
      wash.position.set(cx, 0.03, cz);
      wash.renderOrder = 1;
      wash.visible = false;
      knownGroup.add(wash);
      cellWash.push(wash);

      /* Drawn last and without writing depth, so murk in front of a block
         hides it the way murk should, and two cells of murk stack into
         something you really cannot see through. */
      var cellTex = fogTex.clone();
      cellTex.needsUpdate = true;
      cellTex.wrapS = cellTex.wrapT = THREE.RepeatWrapping;
      cellTex.center.set(0.5, 0.5);
      cellTex.rotation = fogRnd() * Math.PI * 2;
      cellTex.offset.set(fogRnd(), fogRnd());
      var fog = new THREE.Mesh(fogGeo, new THREE.MeshBasicMaterial({
        map: cellTex, color: new THREE.Color(1.15, 1.75, 2.05),
        transparent: true, opacity: 0.9, depthWrite: false, fog: true
      }));
      fog.position.set(cx, 0.51, cz);
      fog.renderOrder = 4;
      knownGroup.add(fog);
      cellFog.push(fog);
    }

    /* ------------------------------------------------------- ground truth
       The optional layer, in two forms.

       "Ground truth" shows the room as it really is: solid walls, lit, no murk
       at all, so a reader can flip to it and read the real layout in one look.
       "Both" keeps the robot's map and marks where it is wrong, which is the
       question worth asking of a map:

         missed wall   believed free, actually occupied — the dangerous error,
                       because it is the one a planner would drive into.
         phantom wall  believed occupied, actually free — the wasteful one:
                       ground walled off for nothing.

       A cell counts as an error only once it is resolved, at the same
       |log-odds| the environment itself calls resolved. A cell still in doubt
       is not wrong yet, it is unfinished, and it is counted separately. */
    var truth = world.true_map || [];
    var truthSolid = new THREE.Group();
    truthSolid.visible = false;
    scene.add(truthSolid);
    var truthGhost = new THREE.Group();
    truthGhost.visible = false;
    scene.add(truthGhost);

    var wallMat = new THREE.MeshStandardMaterial({
      color: 0x3F4C55, roughness: 0.78, metalness: 0.16, envMapIntensity: 0.45
    });
    var blockMat = new THREE.MeshStandardMaterial({
      color: 0x3A4750, roughness: 0.7, metalness: 0.22, envMapIntensity: 0.55
    });
    var capMat = new THREE.MeshStandardMaterial({
      color: 0x55646E, roughness: 0.5, metalness: 0.4, envMapIntensity: 0.7
    });
    /* In "Both" the real walls are translucent volumes, not outlines: a box
       you can see the robot's map through still reads as a wall from across
       the room, where a wireframe does not. */
    var ghostMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.8, 1.0, 1.15), transparent: true, opacity: 0.16,
      blending: THREE.AdditiveBlending, depthWrite: false
    });

    for (var r = 0; r < ROWS; r++) {
      for (var c = 0; c < COLS; c++) {
        if (!truth[r * COLS + c]) continue;
        var boundary = r === 0 || c === 0 || r === ROWS - 1 || c === COLS - 1;
        var th = boundary ? 1.35 : 0.85;

        var solid = new THREE.Mesh(
          new THREE.BoxGeometry(0.99, th, 0.99), boundary ? wallMat : blockMat
        );
        solid.position.set(wx(c), th / 2, wz(r));
        solid.castShadow = true;
        solid.receiveShadow = true;
        truthSolid.add(solid);
        /* A machined cap: two materials at different roughness is most of what
           stops a grid of boxes reading as toy bricks. */
        var cap = new THREE.Mesh(new THREE.BoxGeometry(1.01, 0.07, 1.01), capMat);
        cap.position.set(wx(c), th + 0.035, wz(r));
        cap.castShadow = true;
        truthSolid.add(cap);

        var ghost = new THREE.Mesh(new THREE.BoxGeometry(0.99, th, 0.99), ghostMat);
        ghost.position.set(wx(c), th / 2, wz(r));
        ghost.renderOrder = 6;
        truthGhost.add(ghost);
      }
    }

    /* One error marker per cell, hidden until it is earned. Drawn last and
       ignoring depth, so an error inside unmapped murk is still visible —
       which is exactly the case a reader most needs to see. */
    var errorGroup = new THREE.Group();
    errorGroup.visible = false;
    scene.add(errorGroup);
    var errorMarks = [], errorPips = [];
    var markGeo = new THREE.BoxGeometry(0.82, 1.7, 0.82);
    var pipGeo = new THREE.RingGeometry(0.30, 0.44, 24);
    for (var ei = 0; ei < CELLS; ei++) {
      var mark = new THREE.Mesh(markGeo, new THREE.MeshBasicMaterial({
        color: C_MISSED, transparent: true, opacity: 0.8,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false
      }));
      mark.position.set(wx(ei % COLS), 0.85, wz(Math.floor(ei / COLS)));
      mark.renderOrder = 7;
      mark.visible = false;
      var pip = new THREE.Mesh(pipGeo, new THREE.MeshBasicMaterial({
        color: C_MISSED, transparent: true, opacity: 0.95,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
        side: THREE.DoubleSide
      }));
      pip.rotation.x = -Math.PI / 2;
      pip.position.set(0, -0.80, 0);
      mark.add(pip);
      errorGroup.add(mark);
      errorMarks.push(mark);
      errorPips.push(pip);
    }

    /** Classify the robot's map against the hidden one, and place the marks. */
    function classify(logOdds) {
      var missed = 0, phantom = 0, unresolved = 0;
      for (var i = 0; i < CELLS; i++) {
        var l = logOdds[i];
        var kind = 0;                     /* 0 none, 1 missed, 2 phantom */
        if (Math.abs(l) < RESOLVED) unresolved++;
        else if (l < 0 && truth[i]) { kind = 1; missed++; }
        else if (l > 0 && !truth[i]) { kind = 2; phantom++; }
        var marker = errorMarks[i];
        if (!kind) {
          marker.visible = false;
        } else {
          marker.visible = true;
          var tint = kind === 1 ? C_MISSED : C_PHANTOM;
          marker.material.color.copy(tint);
          errorPips[i].material.color.copy(tint);
        }
      }
      return { missed: missed, phantom: phantom, unresolved: unresolved };
    }

    /* ------------------------------------------------------------- lamps
       Service lamps around the room, real lumens, inverse-square falloff. They
       light the room; the far half stays dim, which is part of why the robot
       has to drive to finish the map rather than seeing it from home. */
    var lamps = [];
    var lampMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.4, 4.2, 4.3) });
    var postMat = new THREE.MeshStandardMaterial({
      color: 0x566874, roughness: 0.4, metalness: 0.85
    });
    [[-1, -1], [-1, 1], [1, -1], [1, 1], [-1, 0], [1, 0], [0, -1], [0, 1]].forEach(
      function (q, qi) {
        var x = q[0] * (HC + 3.4), z = q[1] * (HR + 3.4);
        if (qi > 3) {
          // Bare fill lights between the corners: a mast on every side would
          // be a picket fence around the room.
          var bare = new THREE.PointLight(0xBFE6F0, 1, 30, 2);
          bare.power = LAMP_LUMENS;
          bare.position.set(x, 4.56, z);
          scene.add(bare);
          lamps.push(bare);
          return;
        }
        var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.1, 4.6, 10), postMat);
        mast.position.set(x, 2.3, z);
        mast.castShadow = true;
        scene.add(mast);
        var head = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.26, 0.16, 14), postMat);
        head.position.set(x, 4.72, z);
        scene.add(head);
        var lens = new THREE.Mesh(new THREE.SphereGeometry(0.13, 14, 10), lampMat);
        lens.position.set(x, 4.58, z);
        scene.add(lens);
        var light = new THREE.PointLight(0xBFE6F0, 1, 30, 2);
        light.power = LAMP_LUMENS;
        light.position.set(x, 4.56, z);
        scene.add(light);
        lamps.push(light);
      }
    );

    var fill = new THREE.PointLight(0x9FC6D6, 1, 26, 2);
    fill.power = 5200;
    fill.position.set(0, 7.5, 0);
    scene.add(fill);

    /* One shadow-casting spot, parked over the robot and aimed at it, taking
       its share of power from the nearest lamp. Four shadow-casting point
       lights would be twenty-four shadow passes a frame; this is one. Narrow
       cone plus normalBias, which is the fix for acne a large negative bias is
       not. */
    var shadowLight = new THREE.SpotLight(0xCFEAF2, 1, 20, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.4;
    shadowLight.shadow.camera.far = 20;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, 5.4, 0);
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    /* ------------------------------------------------------------- robot */
    var robot = new THREE.Group();
    scene.add(robot);
    var hull = new THREE.Group();
    robot.add(hull);

    var paintMat = new THREE.MeshStandardMaterial({
      color: COLORS.robot, roughness: 0.42, metalness: 0.34
    });
    var darkMat = new THREE.MeshStandardMaterial({
      color: COLORS.robotDark, roughness: 0.6, metalness: 0.55
    });
    var creamMat = new THREE.MeshStandardMaterial({
      color: COLORS.robotLight, roughness: 0.45, metalness: 0.2
    });

    var chassis = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.17, 0.58), paintMat);
    chassis.position.y = 0.19;
    chassis.castShadow = true;
    hull.add(chassis);

    var skirt = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.08, 0.64), darkMat);
    skirt.position.y = 0.11;
    skirt.castShadow = true;
    hull.add(skirt);

    // The nose wedge, so heading reads from any angle. The robot faces north
    // at heading 0, which is -z.
    var noseShape = new THREE.Shape();
    noseShape.moveTo(0, -0.30); noseShape.lineTo(0.20, 0.06); noseShape.lineTo(-0.20, 0.06);
    noseShape.closePath();
    var nose = new THREE.Mesh(
      new THREE.ExtrudeGeometry(noseShape, {
        depth: 0.07, bevelEnabled: true, bevelSize: 0.012,
        bevelThickness: 0.012, bevelSegments: 1
      }),
      creamMat
    );
    nose.rotation.x = Math.PI / 2;
    nose.position.set(0, 0.30, 0);
    nose.castShadow = true;
    hull.add(nose);

    [-0.28, 0.28].forEach(function (x) {
      var rail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.2, 0.62), darkMat);
      rail.position.set(x, 0.1, 0);
      rail.castShadow = true;
      robot.add(rail);
      for (var t = 0; t < 5; t++) {
        var rung = new THREE.Mesh(
          new THREE.BoxGeometry(0.15, 0.035, 0.055),
          new THREE.MeshStandardMaterial({ color: 0x7A342C, roughness: 0.8, metalness: 0.1 })
        );
        rung.position.set(x, 0.2, -0.24 + t * 0.12);
        robot.add(rung);
      }
    });

    // The sensor dome: what does the scanning, and where every beam is drawn
    // from. It spins, because a 360-degree range finder does.
    var domeBase = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.17, 0.1, 16), darkMat);
    domeBase.position.y = 0.33;
    hull.add(domeBase);
    var dome = new THREE.Mesh(
      new THREE.CylinderGeometry(0.13, 0.13, 0.13, 18),
      new THREE.MeshStandardMaterial({
        color: 0x0F3A38, roughness: 0.25, metalness: 0.5,
        emissive: 0x0B3F3A, emissiveIntensity: 1.0
      })
    );
    dome.position.y = 0.42;
    dome.castShadow = true;
    hull.add(dome);
    var domeEye = new THREE.Mesh(
      new THREE.BoxGeometry(0.04, 0.07, 0.16),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(2.6, 5.0, 4.4) })
    );
    domeEye.position.set(0, 0, 0.1);
    dome.add(domeEye);
    var domeCap = new THREE.Mesh(new THREE.CylinderGeometry(0.135, 0.135, 0.03, 18), creamMat);
    domeCap.position.y = 0.50;
    hull.add(domeCap);

    /* The robot's body is opaque, so murk in front of it draws over it whatever
       the render order. This halo is not: it is drawn last and ignores depth,
       so where the robot is stays legible even inside unmapped space. The pose
       is the one quantity the robot knows exactly, so hiding it would lie. */
    var domeFlare = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: new THREE.Color(1.9, 4.6, 4.0), transparent: true,
      opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false
    }));
    domeFlare.scale.set(1.15, 1.15, 1);
    domeFlare.position.y = 0.44;
    domeFlare.material.depthTest = false;
    domeFlare.renderOrder = 6;
    hull.add(domeFlare);

    var sensorLight = new THREE.PointLight(0x7EF0D0, 1, 4.2, 2);
    sensorLight.power = 110;
    sensorLight.position.y = 0.45;
    robot.add(sensorLight);

    var blob = new THREE.Mesh(
      new THREE.PlaneGeometry(1.15, 1.15),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.45, depthWrite: false
      })
    );
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.01;
    robot.add(blob);

    /* -------------------------------------------------------------- beams
       One line segment per beam, from the sensor dome to where the measured
       range says the beam stopped. The ranges are the noisy ones the episode
       actually drew, so a beam can and does overshoot a wall slightly, and the
       fog retreating along them is what shows where the map came from. */
    var beamPos = new Float32Array(NB * 2 * 3);
    var beamCol = new Float32Array(NB * 2 * 3);
    var beamGeo = new THREE.BufferGeometry();
    beamGeo.setAttribute("position", new THREE.BufferAttribute(beamPos, 3));
    beamGeo.setAttribute("color", new THREE.BufferAttribute(beamCol, 3));
    var beams = new THREE.LineSegments(beamGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.9,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    beams.renderOrder = 5;
    scene.add(beams);

    // A bright mote where each beam stopped: the measurement itself.
    var hitPos = new Float32Array(NB * 3);
    var hitGeo = new THREE.BufferGeometry();
    hitGeo.setAttribute("position", new THREE.BufferAttribute(hitPos, 3));
    var hits = new THREE.Points(hitGeo, new THREE.PointsMaterial({
      size: 0.24, map: partTex, color: 0x9DFFE4, transparent: true, opacity: 1.0,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    }));
    hits.renderOrder = 5;
    scene.add(hits);

    /* -------------------------------------------------------------- trail */
    var trailPos = new Float32Array((payload.poses.length + 1) * 3);
    var trailCol = new Float32Array((payload.poses.length + 1) * 3);
    var trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3));
    trailGeo.setDrawRange(0, 0);
    var trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.85
    }));
    trail.renderOrder = 5;
    scene.add(trail);

    /* ---------------------------------------------------------------- air */
    var MOTES = 220;
    var motePos = new Float32Array(MOTES * 3);
    var moteRnd = mulberry(1357);
    for (var mi = 0; mi < MOTES; mi++) {
      motePos[mi * 3] = (moteRnd() - 0.5) * (COLS + 1);
      motePos[mi * 3 + 1] = 0.1 + moteRnd() * 2.3;
      motePos[mi * 3 + 2] = (moteRnd() - 0.5) * (ROWS + 1);
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    var motes = new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.032, map: partTex, color: 0xB6E4EC, transparent: true, opacity: 0.32,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    }));
    scene.add(motes);

    core.linearize();

    /* ------------------------------------------------------------ episode */
    var view = "map";
    var control = buildViewControl(function (mode) {
      view = mode;
      knownGroup.visible = mode !== "truth";
      truthSolid.visible = mode === "truth";
      truthGhost.visible = mode === "both";
      errorGroup.visible = mode === "both";
    });

    /* Headings unwrapped to a continuous angle, so a left turn from north
       rotates the short way instead of spinning three quarters of a circle. */
    var headingAngle = [payload.poses.length ? payload.poses[0][2] * Math.PI / 2 : 0];
    for (var hi = 1; hi < payload.poses.length; hi++) {
      var d = ((payload.poses[hi][2] - payload.poses[hi - 1][2] + 5) % 4) - 1;
      headingAngle.push(headingAngle[hi - 1] + d * Math.PI / 2);
    }

    var running = [];
    var total = 0;
    for (var ri = 0; ri < trace.steps.length; ri++) {
      var reward = trace.steps[ri].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, ri);
      }
      running.push(total);
    }

    var lerpLogOdds = new Float64Array(CELLS);

    function sampleAt(t) {
      var n = payload.poses.length;
      var i0 = Math.floor(clamp(t, 0, n - 1));
      var i1 = Math.min(i0 + 1, n - 1);
      var f = clamp(t - i0, 0, 1);
      // Ease between recorded steps. The steps are discrete; the easing only
      // decides what the eye sees in between.
      var e = f * f * (3 - 2 * f);
      var a = payload.log_odds[i0], b = payload.log_odds[i1];
      for (var k = 0; k < CELLS; k++) lerpLogOdds[k] = lerp(a[k], b[k], e);
      return {
        i0: i0,
        // The step a reader is being shown: the nearer of the two, so the HUD
        // never claims a scan the robot has not taken yet.
        index: f > 0.5 ? i1 : i0,
        row: lerp(payload.poses[i0][0], payload.poses[i1][0], e),
        col: lerp(payload.poses[i0][1], payload.poses[i1][1], e),
        angle: lerp(headingAngle[i0], headingAngle[i1], e),
        entropy: lerp(payload.entropy_bits[i0], payload.entropy_bits[i1], e),
        logOdds: lerpLogOdds
      };
    }

    var cOcc = new THREE.Color(COLORS.occupied);

    /* Everything a reader sees about the map comes out of this one loop, and
       out of one number per cell. Certainty near zero means murk and nothing
       else; certainty near one means the cell has fully resolved into either
       lit floor or a block. Everything between is genuinely half-resolved. */
    function updateMap(logOdds) {
      for (var i = 0; i < CELLS; i++) {
        var l = logOdds[i];
        var certainty = clamp(Math.abs(l) / CLAMP, 0, 1);
        /* Certainty is spent early: the first sighting of a cell is worth far
           more than the sixth, and the picture should say so. */
        var shown = Math.pow(certainty, 0.62);

        var fog = cellFog[i];
        var veil = 1 - shown;
        if (veil < 0.02) {
          fog.visible = false;
        } else {
          fog.visible = true;
          // Thinner in "Both", where the truth has to read through it.
          fog.material.opacity = (view === "both" ? 0.42 : 0.9) * veil;
        }

        var block = cellBlocks[i], wash = cellWash[i];
        if (l > 0) {
          wash.visible = false;
          block.visible = true;
          var height = Math.max(0.02, shown * 1.15);
          block.scale.y = height;
          block.position.y = height / 2;
          /* The amber the GIF uses for "believed occupied", carried as
             emissive so a confirmed block is lit from inside and a doubtful
             one is not. */
          block.material.emissive.setRGB(
            cOcc.r * 1.15 * shown, cOcc.g * 0.78 * shown, cOcc.b * 0.30 * shown
          );
        } else if (l < 0) {
          block.visible = false;
          wash.visible = true;
          wash.material.opacity = 0.42 * shown;
        } else {
          block.visible = false;
          wash.visible = false;
        }
      }
    }

    function updateBeams(index, row, col, angle, sweep) {
      var scan = payload.ranges[index];
      var show = !!payload.scanned[index] && !!scan;
      beams.visible = show;
      hits.visible = show;
      if (!show) return;

      var ox = wx(col), oz = wz(row), oy = 0.46;
      for (var b = 0; b < NB; b++) {
        var bearing = bearingOf(b) + angle;
        var dRow = -Math.cos(bearing), dCol = Math.sin(bearing);
        var range = clamp(scan[b], 0.12, MAXR);
        var ex = ox + range * dCol, ez = oz + range * dRow;
        /* Beams brighten as the spinning head passes them, which is what a
           range finder taking NB samples per revolution does. */
        var phase = ((b + 0.5) / NB - sweep + 2) % 1;
        var lit = 0.5 + 0.5 * Math.pow(1 - phase, 6);
        // At or above the maximum range the reading is a miss, and the
        // environment treats a miss and a hit at exactly maximum range the
        // same way. A dimmer beam and no mote says "nothing came back".
        var miss = scan[b] >= MAXR - 1e-6;
        var base = miss ? 1.6 : 4.4;
        var k = b * 6;
        beamPos[k] = ox; beamPos[k + 1] = oy; beamPos[k + 2] = oz;
        beamPos[k + 3] = ex; beamPos[k + 4] = 0.30; beamPos[k + 5] = ez;
        beamCol[k] = 0.30 * base * lit * 0.2;
        beamCol[k + 1] = 0.94 * base * lit * 0.2;
        beamCol[k + 2] = 0.82 * base * lit * 0.2;
        beamCol[k + 3] = 0.30 * base * lit;
        beamCol[k + 4] = 0.94 * base * lit;
        beamCol[k + 5] = 0.82 * base * lit;
        hitPos[b * 3] = ex;
        hitPos[b * 3 + 1] = miss ? -9 : 0.30;
        hitPos[b * 3 + 2] = ez;
      }
      beamGeo.attributes.position.needsUpdate = true;
      beamGeo.attributes.color.needsUpdate = true;
      hitGeo.attributes.position.needsUpdate = true;
    }

    function updateTrail(t) {
      var upto = clamp(t, 0, payload.poses.length - 1);
      var whole = Math.floor(upto);
      var count = 0;
      for (var i = 0; i <= whole; i++) {
        trailPos[count * 3] = wx(payload.poses[i][1]);
        trailPos[count * 3 + 1] = 0.05;
        trailPos[count * 3 + 2] = wz(payload.poses[i][0]);
        count++;
      }
      var frac = upto - whole;
      if (frac > 0 && whole + 1 < payload.poses.length) {
        var a = payload.poses[whole], b = payload.poses[whole + 1];
        trailPos[count * 3] = lerp(wx(a[1]), wx(b[1]), frac);
        trailPos[count * 3 + 1] = 0.05;
        trailPos[count * 3 + 2] = lerp(wz(a[0]), wz(b[0]), frac);
        count++;
      }
      for (var j = 0; j < count; j++) {
        var age = count < 2 ? 1 : j / (count - 1);
        trailCol[j * 3] = lerp(0.55, 1.0, age);
        trailCol[j * 3 + 1] = lerp(0.17, 0.54, age);
        trailCol[j * 3 + 2] = lerp(0.10, 0.33, age);
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
      trailGeo.setDrawRange(0, count);
    }

    /* The belief is a cloud of whole maps, so it is reported rather than drawn
       a second time: the log-odds map on the floor is already in every
       particle. What the cloud adds is how concentrated it is and how much the
       hypotheses still disagree about the hidden occupancy — both read off the
       recorded particles and their recorded weights, never re-derived. */
    function describeBelief(index) {
      var belief = payload.beliefs[index];
      if (!belief) return "—";
      if (belief.kind === "particle_batch") {
        // Several beliefs held together for a vectorized planner. Merging its
        // members would describe a cloud that was never anyone's belief.
        return "batch of " + belief.batch_size + " beliefs, not summarised";
      }
      if (belief.kind !== "particles") {
        // A belief class core has no payload for. Named, so the gap is
        // diagnosable, rather than replaced by something plausible.
        return "not recorded (" + (belief.belief_class || belief.kind) + ")";
      }

      var label = belief.num_particles + " whole-map particles";
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " read)";
      }

      var weights = belief.weights || [];
      var sumSquares = 0;
      for (var w = 0; w < weights.length; w++) sumSquares += weights[w] * weights[w];
      if (sumSquares > 0) label += " · ESS " + (1 / sumSquares).toFixed(1);

      // A particle is a whole state vector, so the hidden-occupancy block sits
      // at the layout offset the payload carries. A belief of anything else —
      // a planner run with a different state shape — is reported without this
      // rather than indexed into blindly.
      var offset = LAYOUT.map_offset;
      var first = belief.particles[0];
      if (offset === undefined || !Array.isArray(first) || first.length !== LAYOUT.state_size) {
        return label;
      }
      var disputed = 0;
      for (var cell = 0; cell < CELLS; cell++) {
        var marginal = 0;
        for (var p = 0; p < belief.particles.length; p++) {
          marginal += weights[p] * (belief.particles[p][offset + cell] ? 1 : 0);
        }
        if (marginal > 0.1 && marginal < 0.9) disputed++;
      }
      return label + " · " + disputed + " cells in dispute";
    }

    var extent = Math.max(ROWS, COLS);
    var tmpVec = new THREE.Vector3();

    return {
      steps: payload.poses.length,

      /* Framing scales with the room, because a trace decides how big it is.
         The constant term is headroom for the masts, which are the same height
         on any board, so a small room needs proportionally more of it. */
      camera: {
        board: [0, extent * 0.94 + 2.0, extent * 1.10 + 2.2],
        top: [0, extent * 1.75 + 2.0, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var px = wx(sample.col), pz = wz(sample.row);

        robot.position.set(px, 0, pz);
        robot.rotation.y = -sample.angle;
        hull.position.y = playing ? Math.sin(elapsed * 6.5) * 0.004 : 0;

        var sweep = playing ? (elapsed * 0.9) % 1 : 0.25;
        dome.rotation.y = sweep * Math.PI * 2;

        updateMap(sample.logOdds);
        updateBeams(sample.index, sample.row, sample.col, sample.angle, sweep);
        updateTrail(t);

        // Park the one shadow caster over the robot and take its share of
        // power off the nearest lamp, so the room's total light stays put.
        var nearest = null, nd = Infinity;
        tmpVec.set(px, 1.38, pz);
        for (var li = 0; li < lamps.length; li++) {
          lamps[li].power = LAMP_LUMENS;
          var d2 = lamps[li].position.distanceToSquared(tmpVec);
          if (d2 < nd) { nd = d2; nearest = lamps[li]; }
        }
        if (nearest) {
          shadowLight.position.set(px, 5.4, pz + 0.9);
          shadowLight.target.position.set(px, 0.15, pz);
          shadowLight.target.updateMatrixWorld();
          shadowLight.power = LAMP_LUMENS * 0.5;
          nearest.power = LAMP_LUMENS * 0.5;
        }

        if (playing) {
          for (var m = 0; m < MOTES; m++) {
            motePos[m * 3] += Math.sin(elapsed * 0.3 + m) * 0.0014;
            motePos[m * 3 + 1] += 0.0021;
            if (motePos[m * 3 + 1] > 2.5) motePos[m * 3 + 1] = 0.1;
          }
          moteGeo.attributes.position.needsUpdate = true;
        }

        var pose = payload.poses[sample.index];
        var tally = classify(payload.log_odds[sample.index]);
        control.setNote(
          VIEW_NOTE[view] + "  —  row " + pose[0] + ", col " + pose[1] +
          ", facing " + HEADING_LABEL[pose[2]] + "  ·  " +
          Math.round(100 * (CELLS - tally.unresolved) / CELLS) + "% of cells resolved, " +
          payload.entropy_bits[sample.index].toFixed(1) + " of " +
          world.initial_entropy_bits.toFixed(0) + " bits left (resolved at " +
          world.entropy_threshold_bits.toFixed(0) + ")  ·  " +
          tally.missed + " missed, " + tally.phantom + " phantom, " +
          tally.unresolved + " unresolved"
        );

        var step = trace.steps[sample.index] || {};
        var action = step.action === null || step.action === undefined
          ? "—"
          : (ACTION_LABEL[step.action] || String(step.action));
        return {
          /* The core rig's chase mode measures heading from +x, while this
             world's heading 0 faces -z, so the quarter turn between the two
             conventions is applied here rather than in the shared rig. */
          follow: { x: px, z: pz, heading: sample.angle - Math.PI / 2 },
          step: sample.index,
          action: action,
          x: sample.col,
          y: sample.row,
          reward: step.reward,
          ret: running[sample.index],
          belief: describeBelief(sample.index)
        };
      }
    };
  }

  V.scenes["occupancy_grid_mapping.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // room's lamp power; it is not a knob to remove.
    exposure: 0.085
  };
})(window);
