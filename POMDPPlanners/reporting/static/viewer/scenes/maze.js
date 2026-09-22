/* SPDX-License-Identifier: MIT
 *
 * Maze scene module, for the discrete and continuous variants alike.
 *
 * Builds the generated maze from a trace's `payload.world` block — the same
 * walkable set the episode was walked on, not a map regenerated here — and
 * moves the agent along the trace's recorded states. There is no authored
 * episode and no fallback: given no trace, the player draws nothing and says
 * so.
 *
 * The belief in this environment is not a cloud. The only hidden thing is
 * which of the two corner goals pays, so the belief is a distribution over one
 * bit, and every particle sits exactly where the agent is. Drawing it as a
 * spatial cloud would show a blob on top of the vehicle and say nothing. It is
 * drawn instead as two labelled bars, one per goal corner, holding the mass
 * the run's own belief put on that side — read out of the particles core
 * serialised, split on the state slot the payload names.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  var COLORS = {
    lamp: 0xFFE4B5,
    agent: 0xC74636,
    goal: 0x278365,
    cue: 0x1F77B4,
    start: 0x8C8C8C
  };

  // Lumens. A service lamp bolted to a wall top, not a floodlight: the far
  // corridors have to stay dark, or the maze has nowhere to hide a wrong turn.
  var LAMP_LUMENS = 620;
  var WALL_H = 0.8;

  // The belief bar sits above wall height on purpose: from the board camera a
  // bar rooted on the goal pad is hidden behind the near wall exactly when it
  // matters.
  var BAR_BASE = 0.95;
  var BAR_MAX = 0.9;

  /* The corridor floor: worn pale flagstones on a one-cell pitch, so the grid
     the discrete variant moves on is legible without a painted overlay. The
     normal map is derived from this canvas, so every chip catches the lamps
     from the side the lamp is actually on. */
  function floorCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(70719);
    var cells = 8;
    var step = s / cells;

    g.fillStyle = "#6E6A5E";
    g.fillRect(0, 0, s, s);
    for (var cy = 0; cy < cells; cy++) {
      for (var cx = 0; cx < cells; cx++) {
        var tone = 96 + rnd() * 40;
        g.fillStyle = "rgb(" + (tone | 0) + "," + ((tone * 0.97) | 0) + "," + ((tone * 0.88) | 0) + ")";
        g.fillRect(cx * step + 2, cy * step + 2, step - 4, step - 4);
      }
    }
    for (var i = 0; i < 4200; i++) {
      var r = 1 + rnd() * 7;
      g.fillStyle = rnd() > 0.5 ? "rgba(60,56,48,0.18)" : "rgba(214,209,194,0.14)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
    }
    for (var j = 0; j < 260; j++) {
      g.strokeStyle = "rgba(48,45,40,0.22)";
      g.lineWidth = 0.6 + rnd() * 1.2;
      g.beginPath();
      var sx = rnd() * s, sy = rnd() * s;
      g.moveTo(sx, sy);
      g.lineTo(sx + (rnd() - 0.5) * 70, sy + (rnd() - 0.5) * 70);
      g.stroke();
    }
    // Mortar lines last, dark and deep, so the normal map reads them as joints.
    g.strokeStyle = "rgba(28,26,23,0.85)";
    g.lineWidth = 4;
    for (var k = 0; k <= cells; k++) {
      var p = k * step;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
      g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
    }
    return cv;
  }

  /* The walls get their own coursed-stone texture rather than a tinted copy of
     the floor: they are the subject of this environment. */
  function wallCanvas() {
    var s = 512;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(33017);
    g.fillStyle = "#1A2430";
    g.fillRect(0, 0, s, s);

    var courses = 6;
    var h = s / courses;
    for (var row = 0; row < courses; row++) {
      var x = -(row % 2) * h * 0.9;
      while (x < s) {
        var w = h * (1.2 + rnd() * 1.1);
        var tone = 42 + rnd() * 26;
        g.fillStyle = "rgb(" + (tone | 0) + "," + ((tone * 1.22) | 0) + "," + ((tone * 1.55) | 0) + ")";
        g.fillRect(x + 3, row * h + 3, w - 6, h - 6);
        // A lit top chamfer and a dark underside on every block.
        g.fillStyle = "rgba(150,170,196,0.16)";
        g.fillRect(x + 3, row * h + 3, w - 6, 3);
        g.fillStyle = "rgba(6,9,13,0.5)";
        g.fillRect(x + 3, row * h + h - 7, w - 6, 4);
        x += w;
      }
    }
    for (var i = 0; i < 2600; i++) {
      g.fillStyle = rnd() > 0.55 ? "rgba(8,12,17,0.3)" : "rgba(128,148,172,0.1)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, 0.6 + rnd() * 3.4, 0, Math.PI * 2); g.fill();
    }
    return cv;
  }

  function labelTexture(text, color) {
    var s = 256;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    g.font = "bold 150px 'Chakra Petch', sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillStyle = color;
    g.fillText(text, s / 2, s / 2 + 6);
    return new THREE.CanvasTexture(cv);
  }

  /**
   * Choose where the service lamps hang.
   *
   * The environment has no lighting model, so this is presentation and nothing
   * else — but it cannot be the prototype's hand-picked list, because the map
   * is generated and a run may use any size or seed. Lamps go on wall cells
   * that touch a corridor, taken in a fixed order and thinned to a minimum
   * spacing, so a 7x9 map gets a handful and a 21x21 one gets them spread the
   * same way rather than a hundred shadow-casting candidates.
   *
   * @param {Object} world          The payload's world block.
   * @param {Function} isWalkable   (x, y) => boolean.
   * @returns {Array} Cells to hang a lamp on.
   */
  function lampCells(world, isWalkable) {
    var spacing = 2.4;
    var chosen = [];
    for (var y = 0; y < world.height; y++) {
      for (var x = 0; x < world.width; x++) {
        if (isWalkable(x, y)) continue;
        if (!(isWalkable(x + 1, y) || isWalkable(x - 1, y) ||
              isWalkable(x, y + 1) || isWalkable(x, y - 1))) continue;
        var clear = true;
        for (var c = 0; c < chosen.length && clear; c++) {
          if (Math.hypot(x - chosen[c][0], y - chosen[c][1]) < spacing) clear = false;
        }
        if (clear) chosen.push([x, y]);
      }
    }
    return chosen;
  }

  /**
   * Build the Maze world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind maze.v1.
   * @returns {Object} The scene module the player drives.
   */
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;

    var GW = world.width, GH = world.height;
    var HX = (GW - 1) / 2, HY = (GH - 1) / 2;
    // Grid x runs right; grid y runs *up* the board, which is -z here, so the
    // goals sit at the far end of the frame from the start.
    function wx(x) { return x - HX; }
    function wz(y) { return -(y - HY); }

    var walkableKeys = {};
    world.walkable.forEach(function (cell) { walkableKeys[cell[0] + "," + cell[1]] = true; });
    function isWalkable(x, y) { return walkableKeys[x + "," + y] === true; }

    scene.background = new THREE.Color(0x05070B).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x06080D, 0.018);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x1E2A3E, 0x080706, 0.26));
    var moon = new THREE.DirectionalLight(0x8FA3C8, 0.18);
    moon.position.set(-6, 9, 4);
    scene.add(moon);
    core.buildNightEnvironment();

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);
    var glowTex = V.radialTexture(0.9, 0.3);

    // Floor
    var fCanvas = floorCanvas();
    var floorNormal = V.normalMapFrom(renderer, fCanvas, 2.2);
    var floorAlbedo = new THREE.CanvasTexture(fCanvas);
    floorAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    floorAlbedo.encoding = THREE.sRGBEncoding;

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(GW, GH),
      new THREE.MeshStandardMaterial({
        map: floorAlbedo, normalMap: floorNormal,
        normalScale: new THREE.Vector2(0.9, 0.9),
        color: 0x8A8677, roughness: 0.92, metalness: 0.0, envMapIntensity: 0.3
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Ground carrying on past the maze, so the board is not a slab in a void.
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(90, 90),
      new THREE.MeshStandardMaterial({
        map: floorAlbedo, normalMap: floorNormal,
        normalScale: new THREE.Vector2(0.7, 0.7),
        color: 0x06080C, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.1
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.62;
    outer.receiveShadow = true;
    scene.add(outer);

    var plinth = new THREE.Mesh(
      new THREE.BoxGeometry(GW + 0.5, 0.66, GH + 0.5),
      new THREE.MeshStandardMaterial({ color: 0x040508, roughness: 0.94, metalness: 0.08 })
    );
    plinth.position.y = -0.5;
    plinth.castShadow = true;
    plinth.receiveShadow = true;
    scene.add(plinth);

    /* Walls. One block per wall cell, casting and receiving. The height is the
       whole point: a maze read from above is a picture, a maze with walls is a
       place you cannot see round. */
    var wCanvas = wallCanvas();
    var wallNormal = V.normalMapFrom(renderer, wCanvas, 2.6);
    var wallAlbedo = new THREE.CanvasTexture(wCanvas);
    wallAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    wallAlbedo.encoding = THREE.sRGBEncoding;

    var wallMat = new THREE.MeshStandardMaterial({
      map: wallAlbedo, normalMap: wallNormal,
      normalScale: new THREE.Vector2(1.0, 1.0),
      color: 0xA8B4C4, roughness: 0.88, metalness: 0.04, envMapIntensity: 0.4
    });
    var capMat = new THREE.MeshStandardMaterial({
      map: wallAlbedo, normalMap: wallNormal, color: 0x3A4654,
      roughness: 0.96, metalness: 0.04, envMapIntensity: 0.16
    });
    var wallGeo = new THREE.BoxGeometry(1.0, WALL_H, 1.0);
    var capGeo = new THREE.BoxGeometry(1.04, 0.07, 1.04);

    for (var gy = 0; gy < GH; gy++) {
      for (var gx = 0; gx < GW; gx++) {
        if (isWalkable(gx, gy)) continue;
        var block = new THREE.Mesh(wallGeo, wallMat);
        block.position.set(wx(gx), WALL_H / 2, wz(gy));
        block.castShadow = true; block.receiveShadow = true;
        scene.add(block);
        // A coping stone on top: it catches the lamps and gives every wall a
        // crisp lit edge, which is what sells the height from the raised view.
        var cap = new THREE.Mesh(capGeo, capMat);
        cap.position.set(wx(gx), WALL_H + 0.035, wz(gy));
        cap.castShadow = true; cap.receiveShadow = true;
        scene.add(cap);
      }
    }

    // Lamps. Real lumens with inverse-square falloff, so the far corridors
    // genuinely go dark.
    var LAMP_Y = WALL_H + 1.05;
    var lamps = [];
    var lampLensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(4.2, 3.6, 2.7) });
    var lampMetal = new THREE.MeshStandardMaterial({ color: 0x5C5142, roughness: 0.45, metalness: 0.8 });
    var armGeo = new THREE.CylinderGeometry(0.018, 0.026, 1.05, 8);
    var lampLensGeo = new THREE.SphereGeometry(0.08, 12, 10);
    var hoodGeo = new THREE.ConeGeometry(0.115, 0.13, 14, 1, true);

    lampCells(world, isWalkable).forEach(function (c) {
      var x = wx(c[0]), z = wz(c[1]);
      var post = new THREE.Mesh(armGeo, lampMetal);
      post.position.set(x, WALL_H + 0.52, z);
      post.castShadow = true;
      scene.add(post);

      var hood = new THREE.Mesh(hoodGeo, lampMetal);
      hood.position.set(x, LAMP_Y + 0.09, z);
      hood.castShadow = true;
      scene.add(hood);

      var lens = new THREE.Mesh(lampLensGeo, lampLensMat);
      lens.position.set(x, LAMP_Y, z);
      scene.add(lens);

      var light = new THREE.PointLight(COLORS.lamp, 1, 8.5, 2);
      light.power = LAMP_LUMENS;
      light.position.set(x, LAMP_Y, z);
      scene.add(light);
      lamps.push({ light: light, x: x, z: z });
    });

    /* One shadow-casting spot, not one per lamp. It parks on whichever lamp is
       nearest the agent, aims at it, and takes that share of the lamp's own
       output, so the maze stays evenly lit and the agent always throws a
       shadow that points the right way. */
    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 12, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.4;
    shadowLight.shadow.camera.far = 13;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, LAMP_Y, 0);
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    // The cue cell: the one place in the maze that tells the agent anything.
    var cueGroup = new THREE.Group();
    cueGroup.position.set(wx(world.cue_cell[0]), 0, wz(world.cue_cell[1]));
    scene.add(cueGroup);

    var cueDisc = new THREE.Mesh(
      new THREE.CircleGeometry(0.42, 40),
      new THREE.MeshBasicMaterial({
        color: COLORS.cue, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false
      })
    );
    cueDisc.rotation.x = -Math.PI / 2;
    cueDisc.position.y = 0.03;
    cueGroup.add(cueDisc);

    var cueRing = new THREE.Mesh(
      new THREE.RingGeometry(0.42, 0.48, 48),
      new THREE.MeshBasicMaterial({
        color: COLORS.cue, transparent: true, opacity: 0.85,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    cueRing.rotation.x = -Math.PI / 2;
    cueRing.position.y = 0.055;
    cueGroup.add(cueRing);

    var cueColumn = new THREE.Mesh(
      new THREE.CylinderGeometry(0.2, 0.44, 1.5, 24, 1, true),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: COLORS.cue, transparent: true, opacity: 0.09,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    cueColumn.position.y = 0.76;
    cueGroup.add(cueColumn);

    var cueLight = new THREE.PointLight(COLORS.cue, 1, 4.0, 2);
    cueLight.power = 120;
    cueLight.position.y = 0.5;
    cueGroup.add(cueLight);

    /* The goals. Both corners look the same to the agent: same pad, same
       letter. Only the belief bar above them differs, and the observer star,
       which is hidden from the agent by definition. */
    var goals = {};
    [["left", world.left_goal_cell, "L"], ["right", world.right_goal_cell, "R"]]
      .forEach(function (spec) {
        var side = spec[0], cell = spec[1], letter = spec[2];
        var group = new THREE.Group();
        group.position.set(wx(cell[0]), 0, wz(cell[1]));
        scene.add(group);

        var pad = new THREE.Mesh(
          new THREE.CylinderGeometry(0.44, 0.44, 0.05, 36),
          new THREE.MeshStandardMaterial({
            color: 0x0E2A20, roughness: 0.6, metalness: 0.3, emissive: 0x03140E
          })
        );
        pad.position.y = 0.025;
        pad.receiveShadow = true;
        group.add(pad);

        var rim = new THREE.Mesh(
          new THREE.RingGeometry(0.44, 0.5, 48),
          new THREE.MeshBasicMaterial({
            color: COLORS.goal, transparent: true, opacity: 0.8,
            blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
          })
        );
        rim.rotation.x = -Math.PI / 2;
        rim.position.y = 0.075;
        group.add(rim);

        var label = new THREE.Mesh(
          new THREE.PlaneGeometry(0.5, 0.5),
          new THREE.MeshBasicMaterial({
            map: labelTexture(letter, "#7FE0B8"), transparent: true,
            depthWrite: false, opacity: 0.9
          })
        );
        label.rotation.x = -Math.PI / 2;
        label.position.y = 0.1;
        group.add(label);

        /* The belief bar. Height is the mass the run's belief put on this
           side; the dim sleeve behind it is the full 1.0, so the two corners
           read against one scale rather than against each other. */
        var sleeve = new THREE.Mesh(
          new THREE.BoxGeometry(0.17, BAR_MAX, 0.17),
          new THREE.MeshBasicMaterial({ color: 0x33291A, transparent: true, opacity: 0.5 })
        );
        sleeve.position.y = BAR_BASE + BAR_MAX / 2;
        group.add(sleeve);

        // A stem from the pad up to the bar, so the reading is visibly
        // attached to the corner it is about.
        var stem = new THREE.Mesh(
          new THREE.CylinderGeometry(0.025, 0.025, BAR_BASE, 8),
          new THREE.MeshBasicMaterial({ color: 0x4A3C20 })
        );
        stem.position.y = BAR_BASE / 2;
        group.add(stem);

        var bar = new THREE.Mesh(
          new THREE.BoxGeometry(0.2, 1.0, 0.2),
          new THREE.MeshBasicMaterial({ color: new THREE.Color(2.9, 1.95, 0.42) })
        );
        group.add(bar);

        // The observer's star over the goal that actually pays. It is the
        // GIF's "True goal" caption in three dimensions: legibility for whoever
        // is watching, and never an input to the bars.
        var star = new THREE.Mesh(
          new THREE.SphereGeometry(0.11, 14, 12),
          new THREE.MeshBasicMaterial({ color: new THREE.Color(0.6, 3.2, 1.8) })
        );
        star.position.y = BAR_BASE + BAR_MAX + 0.3;
        star.visible = (side === "left")
          ? payload.true_goal_side === world.goal_left
          : payload.true_goal_side === world.goal_right;
        group.add(star);

        var light = new THREE.PointLight(COLORS.goal, 1, 3.6, 2);
        light.power = 26;
        light.position.y = 0.55;
        group.add(light);

        goals[side] = { bar: bar, star: star, light: light };
      });

    var startPad = new THREE.Mesh(
      new THREE.RingGeometry(0.3, 0.4, 36),
      new THREE.MeshBasicMaterial({
        color: COLORS.start, transparent: true, opacity: 0.55, side: THREE.DoubleSide
      })
    );
    startPad.rotation.x = -Math.PI / 2;
    startPad.position.set(wx(world.start_cell[0]), 0.02, wz(world.start_cell[1]));
    scene.add(startPad);

    // The agent. Scaled down, because a one-cell corridor has to stay a
    // corridor the vehicle fits along rather than fills.
    var rover = new THREE.Group();
    rover.scale.setScalar(0.78);
    scene.add(rover);
    var hull = new THREE.Group();
    rover.add(hull);

    var paintMat = new THREE.MeshStandardMaterial({ color: COLORS.agent, roughness: 0.46, metalness: 0.38 });
    var darkMetal = new THREE.MeshStandardMaterial({ color: 0x2A2522, roughness: 0.55, metalness: 0.75 });

    var chassis = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.17, 0.42), paintMat);
    chassis.position.y = 0.17; chassis.castShadow = true;
    hull.add(chassis);
    var deck = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.09, 0.34),
      new THREE.MeshStandardMaterial({ color: 0x8E241E, roughness: 0.5, metalness: 0.4 }));
    deck.position.set(-0.04, 0.30, 0); deck.castShadow = true;
    hull.add(deck);
    var cabin = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.13, 0.26),
      new THREE.MeshStandardMaterial({
        color: 0x24406E, roughness: 0.12, metalness: 0.3,
        emissive: 0x13203A, emissiveIntensity: 0.8
      }));
    cabin.position.set(0.06, 0.40, 0); cabin.castShadow = true;
    hull.add(cabin);

    // Mast and sensor head: the thing that reads the cue.
    var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.26, 8), darkMetal);
    mast.position.set(-0.16, 0.48, 0);
    hull.add(mast);
    var sensorHead = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.06, 0.05), darkMetal);
    sensorHead.position.set(-0.16, 0.62, 0); sensorHead.castShadow = true;
    hull.add(sensorHead);
    var sensorEye = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 8),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 1.6, 2.6) }));
    sensorEye.position.set(-0.125, 0.62, 0);
    hull.add(sensorEye);

    [0.14, -0.14].forEach(function (z) {
      var rail = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.012, 6, 12, Math.PI), darkMetal);
      rail.position.set(-0.04, 0.34, z);
      rail.rotation.y = Math.PI / 2;
      hull.add(rail);
    });

    var wheels = [];
    var wheelGeo = new THREE.CylinderGeometry(0.115, 0.115, 0.10, 18);
    var wheelMat = new THREE.MeshStandardMaterial({ color: 0x171514, roughness: 0.88, metalness: 0.12 });
    var hubMat = new THREE.MeshStandardMaterial({ color: 0x6E5A3C, roughness: 0.4, metalness: 0.85 });
    [[0.20, 0.22], [0.20, -0.22], [-0.20, 0.22], [-0.20, -0.22]].forEach(function (w) {
      var m = new THREE.Mesh(wheelGeo, wheelMat);
      m.position.set(w[0], 0.115, w[1]);
      m.rotation.x = Math.PI / 2;
      m.castShadow = true;
      rover.add(m);
      wheels.push(m);
      for (var t = 0; t < 8; t++) {
        var a = (t / 8) * Math.PI * 2;
        var tread = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.11, 0.025), wheelMat);
        tread.position.set(Math.cos(a) * 0.108, 0, Math.sin(a) * 0.108);
        tread.rotation.y = -a;
        m.add(tread);
      }
      m.add(new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 0.106, 10), hubMat));
    });

    var headLampMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.6, 3.3, 2.7) });
    [[0.31, 0.13], [0.31, -0.13]].forEach(function (h) {
      var m = new THREE.Mesh(new THREE.SphereGeometry(0.042, 10, 10), headLampMat);
      m.position.set(h[0], 0.23, h[1]);
      hull.add(m);
      var flare = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xFFF1D6, transparent: true, opacity: 0.45,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      flare.scale.set(0.4, 0.4, 1);
      flare.position.set(h[0] + 0.01, 0.23, h[1]);
      hull.add(flare);
    });

    var blob = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 0.8),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.4, depthWrite: false
      }));
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.008;
    rover.add(blob);

    // The headlight is what makes a corridor readable from the chase camera.
    var headlight = new THREE.SpotLight(0xFFF1D6, 1, 8, 0.52, 0.62, 2);
    headlight.power = 760;
    headlight.castShadow = true;
    headlight.shadow.mapSize.set(1024, 1024);
    headlight.shadow.camera.near = 0.2;
    headlight.shadow.camera.far = 8;
    headlight.shadow.normalBias = 0.03;
    headlight.position.set(0.3, 0.26, 0);
    var headTarget = new THREE.Object3D();
    headTarget.position.set(4, -0.1, 0);
    rover.add(headlight);
    rover.add(headTarget);
    headlight.target = headTarget;

    var beamCone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 0.95, 2.6, 20, 1, true),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0xFFE9BE, transparent: true, opacity: 0.026,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    beamCone.rotation.z = Math.PI / 2 + 0.05;
    beamCone.position.set(1.5, 0.22, 0);
    rover.add(beamCone);

    var roverGlow = new THREE.PointLight(0xFFB08A, 0.4, 2.0, 2);
    roverGlow.position.y = 0.3;
    rover.add(roverGlow);

    // Dust, so the lamp beams have something to hang in.
    var MOTES = 220;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(1357);
    for (var mi = 0; mi < MOTES; mi++) {
      motePos[mi * 3] = (moteSeed() - 0.5) * (GW + 1);
      motePos[mi * 3 + 1] = 0.1 + moteSeed() * 1.7;
      motePos[mi * 3 + 2] = (moteSeed() - 0.5) * (GH + 1);
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    scene.add(new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.032, map: partTex, color: 0xFFE3BC, transparent: true, opacity: 0.35,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    })));

    // Trail: the recorded path, drawn up to the current step.
    var TRAIL_MAX = payload.states.length + 1;
    var tPos = new Float32Array(TRAIL_MAX * 3);
    var tCol = new Float32Array(TRAIL_MAX * 3);
    var tGeo = new THREE.BufferGeometry();
    tGeo.setAttribute("position", new THREE.BufferAttribute(tPos, 3));
    tGeo.setAttribute("color", new THREE.BufferAttribute(tCol, 3));
    tGeo.setDrawRange(0, 0);
    scene.add(new THREE.Line(tGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.9
    })));

    /* A line one pixel wide vanishes at this camera height, and WebGL ignores
       linewidth, so the visible trail is a dense strip of glowing markers laid
       along the same polyline. */
    var DOT_SPACING = 0.07;
    var DOT_MAX = Math.min(2400, Math.max(64, Math.ceil(TRAIL_MAX * 2.2 / DOT_SPACING)));
    var dPos = new Float32Array(DOT_MAX * 3);
    var dGeo = new THREE.BufferGeometry();
    dGeo.setAttribute("position", new THREE.BufferAttribute(dPos, 3));
    dGeo.setDrawRange(0, 0);
    scene.add(new THREE.Points(dGeo, new THREE.PointsMaterial({
      size: 0.11, map: partTex, color: new THREE.Color(1.5, 0.42, 0.3),
      transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending,
      depthWrite: false, sizeAttenuation: true
    })));

    core.linearize();

    var running = [];
    var total = 0;
    for (var ri = 0; ri < trace.steps.length; ri++) {
      var reward = trace.steps[ri].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, ri);
      }
      running.push(total);
    }

    function sampleAt(t) {
      var n = payload.states.length;
      var i0 = Math.floor(clamp(t, 0, n - 1));
      var i1 = Math.min(i0 + 1, n - 1);
      var f = clamp(t - i0, 0, 1);
      var a = payload.states[i0], b = payload.states[i1];
      return {
        index: i0, frac: f,
        x: lerp(a[0], b[0], f),
        y: lerp(a[1], b[1], f)
      };
    }

    function rebuildTrail(t) {
      var pts = payload.states;
      var i0 = Math.floor(clamp(t, 0, pts.length - 1));
      var f = clamp(t - i0, 0, 1);
      var count = 0;
      for (var i = 0; i <= i0 && count < TRAIL_MAX; i++) {
        tPos[count * 3] = wx(pts[i][0]);
        tPos[count * 3 + 1] = 0.06;
        tPos[count * 3 + 2] = wz(pts[i][1]);
        count++;
      }
      if (f > 0 && i0 + 1 < pts.length && count < TRAIL_MAX) {
        tPos[count * 3] = wx(lerp(pts[i0][0], pts[i0 + 1][0], f));
        tPos[count * 3 + 1] = 0.06;
        tPos[count * 3 + 2] = wz(lerp(pts[i0][1], pts[i0 + 1][1], f));
        count++;
      }
      for (var c = 0; c < count; c++) {
        var age = count < 2 ? 1 : c / (count - 1);
        tCol[c * 3] = lerp(0.42, 0.92, age);
        tCol[c * 3 + 1] = lerp(0.12, 0.31, age);
        tCol[c * 3 + 2] = lerp(0.10, 0.24, age);
      }
      tGeo.attributes.position.needsUpdate = true;
      tGeo.attributes.color.needsUpdate = true;
      tGeo.setDrawRange(0, Math.max(count, 0));

      var dots = 0;
      for (var s = 1; s < count && dots < DOT_MAX; s++) {
        var ax = tPos[(s - 1) * 3], az = tPos[(s - 1) * 3 + 2];
        var bx = tPos[s * 3], bz = tPos[s * 3 + 2];
        var seg = Math.hypot(bx - ax, bz - az);
        var n = Math.max(1, Math.round(seg / DOT_SPACING));
        for (var q = 0; q < n && dots < DOT_MAX; q++) {
          var u = q / n;
          dPos[dots * 3] = lerp(ax, bx, u);
          dPos[dots * 3 + 1] = 0.05;
          dPos[dots * 3 + 2] = lerp(az, bz, u);
          dots++;
        }
      }
      dGeo.attributes.position.needsUpdate = true;
      dGeo.setDrawRange(0, dots);
    }

    /* ------------------------------------------------------------- belief
       The hidden variable is which corner pays, so the belief is read as the
       mass its particles put on each goal side and drawn as the two bars.
       Nothing is inferred from the true side and nothing is smoothed: a belief
       core could not serialise is reported as missing, and the bars come down
       rather than being replaced by something plausible. */

    /** Split a serialized particle cloud by goal side. */
    function particleSplit(belief) {
      var slot = world.state_goal_index;
      if (!belief.particles.length || !Array.isArray(belief.particles[0])) return null;
      if (belief.particles[0].length <= slot) return null;
      var left = 0, mass = 0;
      for (var i = 0; i < belief.particles.length; i++) {
        var w = belief.weights[i];
        if (w === undefined) continue;
        mass += w;
        if (belief.particles[i][slot] === world.goal_left) left += w;
      }
      if (mass <= 0) return null;
      return left / mass;
    }

    function setBars(pLeft) {
      [["left", pLeft], ["right", 1 - pLeft]].forEach(function (pair) {
        var g = goals[pair[0]], p = pair[1];
        g.bar.visible = true;
        var h = Math.max(0.035, p * BAR_MAX);
        g.bar.scale.y = h;
        g.bar.position.y = BAR_BASE + h / 2;
        g.light.power = 26 + 44 * p;
      });
    }

    function hideBars() {
      goals.left.bar.visible = false;
      goals.right.bar.visible = false;
      goals.left.light.power = 26;
      goals.right.light.power = 26;
    }

    function drawBelief(index) {
      var belief = payload.beliefs[index];
      if (!belief) { hideBars(); return "—"; }

      if (belief.kind === "particles") {
        var pLeft = particleSplit(belief);
        if (pLeft === null) {
          hideBars();
          return "particles carry no goal side";
        }
        setBars(pLeft);
        var label = "L " + pLeft.toFixed(2) + " / R " + (1 - pLeft).toFixed(2) +
          " over " + belief.num_particles + " particles";
        if (belief.num_written < belief.num_particles) {
          label += " (heaviest " + belief.num_written + " read)";
        }
        return label;
      }
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner. It
        // is not one episode's belief, and averaging its members would show a
        // split that was never anyone's belief.
        hideBars();
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }

      // A Gaussian over this state would be a continuous density over a slot
      // that takes two values, and a class core has no payload for cannot be
      // read at all. Both are named rather than guessed at.
      hideBars();
      return "not readable as goal-side mass (" + (belief.belief_class || belief.kind) + ")";
    }

    function formatAction(action) {
      if (action === null || action === undefined) return "—";
      if (Array.isArray(action)) {
        return "[" + action.map(function (v) { return Number(v).toFixed(2); }).join(", ") + "]";
      }
      return String(action);
    }

    var heading = -Math.PI / 2;      // facing up the board at the start
    var wheelSpin = 0;
    var span = Math.max(GW, GH);

    return {
      steps: payload.states.length,

      /* Framing scales with the map, because the trace decides how big it is:
         the distance that frames a 7x9 maze leaves a 21x21 one running off
         the canvas. The constant term is headroom for the props, which do not
         scale — a lamp post is the same height on any map. */
      camera: {
        board: [0, span * 1.0 + 1.2, span * 0.7 + 0.7],
        top: [0, span * 1.2 + 1.6, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var px = wx(sample.x), pz = wz(sample.y);

        /* Heading is taken in world space, not grid space: grid y maps to -z,
           so a grid direction (dx, dy) travels (dx, -dy) here. The body's
           forward axis is local +x and `rotation.y = -heading` sends it to
           (cos, 0, sin), which is the same vector the core rig's chase camera
           sits behind. Rotating by +heading instead leaves east-west legs
           right and drives every north-south leg backwards. */
        var next = payload.states[Math.min(sample.index + 1, payload.states.length - 1)];
        var current = payload.states[sample.index];
        var dx = next[0] - current[0], dy = next[1] - current[1];
        if (dx !== 0 || dy !== 0) {
          var target = Math.atan2(-dy, dx);
          var diff = ((target - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          var turn = diff * clamp(dt * 8, 0, 1);
          heading += turn;
          hull.rotation.z = lerp(hull.rotation.z, clamp(turn * 3.0, -0.15, 0.15),
            clamp(dt * 8, 0, 1));
        }
        rover.position.set(px, 0, pz);
        rover.rotation.y = -heading;

        var speed = playing ? 1 : 0;
        wheelSpin += dt * speed * 6.0;
        for (var w = 0; w < wheels.length; w++) wheels[w].rotation.y = wheelSpin;

        // Move the one shadow caster onto the nearest lamp and take that much
        // light back off it, so the shadow follows without over-lighting.
        var nearest = null, nd = Infinity;
        for (var li = 0; li < lamps.length; li++) {
          var L = lamps[li];
          L.light.power = LAMP_LUMENS;
          var d2 = (L.x - px) * (L.x - px) + (L.z - pz) * (L.z - pz);
          if (d2 < nd) { nd = d2; nearest = L; }
        }
        if (nearest) {
          shadowLight.position.set(nearest.x, LAMP_Y, nearest.z);
          shadowLight.target.position.set(px, 0.1, pz);
          shadowLight.target.updateMatrixWorld();
          var share = 0.6 * clamp(1 - (Math.sqrt(nd) - 1.5) / 5.5, 0.08, 1);
          shadowLight.power = LAMP_LUMENS * share;
          nearest.light.power = LAMP_LUMENS * (1 - share);
        }

        /* The cue is single-use, and the trace says so: it burns while the
           phase is unseen, flares on the step whose state is emitting, and
           goes dark once consumed. That is the environment's rule, not
           decoration — standing on the cue cell does not re-read it. */
        var phase = payload.cue_phases[sample.index];
        var cueGlow = phase === world.cue_emitting
          ? 1.5
          : (phase === world.cue_consumed
            ? 0.12
            : 0.75 + Math.sin(elapsed * 2.4) * 0.18);
        cueDisc.material.opacity = 0.42 * cueGlow;
        cueRing.material.opacity = 0.85 * cueGlow;
        cueColumn.material.opacity = 0.09 * cueGlow;
        cueLight.power = 120 * cueGlow;

        var beliefLabel = drawBelief(sample.index);

        for (var m = 0; m < MOTES; m++) {
          motePos[m * 3] += Math.sin(elapsed * 0.3 + m) * 0.0014;
          motePos[m * 3 + 1] += 0.0018;
          if (motePos[m * 3 + 1] > 1.9) motePos[m * 3 + 1] = 0.1;
        }
        moteGeo.attributes.position.needsUpdate = true;
        var bob = Math.sin(elapsed * 1.6) * 0.05;
        goals.left.star.position.y = BAR_BASE + BAR_MAX + 0.3 + bob;
        goals.right.star.position.y = BAR_BASE + BAR_MAX + 0.3 + bob;

        rebuildTrail(t);

        var step = trace.steps[sample.index] || {};
        return {
          follow: { x: px, z: pz, heading: heading },
          step: sample.index,
          action: formatAction(step.action),
          x: sample.x,
          y: sample.y,
          reward: step.reward,
          ret: running[sample.index],
          belief: beliefLabel
        };
      }
    };
  }

  V.scenes["maze.v1"] = {
    build: build,
    // The lamps carry real lumens, so the camera stops down the way a real one
    // would for a night shot. This single number sets the whole mood; it is
    // tuned for this scene's lamp power, not a knob to remove.
    exposure: 0.21
  };
})(window);
