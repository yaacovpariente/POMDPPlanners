/* SPDX-License-Identifier: MIT
 *
 * Push scene module, for both Push worlds.
 *
 * Builds the yard from a trace's `payload.world` block and moves it from the
 * trace's recorded states and beliefs. Nothing here is invented: there is no
 * fallback episode, no hand-placed waypoint and no synthetic belief. If the
 * player hands this module no trace, it draws nothing and says so.
 *
 * One kind covers the discrete and the continuous variant. They differ in the
 * two things that have to be drawn differently — an obstacle is a circle of
 * `obstacle_radius` in one and an axis-aligned square in the other, and an
 * action is one of four labels in one and a 2-D vector in the other — and the
 * payload names both, so this file branches on the world rather than on two
 * nearly identical scene modules.
 *
 * Two things about this environment drive the whole look.
 *
 * Contact is the task. The rover changes the crate's position only from inside
 * `push_threshold`, and friction means the crate travels (1 - f) of the shove,
 * so the rover closes on the crate every time it pushes and must eventually
 * back out to line up again. The blade, the two contact shadows, the dust at
 * the point of contact and the drag mark behind the crate are all there to
 * make that one metre legible.
 *
 * And the crate travels along the ACTION, not along a contact normal. The
 * environment's transition has no normal in it: if the crate is within
 * `push_threshold` of the post-move robot, it moves along the action and
 * nothing else. So the blade tracks the crate (that is where the two bodies
 * touch) while the red arrow keeps saying which way the crate will actually
 * go, and the two are allowed to disagree.
 *
 * The struggle across broken ground is attitude only. A dangerous area costs
 * its penalty and nothing else: it does not slow the rover, and the step still
 * takes exactly one step. So pitch, roll, ride height, wheel slip, judder and
 * flung grit carry no positional meaning at all, and the one thing that does
 * touch position — a wobble on the along-step parameter — is windowed to
 * exactly zero at both ends of every step. The rover and the crate sit on the
 * recorded state at every step boundary, however badly they are thrown about
 * in between.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* Colours are the environment's own, from push_visualization_utils.py and
     push_visualization_assets.py, so the viewer reads as the same world as the
     GIF the package ships. */
  var C = {
    robot: 0x3A48E0,      // ROBOT (0,0,255), lifted so it reads on sand
    radius: 0xBFBFFF,     // ROBOT_RADIUS
    object: 0xFFA500,     // OBJECT
    objEdge: 0xB8741A,
    target: 0xDAA500,     // TARGET
    tgtEdge: 0xB8860B,    // TARGET_EDGE
    rock: 0xAE583D,       // OBSTACLE
    rockEdge: 0x5E2B1D,   // OBSTACLE_EDGE
    danger: 0xB93A25,
    belief: 0x6FD8FF
  };

  /* The shared post stack's bright pass is tuned for a night scene: it starts
     blooming at a luminance of 0.85, where this sunlit yard would bloom over
     its whole floor. The threshold is not a per-scene knob, so the scene is
     rendered at 0.85/1.6 of the radiance it was designed at and the camera
     opens up by the same factor — identical composite, bloom cut in the same
     place. Every radiance in this file is scaled by RADIANCE on the way in:
     the lights, the background, the sky it reflects, and the colours of the
     unlit materials, which a basic material emits as-is. */
  var RADIANCE = 0.85 / 1.6;
  var EXPOSURE = 0.74 / RADIANCE;

  // Cap on drawn belief particles. The exporter already caps a cloud at 400;
  // more than this is draw cost with no extra information at this scale.
  var MAX_DRAWN_PARTICLES = 400;

  var CRATE = 0.38;       // drawn size of the crate; the state holds a point
  var WHEEL_Y = 0.105;
  var DUST = 210;
  var MARK_MAX = 900;

  function smooth(t) { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); }

  /* Sandstone pan. Four octaves of blotching over the warm base the Pillow
     renderer layers, then grit and pebbles with a lit cap and a dark side,
     which the normal map derived from this canvas turns into relief. */
  function groundCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(20260905);   // the seed stone_texture uses

    g.fillStyle = "#9B896D";
    g.fillRect(0, 0, s, s);
    var oct = [[5, 19], [14, 12], [45, 7], [140, 4]];
    for (var o = 0; o < oct.length; o++) {
      var cells = oct[o][0], amp = oct[o][1] * 2.2, cell = s / cells;
      for (var cy = 0; cy < cells; cy++) {
        for (var cx = 0; cx < cells; cx++) {
          var v = (rnd() - 0.5) * amp;
          g.fillStyle = "rgba(" + (v > 0 ? 255 : 0) + "," + (v > 0 ? 240 : 0) + "," +
            (v > 0 ? 210 : 0) + "," + (Math.abs(v) / 255 * 2.1).toFixed(3) + ")";
          g.fillRect(cx * cell - 1, cy * cell - 1, cell + 2, cell + 2);
        }
      }
    }
    for (var p = 0; p < 1600; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.4 + rnd() * 4.2;
      g.fillStyle = "rgba(58,46,32,0.42)";
      g.beginPath(); g.arc(px + pr * 0.35, py + pr * 0.35, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(214,196,160,0.36)";
      g.beginPath(); g.arc(px, py, pr * 0.85, 0, Math.PI * 2); g.fill();
    }
    for (var j = 0; j < 2600; j++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(236,222,190,0.07)" : "rgba(40,31,21,0.16)";
      g.fillRect(rnd() * s, rnd() * s, 3, 3);
    }
    return cv;
  }

  /* The one-unit grid, painted onto the colour only so it marks scale without
     embossing the sand. The Pillow renderer draws a dark line with a pale line
     beside it; the same pair reads as a scored line here. */
  function paintGrid(cv, cells) {
    var s = cv.width;
    var g = cv.getContext("2d");
    for (var k = 0; k <= cells; k++) {
      var p = (k / cells) * s;
      g.strokeStyle = "rgba(84,74,59,0.55)"; g.lineWidth = 2.5;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
      g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
      g.strokeStyle = "rgba(200,182,148,0.30)"; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(p + 2.5, 0); g.lineTo(p + 2.5, s); g.stroke();
      g.beginPath(); g.moveTo(0, p + 2.5); g.lineTo(s, p + 2.5); g.stroke();
    }
  }

  /* A burn patch rather than a soft glow: nearly solid over most of the disc,
     ragged at the edge, mottled so it does not read as a decal. The shared
     radial sprite's falloff is far too gentle to darken a whole zone. */
  function scorchTexture() {
    var s = 256, cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grad.addColorStop(0.00, "rgba(255,255,255,0.94)");
    grad.addColorStop(0.62, "rgba(255,255,255,0.86)");
    grad.addColorStop(0.86, "rgba(255,255,255,0.40)");
    grad.addColorStop(1.00, "rgba(255,255,255,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, s, s);
    var rnd = mulberry(5309);
    g.globalCompositeOperation = "destination-out";
    for (var i = 0; i < 90; i++) {
      var a = rnd() * Math.PI * 2, r = rnd() * s * 0.46;
      g.fillStyle = "rgba(0,0,0," + (0.10 + rnd() * 0.30).toFixed(2) + ")";
      g.beginPath();
      g.arc(s / 2 + Math.cos(a) * r, s / 2 + Math.sin(a) * r, 3 + rnd() * 16, 0, Math.PI * 2);
      g.fill();
    }
    g.globalCompositeOperation = "source-over";
    return new THREE.CanvasTexture(cv);
  }

  /* A late-afternoon sky to reflect, with a low sun disc in it. Without an
     environment map the rover's paint and the blade's steel have nothing to
     mirror, and PBR metal with nothing to reflect reads as grey plastic. The
     core's own helper builds a night sky and cannot carry the disc, which is
     what a grazing highlight on the blade is a highlight of. */
  function buildYardEnvironment(core) {
    var cv = document.createElement("canvas");
    cv.width = 256; cv.height = 128;
    var g = cv.getContext("2d");
    var grad = g.createLinearGradient(0, 0, 0, 128);
    grad.addColorStop(0.0, "#5C87C8");
    grad.addColorStop(0.42, "#A8BCD2");
    grad.addColorStop(0.52, "#E0BE8C");
    grad.addColorStop(0.62, "#B08E63");
    grad.addColorStop(1.0, "#6A5A42");
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 128);
    var sg = g.createRadialGradient(196, 62, 0, 196, 62, 26);
    sg.addColorStop(0, "rgba(255,244,214,1)");
    sg.addColorStop(1, "rgba(255,214,150,0)");
    g.fillStyle = sg;
    g.fillRect(170, 36, 52, 52);
    var tex = new THREE.CanvasTexture(cv);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    tex.encoding = THREE.sRGBEncoding;
    var pmrem = new THREE.PMREMGenerator(core.renderer);
    pmrem.compileEquirectangularShader();
    core.scene.environment = pmrem.fromEquirectangular(tex).texture;
    pmrem.dispose();
    tex.dispose();
  }

  /* Scale every radiance in the scene, once, after it is built. Standard
     materials take it through their share of the environment light; unlit
     materials emit their colour directly, so the colour itself is scaled. */
  function scaleRadiance(scene, k) {
    scene.traverse(function (obj) {
      var mats = obj.material ? (Array.isArray(obj.material) ? obj.material : [obj.material]) : [];
      mats.forEach(function (m) {
        if (!m || m.__scaled) return;
        m.__scaled = true;
        if (m.isMeshStandardMaterial) {
          m.envMapIntensity = (m.envMapIntensity === undefined ? 1 : m.envMapIntensity) * k;
        } else if (m.color) {
          m.color.multiplyScalar(k);
        }
      });
    });
  }

  /**
   * Build the Push world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind push.v1.
   * @returns {Object} The scene module the player drives.
   */
  // The world is one scene, built in one pass, and splitting it into a dozen
  // helpers that each take the same eight locals would not make it readable.
  // eslint-disable-next-line max-statements
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;

    var reduceMotion = global.matchMedia
      ? global.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;

    var GRID = Math.max(2, world.grid_size);
    var extent = GRID - 1;
    var HALF = extent / 2;
    var PUSH = world.push_threshold;
    var HAZARD_R = world.dangerous_area_radius;
    var GOAL_TOL = world.goal_tolerance;
    var continuous = world.variant === "continuous";
    var ROBOT_RADIUS = world.robot_radius || 0.3;
    var OBST_R = world.obstacle_radius || 0.5;
    var obstacles = world.obstacles || [];
    var hazards = world.dangerous_areas || [];

    // The board spans 0..extent on both axes and is shifted so its centre sits
    // at the origin; world +y runs away from the camera, as it does in the GIF.
    function wx(x) { return x - HALF; }
    function wz(y) { return -(y - HALF); }
    function ex(worldX) { return worldX + HALF; }
    function ey(worldZ) { return HALF - worldZ; }

    scene.background = new THREE.Color(0xE3C79B).convertSRGBToLinear().multiplyScalar(RADIANCE);
    scene.fog = new THREE.FogExp2(0xD9BB8E, 0.0095);
    scene.fog.color.convertSRGBToLinear().multiplyScalar(RADIANCE);

    /* Late afternoon. One low sun does the modelling and, more to the point,
       throws the long shadows that tell you the blade is touching the crate
       and not hovering a hand's width from it. */
    var SUN_DIR = new THREE.Vector3(-5.4, 6.6, 4.6);
    var sun = new THREE.DirectionalLight(0xFFE0AE, 3.15 * RADIANCE);
    sun.position.copy(SUN_DIR).multiplyScalar(1.6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.radius = 2;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 40;
    sun.shadow.camera.left = -8.5;
    sun.shadow.camera.right = 8.5;
    sun.shadow.camera.top = 8.5;
    sun.shadow.camera.bottom = -8.5;
    // A narrow shadow camera plus normalBias, which is the fix for acne that a
    // large negative bias is not.
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.035;
    scene.add(sun);
    scene.add(sun.target);

    // Sky fill and warm bounce off the sand, so shadow sides are not black.
    scene.add(new THREE.HemisphereLight(0x9FC2F0, 0x6A5334, 0.85 * RADIANCE));
    var bounce = new THREE.DirectionalLight(0xFFCF9C, 0.35 * RADIANCE);
    bounce.position.set(5, 2.4, -4);
    scene.add(bounce);

    buildYardEnvironment(core);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);

    // The grit is painted first and the normal map derived from it, so every
    // pebble catches the sun from the side the sun is actually on. The grid is
    // painted afterwards, onto colour only.
    var gCanvas = groundCanvas();
    var groundNormal = V.normalMapFrom(renderer, gCanvas, 2.1);
    paintGrid(gCanvas, extent);
    var groundAlbedo = new THREE.CanvasTexture(gCanvas);
    groundAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    groundAlbedo.encoding = THREE.sRGBEncoding;

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(extent, extent),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.9, 0.9),
        roughness: 0.96, metalness: 0.0, envMapIntensity: 0.4
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    /* Ground carrying on past the board, dropped well clear of the floor so
       the 16-bit depth buffer never has to separate two coplanar slabs. */
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(120, 120),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.8, 0.8),
        color: 0x8A7658, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.2
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.62;
    outer.receiveShadow = true;
    scene.add(outer);

    var plinth = new THREE.Mesh(
      new THREE.BoxGeometry(extent + 0.7, 0.66, extent + 0.7),
      new THREE.MeshStandardMaterial({ color: 0x4A3B29, roughness: 0.92, metalness: 0.05 })
    );
    plinth.position.y = -0.47;
    plinth.castShadow = true;
    plinth.receiveShadow = true;
    scene.add(plinth);

    var kerbMat = new THREE.MeshStandardMaterial({
      color: 0x6A5740, roughness: 0.9, metalness: 0.06
    });
    var kerbLen = extent + 0.36;
    [[0, -HALF - 0.18], [0, HALF + 0.18]].forEach(function (o) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(kerbLen, 0.16, 0.36), kerbMat);
      m.position.set(o[0], 0.08, o[1]);
      m.receiveShadow = true; m.castShadow = true;
      scene.add(m);
    });
    [[-HALF - 0.18, 0], [HALF + 0.18, 0]].forEach(function (o) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.16, kerbLen), kerbMat);
      m.position.set(o[0], 0.08, o[1]);
      m.receiveShadow = true; m.castShadow = true;
      scene.add(m);
    });

    // Loose rock, for parallax and cast shadow. Kept off the recorded route of
    // both bodies, off the obstacles and off the target.
    var rockGeos = [
      new THREE.DodecahedronGeometry(0.15, 0),
      new THREE.IcosahedronGeometry(0.12, 0),
      new THREE.TetrahedronGeometry(0.18, 0)
    ];
    var pebbleMat = new THREE.MeshStandardMaterial({
      color: 0x7B6A4E, roughness: 0.95, metalness: 0.04, flatShading: true
    });
    var rockRnd = mulberry(5150);
    var line = payload.states;
    var placed = 0, guard = 0;
    while (placed < 34 && guard++ < 900) {
      var rx0 = rockRnd() * extent, ry0 = rockRnd() * extent;
      var clear = true;
      for (var oi = 0; oi < obstacles.length && clear; oi++) {
        if (Math.hypot(rx0 - obstacles[oi][0], ry0 - obstacles[oi][1]) < 1.1) clear = false;
      }
      for (var hi = 0; hi < hazards.length && clear; hi++) {
        if (Math.hypot(rx0 - hazards[hi][0], ry0 - hazards[hi][1]) < HAZARD_R + 0.3) clear = false;
      }
      for (var li = 0; li < line.length && clear; li++) {
        if (Math.hypot(rx0 - line[li][0], ry0 - line[li][1]) < 0.7) clear = false;
        if (Math.hypot(rx0 - line[li][2], ry0 - line[li][3]) < 0.7) clear = false;
      }
      if (Math.hypot(rx0 - world.target[0], ry0 - world.target[1]) < 1.2) clear = false;
      if (!clear) continue;
      var rock = new THREE.Mesh(rockGeos[placed % rockGeos.length], pebbleMat);
      var sc = 0.35 + rockRnd() * 0.75;
      rock.scale.set(sc, sc * (0.5 + rockRnd() * 0.4), sc);
      rock.position.set(wx(rx0), 0.04 * sc, wz(ry0));
      rock.rotation.set(rockRnd() * 3, rockRnd() * 3, rockRnd() * 3);
      rock.castShadow = true; rock.receiveShadow = true;
      scene.add(rock);
      placed++;
    }

    /* Obstacles. The two worlds really do have different obstacle geometry:
       circles of obstacle_radius in the discrete one, axis-aligned squares in
       the continuous one, and both are walls to the rover and the crate. */
    var rockMat = new THREE.MeshStandardMaterial({
      color: C.rock, roughness: 0.88, metalness: 0.05, flatShading: true
    });
    obstacles.forEach(function (o, idx) {
      var x = wx(o[0]), z = wz(o[1]);
      if (continuous) {
        var hx = o[2], hy = o[3];
        var block = new THREE.Mesh(new THREE.BoxGeometry(2 * hx, 0.52, 2 * hy), rockMat);
        block.position.set(x, 0.26, z);
        block.castShadow = true; block.receiveShadow = true;
        scene.add(block);
        var cap = new THREE.Mesh(
          new THREE.BoxGeometry(2 * hx + 0.08, 0.08, 2 * hy + 0.08), rockMat
        );
        cap.position.set(x, 0.56, z);
        cap.castShadow = true; cap.receiveShadow = true;
        scene.add(cap);
        return;
      }
      var boulder = new THREE.Mesh(new THREE.IcosahedronGeometry(OBST_R, 1), rockMat);
      boulder.position.set(x, 0.30 * (OBST_R / 0.5), z);
      boulder.scale.set(1, 0.86, 1);
      boulder.rotation.set(0.4 + idx, 0.9 * idx, 0.2);
      boulder.castShadow = true; boulder.receiveShadow = true;
      scene.add(boulder);
      // The collision circle itself, so the footprint that blocks movement is
      // visible and not guessed from the boulder's silhouette.
      var skirt = new THREE.Mesh(
        new THREE.RingGeometry(OBST_R - 0.02, OBST_R + 0.03, 40),
        new THREE.MeshBasicMaterial({
          color: C.rockEdge, transparent: true, opacity: 0.5, side: THREE.DoubleSide
        })
      );
      skirt.rotation.x = -Math.PI / 2;
      skirt.position.set(x, 0.012, z);
      scene.add(skirt);
    });

    /* Dangerous areas. A penalty region, not an obstacle: both bodies cross it
       freely and it simply costs the penalty. A flat red disc said "scoring
       region", so the ground inside is built as genuinely broken terrain —
       fractured slabs heaved out of the pan, fissures between them, sharp
       upthrust wedges and loose scree, all scorched. Nothing is tall enough to
       read as a wall; the point is that you could drive over it and would
       regret it. The low sun does the work: every slab casts, so a fissure is
       a real dark line and an upthrust edge has a real shadow behind it.

       Two rules govern the build. Nothing may cross the true penalty radius,
       so every piece is placed by rejection sampling against its own worst-case
       reach with 4 cm of margin, and the red ring stays exactly on the radius
       the environment penalises. And nothing thin sits coplanar with the yard
       floor: the stain and the fissure beds are the only ground-hugging quads
       and both carry polygonOffset. */
    var scorchTex = scorchTexture();
    var stainMat = new THREE.MeshBasicMaterial({
      map: scorchTex, color: new THREE.Color(0x241811).convertSRGBToLinear(),
      transparent: true, opacity: 1.0, depthWrite: false,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2
    });
    var fissureMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0x120C08).convertSRGBToLinear(), transparent: true, opacity: 0.95,
      depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3
    });
    var shaleMat = new THREE.MeshStandardMaterial({
      color: 0x33231C, roughness: 0.98, metalness: 0.03, flatShading: true
    });
    var shaleHot = new THREE.MeshStandardMaterial({
      color: 0x6A452F, roughness: 0.93, metalness: 0.03, flatShading: true
    });
    var screeMat = new THREE.MeshStandardMaterial({
      color: 0x402E24, roughness: 0.99, metalness: 0.02, flatShading: true
    });
    var slabGeo = new THREE.BoxGeometry(1, 1, 1);
    var wedgeGeo = new THREE.TetrahedronGeometry(0.5, 0);
    var screeGeo = new THREE.IcosahedronGeometry(0.5, 0);
    var stainGeo = new THREE.PlaneGeometry(2 * HAZARD_R * 0.98, 2 * HAZARD_R * 0.98);

    /* Place one piece inside the zone. `reach` is the piece's own worst-case
       horizontal half-extent once it is rotated and tilted, so a piece is only
       accepted where every corner of it still falls inside the radius. */
    function placeInZone(rnd, reach) {
      var limit = HAZARD_R - 0.04 - reach;
      if (limit <= 0) return null;         // too big to fit inside the radius
      // sqrt() spreads samples evenly over the disc instead of crowding the
      // middle; the 0.08 floor keeps the centre from going bare.
      var r = limit * Math.sqrt(0.08 + 0.92 * rnd());
      var a = rnd() * Math.PI * 2;
      return { x: Math.cos(a) * r, z: Math.sin(a) * r };
    }

    function buildHazardTerrain(cx, cz, seed) {
      var rnd = mulberry(seed);

      // Scorching on the pan itself, so the zone still reads from straight above.
      var stain = new THREE.Mesh(stainGeo, stainMat);
      stain.rotation.x = -Math.PI / 2;
      stain.rotation.z = rnd() * Math.PI;
      stain.position.set(cx, 0.008, cz);
      scene.add(stain);

      /* Fissures. A dark bed just off the floor with a low lip down each side,
         so the crack has an edge for the sun to cut a shadow into rather than
         being a black sticker. */
      for (var f = 0; f < 3; f++) {
        var flen = 0.20 + rnd() * 0.30;
        var fwid = 0.040 + rnd() * 0.038;
        var fang = rnd() * Math.PI;
        var spot = placeInZone(rnd, flen / 2 + 0.05);
        if (!spot) continue;

        var bed = new THREE.Mesh(slabGeo, fissureMat);
        bed.scale.set(flen, 0.010, fwid);
        bed.position.set(cx + spot.x, 0.010, cz + spot.z);
        bed.rotation.y = fang;
        scene.add(bed);

        /* eslint-disable no-loop-func */
        [-1, 1].forEach(function (side) {
          var lip = new THREE.Mesh(slabGeo, side > 0 ? shaleHot : shaleMat);
          lip.scale.set(flen * (0.8 + rnd() * 0.2), 0.034 + rnd() * 0.030, 0.030 + rnd() * 0.020);
          lip.position.set(
            cx + spot.x + Math.sin(fang) * side * (fwid / 2 + 0.018),
            lip.scale.y / 2,
            cz + spot.z + Math.cos(fang) * side * (fwid / 2 + 0.018)
          );
          lip.rotation.y = fang;
          lip.rotation.x = side * (0.06 + rnd() * 0.10);
          lip.castShadow = true;
          lip.receiveShadow = true;
          scene.add(lip);
        });
        /* eslint-enable no-loop-func */
      }

      // Fractured slabs, heaved out of the pan and tipped over each other.
      for (var s = 0; s < 11; s++) {
        var sx = 0.09 + rnd() * 0.15, sz = 0.07 + rnd() * 0.13;
        var sy = 0.022 + rnd() * 0.048;
        var tilt = 0.10 + rnd() * 0.45;
        var p = placeInZone(rnd, 0.5 * Math.hypot(sx, sz) + 0.5 * sy);
        if (!p) continue;
        var slab = new THREE.Mesh(slabGeo, rnd() > 0.45 ? shaleMat : shaleHot);
        slab.scale.set(sx, sy, sz);
        slab.position.set(cx + p.x, sy * 0.5 + rnd() * 0.012, cz + p.z);
        slab.rotation.y = rnd() * Math.PI;
        slab.rotation.z = (rnd() > 0.5 ? 1 : -1) * tilt;
        slab.rotation.x = (rnd() - 0.5) * tilt;
        slab.castShadow = true;
        slab.receiveShadow = true;
        scene.add(slab);
      }

      /* Upthrust wedges: the sharp ones. Tall enough to catch the sun on one
         face and throw a hard shadow off the other, low enough that the rover
         plainly could drive over them. */
      for (var w = 0; w < 4; w++) {
        var ws = 0.10 + rnd() * 0.09;
        var wp = placeInZone(rnd, ws * 0.75);
        if (!wp) continue;
        var wedge = new THREE.Mesh(wedgeGeo, rnd() > 0.5 ? shaleHot : shaleMat);
        wedge.scale.set(ws, ws * (1.3 + rnd() * 0.7), ws);
        wedge.position.set(cx + wp.x, ws * 0.42, cz + wp.z);
        wedge.rotation.set((rnd() - 0.5) * 0.7, rnd() * Math.PI, (rnd() - 0.5) * 0.7);
        wedge.castShadow = true;
        wedge.receiveShadow = true;
        scene.add(wedge);
      }

      // Scree: the rubble that fills the gaps and says the rock shattered.
      for (var g = 0; g < 20; g++) {
        var gs = 0.016 + rnd() * 0.034;
        var gp = placeInZone(rnd, gs);
        if (!gp) continue;
        var bit = new THREE.Mesh(screeGeo, screeMat);
        bit.scale.set(gs, gs * (0.45 + rnd() * 0.5), gs * (0.7 + rnd() * 0.5));
        bit.position.set(cx + gp.x, gs * 0.3, cz + gp.z);
        bit.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
        bit.receiveShadow = true;   // too small to cast a readable shadow
        scene.add(bit);
      }
    }

    var hazardRims = [];
    hazards.forEach(function (h, idx) {
      var x = wx(h[0]), z = wz(h[1]);
      buildHazardTerrain(x, z, 6200 + idx * 137);

      /* The boundary ring. It sits on the radius the environment actually
         penalises, and it is drawn above the rubble so it stays readable from
         every camera. */
      var rim = new THREE.Mesh(
        new THREE.RingGeometry(HAZARD_R - 0.05, HAZARD_R + 0.02, 52),
        new THREE.MeshBasicMaterial({
          color: C.danger, transparent: true, opacity: 0.75, side: THREE.DoubleSide,
          depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4
        })
      );
      rim.rotation.x = -Math.PI / 2;
      rim.position.set(x, 0.030, z);
      scene.add(rim);
      hazardRims.push(rim);

      // Three short stakes on the boundary. Markers, not a fence.
      for (var p = 0; p < 3; p++) {
        var a = p / 3 * Math.PI * 2 + 0.6 + idx;
        var post = new THREE.Mesh(
          new THREE.CylinderGeometry(0.019, 0.026, 0.21, 6),
          new THREE.MeshStandardMaterial({ color: 0x8E2D1E, roughness: 0.6, metalness: 0.2 })
        );
        post.position.set(x + Math.cos(a) * HAZARD_R, 0.105, z + Math.sin(a) * HAZARD_R);
        post.castShadow = true;
        scene.add(post);
      }
    });

    /* Target: the pad, the star from the GIF, and a ring at exactly the goal
       tolerance that ends the episode, so the terminal condition is a thing
       you can see. */
    var goalGroup = new THREE.Group();
    goalGroup.position.set(wx(world.target[0]), 0, wz(world.target[1]));
    scene.add(goalGroup);

    var pad = new THREE.Mesh(
      new THREE.CylinderGeometry(GOAL_TOL, GOAL_TOL, 0.05, 40),
      new THREE.MeshStandardMaterial({ color: 0x6B5320, roughness: 0.75, metalness: 0.25 })
    );
    pad.position.y = 0.025;
    pad.receiveShadow = true;
    goalGroup.add(pad);

    var tolRing = new THREE.Mesh(
      new THREE.RingGeometry(GOAL_TOL - 0.04, GOAL_TOL + 0.01, 56),
      new THREE.MeshBasicMaterial({
        color: C.target, transparent: true, opacity: 0.85,
        side: THREE.DoubleSide, depthWrite: false
      })
    );
    tolRing.rotation.x = -Math.PI / 2;
    tolRing.position.y = 0.055;
    goalGroup.add(tolRing);

    var starShape = new THREE.Shape();
    for (var s5 = 0; s5 < 10; s5++) {
      var ang5 = (Math.PI / 5) * s5 - Math.PI / 2;
      var rad5 = s5 % 2 === 0 ? 0.30 : 0.126;   // the GIF star's outer/inner ratio
      var stx = Math.cos(ang5) * rad5, sty = Math.sin(ang5) * rad5;
      if (s5 === 0) starShape.moveTo(stx, sty); else starShape.lineTo(stx, sty);
    }
    starShape.closePath();
    var star = new THREE.Mesh(
      new THREE.ExtrudeGeometry(starShape, {
        depth: 0.08, bevelEnabled: true, bevelSize: 0.022,
        bevelThickness: 0.02, bevelSegments: 1
      }),
      new THREE.MeshStandardMaterial({
        color: C.target, roughness: 0.34, metalness: 0.75, emissive: 0x2A1E00
      })
    );
    star.rotation.x = -Math.PI / 2;
    star.position.y = 0.36;
    star.castShadow = true;
    goalGroup.add(star);
    var starPost = new THREE.Mesh(
      new THREE.CylinderGeometry(0.028, 0.038, 0.37, 10),
      new THREE.MeshStandardMaterial({ color: C.tgtEdge, roughness: 0.4, metalness: 0.8 })
    );
    starPost.position.y = 0.2;
    starPost.castShadow = true;
    goalGroup.add(starPost);

    /* The crate. A point in the state vector; here a braced wooden box, the
       crate the GIF's sprite sheet draws. */
    var crate = new THREE.Group();
    scene.add(crate);
    var crateBody = new THREE.Group();
    crate.add(crateBody);

    var woodMat = new THREE.MeshStandardMaterial({
      color: C.object, roughness: 0.72, metalness: 0.04
    });
    var frameMat = new THREE.MeshStandardMaterial({
      color: C.objEdge, roughness: 0.66, metalness: 0.08
    });
    var box = new THREE.Mesh(new THREE.BoxGeometry(CRATE, CRATE, CRATE), woodMat);
    box.position.y = CRATE / 2;
    box.castShadow = true; box.receiveShadow = true;
    crateBody.add(box);
    [[1, 1], [1, -1], [-1, 1], [-1, -1]].forEach(function (c) {
      var postm = new THREE.Mesh(new THREE.BoxGeometry(0.06, CRATE + 0.01, 0.06), frameMat);
      postm.position.set(c[0] * CRATE / 2, CRATE / 2, c[1] * CRATE / 2);
      postm.castShadow = true;
      crateBody.add(postm);
    });
    [0, Math.PI / 2].forEach(function (rot) {
      [1, -1].forEach(function (side) {
        [1, -1].forEach(function (diag) {
          var brace = new THREE.Mesh(new THREE.BoxGeometry(CRATE * 1.28, 0.042, 0.018), frameMat);
          brace.position.set(0, CRATE / 2, 0);
          brace.rotation.z = diag * Math.PI / 4;
          var holder = new THREE.Group();
          holder.add(brace);
          holder.rotation.y = rot;
          brace.position.z = side * (CRATE / 2 + 0.008);
          crateBody.add(holder);
        });
      });
    });

    // Contact shadow. The sun's shadow map handles the cast shadow; this is
    // the dark seam directly under the crate, and it tightens as weight settles.
    var crateBlob = new THREE.Mesh(
      new THREE.PlaneGeometry(1.15, 1.15),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.42, depthWrite: false
      })
    );
    crateBlob.rotation.x = -Math.PI / 2;
    crateBlob.position.y = 0.006;
    crate.add(crateBlob);

    // The rover
    var rover = new THREE.Group();
    scene.add(rover);
    var hull = new THREE.Group();
    rover.add(hull);

    var paintMat = new THREE.MeshStandardMaterial({
      color: C.robot, roughness: 0.42, metalness: 0.35
    });
    var darkMetal = new THREE.MeshStandardMaterial({
      color: 0x2A2724, roughness: 0.5, metalness: 0.8
    });
    var steelMat = new THREE.MeshStandardMaterial({
      color: 0x9AA0A8, roughness: 0.38, metalness: 0.62
    });

    var chassis = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.15, 0.36), paintMat);
    chassis.position.y = 0.19;
    chassis.castShadow = true;
    hull.add(chassis);

    var deck = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.10, 0.30),
      new THREE.MeshStandardMaterial({ color: 0x2733A8, roughness: 0.45, metalness: 0.45 }));
    deck.position.set(-0.03, 0.31, 0);
    deck.castShadow = true;
    hull.add(deck);

    var cabin = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.11, 0.22),
      new THREE.MeshStandardMaterial({ color: 0x16233F, roughness: 0.1, metalness: 0.35 }));
    cabin.position.set(0.05, 0.41, 0);
    cabin.castShadow = true;
    hull.add(cabin);

    var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.22, 8), darkMetal);
    mast.position.set(-0.13, 0.46, 0.09);
    hull.add(mast);
    var sensorHead = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.055, 0.05), darkMetal);
    sensorHead.position.set(-0.13, 0.58, 0.09);
    sensorHead.castShadow = true;
    hull.add(sensorHead);
    var sensorEye = new THREE.Mesh(new THREE.SphereGeometry(0.016, 8, 8),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.7, 2.6, 3.2) }));
    sensorEye.position.set(-0.095, 0.58, 0.09);
    hull.add(sensorEye);

    /* The blade, on a turret. It is mounted on two arms that slide, and the
       arms are what sell the contact: the blade rests against the crate and is
       pressed back into its mounts as the rover leans into the shove.

       The turret is the honest way to draw a push this environment actually
       performs. The transition has no contact normal in it — the crate moves
       along the action whenever it is within push_threshold of the post-move
       robot — so a rover with a fixed front blade would sometimes be shown
       shoving with its flank. Letting the blade track the crate keeps the
       contact where the two bodies really touch, and the red arrow above the
       rover still says which way the crate will travel. */
    var bladePivot = new THREE.Group();
    hull.add(bladePivot);
    var bladeRig = new THREE.Group();
    bladePivot.add(bladeRig);
    var blade = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.20, 0.50), steelMat);
    blade.position.set(0, 0.13, 0);
    blade.castShadow = true;
    bladeRig.add(blade);
    var bladeLip = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.045, 0.52),
      new THREE.MeshStandardMaterial({ color: 0xC8CCD2, roughness: 0.25, metalness: 0.75 }));
    bladeLip.position.set(0.006, 0.038, 0);
    bladeLip.castShadow = true;
    bladeRig.add(bladeLip);
    var arms = [];
    [0.20, -0.20].forEach(function (z) {
      var arm = new THREE.Mesh(new THREE.BoxGeometry(0.30, 0.035, 0.035), darkMetal);
      arm.position.set(-0.15, 0.18, z);
      arm.castShadow = true;
      bladeRig.add(arm);
      arms.push(arm);
    });

    var wheels = [];
    var wheelGeo = new THREE.CylinderGeometry(0.105, 0.105, 0.09, 18);
    var wheelMat = new THREE.MeshStandardMaterial({
      color: 0x191716, roughness: 0.9, metalness: 0.1
    });
    var hubMat = new THREE.MeshStandardMaterial({
      color: 0x8E9299, roughness: 0.35, metalness: 0.9
    });
    [[0.15, 0.20], [0.15, -0.20], [-0.16, 0.20], [-0.16, -0.20]].forEach(function (w) {
      var m = new THREE.Mesh(wheelGeo, wheelMat);
      m.position.set(w[0], WHEEL_Y, w[1]);
      m.rotation.x = Math.PI / 2;
      m.castShadow = true;
      m.userData.ox = w[0];
      m.userData.oz = w[1];
      rover.add(m);
      wheels.push(m);
      for (var t = 0; t < 9; t++) {
        var a = (t / 9) * Math.PI * 2;
        var tread = new THREE.Mesh(new THREE.BoxGeometry(0.032, 0.10, 0.022), wheelMat);
        tread.position.set(Math.cos(a) * 0.099, 0, Math.sin(a) * 0.099);
        tread.rotation.y = -a;
        m.add(tread);
      }
      m.add(new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.038, 0.095, 10), hubMat));
    });

    var roverBlob = new THREE.Mesh(
      new THREE.PlaneGeometry(1.25, 1.0),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.34, depthWrite: false
      })
    );
    roverBlob.rotation.x = -Math.PI / 2;
    roverBlob.position.y = 0.006;
    rover.add(roverBlob);

    /* The push radius, a real parameter of both variants, drawn on the ground
       around the rover. The crate moves if and only if it is inside this disc. */
    var pushRing = new THREE.Mesh(
      new THREE.RingGeometry(PUSH - 0.028, PUSH + 0.005, 72),
      new THREE.MeshBasicMaterial({
        color: C.radius, transparent: true, opacity: 0.55,
        side: THREE.DoubleSide, depthWrite: false
      })
    );
    pushRing.rotation.x = -Math.PI / 2;
    pushRing.position.y = 0.018;
    rover.add(pushRing);
    var pushFill = new THREE.Mesh(
      new THREE.CircleGeometry(PUSH, 64),
      new THREE.MeshBasicMaterial({
        color: C.radius, transparent: true, opacity: 0.07, depthWrite: false
      })
    );
    pushFill.rotation.x = -Math.PI / 2;
    pushFill.position.y = 0.012;
    rover.add(pushFill);

    // The continuous variant's body radius: the circle that collides with walls.
    var bodyDisc = new THREE.Mesh(
      new THREE.RingGeometry(ROBOT_RADIUS - 0.025, ROBOT_RADIUS, 40),
      new THREE.MeshBasicMaterial({
        color: C.radius, transparent: true, opacity: 0.8,
        side: THREE.DoubleSide, depthWrite: false
      })
    );
    bodyDisc.rotation.x = -Math.PI / 2;
    bodyDisc.position.y = 0.02;
    bodyDisc.visible = continuous;
    rover.add(bodyDisc);

    // The action arrow, the red arrow of the GIF, floating over the rover. It
    // points along the action, which is the direction the crate travels.
    var arrowGroup = new THREE.Group();
    arrowGroup.position.y = 0.60;
    rover.add(arrowGroup);
    var arrowMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.25, 0.10, 0.05) });
    var arrowShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.5, 8), arrowMat);
    arrowShaft.rotation.z = -Math.PI / 2;
    arrowShaft.position.x = 0.25;
    arrowGroup.add(arrowShaft);
    var arrowHead = new THREE.Mesh(new THREE.ConeGeometry(0.038, 0.11, 10), arrowMat);
    arrowHead.rotation.z = -Math.PI / 2;
    arrowHead.position.x = 0.58;
    arrowGroup.add(arrowHead);

    /* Belief. The crate's position is the only hidden part of the state, so
       this cloud is the belief about the crate and about nothing else: each
       particle is a whole recorded state, and what is plotted is its object
       slice, at the weight the run's own belief gave it. */
    var maxParticles = 1;
    payload.beliefs.forEach(function (b) {
      if (b.kind === "particles") maxParticles = Math.max(maxParticles, b.particles.length);
      if (b.kind === "gaussian" || b.kind === "gaussian_mixture") {
        maxParticles = Math.max(maxParticles, MAX_DRAWN_PARTICLES);
      }
    });
    maxParticles = Math.min(maxParticles, MAX_DRAWN_PARTICLES);
    var pPos = new Float32Array(maxParticles * 3);
    var pCol = new Float32Array(maxParticles * 3);
    var pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    pGeo.setAttribute("color", new THREE.BufferAttribute(pCol, 3));
    pGeo.setDrawRange(0, 0);
    var particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
      size: 0.105, map: partTex, vertexColors: true, transparent: true,
      opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false,
      sizeAttenuation: true
    }));
    scene.add(particles);

    /* The cloud alone is hard to size by eye from the raised view, so the same
       belief is also drawn flat on the ground as a two-sigma ring, sized from
       the spread of whatever was plotted. It is the honest shape of the
       uncertainty, not a glow. */
    var beliefRing = new THREE.Mesh(
      new THREE.RingGeometry(0.94, 1.0, 56),
      new THREE.MeshBasicMaterial({
        color: C.belief, transparent: true, opacity: 0.6,
        side: THREE.DoubleSide, depthWrite: false
      })
    );
    beliefRing.rotation.x = -Math.PI / 2;
    beliefRing.position.y = 0.024;
    scene.add(beliefRing);

    // The noisy reading of the crate the environment handed the agent.
    var obsMark = new THREE.Group();
    scene.add(obsMark);
    [0, Math.PI / 2].forEach(function (r) {
      var bar = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.011, 0.011),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(0.35, 1.55, 1.95) }));
      bar.rotation.y = r;
      obsMark.add(bar);
    });

    /* Dust: a squirt at the blade when the crate is shoved, a light plume from
       the wheels, and much more of both off broken ground. All of it is about
       weight — a shove should cost something — and none of it is position. */
    var dustPos = new Float32Array(DUST * 3);
    var dustVel = new Float32Array(DUST * 3);
    var dustLife = new Float32Array(DUST);
    for (var d0 = 0; d0 < DUST; d0++) dustPos[d0 * 3 + 1] = -6;
    var dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
    var dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({
      size: 0.15, map: partTex, color: 0xCBB68C, transparent: true, opacity: 0.4,
      depthWrite: false, sizeAttenuation: true
    }));
    scene.add(dust);
    var dustCursor = 0;
    var dustRnd = mulberry(77123);
    function spawnDust(x, z, vx, vz, n, up) {
      for (var i = 0; i < n; i++) {
        var k = dustCursor++ % DUST;
        dustPos[k * 3] = x + (dustRnd() - 0.5) * 0.22;
        dustPos[k * 3 + 1] = 0.04 + dustRnd() * 0.06;
        dustPos[k * 3 + 2] = z + (dustRnd() - 0.5) * 0.22;
        dustVel[k * 3] = vx * (0.4 + dustRnd() * 0.8) + (dustRnd() - 0.5) * 0.3;
        dustVel[k * 3 + 1] = up * (0.5 + dustRnd());
        dustVel[k * 3 + 2] = vz * (0.4 + dustRnd() * 0.8) + (dustRnd() - 0.5) * 0.3;
        dustLife[k] = 1;
      }
    }

    /* Drag marks. The crate scuffs the sand it is dragged over and the rover
       leaves wheel tracks: thin dark ribbons that accumulate with the
       playback, which is the cheapest honest record of where weight has been. */
    function makeRibbon(color, opacity) {
      var pos = new Float32Array(MARK_MAX * 3);
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setDrawRange(0, 0);
      var lineMesh = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: color, transparent: true, opacity: opacity
      }));
      scene.add(lineMesh);
      return { pos: pos, geo: geo, n: 0 };
    }
    var crateMark = makeRibbon(0x4A3A22, 0.8);
    var trackL = makeRibbon(0x51422C, 0.45);
    var trackR = makeRibbon(0x51422C, 0.45);
    function ribbonPush(rb, x, y, z) {
      if (rb.n >= MARK_MAX) return;
      rb.pos[rb.n * 3] = x; rb.pos[rb.n * 3 + 1] = y; rb.pos[rb.n * 3 + 2] = z;
      rb.n++;
      rb.geo.attributes.position.needsUpdate = true;
      rb.geo.setDrawRange(0, rb.n);
    }
    function ribbonReset(rb) { rb.n = 0; rb.geo.setDrawRange(0, 0); }

    core.linearize();
    scaleRadiance(scene, RADIANCE);
    core.composite.uniforms.bloomStrength.value = 0.30;
    core.composite.uniforms.grain.value = 0.013;
    core.composite.uniforms.aberration.value = 0.0014;

    /* ---------------------------------------------------------- the episode */

    var ACTION_LABELS = world.action_vectors || {};

    /** The direction a step's action pushes in, as a unit vector. */
    function actionDirection(action) {
      if (action === null || action === undefined) return null;
      var vector = null;
      if (typeof action === "string") {
        vector = ACTION_LABELS[action] || null;
      } else if (Array.isArray(action) && action.length >= 2) {
        vector = action;
      }
      if (!vector) return null;
      var n = Math.hypot(vector[0], vector[1]);
      if (n < 1e-12) return null;
      return [vector[0] / n, vector[1] / n, n];
    }

    function actionLabel(action) {
      if (action === null || action === undefined) return "—";
      if (typeof action === "string") return action;
      if (Array.isArray(action) && action.length >= 2) {
        return "(" + Number(action[0]).toFixed(2) + ", " + Number(action[1]).toFixed(2) + ")";
      }
      return String(action);
    }

    /* Per-step facts read off the trace once: which way the action pointed, and
       whether the crate actually moved. "Pushed" is not recorded anywhere and
       is not guessed either — it is exactly whether the recorded object
       position changed between this state and the next. */
    var stepFacts = [];
    for (var si = 0; si < payload.states.length; si++) {
      var here = payload.states[si];
      var there = payload.states[Math.min(si + 1, payload.states.length - 1)];
      var envelope = trace.steps[si] || {};
      stepFacts.push({
        dir: actionDirection(envelope.action),
        label: actionLabel(envelope.action),
        moved: Math.hypot(there[2] - here[2], there[3] - here[3])
      });
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

    /* How deep into a dangerous area a point is: 0 outside, 1 well inside. It
       ramps across the rim rather than switching on, so the rover starts to
       labour as it climbs in rather than lurching the instant it crosses. */
    function zoneDepth(px, py) {
      var worst = 0;
      for (var i = 0; i < hazards.length; i++) {
        var d = Math.hypot(px - hazards[i][0], py - hazards[i][1]);
        var v = clamp((HAZARD_R + 0.10 - d) / 0.42, 0, 1);
        if (v > worst) worst = v;
      }
      return smooth(worst);
    }

    /* Height of the rubble under a point. Deterministic in world position, so
       a wheel climbing the same slab twice climbs the same slab, and scrubbing
       backwards retraces the same ride rather than reshuffling the ground. */
    function rubbleHeight(px, py) {
      var depth = zoneDepth(px, py);
      if (depth <= 0) return 0;
      var n = 0.5 + 0.5 * Math.sin(13.7 * px + 2.1) * Math.cos(11.3 * py - 0.7)
        + 0.28 * Math.sin(23.1 * px - 4.2) * Math.sin(19.7 * py + 1.3);
      return depth * 0.058 * clamp(n, 0, 1.35);
    }

    /* One step, unpacked into something continuous. The rover eases across the
       whole step; the crate only moves while the blade is on it, which is a
       later and shorter window, so a shove reads as a shove. */
    function sampleAt(t) {
      var S = payload.states;
      var i0 = Math.floor(clamp(t, 0, S.length - 1));
      var i1 = Math.min(i0 + 1, S.length - 1);
      var f = clamp(t - i0, 0, 1);
      var a = S[i0], b = S[i1];
      var fact = stepFacts[i0];
      var pushed = fact.moved > 1e-9;

      var er = smooth(clamp(f * 1.12, 0, 1));                  // rover motion
      var contactStart = 0.30, contactEnd = 0.86;
      var ec = smooth(clamp((f - contactStart) / (contactEnd - contactStart), 0, 1));
      // A shove overshoots a hair and settles back: the crate has mass.
      var over = pushed
        ? Math.sin(clamp((f - contactStart) / (contactEnd - contactStart), 0, 1) * Math.PI) * 0.055
        : 0;

      /* Rough ground makes the crossing uneven, not slow. The window is
         sin(pi f): exactly zero at f = 0 and f = 1, so the wobble can only
         redistribute progress within the step and can never move where the
         step starts or ends. Phases are keyed to the step index, so the same
         step always lurches the same way. */
      var win = Math.sin(Math.PI * f);
      var wob = win * (0.62 * Math.sin(f * 6.9 + i0 * 2.399)
        + 0.38 * Math.sin(f * 12.7 + i0 * 5.137));

      var roughR = zoneDepth(lerp(a[0], b[0], er), lerp(a[1], b[1], er));
      var erS = clamp(er + roughR * 0.095 * wob, 0, 1);
      var roughO = zoneDepth(lerp(a[2], b[2], ec), lerp(a[3], b[3], ec));
      var ecS = clamp(ec + over + roughO * 0.085 * wob, 0, 1);

      return {
        index: i0, f: f, fact: fact, pushed: pushed,
        rough: roughR, crateRough: roughO, win: win,
        rx: lerp(a[0], b[0], erS),
        ry: lerp(a[1], b[1], erS),
        ox: lerp(a[2], b[2], ecS),
        oy: lerp(a[3], b[3], ecS),
        pushing: pushed && f > contactStart * 0.6 && f < 0.98,
        lean: pushed ? Math.sin(clamp((f - 0.18) / 0.7, 0, 1) * Math.PI) : 0,
        moving: f < 0.97
      };
    }

    /* ----------------------------------------------------------- the belief */

    /* A Gaussian carries no particles, so it is sampled for display only —
       with a fixed seed, and from the run's own mean and covariance, so what
       is drawn is that belief and nothing else. */
    var gaussianRnd = mulberry(20260920);
    var gaussianDraws = [];
    for (var gi = 0; gi < MAX_DRAWN_PARTICLES; gi++) {
      var u1 = Math.max(1e-6, gaussianRnd()), u2 = gaussianRnd();
      var mag = Math.sqrt(-2 * Math.log(u1));
      gaussianDraws.push([mag * Math.cos(2 * Math.PI * u2), mag * Math.sin(2 * Math.PI * u2)]);
    }

    // Weighted mean and spread of what was plotted, for the two-sigma ring.
    var spread = { x: 0, y: 0, sigma: 0, count: 0 };

    /* The cloud floats at crate height rather than on the floor. A belief that
       has locked onto the crate sits exactly where the crate is, and at ground
       level the crate's own box swallows it — the one moment the belief is
       most worth seeing is the one where it would disappear. Only the height
       is chosen; the two coordinates that mean anything are the particle's. */
    var CLOUD_Y = CRATE + 0.06;

    function writeParticle(slot, x, y, relativeWeight) {
      pPos[slot * 3] = wx(x);
      pPos[slot * 3 + 1] = CLOUD_Y + (slot % 5) * 0.022;
      pPos[slot * 3 + 2] = wz(y);
      // Cyan where the mass is, deep blue in the tail. Above 1.0 so the heavy
      // end blooms, and scaled with everything else that emits light here.
      var hot = 1.85 * RADIANCE;
      pCol[slot * 3] = lerp(0.04, 0.30, relativeWeight) * hot;
      pCol[slot * 3 + 1] = lerp(0.48, 0.92, relativeWeight) * hot;
      pCol[slot * 3 + 2] = lerp(0.72, 1.00, relativeWeight) * hot;
    }

    function commit(count) {
      pGeo.attributes.position.needsUpdate = true;
      pGeo.attributes.color.needsUpdate = true;
      pGeo.setDrawRange(0, count);
      spread.count = count;
    }

    /** Mean and standard deviation of the drawn cloud, at its own weights. */
    function measure(points, weights) {
      var mx = 0, my = 0, mass = 0;
      for (var i = 0; i < points.length; i++) {
        var w = weights ? weights[i] : 1;
        mx += points[i][0] * w; my += points[i][1] * w; mass += w;
      }
      if (mass <= 0) return null;
      mx /= mass; my /= mass;
      var variance = 0;
      for (var j = 0; j < points.length; j++) {
        var wj = weights ? weights[j] : 1;
        variance += wj * ((points[j][0] - mx) * (points[j][0] - mx)
          + (points[j][1] - my) * (points[j][1] - my));
      }
      // Trace of the covariance halved: the radius of a circle with the same
      // area as the cloud's own ellipse, which is what a single ring can say.
      return { x: mx, y: my, sigma: Math.sqrt(Math.max(variance / mass, 0) / 2) };
    }

    /* The belief is drawn from the kind core serialised it as, never from the
       belief class that produced it. Every particle belief in the package —
       weighted, unweighted, incremental, vectorized — arrives as "particles". */
    function drawParticles(belief) {
      var count = Math.min(belief.particles.length, maxParticles);
      if (count && (!Array.isArray(belief.particles[0]) || belief.particles[0].length < 4)) {
        // A particle that is not a Push state. This scene plots object
        // positions, so it says so rather than plotting something else.
        pGeo.setDrawRange(0, 0);
        return belief.num_particles + " particles this scene cannot place";
      }
      var maxWeight = 0;
      for (var w = 0; w < belief.weights.length; w++) {
        if (belief.weights[w] > maxWeight) maxWeight = belief.weights[w];
      }
      var points = [], mass = [];
      for (var p = 0; p < count; p++) {
        // An unweighted belief is uniform by construction, so shading it by
        // weight would imply structure the belief does not have.
        var rel = belief.weighted === false
          ? 1
          : (maxWeight > 0 ? belief.weights[p] / maxWeight : 0);
        // Columns 2 and 3 of the state are the object: the only hidden part.
        writeParticle(p, belief.particles[p][2], belief.particles[p][3], rel);
        points.push([belief.particles[p][2], belief.particles[p][3]]);
        mass.push(belief.weights[p]);
      }
      commit(count);
      var stats = measure(points, belief.weighted === false ? null : mass);
      if (stats) { spread.x = stats.x; spread.y = stats.y; spread.sigma = stats.sigma; }
      var label = belief.num_particles
        + (belief.weighted === false ? " uniform particles" : " particles");
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " drawn)";
      }
      return label;
    }

    /** Lower-triangular Cholesky of a 2x2, so a drawn cloud has the real shape. */
    function cholesky2(covariance) {
      var l11 = Math.sqrt(Math.max(covariance[0][0], 1e-9));
      var l21 = covariance[1][0] / l11;
      var l22 = Math.sqrt(Math.max(covariance[1][1] - l21 * l21, 1e-9));
      return [l11, l21, l22];
    }

    /** The object block of a Push state's mean and covariance. */
    function objectBlock(mean, covariance) {
      return {
        mean: [mean[2], mean[3]],
        covariance: [
          [covariance[2][2], covariance[2][3]],
          [covariance[3][2], covariance[3][3]]
        ]
      };
    }

    function writeGaussian(mean, covariance, slot, howMany, weightScale, points) {
      var L = cholesky2(covariance);
      for (var q = 0; q < howMany; q++) {
        var z1 = gaussianDraws[q][0], z2 = gaussianDraws[q][1];
        var x = mean[0] + L[0] * z1;
        var y = mean[1] + L[1] * z1 + L[2] * z2;
        writeParticle(slot + q, x, y, Math.exp(-(z1 * z1 + z2 * z2) * 0.5) * weightScale);
        points.push([x, y]);
      }
      return howMany;
    }

    function drawGaussian(belief) {
      if (belief.mean.length < 4) { pGeo.setDrawRange(0, 0); return "Gaussian, not a Push state"; }
      var block = objectBlock(belief.mean, belief.covariance);
      var points = [];
      var count = Math.min(MAX_DRAWN_PARTICLES, maxParticles);
      commit(writeGaussian(block.mean, block.covariance, 0, count, 1, points));
      var stats = measure(points, null);
      if (stats) { spread.x = stats.x; spread.y = stats.y; spread.sigma = stats.sigma; }
      return "Gaussian";
    }

    /* A mixture is drawn as its components, each given a share of the points
       in proportion to its weight. Collapsing it to one cloud would place the
       belief's mass between its modes, where the belief says nothing is. */
    function drawGaussianMixture(belief) {
      var budget = Math.min(MAX_DRAWN_PARTICLES, maxParticles);
      var points = [];
      var slot = 0;
      for (var k = 0; k < belief.components.length && slot < budget; k++) {
        var component = belief.components[k];
        if (component.mean.length < 4) continue;
        var block = objectBlock(component.mean, component.covariance);
        var share = Math.max(1, Math.round(component.weight * budget));
        share = Math.min(share, budget - slot, gaussianDraws.length);
        slot += writeGaussian(
          block.mean, block.covariance, slot, share, component.weight, points
        );
      }
      commit(slot);
      var stats = measure(points, null);
      if (stats) { spread.x = stats.x; spread.y = stats.y; spread.sigma = stats.sigma; }
      return belief.components.length + "-component mixture";
    }

    function drawBelief(index) {
      var belief = payload.beliefs[index];
      spread.sigma = 0;
      if (!belief) { pGeo.setDrawRange(0, 0); spread.count = 0; return "—"; }

      if (belief.kind === "particles") return drawParticles(belief);
      if (belief.kind === "gaussian") return drawGaussian(belief);
      if (belief.kind === "gaussian_mixture") return drawGaussianMixture(belief);
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner; it
        // is not one episode's belief, and merging its members would show a
        // cloud that was never anyone's belief.
        pGeo.setDrawRange(0, 0); spread.count = 0;
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }
      // A belief class core has no payload for yet. Named, so the gap is
      // diagnosable, and drawn as nothing, so it is not invented.
      pGeo.setDrawRange(0, 0); spread.count = 0;
      return "not recorded (" + (belief.belief_class || belief.kind) + ")";
    }

    /* ---------------------------------------------------------- the playback */

    var heading = 0;
    var wheelSpin = 0;
    var lastMarkT = -1;

    function update(t, dt, elapsed, playing) {
      var sm = sampleAt(t);
      var rx = wx(sm.rx), rz = wz(sm.ry);
      var ox = wx(sm.ox), oz = wz(sm.oy);

      // Heading: face the way the action points, turning rather than snapping.
      var dir = sm.fact.dir;
      if (dir) {
        var target = Math.atan2(-dir[1], dir[0]);
        var diff = ((target - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
        heading += diff * clamp(dt * 6.5, 0, 1);
      }
      rover.position.set(rx, 0, rz);
      rover.rotation.y = -heading;

      var gap = Math.hypot(sm.rx - sm.ox, sm.ry - sm.oy);
      var inRange = gap < PUSH;

      /* The blade slides out to meet the crate and is pressed back in as the
         rover leans on it. It never interpenetrates, and the compression is
         the visual weight of the shove. */
      var wantReach = clamp(gap - CRATE / 2 - 0.02, 0.15, 0.66);
      bladeRig.position.x = lerp(bladeRig.position.x, wantReach, clamp(dt * 10, 0, 1));
      var toCrate = Math.atan2(oz - rz, ox - rx);
      var wantPhi = gap < PUSH * 1.35 ? heading - toCrate : 0;
      var dphi = ((wantPhi - bladePivot.rotation.y + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
      bladePivot.rotation.y += dphi * clamp(dt * 7, 0, 1);
      bladeRig.position.x -= sm.pushing ? 0.024 : 0;
      for (var ai = 0; ai < arms.length; ai++) arms[ai].scale.x = clamp(wantReach / 0.5, 0.6, 1.5);

      /* Weight. The hull pitches back into the shove and rolls into a turn,
         and on rubble it also rides the ground: each wheel sits on its own
         bump, the body pitches and rolls across the wheelbase, and the
         suspension takes up the difference. Attitude and ride height only. */
      var wheelH = [];
      var sumH = 0;
      for (var wi = 0; wi < wheels.length; wi++) {
        var lo = wheels[wi].userData;
        // rover local +x is (cos h, sin h) in world; local +z is (-sin h, cos h)
        var wxw = rx + Math.cos(heading) * lo.ox - Math.sin(heading) * lo.oz;
        var wzw = rz + Math.sin(heading) * lo.ox + Math.cos(heading) * lo.oz;
        var hgt = rubbleHeight(ex(wxw), ey(wzw));
        wheelH.push(hgt);
        sumH += hgt;
        wheels[wi].position.y = WHEEL_Y + hgt;
      }
      var lift = sumH / wheelH.length;
      // front pair minus rear pair over the wheelbase, and left minus right.
      var terrainPitch = ((wheelH[0] + wheelH[1]) - (wheelH[2] + wheelH[3])) / 2 / 0.31;
      var terrainRoll = ((wheelH[0] + wheelH[2]) - (wheelH[1] + wheelH[3])) / 2 / 0.40;

      hull.rotation.z = lerp(
        hull.rotation.z, -sm.lean * 0.085 - terrainPitch * 0.85, clamp(dt * 12, 0, 1)
      );
      hull.rotation.x = lerp(hull.rotation.x, terrainRoll * 0.9, clamp(dt * 12, 0, 1));
      // Suspension: it does not take the full bump, and it rings afterwards.
      var judder = reduceMotion ? 0 : sm.rough * 0.010 * Math.sin(elapsed * 31.0 + sm.f * 9.0);
      hull.position.y = (reduceMotion ? 0 : Math.sin(elapsed * 9) * 0.004 * (sm.moving ? 1 : 0.15))
        + lift * 0.72 + judder;

      var speed = playing ? 1 : 0;
      /* Wheels lose grip: they turn faster than the ground is passing, and
         they judder instead of rolling smoothly. Spin is not position, so this
         says "slipping" without claiming the rover went anywhere it did not. */
      var slip = 1 + sm.rough * 2.6;
      wheelSpin += dt * speed * (sm.moving ? 5.4 : 0) * (1 + sm.lean * 0.6) * slip;
      for (var w2 = 0; w2 < wheels.length; w2++) {
        var stutter = reduceMotion ? 0
          : sm.rough * (0.30 * Math.sin(elapsed * 27 + w2 * 1.9) + 0.16 * Math.sin(elapsed * 44 + w2));
        wheels[w2].rotation.y = wheelSpin + stutter;
      }

      // Crate: rocks forward on the shove and settles back. Dragged over
      // rubble it catches and tips instead of sliding, windowed like the
      // rover's wobble so it is square again by the end of the step.
      crate.position.set(ox, 0, oz);
      var rock = sm.pushing ? Math.sin(clamp((sm.f - 0.3) / 0.6, 0, 1) * Math.PI) * 0.10 : 0;
      var cr = sm.crateRough;
      var snag = cr * (0.055 * Math.sin(sm.f * 15.3 + sm.index * 1.7) + 0.035 * Math.sin(sm.f * 27.1));
      crateBody.rotation.z = lerp(
        crateBody.rotation.z, -rock * Math.cos(heading) + snag * 1.6, clamp(dt * 11, 0, 1)
      );
      crateBody.rotation.x = lerp(
        crateBody.rotation.x, rock * Math.sin(heading) - snag * 1.2, clamp(dt * 11, 0, 1)
      );
      crateBody.rotation.y = 0.18 + cr * sm.win * 0.16 * Math.sin(sm.f * 9.7 + sm.index);
      crateBody.position.y = cr * (0.012 + 0.010 * Math.sin(sm.f * 19.0 + sm.index * 2.3));

      /* Contact shadows. Both blobs tighten and darken as the two bodies
         close, which says "touching" more strongly than any highlight. */
      var close = clamp(1 - (gap - 0.45) / 0.9, 0, 1);
      var cs = lerp(1.15, 0.82, close);
      crateBlob.scale.set(cs, cs, 1);
      crateBlob.material.opacity = lerp(0.34, 0.55, close);
      roverBlob.material.opacity = lerp(0.28, 0.44, close);

      pushRing.material.opacity = inRange ? 0.85 : 0.4;
      pushFill.material.opacity = inRange ? 0.13 : 0.05;

      // The action arrow, along the action and scaled by its magnitude.
      arrowGroup.visible = !!dir && sm.f < 0.9;
      if (arrowGroup.visible) {
        var alen = continuous ? clamp(dir[2], 0.3, 2) * 0.34 : 0.42;
        arrowShaft.scale.y = alen / 0.5;
        arrowShaft.position.x = alen / 2;
        arrowHead.position.x = alen + 0.055;
      }

      if (!reduceMotion && speed > 0) {
        if (sm.pushing) {
          spawnDust(lerp(rx, ox, 0.72), lerp(rz, oz, 0.72),
            Math.cos(heading) * 0.5, Math.sin(heading) * 0.5, 2, 0.5);
        }
        if (sm.moving) {
          spawnDust(rx - Math.cos(heading) * 0.22, rz - Math.sin(heading) * 0.22, 0, 0, 1, 0.3);
        }
        // Rubble: grit flung backwards off the slipping wheels, and the crate
        // scraping. Much more of it than the pan throws up.
        if (sm.rough > 0.03) {
          var n = 1 + Math.round(sm.rough * 3);
          for (var gd = 0; gd < n; gd++) {
            var side = gd % 2 ? 0.2 : -0.2;
            spawnDust(
              rx - Math.cos(heading) * 0.16 - Math.sin(heading) * side,
              rz - Math.sin(heading) * 0.16 + Math.cos(heading) * side,
              -Math.cos(heading) * (1.1 + sm.rough), -Math.sin(heading) * (1.1 + sm.rough),
              1, 0.9 + sm.rough * 0.8
            );
          }
        }
        if (sm.crateRough > 0.05 && sm.pushing) {
          spawnDust(ox, oz, -Math.cos(heading) * 0.5, -Math.sin(heading) * 0.5, 1, 0.55);
        }
      }
      for (var di = 0; di < DUST; di++) {
        if (dustLife[di] <= 0) continue;
        dustLife[di] -= dt * 1.25;
        dustPos[di * 3] += dustVel[di * 3] * dt;
        dustPos[di * 3 + 1] += dustVel[di * 3 + 1] * dt;
        dustPos[di * 3 + 2] += dustVel[di * 3 + 2] * dt;
        dustVel[di * 3 + 1] -= dt * 0.55;
        if (dustPos[di * 3 + 1] < 0.02 || dustLife[di] <= 0) {
          dustLife[di] = 0; dustPos[di * 3 + 1] = -6;
        }
      }
      dustGeo.attributes.position.needsUpdate = true;

      // Drag marks, sampled as the playback advances and dropped on a rewind.
      if (t < lastMarkT) {
        ribbonReset(crateMark); ribbonReset(trackL); ribbonReset(trackR);
        lastMarkT = -1;
      }
      if (t > lastMarkT + 0.02) {
        lastMarkT = t;
        ribbonPush(crateMark, ox, 0.009, oz);
        var px = Math.sin(heading) * 0.2, pz = Math.cos(heading) * 0.2;
        ribbonPush(trackL, rx + px, 0.008, rz + pz);
        ribbonPush(trackR, rx - px, 0.008, rz - pz);
      }

      var beliefLabel = drawBelief(sm.index);
      particles.visible = spread.count > 0;
      beliefRing.visible = spread.count > 0 && spread.sigma > 0;
      if (beliefRing.visible) {
        var twoSig = Math.max(0.12, spread.sigma * 2);
        beliefRing.position.set(wx(spread.x), 0.024, wz(spread.y));
        beliefRing.scale.set(twoSig, twoSig, 1);
      }

      // The noisy reading of the crate this step, when the episode recorded one.
      var observation = payload.observations[sm.index];
      obsMark.visible = Array.isArray(observation) && observation.length >= 4;
      if (obsMark.visible) {
        // Above the belief cloud, so the true crate, what the agent believes
        // and what it last read stack up legibly instead of overlapping.
        obsMark.position.set(wx(observation[2]), CLOUD_Y + 0.22, wz(observation[3]));
        obsMark.rotation.y = elapsed * 0.9;
      }

      /* The sun's shadow camera follows the action, so 2048 pixels of shadow
         map land where the contact is rather than spread over the whole yard. */
      sun.target.position.set(lerp(rx, ox, 0.5), 0, lerp(rz, oz, 0.5));
      sun.position.copy(sun.target.position).add(SUN_DIR);
      sun.target.updateMatrixWorld();

      if (!reduceMotion) {
        star.rotation.z = elapsed * 0.45;
        var pulse = 0.55 + Math.sin(elapsed * 2.2) * 0.2;
        for (var hr = 0; hr < hazardRims.length; hr++) hazardRims[hr].material.opacity = pulse;
      }

      var step = trace.steps[sm.index] || {};
      return {
        /* What the chase camera follows is the midpoint of the two bodies,
           not the rover. This environment is two bodies whose relationship is
           the task, and a camera locked to the rover alone loses the crate
           exactly when the shove happens. The core rig's chase mode takes one
           point and a heading and then looks 1.6 ahead of it, so the point
           handed over is backed off by that much and the shot lands on the
           contact. It is the contact camera expressed in the terms the shared
           rig already has, rather than a fifth camera mode the page has no
           button for. */
        follow: {
          x: lerp(rx, ox, 0.5) - Math.cos(heading) * 1.6,
          z: lerp(rz, oz, 0.5) - Math.sin(heading) * 1.6,
          heading: heading
        },
        step: sm.index,
        action: sm.fact.label,
        /* The position readout is both bodies and the distance between them,
           because in this environment one position on its own says nothing:
           the rover changes the crate only from inside push_threshold, so the
           gap is the number that decides whether a step did anything. */
        pos: "robot " + sm.rx.toFixed(2) + ", " + sm.ry.toFixed(2)
          + "   crate " + sm.ox.toFixed(2) + ", " + sm.oy.toFixed(2)
          + "   gap " + gap.toFixed(2) + (inRange ? " (in push radius)" : " (out of range)"),
        x: sm.rx,
        y: sm.ry,
        reward: step.reward,
        ret: running[sm.index],
        belief: beliefLabel
      };
    }

    return {
      steps: payload.states.length,

      /* Framing scales with the world, because a trace decides how big the
         board is. The constant term is the margin for the props, which do not
         scale with the grid — a kerb and a target post are the same size on
         any board. The overhead distance is not a taste: the core camera is a
         36 mm lens, so it has to stand off by (extent + margin) / (2 tan 18°)
         or the far rows fall outside the frame. */
      camera: {
        board: [0, extent * 1.11 + 0.9, extent * 1.06 + 0.95],
        top: [0, (extent + 2.4) / (2 * Math.tan(Math.PI * 18 / 180)), 0.01]
      },

      update: update
    };
  }

  V.scenes["push.v1"] = {
    build: build,
    // A sunlit yard stops down hard, and then opens back up by the factor the
    // whole scene's radiance was scaled by so the shared bright pass cuts in
    // the right place. See RADIANCE above; this is not a knob to remove.
    exposure: EXPOSURE
  };
})(window);
