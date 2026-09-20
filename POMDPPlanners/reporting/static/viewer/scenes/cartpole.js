/* SPDX-License-Identifier: MIT
 *
 * CartPole scene module.
 *
 * Builds the rig from a trace's `payload.world` block and moves it from the
 * trace's recorded states and beliefs. The physics is deliberately not here:
 * the episode already happened, the states are in the file, and a viewer that
 * re-integrated the plant would be drawing its own rollout next to the real
 * episode's rewards. So this module interpolates between recorded states and
 * integrates nothing.
 *
 * The belief is the run's own. CartPole's beliefs are usually Gaussian (EKF or
 * UKF over the four-dimensional state), so the fan of ghost poles is sampled
 * from the recorded mean and covariance — from the [x, theta] marginal, which
 * is exactly the 2x2 sub-block of the 4x4. A particle belief is drawn as its
 * particles instead, and a belief core could not serialise is named and left
 * undrawn rather than replaced by something plausible.
 *
 * The world is built with the pole's pivot at the origin. That is not
 * decoration: the shared camera rig's board mode looks at a fixed point 0.3
 * above the origin, so putting the pivot there is what makes the default view
 * low and side-on, level with the hinge. The pole angle is the quantity a
 * reader reads, and an angle is only legible from the side.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* The palette of cartpole_visualizer.py, so the viewer and the GIF are
     recognisably the same rig. */
  var C = {
    ink: 0x242B32,
    blue: 0x1B53A4,     // cart enamel, action arrow
    red: 0xBF2D20,      // pole, limit markers
    flame: 0xEF6747,    // the pole's lit edge
    tip: 0xE04E35,
    lamp: 0xFFE6C0
  };

  // Heights of the rig, in metres, measured from the pivot at y = 0.
  var PIVOT_Y = 0.0;
  var RAIL_Y = -0.145;          // top face of the rail beam
  var TABLE_Y = -0.29;
  var FLOOR_Y = -1.045;
  var RAIL_HALF = 3.2;
  var CART_H = 0.15;

  // Ghost poles drawn for a Gaussian or mixture belief. More than this is a
  // draw cost that adds no readable width to the fan.
  var MAX_GHOSTS = 64;

  /* The rig is lit like a workshop: three wall washers on the back wall plus
     one shadow-casting work light over the cart. Real lumens, so the camera
     stops down in the composite instead of the lights being turned down. */
  var WASHER_LUMENS = 3600;
  var WORK_LUMENS = 2100;

  /** Grain-and-speck albedo for the floor and the wall. */
  function surfaceCanvas(base, specks, seed) {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = base;
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 700; i++) {
      var r = 20 + rnd() * 150;
      g.fillStyle = "rgba(" + (110 + rnd() * 60 | 0) + "," + (108 + rnd() * 56 | 0) + "," +
        (102 + rnd() * 52 | 0) + ",0.055)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
    }
    for (var j = 0; j < specks; j++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(235,232,226,0.10)" : "rgba(10,10,10,0.16)";
      var px = rnd() * s, py = rnd() * s, w = 1 + rnd() * 3;
      g.fillRect(px, py, w, w);
    }
    return cv;
  }

  /** Lower-triangular Cholesky of a 2x2, so a sampled cloud has the real shape. */
  function cholesky2(a, b, c) {
    var l11 = Math.sqrt(Math.max(a, 1e-12));
    var l21 = b / l11;
    var l22 = Math.sqrt(Math.max(c - l21 * l21, 1e-12));
    return [l11, l21, l22];
  }

  /**
   * Build the CartPole rig from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind cartpole.v1.
   * @returns {Object} The scene module the player drives.
   */
  function build(core, trace) {
    var world = trace.payload.world;
    var payload = trace.payload;
    var scene = core.scene;
    var renderer = core.renderer;

    // `length` is half the pole, which is how CartPolePOMDP defines it. The
    // drawn pole is the whole thing.
    var poleLen = 2 * world.length;
    var xLimit = world.x_threshold;
    var thetaLimit = world.theta_threshold_radians;

    scene.background = new THREE.Color(0x0A0C0D).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x0C0E10, 0.026);
    scene.fog.color.convertSRGBToLinear();

    /* Sky-and-bounce fill. A workshop is not a void: the washers throw a lot
       of light at a pale wall and it comes back. */
    scene.add(new THREE.HemisphereLight(0x3D444C, 0x1A1713, 0.95));
    [[-6.0, 1.56, 3.2], [6.0, 1.56, 3.2]].forEach(function (rp) {
      var room = new THREE.PointLight(0xFFE9C9, 1, 16, 2);
      room.power = 1700;
      room.position.set(rp[0], rp[1], rp[2]);
      scene.add(room);
    });

    /* An interior to reflect. Without an environment map the chrome guide rods
       and the cart's trim have nothing to mirror, and PBR metal with nothing
       to reflect reads as flat grey plastic. */
    core.buildNightEnvironment([
      [0.0, "#2A2A28"],   // ceiling, catching the uplight
      [0.42, "#585550"],
      [0.62, "#3A3835"],  // wall
      [1.0, "#131313"]    // floor
    ]);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);

    // Polished concrete: low roughness, so the rig gets a soft reflection.
    var floorCv = surfaceCanvas("#3C3A37", 2200, 8821);
    var floorNormal = V.normalMapFrom(renderer, floorCv, 1.1);
    var floorAlbedo = new THREE.CanvasTexture(floorCv);
    floorAlbedo.encoding = THREE.sRGBEncoding;
    floorAlbedo.wrapS = floorAlbedo.wrapT = THREE.RepeatWrapping;
    floorAlbedo.repeat.set(3, 3);
    floorAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    floorNormal.wrapS = floorNormal.wrapT = THREE.RepeatWrapping;
    floorNormal.repeat.set(3, 3);

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(34, 34),
      new THREE.MeshStandardMaterial({
        map: floorAlbedo, normalMap: floorNormal,
        normalScale: new THREE.Vector2(0.4, 0.4),
        color: 0x6A6660, roughness: 0.34, metalness: 0.0, envMapIntensity: 0.55
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = FLOOR_Y;
    floor.receiveShadow = true;
    scene.add(floor);

    // Plaster wall. The three pools on it are real spot lights, not paint.
    var wallCv = surfaceCanvas("#67635C", 1400, 3313);
    var wallNormal = V.normalMapFrom(renderer, wallCv, 2.4);
    var wallAlbedo = new THREE.CanvasTexture(wallCv);
    wallAlbedo.encoding = THREE.sRGBEncoding;
    wallAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();

    var wall = new THREE.Mesh(
      new THREE.PlaneGeometry(26, 6.4),
      new THREE.MeshStandardMaterial({
        map: wallAlbedo, normalMap: wallNormal,
        normalScale: new THREE.Vector2(0.65, 0.65),
        color: 0x8C877E, roughness: 0.92, metalness: 0.0, envMapIntensity: 0.22
      })
    );
    wall.position.set(0, FLOOR_Y + 3.2, -3.0);
    wall.receiveShadow = true;
    scene.add(wall);

    // Skirting, set well clear of the wall plane: two coplanar surfaces a
    // centimetre apart is what stripes a 16-bit depth buffer.
    var skirt = new THREE.Mesh(
      new THREE.BoxGeometry(26, 0.16, 0.09),
      new THREE.MeshStandardMaterial({ color: 0x4D4A45, roughness: 0.7, metalness: 0.15 })
    );
    skirt.position.set(0, FLOOR_Y + 0.08, -2.94);
    skirt.receiveShadow = true;
    scene.add(skirt);

    // Ceiling: mostly out of frame, but it stops the light fixtures hanging in
    // a void when the orbit camera looks up.
    var ceiling = new THREE.Mesh(
      new THREE.PlaneGeometry(26, 14),
      new THREE.MeshStandardMaterial({ color: 0x1B1B1A, roughness: 0.95, metalness: 0.0 })
    );
    ceiling.rotation.x = Math.PI / 2;
    ceiling.position.set(0, FLOOR_Y + 3.9, 1.0);
    scene.add(ceiling);

    // Three wall washers, aimed at the wall rather than at the rig, so the rig
    // is lit by what bounces plus its own work light.
    [-3.5, 0, 3.5].forEach(function (lx) {
      var can = new THREE.Mesh(
        new THREE.CylinderGeometry(0.10, 0.12, 0.16, 14),
        new THREE.MeshStandardMaterial({ color: 0x24262A, roughness: 0.4, metalness: 0.8 })
      );
      can.position.set(lx, FLOOR_Y + 3.78, -2.35);
      scene.add(can);

      var lens = new THREE.Mesh(
        new THREE.CircleGeometry(0.095, 14),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(3.2, 2.9, 2.4), side: THREE.DoubleSide
        })
      );
      lens.rotation.x = Math.PI / 2;
      lens.position.set(lx, FLOOR_Y + 3.695, -2.35);
      scene.add(lens);

      var spot = new THREE.SpotLight(C.lamp, 1, 12, 0.58, 0.75, 2);
      spot.power = WASHER_LUMENS;
      spot.position.set(lx, FLOOR_Y + 3.7, -2.35);
      spot.target.position.set(lx, FLOOR_Y + 2.2, -2.99);
      scene.add(spot);
      scene.add(spot.target);
    });

    // Benches and pegboards at the edges of the frame: enough to say
    // "workshop", not enough to compete with the rig.
    var benchTop = new THREE.MeshStandardMaterial({
      color: 0x3E3B36, roughness: 0.55, metalness: 0.25
    });
    var benchBody = new THREE.MeshStandardMaterial({
      color: 0x2C2E31, roughness: 0.6, metalness: 0.45
    });
    var pegMat = new THREE.MeshStandardMaterial({
      color: 0x33322E, roughness: 0.85, metalness: 0.1
    });
    var toolMat = new THREE.MeshStandardMaterial({
      color: 0x8C8983, roughness: 0.35, metalness: 0.75
    });
    [-5.0, 5.0].forEach(function (bx, bi) {
      var cab = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.86, 0.72), benchBody);
      cab.position.set(bx, FLOOR_Y + 0.43, -2.45);
      cab.castShadow = true; cab.receiveShadow = true;
      scene.add(cab);
      var top = new THREE.Mesh(new THREE.BoxGeometry(1.62, 0.06, 0.80), benchTop);
      top.position.set(bx, FLOOR_Y + 0.89, -2.45);
      top.castShadow = true; top.receiveShadow = true;
      scene.add(top);

      var board = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.5, 0.05), pegMat);
      board.position.set(bx, FLOOR_Y + 1.95, -2.86);
      board.receiveShadow = true;
      scene.add(board);

      var trnd = mulberry(1000 + bi);
      for (var k = 0; k < 9; k++) {
        var tool = new THREE.Mesh(
          new THREE.BoxGeometry(0.035 + trnd() * 0.03, 0.16 + trnd() * 0.22, 0.03), toolMat
        );
        tool.position.set(bx - 0.6 + k * 0.15, FLOOR_Y + 2.25 - trnd() * 0.55, -2.81);
        tool.castShadow = true;
        scene.add(tool);
      }
      for (var j = 0; j < 3; j++) {
        var jar = new THREE.Mesh(
          new THREE.CylinderGeometry(0.055, 0.055, 0.13 + trnd() * 0.08, 12),
          new THREE.MeshStandardMaterial({ color: 0x55524B, roughness: 0.3, metalness: 0.6 })
        );
        jar.position.set(bx - 0.5 + j * 0.28, FLOOR_Y + 0.99, -2.3);
        jar.castShadow = true;
        scene.add(jar);
      }
    });

    /* ---------------------------------------------------------------- rig */
    var steel = new THREE.MeshStandardMaterial({
      color: 0x4A4F54, roughness: 0.45, metalness: 0.82
    });
    var darkSteel = new THREE.MeshStandardMaterial({
      color: 0x1F2427, roughness: 0.5, metalness: 0.7
    });
    var chrome = new THREE.MeshStandardMaterial({
      color: 0x9AA0A5, roughness: 0.28, metalness: 0.92, envMapIntensity: 0.6
    });
    var pale = new THREE.MeshStandardMaterial({
      color: 0xB4B0A6, roughness: 0.62, metalness: 0.15
    });

    var table = new THREE.Mesh(new THREE.BoxGeometry(7.0, 0.09, 0.62), steel);
    table.position.set(0, TABLE_Y, 0);
    table.castShadow = true; table.receiveShadow = true;
    scene.add(table);
    [-3.2, -1.05, 1.05, 3.2].forEach(function (lx) {
      [-0.22, 0.22].forEach(function (lz) {
        var leg = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.71, 0.07), darkSteel);
        leg.position.set(lx, FLOOR_Y + 0.355, lz);
        leg.castShadow = true; leg.receiveShadow = true;
        scene.add(leg);
      });
    });
    var brace = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.05, 0.05), darkSteel);
    brace.position.set(0, FLOOR_Y + 0.24, 0);
    brace.castShadow = true;
    scene.add(brace);

    // The rail: a machined beam with a pale ruler face, then two chrome guide
    // rods the cart rides. Drawn here, never baked into art.
    var beam = new THREE.Mesh(new THREE.BoxGeometry(2 * RAIL_HALF, 0.07, 0.30), darkSteel);
    beam.position.set(0, RAIL_Y - 0.035, 0);
    beam.castShadow = true; beam.receiveShadow = true;
    scene.add(beam);

    var rulerFace = new THREE.Mesh(new THREE.BoxGeometry(2 * RAIL_HALF, 0.052, 0.022), pale);
    rulerFace.position.set(0, RAIL_Y - 0.036, 0.152);
    rulerFace.receiveShadow = true;
    scene.add(rulerFace);

    var rodMat = new THREE.MeshStandardMaterial({
      color: 0x7E848A, roughness: 0.42, metalness: 0.85, envMapIntensity: 0.4
    });
    [-0.085, 0.085].forEach(function (rz) {
      var rod = new THREE.Mesh(
        new THREE.CylinderGeometry(0.013, 0.013, 2 * RAIL_HALF, 12), rodMat
      );
      rod.rotation.z = Math.PI / 2;
      rod.position.set(0, RAIL_Y + 0.013, rz);
      rod.castShadow = true;
      scene.add(rod);
    });

    // Ticks every 0.5 m, tall at the metres, the same scale the GIF rules.
    var tickMat = new THREE.MeshStandardMaterial({
      color: 0x2A2F34, roughness: 0.7, metalness: 0.1
    });
    for (var tk = -6; tk <= 6; tk++) {
      var xm = tk * 0.5;
      if (Math.abs(xm) > RAIL_HALF - 0.1) continue;
      var tall = tk % 2 === 0;
      var tick = new THREE.Mesh(
        new THREE.BoxGeometry(0.011, tall ? 0.055 : 0.03, 0.006), tickMat
      );
      tick.position.set(xm, RAIL_Y - (tall ? 0.034 : 0.026), 0.166);
      scene.add(tick);
    }

    // Limit posts at |x| = x_threshold: cross one and the episode ends.
    var limitMat = new THREE.MeshStandardMaterial({
      color: C.red, roughness: 0.4, metalness: 0.2, emissive: 0x3A0C07, emissiveIntensity: 1.0
    });
    [-xLimit, xLimit].forEach(function (lx) {
      var post = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.20, 0.025), limitMat);
      post.position.set(lx, RAIL_Y + 0.10, 0.10);
      post.castShadow = true;
      scene.add(post);
      var flag = new THREE.Mesh(
        new THREE.BoxGeometry(0.012, 0.10, 0.11),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(1.5, 0.22, 0.14) })
      );
      flag.position.set(lx, RAIL_Y + 0.185, 0.10);
      scene.add(flag);
    });

    [-RAIL_HALF, RAIL_HALF].forEach(function (lx) {
      var stop = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.17, 0.30), steel);
      stop.position.set(lx, RAIL_Y + 0.05, 0);
      stop.castShadow = true; stop.receiveShadow = true;
      scene.add(stop);
    });

    /* --------------------------------------------------------------- cart */
    var cart = new THREE.Group();
    scene.add(cart);

    var enamel = new THREE.MeshStandardMaterial({
      color: C.blue, roughness: 0.28, metalness: 0.35
    });
    var body = new THREE.Mesh(new THREE.BoxGeometry(0.44, CART_H, 0.26), enamel);
    body.position.y = RAIL_Y + 0.075;
    body.castShadow = true; body.receiveShadow = true;
    cart.add(body);

    var trim = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.018, 0.275), chrome);
    trim.position.y = RAIL_Y + 0.145;
    trim.castShadow = true;
    cart.add(trim);
    var trimLow = new THREE.Mesh(new THREE.BoxGeometry(0.455, 0.012, 0.268), chrome);
    trimLow.position.y = RAIL_Y + 0.012;
    cart.add(trimLow);

    var wheels = [];
    var wheelMat = new THREE.MeshStandardMaterial({
      color: 0x151719, roughness: 0.75, metalness: 0.3
    });
    [[-0.15, -0.085], [-0.15, 0.085], [0.15, -0.085], [0.15, 0.085]].forEach(function (w) {
      var m = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.036, 0.03, 14), wheelMat);
      m.rotation.x = Math.PI / 2;
      m.position.set(w[0], RAIL_Y + 0.013, w[1]);
      m.castShadow = true;
      cart.add(m);
      wheels.push(m);
      var hub = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.034, 8), chrome);
      hub.rotation.x = Math.PI / 2;
      hub.position.copy(m.position);
      cart.add(hub);
    });

    // The bearing the pole turns in. The pivot is its centre, at y = 0.
    var housing = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.052, 0.045, 18), chrome);
    housing.position.y = RAIL_Y + 0.168;
    housing.rotation.x = Math.PI / 2;
    housing.castShadow = true;
    cart.add(housing);
    var race = new THREE.Mesh(new THREE.TorusGeometry(0.032, 0.007, 8, 20), darkSteel);
    race.position.set(0, PIVOT_Y, 0.026);
    cart.add(race);

    /* --------------------------------------------------------------- pole
       A group pivoted at the bearing. `rotation.z = -theta` reproduces the
       sign convention of the GIF's pole_geometry(): positive radians lean
       towards +x. */
    var pole = new THREE.Group();
    pole.position.set(0, PIVOT_Y, 0);
    cart.add(pole);

    var shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.021, 0.024, poleLen, 16),
      new THREE.MeshStandardMaterial({
        color: C.red, roughness: 0.34, metalness: 0.2,
        emissive: 0x3A0B06, emissiveIntensity: 1.0
      })
    );
    shaft.position.y = poleLen / 2;
    shaft.castShadow = true;
    pole.add(shaft);

    // The lit edge the GIF paints as a highlight line, here as real geometry.
    var edgeStrip = new THREE.Mesh(
      new THREE.BoxGeometry(0.008, poleLen * 0.96, 0.008),
      new THREE.MeshStandardMaterial({ color: C.flame, roughness: 0.3, metalness: 0.15 })
    );
    edgeStrip.position.set(-0.019, poleLen / 2, 0.012);
    pole.add(edgeStrip);

    var tipCap = new THREE.Mesh(
      new THREE.SphereGeometry(0.028, 16, 12),
      new THREE.MeshStandardMaterial({
        color: C.tip, roughness: 0.28, metalness: 0.2,
        emissive: 0x5A1408, emissiveIntensity: 1.0
      })
    );
    tipCap.position.y = poleLen;
    tipCap.castShadow = true;
    pole.add(tipCap);

    var collar = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.03, 14), chrome);
    collar.position.y = 0.05;
    pole.add(collar);

    /* The threshold wedge and its two edges ride with the cart, so they mean
       what they mean in the environment: an angle, not a place on the rail. */
    var guides = new THREE.Group();
    cart.add(guides);
    var guideMat = new THREE.MeshBasicMaterial({
      color: 0xB7A37E, transparent: true, opacity: 0.55, depthWrite: false
    });
    [-thetaLimit, thetaLimit].forEach(function (ang) {
      var g = new THREE.Mesh(new THREE.BoxGeometry(0.006, poleLen, 0.006), guideMat);
      g.geometry.translate(0, poleLen / 2, 0);
      g.position.set(0, PIVOT_Y, 0);
      g.rotation.z = -ang;
      guides.add(g);
    });
    var wedge = new THREE.Mesh(
      new THREE.CircleGeometry(poleLen, 24, Math.PI / 2 - thetaLimit, 2 * thetaLimit),
      new THREE.MeshBasicMaterial({
        color: 0x8A7448, transparent: true, opacity: 0.07,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    wedge.position.set(0, PIVOT_Y, -0.02);
    guides.add(wedge);

    /* ------------------------------------------------------- action arrow */
    var arrow = new THREE.Group();
    cart.add(arrow);
    var arrowMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.55, 1.5, 3.4), transparent: true, opacity: 0.95, depthWrite: false
    });
    var arrowShaft = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.022, 0.022), arrowMat);
    arrowShaft.position.set(0.10, 0, 0);
    arrow.add(arrowShaft);
    var arrowHead = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.09, 12), arrowMat);
    arrowHead.rotation.z = -Math.PI / 2;
    arrowHead.position.set(0.245, 0, 0);
    arrow.add(arrowHead);
    arrow.position.set(0, RAIL_Y + 0.21, 0.16);

    /* ----------------------------------------------------------- work lamp
       One shadow caster, following the cart, so the cart and the pole always
       throw a shadow that points the right way and the rest of the room is
       lit by the washers alone. Narrow cone plus normalBias: the pole is a
       thin cylinder, and a depth bias large enough to clear it would detach
       its shadow from the cart. */
    var workLight = new THREE.SpotLight(C.lamp, 1, 9, 0.52, 0.65, 2);
    workLight.power = WORK_LUMENS;
    workLight.castShadow = true;
    workLight.shadow.mapSize.set(2048, 2048);
    workLight.shadow.radius = 2;
    workLight.shadow.camera.near = 0.5;
    workLight.shadow.camera.far = 9;
    workLight.shadow.bias = -0.0006;
    workLight.shadow.normalBias = 0.035;
    workLight.position.set(0, FLOOR_Y + 3.3, 0.9);
    scene.add(workLight);
    scene.add(workLight.target);

    var rigKey = new THREE.PointLight(0xFFF0DA, 1, 8, 2);
    rigKey.power = 850;
    rigKey.position.set(0, FLOOR_Y + 1.9, 2.3);
    scene.add(rigKey);

    var workCan = new THREE.Mesh(
      new THREE.CylinderGeometry(0.11, 0.14, 0.18, 16),
      new THREE.MeshStandardMaterial({ color: 0x24262A, roughness: 0.4, metalness: 0.8 })
    );
    workCan.position.set(0, FLOOR_Y + 3.42, 0.9);
    scene.add(workCan);
    var workLens = new THREE.Mesh(
      new THREE.CircleGeometry(0.105, 16),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(4.0, 3.6, 2.9), side: THREE.DoubleSide
      })
    );
    workLens.rotation.x = Math.PI / 2;
    workLens.position.set(0, FLOOR_Y + 3.325, 0.9);
    scene.add(workLens);

    var workCone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.12, 1.5, 2.4, 22, 1, true),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: C.lamp, transparent: true, opacity: 0.020,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    workCone.position.set(0, FLOOR_Y + 2.14, 0.9);
    scene.add(workCone);

    /* ------------------------------------------------------ belief: ghosts
       One faint pole per belief sample, drawn from that sample's own pivot to
       its own tip. Lines, not a blob: a belief with two modes shows as two
       fans, which a fitted single Gaussian could not. The same samples are
       projected down onto the ruler face as the belief over the cart position
       alone, where the GIF puts its row of dots. */
    var ghostBudget = MAX_GHOSTS;
    payload.beliefs.forEach(function (b) {
      if (b && b.kind === "particles" && b.particles) {
        ghostBudget = Math.max(ghostBudget, Math.min(b.particles.length, 400));
      }
    });

    var ghostPos = new Float32Array(ghostBudget * 2 * 3);
    var ghostCol = new Float32Array(ghostBudget * 2 * 3);
    var ghostGeo = new THREE.BufferGeometry();
    ghostGeo.setAttribute("position", new THREE.BufferAttribute(ghostPos, 3));
    ghostGeo.setAttribute("color", new THREE.BufferAttribute(ghostCol, 3));
    ghostGeo.setDrawRange(0, 0);
    var ghosts = new THREE.LineSegments(ghostGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.45, depthWrite: false
    }));
    scene.add(ghosts);

    var spreadPos = new Float32Array(ghostBudget * 3);
    var spreadGeo = new THREE.BufferGeometry();
    spreadGeo.setAttribute("position", new THREE.BufferAttribute(spreadPos, 3));
    spreadGeo.setDrawRange(0, 0);
    var spread = new THREE.Points(spreadGeo, new THREE.PointsMaterial({
      size: 0.075, map: partTex, color: new THREE.Color(0.20, 1.5, 2.6),
      transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending,
      depthWrite: false, sizeAttenuation: true
    }));
    scene.add(spread);

    // The belief's own centre line, so the fan has a readable middle.
    var meanPos = new Float32Array(6);
    var meanGeo = new THREE.BufferGeometry();
    meanGeo.setAttribute("position", new THREE.BufferAttribute(meanPos, 3));
    var meanLine = new THREE.Line(meanGeo, new THREE.LineBasicMaterial({
      color: new THREE.Color(0.9, 6.0, 10.5), transparent: true, opacity: 0.95,
      depthWrite: false
    }));
    meanLine.visible = false;
    scene.add(meanLine);

    /* A Gaussian belief carries no particles, so it is sampled for display
       only — with a fixed seed, and from the run's own mean and covariance, so
       what is drawn is that belief and nothing else. */
    var gaussianRnd = mulberry(20260920);
    var gaussianDraws = [];
    for (var gi = 0; gi < MAX_GHOSTS; gi++) {
      var u1 = Math.max(1e-6, gaussianRnd()), u2 = gaussianRnd();
      var mag = Math.sqrt(-2 * Math.log(u1));
      gaussianDraws.push([mag * Math.cos(2 * Math.PI * u2), mag * Math.sin(2 * Math.PI * u2)]);
    }

    /* ---------------------------------------------------------------- air
       Motes in the lamp cone: cheap, and they are what tells the eye the work
       light is a beam through air rather than a bright patch on the floor. */
    var MOTES = 220;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(1357);
    for (var mi = 0; mi < MOTES; mi++) {
      motePos[mi * 3] = (moteSeed() - 0.5) * 9;
      motePos[mi * 3 + 1] = FLOOR_Y + 0.5 + moteSeed() * 2.8;
      motePos[mi * 3 + 2] = -2.2 + moteSeed() * 3.4;
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    var motes = new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.016, map: partTex, color: 0xFFE9C6, transparent: true, opacity: 0.35,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    }));
    scene.add(motes);

    /* -------------------------------------------------------------- trail
       Where the cart has been, painted along the ruler face. */
    var steps = payload.states.length;
    var trailPos = new Float32Array(steps * 3);
    var trailCol = new Float32Array(steps * 3);
    var trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3));
    trailGeo.setDrawRange(0, 0);
    var trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.8,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    scene.add(trail);

    core.linearize();

    // Discounted return up to and including each step, so the HUD reports the
    // episode's own arithmetic rather than counting steps.
    var running = [];
    var total = 0;
    for (var ri = 0; ri < trace.steps.length; ri++) {
      var reward = trace.steps[ri].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, ri);
      }
      running.push(total);
    }

    /* The states are the episode's, so between two of them the viewer
       interpolates rather than integrating. At tau = 0.02 s the gap is small
       and a straight line across it is honest; re-running the plant would not
       be, because the result would be this viewer's trajectory. */
    function sampleAt(t) {
      var i0 = Math.floor(clamp(t, 0, steps - 1));
      var i1 = Math.min(i0 + 1, steps - 1);
      var f = clamp(t - i0, 0, 1);
      var a = payload.states[i0], b = payload.states[i1];
      return {
        index: i0,
        x: lerp(a[0], b[0], f),
        v: lerp(a[1], b[1], f),
        theta: lerp(a[2], b[2], f),
        omega: lerp(a[3], b[3], f)
      };
    }

    /** Write one ghost pole and its rail dot, shaded by its share of the mass. */
    function writeGhost(slot, x, theta, relativeWeight) {
      var base = slot * 6;
      ghostPos[base] = x;
      ghostPos[base + 1] = PIVOT_Y;
      ghostPos[base + 2] = -0.006;
      ghostPos[base + 3] = x + poleLen * Math.sin(theta);
      ghostPos[base + 4] = PIVOT_Y + poleLen * Math.cos(theta);
      ghostPos[base + 5] = -0.006;
      // Brighter at the tip, so the fan reads as poles rather than a haze, and
      // brighter overall where the belief puts its mass. Above 1.0 so the
      // heavy end blooms.
      var hot = 0.35 + 0.65 * relativeWeight;
      ghostCol[base] = 0.30 * hot; ghostCol[base + 1] = 1.60 * hot; ghostCol[base + 2] = 2.90 * hot;
      ghostCol[base + 3] = 0.70 * hot;
      ghostCol[base + 4] = 4.20 * hot;
      ghostCol[base + 5] = 7.60 * hot;

      spreadPos[slot * 3] = x;
      spreadPos[slot * 3 + 1] = RAIL_Y - 0.036;
      spreadPos[slot * 3 + 2] = 0.196;
    }

    function commit(count) {
      ghostGeo.attributes.position.needsUpdate = true;
      ghostGeo.attributes.color.needsUpdate = true;
      ghostGeo.setDrawRange(0, count * 2);
      spreadGeo.attributes.position.needsUpdate = true;
      spreadGeo.setDrawRange(0, count);
    }

    function showMean(x, theta) {
      meanPos[0] = x; meanPos[1] = PIVOT_Y; meanPos[2] = -0.007;
      meanPos[3] = x + poleLen * Math.sin(theta);
      meanPos[4] = PIVOT_Y + poleLen * Math.cos(theta);
      meanPos[5] = -0.007;
      meanGeo.attributes.position.needsUpdate = true;
      meanLine.visible = true;
    }

    function clearBelief() {
      ghostGeo.setDrawRange(0, 0);
      spreadGeo.setDrawRange(0, 0);
      meanLine.visible = false;
    }

    /* A 4-D Gaussian's [x, theta] marginal is exactly the 2x2 sub-block of its
       covariance at rows and columns 0 and 2, so the fan is sampled from that
       rather than from a covariance the belief never had. */
    function writeGaussianAt(mean, covariance, slot, howMany, weightScale) {
      var L = cholesky2(covariance[0][0], covariance[2][0], covariance[2][2]);
      for (var q = 0; q < howMany; q++) {
        var z1 = gaussianDraws[q][0], z2 = gaussianDraws[q][1];
        writeGhost(
          slot + q,
          mean[0] + L[0] * z1,
          mean[2] + L[1] * z1 + L[2] * z2,
          Math.exp(-(z1 * z1 + z2 * z2) * 0.5) * weightScale
        );
      }
      return howMany;
    }

    function drawGaussian(belief) {
      var count = Math.min(MAX_GHOSTS, ghostBudget);
      commit(writeGaussianAt(belief.mean, belief.covariance, 0, count, 1));
      showMean(belief.mean[0], belief.mean[2]);
      // The angle's own standard deviation is the number that says whether the
      // agent can tell an upright pole from a falling one.
      var sigma = Math.sqrt(Math.max(belief.covariance[2][2], 0)) * 180 / Math.PI;
      return "Gaussian, σθ " + sigma.toFixed(1) + "°";
    }

    /* A mixture is drawn as its components, each given a share of the fan in
       proportion to its weight. Collapsing it to one cloud would put the
       belief's mass between its modes, where the belief says nothing is. */
    function drawGaussianMixture(belief) {
      var budget = Math.min(MAX_GHOSTS, ghostBudget);
      var slot = 0;
      for (var k = 0; k < belief.components.length && slot < budget; k++) {
        var component = belief.components[k];
        var share = Math.max(1, Math.round(component.weight * budget));
        share = Math.min(share, budget - slot, gaussianDraws.length);
        slot += writeGaussianAt(component.mean, component.covariance, slot, share, component.weight);
      }
      commit(slot);
      return belief.components.length + "-component mixture";
    }

    function drawParticles(belief) {
      var count = Math.min(belief.particles.length, ghostBudget);
      if (count && (!Array.isArray(belief.particles[0]) || belief.particles[0].length < 4)) {
        // Not a four-component state. This scene draws poles, so it says so
        // rather than reading an angle out of whatever arrived.
        clearBelief();
        return belief.num_particles + " particles in an unreadable shape";
      }
      var maxWeight = 0;
      for (var w = 0; w < belief.weights.length; w++) {
        if (belief.weights[w] > maxWeight) maxWeight = belief.weights[w];
      }
      var meanX = 0, meanTheta = 0;
      for (var p = 0; p < count; p++) {
        // An unweighted belief is uniform by construction, so shading it by
        // weight would imply structure the belief does not have.
        var rel = belief.weighted === false
          ? 1
          : (maxWeight > 0 ? belief.weights[p] / maxWeight : 0);
        writeGhost(p, belief.particles[p][0], belief.particles[p][2], rel);
        meanX += belief.weights[p] * belief.particles[p][0];
        meanTheta += belief.weights[p] * belief.particles[p][2];
      }
      commit(count);
      showMean(meanX, meanTheta);
      var label = belief.num_particles +
        (belief.weighted === false ? " uniform particles" : " particles");
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " written)";
      }
      return label;
    }

    function drawBelief(index) {
      var belief = payload.beliefs[index];
      if (!belief) { clearBelief(); return "—"; }

      if (belief.kind === "gaussian") return drawGaussian(belief);
      if (belief.kind === "gaussian_mixture") return drawGaussianMixture(belief);
      if (belief.kind === "particles") return drawParticles(belief);
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner;
        // it is not one episode's belief, and merging its members would show a
        // fan that was never anyone's belief.
        clearBelief();
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }

      // A belief class core has no payload for yet. Named, so the gap is
      // diagnosable, and drawn as nothing, so it is not invented.
      clearBelief();
      return "not recorded (" + (belief.belief_class || belief.kind) + ")";
    }

    function drawTrail(index) {
      var count = index + 1;
      for (var i = 0; i < count; i++) {
        trailPos[i * 3] = payload.states[i][0];
        trailPos[i * 3 + 1] = RAIL_Y - 0.064;
        trailPos[i * 3 + 2] = 0.19;
        var age = count > 1 ? i / (count - 1) : 1;
        trailCol[i * 3] = lerp(0.18, 0.95, age);
        trailCol[i * 3 + 1] = lerp(0.06, 0.30, age);
        trailCol[i * 3 + 2] = lerp(0.05, 0.22, age);
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
      trailGeo.setDrawRange(0, Math.max(0, count));
    }

    return {
      steps: steps,

      /* Low and side-on, level with the hinge. The board camera's look-at is
         0.3 above the origin and the pivot is at the origin, so this puts the
         lens just above the bearing: the pole angle is the quantity a reader
         reads, and an angle is only legible from the side. The rail is 6.4 m
         of scene, so the camera sits back far enough to hold both limit posts. */
      camera: {
        board: [0.45, 0.30, 4.10],
        top: [0, 4.60, 0.01]
      },

      /**
       * Advance the rig to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);

        cart.position.x = sample.x;
        pole.rotation.z = -sample.theta;

        // Wheels turn from the cart's own recorded velocity, not from how fast
        // the browser happens to be playing back.
        var spin = -sample.v * dt * 26;
        for (var w = 0; w < wheels.length; w++) wheels[w].rotation.y += spin;

        var step = trace.steps[sample.index] || {};
        var action = step.action;
        var hasAction = action !== null && action !== undefined;
        arrow.visible = hasAction;
        if (hasAction) {
          // Action 1 is the rightward force, 0 the leftward one.
          arrow.scale.x = Number(action) === 1 ? 1 : -1;
          var hot = playing ? 0.75 + Math.sin(elapsed * 9) * 0.25 : 1.0;
          arrowMat.opacity = 0.55 + hot * 0.4;
        }

        workLight.position.set(sample.x * 0.55, FLOOR_Y + 3.3, 0.9);
        workLight.target.position.set(sample.x, RAIL_Y, 0);
        workLight.target.updateMatrixWorld();
        rigKey.position.x = sample.x * 0.5;
        workCan.position.x = sample.x * 0.55;
        workLens.position.x = sample.x * 0.55;
        workCone.position.x = sample.x * 0.55;

        if (playing) {
          for (var m = 0; m < MOTES; m++) {
            motePos[m * 3] += Math.sin(elapsed * 0.3 + m) * 0.0011;
            motePos[m * 3 + 1] += 0.0013;
            if (motePos[m * 3 + 1] > FLOOR_Y + 3.3) motePos[m * 3 + 1] = FLOOR_Y + 0.5;
          }
          moteGeo.attributes.position.needsUpdate = true;
        }

        var beliefLabel = drawBelief(sample.index);
        drawTrail(sample.index);

        return {
          // The chase camera follows the cart down the rail. A fixed heading
          // of -pi/2 puts it in front of the rig rather than behind it, which
          // is the only side an angle can be read from.
          follow: { x: sample.x, z: 0, heading: -Math.PI / 2 },
          step: sample.index,
          action: hasAction
            ? String(action) + (Number(action) === 1 ? " right" : " left")
            : "—",
          x: sample.x,
          // The player's second readout is labelled "y". CartPole has no y:
          // this is the pole angle in degrees, which is the state component
          // the episode actually turns on.
          y: sample.theta * 180 / Math.PI,
          reward: step.reward,
          ret: running[sample.index],
          belief: beliefLabel
        };
      }
    };
  }

  V.scenes["cartpole.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // scene's lamp power; it is not a knob to remove.
    exposure: 0.165
  };
})(window);
