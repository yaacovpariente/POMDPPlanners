/* SPDX-License-Identifier: MIT
 *
 * Mountain Car scene module.
 *
 * Builds the valley from a trace's `payload.world` block and moves the car
 * along it from the trace's recorded states. Nothing here is invented: the
 * hill's shape arrives in the payload rather than being assumed, the car is
 * placed at the positions the episode actually visited, and no step of the
 * physics is re-integrated. A viewer that re-ran the plant would be showing
 * its own rollout, which is the one thing a recorded episode exists to
 * exclude.
 *
 * The belief needs two dimensions and the valley only offers one. Mountain
 * Car's state is a position *and* a velocity, and a cloud drawn along the
 * slope alone would collapse the half of the belief that the whole problem
 * turns on — the car is underpowered, so the agent is estimating momentum, not
 * whereabouts. So the road is given a width that *is* the velocity axis: a
 * particle's distance across the road is its velocity, the rails mark the
 * speed limit the environment clamps at, and the car itself rides at its own
 * recorded velocity. A cloud strung along the road is a belief that agrees
 * about where the car is and disagrees about where it is going.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  var COLORS = {
    lamp: 0xFFE4B4,
    car: 0xC6432A,
    goal: 0x2EBE4A,
    rail: 0x3C69DC,
    rock: 0x4A4036
  };

  // Lumens. Service lamps on posts along the valley, not floodlights: the far
  // end of the road has to fall away into the dark, or the hill reads flat.
  var LAMP_LUMENS = 820;

  // Cap on drawn particles. A trace may carry several hundred; more than this
  // costs draw time and adds no information at this scale.
  var MAX_DRAWN_PARTICLES = 420;

  // World units per unit of car position, and per unit of hill height. The
  // valley is 1.8 wide and about 0.9 tall in the environment's own units,
  // which is a strip a camera cannot frame; these put it on the same scale as
  // every other board in the viewer.
  var SPAN_X = 7.2;
  var SPAN_Y = 3.6;

  // Half the road's width, in world units. This is the velocity axis: the
  // rails sit at the environment's own ±max_speed.
  var ROAD_HALF = 1.15;

  // How far the rock runs below the valley's lowest point. Deep enough that no
  // camera angle sees under the hill and finds it hollow.
  var BODY_DEPTH = 3.4;

  /** A painted rock face, and the normal map that turns it into stone. */
  function rockCanvas(seed) {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#2A241E";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 900; i++) {
      var r = 12 + rnd() * 110;
      g.fillStyle = "rgba(" + (58 + rnd() * 40 | 0) + "," + (48 + rnd() * 30 | 0) + "," +
        (38 + rnd() * 24 | 0) + ",0.24)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
    }
    // Strata: this rock is a cut hillside, and horizontal banding is what
    // reads as one at a distance.
    for (var b = 0; b < 40; b++) {
      var y = rnd() * s;
      g.strokeStyle = "rgba(24,20,16," + (0.10 + rnd() * 0.22).toFixed(2) + ")";
      g.lineWidth = 1 + rnd() * 5;
      g.beginPath();
      g.moveTo(0, y);
      for (var x = 0; x <= s; x += 64) g.lineTo(x, y + (rnd() - 0.5) * 12);
      g.stroke();
    }
    for (var p = 0; p < 1200; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.5 + rnd() * 5;
      g.fillStyle = "rgba(18,15,12,0.5)";
      g.beginPath(); g.arc(px + pr * 0.3, py + pr * 0.3, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(160,142,118,0.3)";
      g.beginPath(); g.arc(px, py, pr * 0.85, 0, Math.PI * 2); g.fill();
    }
    return cv;
  }

  /** The road surface: tarmac, a centre line, and the two velocity rails. */
  function roadCanvas(seed) {
    var w = 1024, h = 256;
    var cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#1E1C1B";
    g.fillRect(0, 0, w, h);
    for (var i = 0; i < 5000; i++) {
      g.fillStyle = "rgba(" + (44 + rnd() * 34 | 0) + "," + (42 + rnd() * 30 | 0) + "," +
        (40 + rnd() * 28 | 0) + ",0.5)";
      g.fillRect(rnd() * w, rnd() * h, 1 + rnd() * 2, 1 + rnd() * 2);
    }
    // Centre line: zero velocity. A particle sitting on it is a particle that
    // believes the car is stationary.
    g.strokeStyle = "rgba(226,206,163,0.30)";
    g.lineWidth = 3;
    g.setLineDash([26, 22]);
    g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
    g.setLineDash([]);
    // Edges: the clamp. The environment will not let velocity past these.
    g.strokeStyle = "rgba(120,150,230,0.34)";
    g.lineWidth = 5;
    [6, h - 6].forEach(function (y) {
      g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
    });
    return cv;
  }

  /** A small lit signboard, so a label in the world needs no HTML overlay. */
  function signTexture(text, color) {
    var w = 256, h = 64;
    var cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    var g = cv.getContext("2d");
    g.fillStyle = "rgba(10,12,16,0.0)";
    g.fillRect(0, 0, w, h);
    g.font = "600 34px ui-sans-serif, system-ui, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillStyle = color;
    g.fillText(text, w / 2, h / 2 + 2);
    var tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }

  /**
   * Build the Mountain Car world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind mountain_car.v1.
   * @returns {Object} The scene module the player drives.
   */
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;

    var hill = world.hill;
    var minP = world.min_position;
    var maxP = world.max_position;
    var midP = (minP + maxP) / 2;
    var maxSpeed = Math.max(1e-6, world.max_speed);

    /** The valley floor, in the environment's own height units. */
    function height(position) {
      return hill.amplitude * Math.sin(hill.frequency * position) + hill.offset;
    }
    /** Position along the valley, in world units, centred on the origin. */
    function wx(position) { return (position - midP) * SPAN_X; }
    /* Height in world units, measured from the valley's own midline rather
       than from its floor. The camera rig frames the origin, so a road whose
       every point sits above it leaves the bottom half of the canvas empty —
       which is what this scene did before the datum was moved. */
    function wy(position) { return (height(position) - hill.offset) * SPAN_Y; }
    /* Velocity across the road, in world units. This is the belief's second
       axis: the environment clamps speed to ±max_speed, so the rails are the
       real limit and a particle never has to be drawn off the tarmac. */
    function wz(velocity) { return clamp(velocity / maxSpeed, -1, 1) * ROAD_HALF; }

    /** Slope of the valley at a position, as an angle to pitch the car by. */
    function slope(position) {
      var dydx = hill.amplitude * hill.frequency * Math.cos(hill.frequency * position);
      return Math.atan2(dydx * SPAN_Y, SPAN_X);
    }

    var extentX = (maxP - minP) * SPAN_X;

    scene.background = new THREE.Color(0x05070C).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x070810, 0.018);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x223046, 0x0A0806, 0.55));
    var moon = new THREE.DirectionalLight(0x8FA3C8, 0.38);
    moon.position.set(-8, 12, 6);
    scene.add(moon);
    core.buildNightEnvironment();

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);

    /* The hill itself, extruded from its own profile. The profile is sampled
       from the payload's formula rather than approximated by a few control
       points, so the surface the car drives on and the surface drawn are the
       same curve. */
    var SAMPLES = 220;
    var profile = [];
    for (var s = 0; s <= SAMPLES; s++) {
      var p = minP + (maxP - minP) * (s / SAMPLES);
      profile.push([p, wx(p), wy(p)]);
    }

    var rockCv = rockCanvas(20260922);
    var rockNormal = V.normalMapFrom(renderer, rockCv, 2.2);
    var rockAlbedo = new THREE.CanvasTexture(rockCv);
    rockAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    rockAlbedo.encoding = THREE.sRGBEncoding;
    rockAlbedo.wrapS = rockAlbedo.wrapT = THREE.RepeatWrapping;
    rockAlbedo.repeat.set(4, 2);
    rockNormal.wrapS = rockNormal.wrapT = THREE.RepeatWrapping;
    rockNormal.repeat.set(4, 2);

    var rockMat = new THREE.MeshStandardMaterial({
      map: rockAlbedo, normalMap: rockNormal,
      normalScale: new THREE.Vector2(0.9, 0.9),
      roughness: 0.96, metalness: 0.02, envMapIntensity: 0.28
    });

    // The mountain body: the profile closed downwards into a solid, extruded
    // across the road's width plus the shoulders it sits in.
    var bodyShape = new THREE.Shape();
    bodyShape.moveTo(profile[0][1], -BODY_DEPTH);
    profile.forEach(function (point) { bodyShape.lineTo(point[1], point[2]); });
    bodyShape.lineTo(profile[profile.length - 1][1], -BODY_DEPTH);
    bodyShape.closePath();
    var body = new THREE.Mesh(
      new THREE.ExtrudeGeometry(bodyShape, { depth: ROAD_HALF * 2 + 2.4, bevelEnabled: false }),
      rockMat
    );
    body.position.z = -(ROAD_HALF + 1.2);
    body.receiveShadow = true;
    body.castShadow = true;
    scene.add(body);

    // The road: a ribbon following the same profile, laid on the hill's crest.
    var roadCv = roadCanvas(4242);
    var roadAlbedo = new THREE.CanvasTexture(roadCv);
    roadAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    roadAlbedo.encoding = THREE.sRGBEncoding;
    var roadNormal = V.normalMapFrom(renderer, roadCv, 1.1);
    var roadGeo = new THREE.PlaneGeometry(1, 1, SAMPLES, 1);
    var roadPos = roadGeo.attributes.position;
    for (var v = 0; v < roadPos.count; v++) {
      var column = v % (SAMPLES + 1);
      var side = v < (SAMPLES + 1) ? 1 : -1;
      var point = profile[column];
      roadPos.setXYZ(v, point[1], point[2] + 0.02, side * ROAD_HALF);
    }
    roadPos.needsUpdate = true;
    roadGeo.computeVertexNormals();
    var road = new THREE.Mesh(roadGeo, new THREE.MeshStandardMaterial({
      map: roadAlbedo, normalMap: roadNormal,
      normalScale: new THREE.Vector2(0.5, 0.5),
      roughness: 0.88, metalness: 0.04, envMapIntensity: 0.3,
      side: THREE.DoubleSide
    }));
    road.receiveShadow = true;
    scene.add(road);

    /* The rails. They are the velocity clamp, drawn where the clamp is, and
       labelled with the environment's own number — a reader should not have to
       take the scale on trust. */
    var railMat = new THREE.MeshStandardMaterial({
      color: 0x2C3E66, roughness: 0.4, metalness: 0.8, emissive: 0x0A1428
    });
    /* The labels are parked at the valley's floor: the one stretch of road the
       car crosses on every oscillation, and so the place a reader is already
       looking. */
    var lowP = minP;
    for (var lp = 0; lp <= SAMPLES; lp++) {
      var cand = minP + (maxP - minP) * (lp / SAMPLES);
      if (height(cand) < height(lowP)) lowP = cand;
    }
    [1, -1].forEach(function (side) {
      var points = profile.map(function (point) {
        return new THREE.Vector3(point[1], point[2] + 0.16, side * ROAD_HALF);
      });
      var rail = new THREE.Mesh(
        new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), SAMPLES, 0.035, 6, false),
        railMat
      );
      rail.castShadow = true;
      scene.add(rail);

      /* One label per rail, saying what that edge of the road means. It is
         painted flat on the shoulder beyond the rail: readable from the two
         cameras that look down at the valley, and out from underfoot in the
         two that do not. Billboarding it was tried and is worse — a quad that
         turns to face the camera fills half the frame the moment the chase
         camera passes it. */
      var sign = new THREE.Mesh(
        new THREE.PlaneGeometry(0.95, 0.24),
        new THREE.MeshBasicMaterial({
          map: signTexture((side > 0 ? "+" : "−") + maxSpeed.toFixed(3) + " m/step",
            "rgba(150,180,255,0.95)"),
          transparent: true, depthWrite: false
        })
      );
      sign.rotation.x = -Math.PI / 2;
      sign.rotation.z = Math.PI / 2;
      sign.position.set(wx(lowP), wy(lowP) + 0.06, side * (ROAD_HALF + 0.42));
      scene.add(sign);
    });

    /* Lamp posts along the shoulder. They are the scene's light and its only
       sense of distance: the far end of the valley is dark because a lamp this
       size cannot reach it, not because anything was faded out. */
    var lampLights = [];
    var postMat = new THREE.MeshStandardMaterial({ color: 0x6B5436, roughness: 0.42, metalness: 0.85 });
    var lensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(4.4, 3.9, 3.0) });
    for (var lamp = 0; lamp <= 4; lamp++) {
      var lp = minP + (maxP - minP) * (lamp / 4);
      var lx = wx(lp), ly = wy(lp);
      var post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, 1.3, 10), postMat);
      post.position.set(lx, ly + 0.65, -(ROAD_HALF + 0.35));
      post.castShadow = true;
      scene.add(post);
      var lens = new THREE.Mesh(new THREE.SphereGeometry(0.11, 12, 10), lensMat);
      lens.position.set(lx, ly + 1.34, -(ROAD_HALF + 0.35));
      scene.add(lens);
      var light = new THREE.PointLight(COLORS.lamp, 1, 9.5, 2);
      light.power = LAMP_LUMENS;
      light.position.set(lx, ly + 1.36, -(ROAD_HALF + 0.35));
      scene.add(light);
      lampLights.push({ light: light, x: lx, y: ly + 1.36, z: -(ROAD_HALF + 0.35) });
    }

    /* One shadow caster, parked on whichever lamp is nearest the car and aimed
       at it, with that lamp's own output taken down by the same amount. Sixteen
       shadow-casting lights would cost more than the whole rest of the frame
       and would look no different. */
    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 14, 0.5, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.4;
    shadowLight.shadow.camera.far = 15;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    // Scattered boulders on the shoulders. Deterministic, and kept clear of
    // the road so nothing appears to block a car that was never blocked.
    var boulderMat = new THREE.MeshStandardMaterial({
      color: COLORS.rock, roughness: 0.95, metalness: 0.06, flatShading: true
    });
    var boulderGeos = [
      new THREE.DodecahedronGeometry(0.2, 0),
      new THREE.IcosahedronGeometry(0.16, 0),
      new THREE.TetrahedronGeometry(0.24, 0)
    ];
    var rockRnd = mulberry(70707);
    for (var bo = 0; bo < 34; bo++) {
      var bp = minP + (maxP - minP) * rockRnd();
      var bside = rockRnd() < 0.5 ? -1 : 1;
      var boulder = new THREE.Mesh(boulderGeos[bo % boulderGeos.length], boulderMat);
      var bs = 0.4 + rockRnd() * 0.9;
      boulder.scale.set(bs, bs * (0.5 + rockRnd() * 0.5), bs);
      boulder.position.set(
        wx(bp), wy(bp) + 0.04 * bs,
        bside * (ROAD_HALF + 0.45 + rockRnd() * 0.75)
      );
      boulder.rotation.set(rockRnd() * 3, rockRnd() * 3, rockRnd() * 3);
      boulder.castShadow = true;
      boulder.receiveShadow = true;
      scene.add(boulder);
    }

    /* The goal: a flag at the position the environment calls terminal, on the
       slope where it actually sits. It is the only thing in this world worth
       reaching, and the episode is scored by whether the car got here. */
    var goalGroup = new THREE.Group();
    goalGroup.position.set(wx(world.goal_position), wy(world.goal_position), 0);
    scene.add(goalGroup);
    var mast = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.04, 1.5, 10),
      new THREE.MeshStandardMaterial({ color: 0x8A8378, roughness: 0.4, metalness: 0.8 })
    );
    mast.position.y = 0.75;
    mast.castShadow = true;
    goalGroup.add(mast);
    var flag = new THREE.Mesh(
      new THREE.PlaneGeometry(0.62, 0.36, 12, 1),
      new THREE.MeshStandardMaterial({
        color: COLORS.goal, roughness: 0.6, metalness: 0.1,
        emissive: 0x0A3315, side: THREE.DoubleSide
      })
    );
    flag.position.set(0.33, 1.3, 0);
    goalGroup.add(flag);
    var flagBase = flag.geometry.attributes.position.array.slice();
    var goalLight = new THREE.PointLight(COLORS.goal, 1, 5.5, 2);
    goalLight.power = 55;
    goalLight.position.set(0, 1.3, 0);
    goalGroup.add(goalLight);

    // A line down the goal position, so it is readable from the top camera too.
    var goalGate = new THREE.Mesh(
      new THREE.PlaneGeometry(0.1, ROAD_HALF * 2),
      new THREE.MeshBasicMaterial({
        color: COLORS.goal, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    goalGate.rotation.x = -Math.PI / 2;
    goalGate.position.set(wx(world.goal_position), wy(world.goal_position) + 0.05, 0);
    scene.add(goalGate);

    /* The car. Underpowered, and drawn as such: a small hatchback, not a
       rally machine. Its pitch is the slope under it, which is what makes the
       first failed climb legible — the car is visibly nose-up and losing. */
    var car = new THREE.Group();
    scene.add(car);
    var shell = new THREE.Group();
    car.add(shell);
    var paint = new THREE.MeshStandardMaterial({ color: COLORS.car, roughness: 0.38, metalness: 0.45 });
    var trim = new THREE.MeshStandardMaterial({ color: 0x241F1D, roughness: 0.6, metalness: 0.7 });
    var glassMat = new THREE.MeshStandardMaterial({
      color: 0x2A4468, roughness: 0.1, metalness: 0.4,
      emissive: 0x111E33, emissiveIntensity: 0.8
    });
    var hull = new THREE.Mesh(V.roundedBox(0.72, 0.2, 0.42), paint);
    hull.position.y = 0.2; hull.castShadow = true;
    shell.add(hull);
    var cabin = new THREE.Mesh(V.roundedBox(0.4, 0.18, 0.38), paint);
    cabin.position.set(-0.04, 0.38, 0); cabin.castShadow = true;
    shell.add(cabin);
    var glass = new THREE.Mesh(V.roundedBox(0.3, 0.13, 0.39, 0.02), glassMat);
    glass.position.set(-0.02, 0.4, 0);
    shell.add(glass);
    var bumper = new THREE.Mesh(V.roundedBox(0.06, 0.1, 0.44, 0.02), trim);
    bumper.position.set(0.37, 0.16, 0);
    shell.add(bumper);

    var wheels = [];
    var wheelGeo = new THREE.CylinderGeometry(0.125, 0.125, 0.1, 16);
    var wheelMat = new THREE.MeshStandardMaterial({ color: 0x141211, roughness: 0.9, metalness: 0.1 });
    var hubMat = new THREE.MeshStandardMaterial({ color: 0x7A6748, roughness: 0.35, metalness: 0.9 });
    [[0.24, 0.22], [0.24, -0.22], [-0.24, 0.22], [-0.24, -0.22]].forEach(function (w) {
      var wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.position.set(w[0], 0.125, w[1]);
      wheel.rotation.x = Math.PI / 2;
      wheel.castShadow = true;
      shell.add(wheel);
      wheels.push(wheel);
      var hub = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.105, 10), hubMat);
      hub.position.copy(wheel.position);
      hub.rotation.x = Math.PI / 2;
      shell.add(hub);
    });

    // Headlamps, and the beam that makes the dark road ahead readable from the
    // chase camera.
    var headMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.4, 3.1, 2.6) });
    [[0.38, 0.14], [0.38, -0.14]].forEach(function (h) {
      var bulb = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 10), headMat);
      bulb.position.set(h[0], 0.26, h[1]);
      shell.add(bulb);
    });
    var headlight = new THREE.SpotLight(0xFFF1D6, 1, 10, 0.5, 0.6, 2);
    headlight.power = 700;
    headlight.position.set(0.36, 0.28, 0);
    var headTarget = new THREE.Object3D();
    headTarget.position.set(4, -0.3, 0);
    shell.add(headlight);
    shell.add(headTarget);
    headlight.target = headTarget;

    var contact = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.8),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.42, depthWrite: false
      }));
    contact.rotation.x = -Math.PI / 2;
    contact.position.y = 0.01;
    car.add(contact);

    /* Belief cloud. One Points object, sized for the largest step's cloud and
       draw-ranged per frame, so a step with fewer particles costs nothing. */
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
      size: 0.2, map: partTex, vertexColors: true, transparent: true,
      opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false,
      sizeAttenuation: true
    }));
    scene.add(particles);

    /* A Gaussian belief carries no particles, so it is sampled for display
       only — with a fixed seed, and from the run's own mean and covariance, so
       what is drawn is that belief and nothing else. */
    var gaussianRnd = mulberry(20260922);
    var gaussianDraws = [];
    for (var gi = 0; gi < MAX_DRAWN_PARTICLES; gi++) {
      var u1 = Math.max(1e-6, gaussianRnd()), u2 = gaussianRnd();
      var mag = Math.sqrt(-2 * Math.log(u1));
      gaussianDraws.push([mag * Math.cos(2 * Math.PI * u2), mag * Math.sin(2 * Math.PI * u2)]);
    }

    // Trail: the recorded path, drawn up to the current step.
    var trailPos = new Float32Array(payload.states.length * 3);
    var trailCol = new Float32Array(payload.states.length * 3);
    var trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3));
    trailGeo.setDrawRange(0, 0);
    var trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.9
    }));
    scene.add(trail);

    core.linearize();

    var wheelSpin = 0;
    var facing = 1;
    // Where the car is this frame, in world units. The chase camera is
    // owned by this scene and reads it; see the camera block below.
    var carAt = { x: 0, y: 0, z: 0, position: minP };
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
        index: i0,
        position: lerp(a[0], b[0], f),
        velocity: lerp(a[1], b[1], f)
      };
    }

    /** Write one particle into the buffer, shaded by its share of the mass. */
    function writeParticle(slot, point, relativeWeight) {
      pPos[slot * 3] = wx(point[0]);
      pPos[slot * 3 + 1] = wy(point[0]) + 0.3;
      pPos[slot * 3 + 2] = wz(point[1]);
      // Bright amber where the mass is, deep red in the tail. Above 1.0 so the
      // heavy end blooms.
      var hot = 3.4;
      pCol[slot * 3] = lerp(0.86, 1.0, relativeWeight) * hot;
      pCol[slot * 3 + 1] = lerp(0.16, 0.84, relativeWeight) * hot;
      pCol[slot * 3 + 2] = lerp(0.13, 0.18, relativeWeight) * hot;
    }

    function commit(count) {
      pGeo.attributes.position.needsUpdate = true;
      pGeo.attributes.color.needsUpdate = true;
      pGeo.setDrawRange(0, count);
    }

    /* The belief is drawn from the kind core serialised it as, never from the
       belief class that produced it. Every particle belief in the package
       arrives as "particles", so this widget draws them once rather than once
       per class. */
    function drawParticles(belief) {
      var count = Math.min(belief.particles.length, maxParticles);
      if (count && !Array.isArray(belief.particles[0])) {
        // A particle that is not a (position, velocity) pair. This scene plots
        // states on the valley, so it says so rather than plotting something
        // it cannot place.
        pGeo.setDrawRange(0, 0);
        return belief.num_particles + " non-spatial particles";
      }
      var maxWeight = 0;
      for (var w = 0; w < belief.weights.length; w++) {
        if (belief.weights[w] > maxWeight) maxWeight = belief.weights[w];
      }
      for (var p = 0; p < count; p++) {
        // An unweighted belief is uniform by construction, so shading it by
        // weight would imply structure the belief does not have.
        var rel = belief.weighted === false ? 1 : (maxWeight > 0 ? belief.weights[p] / maxWeight : 0);
        writeParticle(p, belief.particles[p], rel);
      }
      commit(count);
      var label = belief.num_particles +
        (belief.weighted === false ? " uniform particles" : " particles");
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " drawn)";
      }
      return label;
    }

    /** Lower-triangular Cholesky of a 2x2, so a drawn cloud has the real shape. */
    function cholesky2(covariance) {
      var l11 = Math.sqrt(Math.max(covariance[0][0], 1e-12));
      var l21 = covariance[1][0] / l11;
      var l22 = Math.sqrt(Math.max(covariance[1][1] - l21 * l21, 1e-12));
      return [l11, l21, l22];
    }

    function writeGaussian(mean, covariance, slot, howMany, weightScale) {
      var L = cholesky2(covariance);
      for (var q = 0; q < howMany; q++) {
        var z1 = gaussianDraws[q][0], z2 = gaussianDraws[q][1];
        writeParticle(
          slot + q,
          [mean[0] + L[0] * z1, mean[1] + L[1] * z1 + L[2] * z2],
          Math.exp(-(z1 * z1 + z2 * z2) * 0.5) * weightScale
        );
      }
      return howMany;
    }

    function drawGaussian(belief) {
      var count = Math.min(MAX_DRAWN_PARTICLES, maxParticles);
      commit(writeGaussian(belief.mean, belief.covariance, 0, count, 1));
      return "Gaussian";
    }

    /* A mixture is drawn as its components, each given a share of the points
       in proportion to its weight. Collapsing it to one cloud would put the
       belief's mass between its modes, where the belief says nothing is. */
    function drawGaussianMixture(belief) {
      var budget = Math.min(MAX_DRAWN_PARTICLES, maxParticles);
      var slot = 0;
      for (var k = 0; k < belief.components.length && slot < budget; k++) {
        var component = belief.components[k];
        var share = Math.max(1, Math.round(component.weight * budget));
        share = Math.min(share, budget - slot, gaussianDraws.length);
        slot += writeGaussian(component.mean, component.covariance, slot, share, component.weight);
      }
      commit(slot);
      return belief.components.length + "-component mixture";
    }

    function drawBelief(index) {
      var belief = payload.beliefs[index];
      if (!belief) { pGeo.setDrawRange(0, 0); return "—"; }

      if (belief.kind === "particles") return drawParticles(belief);
      if (belief.kind === "gaussian") return drawGaussian(belief);
      if (belief.kind === "gaussian_mixture") return drawGaussianMixture(belief);
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner;
        // it is not one episode's belief, and merging its members would show a
        // cloud that was never anyone's belief.
        pGeo.setDrawRange(0, 0);
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }

      // A belief class core has no payload for yet. Named, so the gap is
      // diagnosable, and drawn as nothing, so it is not invented.
      pGeo.setDrawRange(0, 0);
      return "not recorded (" + (belief.belief_class || belief.kind) + ")";
    }

    function drawTrail(sample) {
      var count = sample.index + 1;
      for (var i = 0; i < count; i++) {
        var state = payload.states[i];
        trailPos[i * 3] = wx(state[0]);
        trailPos[i * 3 + 1] = wy(state[0]) + 0.2;
        trailPos[i * 3 + 2] = wz(state[1]);
        var age = count > 1 ? i / (count - 1) : 1;
        trailCol[i * 3] = lerp(0.5, 0.95, age);
        trailCol[i * 3 + 1] = lerp(0.14, 0.34, age);
        trailCol[i * 3 + 2] = lerp(0.1, 0.22, age);
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
      trailGeo.setDrawRange(0, Math.max(0, count));
    }

    return {
      steps: payload.states.length,

      /* Framing scales with the valley, because the payload decides how wide
         it is: a distance that frames the default -1.2..0.6 road leaves a
         narrower one as a smudge in the middle of the canvas.

         The multipliers are not guesses. The core's camera is a long lens —
         about 60 degrees across a 16:9 canvas — so framing a road `extentX`
         wide needs a standoff of `extentX / 2 / tan(30 deg)`, which is
         `extentX * 0.87`. These sit above that, and the constant term is
         headroom for the props: a lamp post is the same height on any road, so
         a short valley needs proportionally more of it, not less. */
      camera: {
        board: [0, extentX * 0.30 + 1.9, extentX * 0.92 + 2.6],
        top: [0, extentX * 1.16 + 3.0, 0.01],

        /* Chase is this scene's own, not the rig's. The rig's version stands
           off a followed point and looks at a fixed height above the ground
           plane, which is right for a board laid flat and wrong here: this
           road climbs and drops through nearly four world units, so a fixed
           look height puts the camera underground in the trough and pointing
           at sky on the crest. Standing off along the road and looking at the
           road ahead is the same shot done in the valley's own frame. */
        modes: {
          chase: function (ctx) {
            var ahead = clamp(carAt.position + facing * 0.22, minP, maxP);
            ctx.pos.lerp(new ctx.THREE.Vector3(
              carAt.x - facing * 3.1,
              carAt.y + 1.45,
              carAt.z + 0.9
            ), clamp(ctx.dt * 3.2, 0, 1));
            ctx.look.lerp(new ctx.THREE.Vector3(
              wx(ahead), wy(ahead) + 0.45, carAt.z
            ), clamp(ctx.dt * 4.5, 0, 1));
          }
        }
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var px = wx(sample.position);
        var py = wy(sample.position);
        var pz = wz(sample.velocity);

        car.position.set(px, py, pz);
        carAt.x = px; carAt.y = py; carAt.z = pz; carAt.position = sample.position;
        /* Facing: the car points the way it is travelling, and keeps the
           heading it had at a standstill, which is what a stopped car does. */
        if (Math.abs(sample.velocity) > 1e-5) facing = sample.velocity < 0 ? -1 : 1;
        shell.rotation.y = facing < 0 ? Math.PI : 0;
        /* Pitch is the hill's own gradient, so the car leans the way the slope
           actually runs rather than by an animation curve. The sign flips with
           the facing: the yaw has already turned the body around, so the same
           roll angle would otherwise stand the car nose-down into the hill. */
        shell.rotation.z = slope(sample.position) * facing;

        // Wheels turn with the recorded speed, not with wall-clock time: a
        // paused frame has still wheels and a fast descent has fast ones.
        wheelSpin += dt * (playing ? 1 : 0) * (sample.velocity / maxSpeed) * 9;
        for (var w = 0; w < wheels.length; w++) wheels[w].rotation.y = wheelSpin;

        // The lamp nearest the car takes the shadow duty, and gives up the
        // light it lends to the caster so the valley stays evenly lit.
        var nearest = null, nd = Infinity;
        for (var li = 0; li < lampLights.length; li++) {
          var L = lampLights[li];
          L.light.power = LAMP_LUMENS;
          var d2 = (L.x - px) * (L.x - px) + (L.y - py) * (L.y - py);
          if (d2 < nd) { nd = d2; nearest = L; }
        }
        if (nearest) {
          shadowLight.position.set(nearest.x, nearest.y, nearest.z);
          shadowLight.target.position.set(px, py + 0.1, pz);
          shadowLight.target.updateMatrixWorld();
          var share = 0.62 * clamp(1 - (Math.sqrt(nd) - 1.5) / 6.5, 0.08, 1);
          shadowLight.power = LAMP_LUMENS * share;
          nearest.light.power = LAMP_LUMENS * (1 - share);
        }

        // The flag ripples. It is the only thing in the scene allowed to move
        // on wall-clock time, because nothing about it is recorded.
        var fp = flag.geometry.attributes.position;
        for (var fv = 0; fv < fp.count; fv++) {
          var bx = flagBase[fv * 3];
          fp.setZ(fv, Math.sin(elapsed * 3.4 + bx * 7.5) * 0.06 * (bx + 0.31));
        }
        fp.needsUpdate = true;

        var beliefLabel = drawBelief(sample.index);
        drawTrail(sample);

        var step = trace.steps[sample.index] || {};
        return {
          // Chase follows along the road, so its heading is the direction of
          // travel in the XZ plane, not the car's pitch.
          // Kept for any rig that drives this scene without the chase mode
          // above: the standoff and the height are the same shot's numbers.
          follow: {
            x: px, z: pz,
            heading: facing < 0 ? Math.PI : 0,
            distance: 3.1, height: py + 1.45
          },
          step: sample.index,
          action: step.action === null || step.action === undefined ? "—" : String(step.action),
          // Position and velocity, named: "x 0.31  y -0.04" would read as a
          // pair of coordinates, and only one of these is one.
          pos: "position " + sample.position.toFixed(3) +
            "  velocity " + sample.velocity.toFixed(4),
          x: sample.position,
          y: sample.velocity,
          reward: step.reward,
          ret: running[sample.index],
          belief: beliefLabel
        };
      }
    };
  }

  V.scenes["mountain_car.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // scene's lamp power; it is not a knob to remove.
    exposure: 0.155
  };
})(window);
