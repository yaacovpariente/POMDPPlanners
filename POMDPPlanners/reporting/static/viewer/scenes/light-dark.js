/* SPDX-License-Identifier: MIT
 *
 * Light-Dark scene module.
 *
 * Builds the world from a trace's `payload.world` block and moves it from the
 * trace's recorded states and beliefs. Nothing here is invented: there is no
 * fallback episode, no hand-placed waypoint and no synthetic belief. If the
 * player hands this module no trace, it draws nothing and says so.
 *
 * The belief is drawn as the run's own particles, at the run's own weights —
 * bright at the heavy end of the cloud, dim in the tail. A Gaussian belief is
 * drawn as its recorded mean and covariance. A belief the exporter could not
 * serialise is reported as missing rather than replaced by something that
 * looks plausible.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  var COLORS = {
    lamp: 0xFFE9BE,
    rover: 0xB02C24,
    hazard: 0xD62C26,
    goal: 0x2EBE4A,
    beacon: 0x3C69DC
  };

  // Lumens. A service lamp on a post, not a stadium light: the far half of the
  // board has to stay dark, or the environment's whole point is lost.
  var BEACON_LUMENS = 780;

  // Cap on drawn particles. A trace may carry several hundred; more than this
  // is a draw cost with no extra information at this scale.
  var MAX_DRAWN_PARTICLES = 420;

  function groundCanvas(seed) {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#241F1B";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 1100; i++) {
      var r = 10 + rnd() * 95;
      g.fillStyle = "rgba(" + (54 + rnd() * 42 | 0) + "," + (45 + rnd() * 32 | 0) + "," +
        (37 + rnd() * 26 | 0) + ",0.26)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
    }
    // Pebbles: a lit cap and a dark side, which the normal map turns into relief.
    for (var p = 0; p < 1400; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.5 + rnd() * 4.5;
      g.fillStyle = "rgba(20,17,15,0.55)";
      g.beginPath(); g.arc(px + pr * 0.35, py + pr * 0.35, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(168,150,124,0.34)";
      g.beginPath(); g.arc(px, py, pr * 0.85, 0, Math.PI * 2); g.fill();
    }
    return cv;
  }

  function paintGrid(cv, cells) {
    var s = cv.width;
    var g = cv.getContext("2d");
    g.strokeStyle = "rgba(255,233,190,0.05)";
    g.lineWidth = 2;
    for (var k = 0; k <= cells; k++) {
      var p = (k / cells) * s;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
      g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
    }
  }

  /**
   * Build the Light-Dark world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind light_dark.v1.
   * @returns {Object} The scene module the player drives.
   */
  function build(core, trace) {
    var world = trace.payload.world;
    var payload = trace.payload;
    var scene = core.scene;
    var renderer = core.renderer;

    // The field spans 0..extent on both axes; the scene is shifted so its
    // centre sits at the origin, which keeps every camera mode simple.
    var extent = Math.max(1, world.grid_size - 1);
    var half = extent / 2;
    function wx(x) { return x - half; }
    function wz(y) { return y - half; }

    scene.background = new THREE.Color(0x05060A).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x06070B, 0.021);
    scene.fog.color.convertSRGBToLinear();

    // A night scene lit by lamps: the sky is deliberately weak.
    scene.add(new THREE.HemisphereLight(0x223046, 0x0A0806, 0.55));
    var moon = new THREE.DirectionalLight(0x8FA3C8, 0.35);
    moon.position.set(-6, 9, -4);
    scene.add(moon);
    core.buildNightEnvironment();

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);

    // Floor. The grit is painted first and the normal map derived from it, so
    // every pebble catches a lamp from the side the lamp is actually on. The
    // grid is painted afterwards, onto colour only: it marks scale and must
    // not emboss the rock.
    var gCanvas = groundCanvas(20260919);
    var groundNormal = V.normalMapFrom(renderer, gCanvas, 2.0);
    paintGrid(gCanvas, extent);
    var groundAlbedo = new THREE.CanvasTexture(gCanvas);
    groundAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    groundAlbedo.encoding = THREE.sRGBEncoding;

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(extent, extent),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.85, 0.85),
        roughness: 0.95, metalness: 0.0, envMapIntensity: 0.35
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Terrain carrying on past the field at exactly the same height, so there
    // is no kerb or plinth: the ground simply runs out of light. It is
    // deliberately coplanar, which would z-fight in a 16-bit depth buffer, so
    // polygonOffset biases it back at raster time instead.
    var outerAlbedo = groundAlbedo.clone();
    outerAlbedo.needsUpdate = true;
    outerAlbedo.wrapS = outerAlbedo.wrapT = THREE.RepeatWrapping;
    outerAlbedo.repeat.set(9, 9);
    var outerNormal = groundNormal.clone();
    outerNormal.needsUpdate = true;
    outerNormal.wrapS = outerNormal.wrapT = THREE.RepeatWrapping;
    outerNormal.repeat.set(9, 9);
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(90, 90),
      new THREE.MeshStandardMaterial({
        map: outerAlbedo, normalMap: outerNormal,
        normalScale: new THREE.Vector2(0.8, 0.8),
        roughness: 1.0, metalness: 0.0, envMapIntensity: 0.12,
        polygonOffset: true, polygonOffsetFactor: 4, polygonOffsetUnits: 4
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.receiveShadow = true;
    scene.add(outer);

    var rockMat = new THREE.MeshStandardMaterial({
      color: 0x4A4036, roughness: 0.95, metalness: 0.06, flatShading: true
    });
    var rockGeos = [
      new THREE.DodecahedronGeometry(0.16, 0),
      new THREE.IcosahedronGeometry(0.13, 0),
      new THREE.TetrahedronGeometry(0.19, 0)
    ];

    // Scattered rock: parallax from the raised view, and shadows that give the
    // floor depth. Deterministic, and kept off the recorded trajectory.
    var rockRnd = mulberry(90210);
    var line = payload.states;
    var placed = 0, guard = 0;
    while (placed < 30 && guard++ < 800) {
      var rx = rockRnd() * extent, ry = rockRnd() * extent;
      var tooClose = false;
      for (var q = 0; q < line.length && !tooClose; q++) {
        if (Math.hypot(rx - line[q][0], ry - line[q][1]) < 0.85) tooClose = true;
      }
      for (var q2 = 0; q2 < world.beacons.length && !tooClose; q2++) {
        if (Math.hypot(rx - world.beacons[q2][0], ry - world.beacons[q2][1]) < 0.55) tooClose = true;
      }
      if (tooClose) continue;
      var rock = new THREE.Mesh(rockGeos[placed % rockGeos.length], rockMat);
      var sc = 0.35 + rockRnd() * 0.7;
      rock.scale.set(sc, sc * (0.5 + rockRnd() * 0.4), sc);
      rock.position.set(wx(rx), 0.045 * sc, wz(ry));
      rock.rotation.set(rockRnd() * 3, rockRnd() * 3, rockRnd() * 3);
      rock.castShadow = true; rock.receiveShadow = true;
      scene.add(rock);
      placed++;
    }

    // Beacons. The lit pool on the floor is the lamp's own inverse-square
    // falloff, not a painted sticker; the cobalt ring is the true sensing
    // radius, which is much smaller than the pool and is what the agent gets.
    var senseRings = [];
    var beaconLights = [];
    var lensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(4.6, 4.0, 3.1) });
    var postMat = new THREE.MeshStandardMaterial({ color: 0x6B5436, roughness: 0.42, metalness: 0.85 });
    var baseMat = new THREE.MeshStandardMaterial({ color: 0x3A2F22, roughness: 0.6, metalness: 0.7 });
    var postGeo = new THREE.CylinderGeometry(0.055, 0.075, 1.05, 12);
    var footGeo = new THREE.CylinderGeometry(0.19, 0.24, 0.1, 16);
    var lensGeo = new THREE.SphereGeometry(0.13, 16, 12);
    var cageGeo = new THREE.TorusGeometry(0.15, 0.022, 8, 18);
    var ringGeo = new THREE.RingGeometry(
      Math.max(0.02, world.beacon_radius - 0.035), world.beacon_radius + 0.035, 56
    );

    world.beacons.forEach(function (b) {
      var x = wx(b[0]), z = wz(b[1]);
      var foot = new THREE.Mesh(footGeo, baseMat);
      foot.position.set(x, 0.05, z); foot.castShadow = true; foot.receiveShadow = true;
      scene.add(foot);
      var post = new THREE.Mesh(postGeo, postMat);
      post.position.set(x, 0.57, z); post.castShadow = true;
      scene.add(post);
      var lens = new THREE.Mesh(lensGeo, lensMat);
      lens.position.set(x, 1.12, z);
      scene.add(lens);
      var cage = new THREE.Mesh(cageGeo, postMat);
      cage.position.set(x, 1.12, z); cage.rotation.x = Math.PI / 2; cage.castShadow = true;
      scene.add(cage);

      var light = new THREE.PointLight(COLORS.lamp, 1, 9.0, 2);
      light.power = BEACON_LUMENS;
      light.position.set(x, 1.15, z);
      scene.add(light);
      beaconLights.push({ light: light, x: x, z: z });

      var ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
        color: COLORS.beacon, transparent: true, opacity: 0.22,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, 0.022, z);
      scene.add(ring);
      senseRings.push(ring);
    });

    // One shadow caster, not one per lamp. It is parked on whichever beacon is
    // nearest the rover and aimed at it, and that beacon's own light is dimmed
    // by the same amount, so the field stays evenly lit. Narrow cone plus
    // normalBias, which is the fix for acne that a big negative bias is not.
    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 13, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.4;
    shadowLight.shadow.camera.far = 14;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, 1.15, 0);
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    // Hazards. Kept see-through and crossable: in this environment a hazard is
    // a probability field the rover may drive into, not a wall. The floor
    // inside one takes the same light as the floor outside it, because
    // brightness is the observation model and a hazard does not blind you.
    var hazardRims = [];
    var hazardRadius = world.obstacle_radius;
    world.obstacles.forEach(function (o) {
      var x = wx(o[0]), z = wz(o[1]);
      var column = new THREE.Mesh(
        new THREE.CylinderGeometry(hazardRadius, hazardRadius, 1.1, 40, 1, true),
        new THREE.MeshBasicMaterial({
          color: COLORS.hazard, transparent: true, opacity: 0.06,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      column.position.set(x, 0.55, z);
      scene.add(column);

      var hazRnd = mulberry(4400 + o[0] * 131 + o[1] * 17);
      // Upthrust slabs: thin, tilted, and the thing that catches a lamp and
      // throws a shadow, which is what reads as broken ground at this range.
      for (var s = 0; s < 9; s++) {
        var sa = hazRnd() * Math.PI * 2;
        var sr = (0.18 + hazRnd() * 0.58) * hazardRadius;
        var slab = new THREE.Mesh(new THREE.BoxGeometry(
          0.10 + hazRnd() * 0.20, 0.16 + hazRnd() * 0.26, 0.05 + hazRnd() * 0.07
        ), rockMat);
        slab.position.set(x + Math.cos(sa) * sr, 0.05 + hazRnd() * 0.05, z + Math.sin(sa) * sr);
        slab.rotation.set((hazRnd() - 0.5) * 0.9, hazRnd() * Math.PI, (hazRnd() - 0.5) * 1.1);
        slab.castShadow = true; slab.receiveShadow = true;
        scene.add(slab);
      }
      for (var g = 0; g < 22; g++) {
        var ga = hazRnd() * Math.PI * 2;
        var gr = hazRnd() * 0.88 * hazardRadius;
        var grit = new THREE.Mesh(rockGeos[g % rockGeos.length], rockMat);
        var gs = 0.16 + hazRnd() * 0.34;
        grit.scale.set(gs, gs * (0.5 + hazRnd() * 0.5), gs);
        grit.position.set(x + Math.cos(ga) * gr, 0.02, z + Math.sin(ga) * gr);
        grit.rotation.set(hazRnd() * 3, hazRnd() * 3, hazRnd() * 3);
        grit.castShadow = true;
        scene.add(grit);
      }

      var rim = new THREE.Mesh(
        new THREE.RingGeometry(hazardRadius - 0.05, hazardRadius + 0.05, 56),
        new THREE.MeshBasicMaterial({
          color: COLORS.hazard, transparent: true, opacity: 0.38,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      rim.rotation.x = -Math.PI / 2;
      rim.position.set(x, 0.03, z);
      scene.add(rim);
      hazardRims.push(rim);
    });

    // Goal
    var goalGroup = new THREE.Group();
    goalGroup.position.set(wx(world.goal_state[0]), 0, wz(world.goal_state[1]));
    scene.add(goalGroup);
    var pad = new THREE.Mesh(
      new THREE.CylinderGeometry(0.62, 0.62, 0.06, 32),
      new THREE.MeshStandardMaterial({ color: 0x0E2A14, roughness: 0.7, metalness: 0.3 })
    );
    pad.position.y = 0.03;
    goalGroup.add(pad);
    var starShape = new THREE.Shape();
    for (var si = 0; si < 10; si++) {
      var ang = (Math.PI / 5) * si - Math.PI / 2;
      var rad = si % 2 === 0 ? 0.42 : 0.18;
      var sx = Math.cos(ang) * rad, sy = Math.sin(ang) * rad;
      if (si === 0) starShape.moveTo(sx, sy); else starShape.lineTo(sx, sy);
    }
    starShape.closePath();
    var star = new THREE.Mesh(
      new THREE.ExtrudeGeometry(starShape, {
        depth: 0.09, bevelEnabled: true, bevelSize: 0.02,
        bevelThickness: 0.02, bevelSegments: 1
      }),
      new THREE.MeshStandardMaterial({
        color: 0x1E7A33, roughness: 0.5, metalness: 0.2, emissive: 0x06220C
      })
    );
    star.rotation.x = -Math.PI / 2;
    star.position.y = 0.1;
    goalGroup.add(star);
    var goalLight = new THREE.PointLight(COLORS.goal, 1, 4.5, 2);
    goalLight.power = 38;
    goalLight.position.y = 0.7;
    goalGroup.add(goalLight);

    var startPad = new THREE.Mesh(
      new THREE.RingGeometry(0.26, 0.36, 32),
      new THREE.MeshBasicMaterial({
        color: 0xDE342E, transparent: true, opacity: 0.8, side: THREE.DoubleSide
      })
    );
    startPad.rotation.x = -Math.PI / 2;
    startPad.position.set(wx(world.start_state[0]), 0.02, wz(world.start_state[1]));
    scene.add(startPad);

    // Rover
    var rover = new THREE.Group();
    scene.add(rover);
    var hull = new THREE.Group();
    rover.add(hull);
    var paintMat = new THREE.MeshStandardMaterial({ color: COLORS.rover, roughness: 0.46, metalness: 0.38 });
    var darkMetal = new THREE.MeshStandardMaterial({ color: 0x2A2522, roughness: 0.55, metalness: 0.75 });
    var chassis = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.17, 0.42), paintMat);
    chassis.position.y = 0.17; chassis.castShadow = true;
    hull.add(chassis);
    var deck = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.09, 0.34),
      new THREE.MeshStandardMaterial({ color: 0x8E241E, roughness: 0.5, metalness: 0.4 }));
    deck.position.set(-0.04, 0.30, 0); deck.castShadow = true;
    hull.add(deck);
    var panel = new THREE.Mesh(new THREE.BoxGeometry(0.30, 0.02, 0.30),
      new THREE.MeshStandardMaterial({ color: 0x1B2A46, roughness: 0.18, metalness: 0.65 }));
    panel.position.set(-0.10, 0.36, 0); panel.rotation.z = -0.06; panel.castShadow = true;
    hull.add(panel);
    var cabin = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.13, 0.26),
      new THREE.MeshStandardMaterial({
        color: 0x24406E, roughness: 0.12, metalness: 0.3,
        emissive: 0x13203A, emissiveIntensity: 0.8
      }));
    cabin.position.set(0.06, 0.40, 0); cabin.castShadow = true;
    hull.add(cabin);
    var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.26, 8), darkMetal);
    mast.position.set(-0.16, 0.48, 0);
    hull.add(mast);
    var sensorHead = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.06, 0.05), darkMetal);
    sensorHead.position.set(-0.16, 0.62, 0); sensorHead.castShadow = true;
    hull.add(sensorHead);

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

    var lampMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.6, 3.3, 2.7) });
    [[0.31, 0.13], [0.31, -0.13]].forEach(function (h) {
      var m = new THREE.Mesh(new THREE.SphereGeometry(0.042, 10, 10), lampMat);
      m.position.set(h[0], 0.23, h[1]);
      hull.add(m);
    });

    var blob = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 0.8),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.4, depthWrite: false
      }));
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.008;
    rover.add(blob);

    // The headlight is what makes the dark readable from the chase camera.
    var headlight = new THREE.SpotLight(0xFFF1D6, 1, 9, 0.5, 0.6, 2);
    headlight.power = 900;
    headlight.castShadow = true;
    headlight.shadow.mapSize.set(1024, 1024);
    headlight.shadow.camera.near = 0.2;
    headlight.shadow.camera.far = 9;
    headlight.shadow.normalBias = 0.03;
    headlight.position.set(0.3, 0.26, 0);
    var headTarget = new THREE.Object3D();
    headTarget.position.set(4, -0.18, 0);
    rover.add(headlight);
    rover.add(headTarget);
    headlight.target = headTarget;

    var beamCone = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 1.15, 3.1, 20, 1, true),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0xFFE9BE, transparent: true, opacity: 0.028,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    beamCone.rotation.z = Math.PI / 2 + 0.06;
    beamCone.position.set(1.75, 0.22, 0);
    rover.add(beamCone);

    // Belief cloud. One Points object, sized for the largest step's cloud and
    // then draw-ranged per frame, so a step with fewer particles costs nothing.
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
      size: 0.23, map: partTex, vertexColors: true, transparent: true,
      opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false,
      sizeAttenuation: true
    }));
    scene.add(particles);

    // A Gaussian belief has no particles to draw, so it is sampled for display
    // only — with a fixed seed, and from the run's own mean and covariance, so
    // what is drawn is that belief and nothing else.
    var gaussianRnd = mulberry(20260920);
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

    var heading = 0;
    var wheelSpin = 0;
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

    /* The belief is drawn from the kind core serialised it as, never from the
       belief class that produced it. Every particle belief in the package —
       weighted, unweighted, incremental, vectorized — arrives as "particles",
       so this widget draws them once rather than once per class. */

    /** Write one particle into the buffer, shaded by its share of the mass. */
    function writeParticle(slot, point, relativeWeight) {
      pPos[slot * 3] = wx(point[0]);
      pPos[slot * 3 + 1] = 0.2 + (slot % 5) * 0.02;
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

    /** Draw a cloud of particles with their normalized weights. */
    function drawParticles(belief) {
      var count = Math.min(belief.particles.length, maxParticles);
      if (count && !Array.isArray(belief.particles[0])) {
        // A particle that is not a coordinate pair — a discrete state label,
        // say. This scene plots positions, so it says so rather than plotting
        // the first character of a string.
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
      var label = belief.num_particles + (belief.weighted === false ? " uniform particles" : " particles");
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

    /* A Gaussian carries no particles, so it is sampled for display only —
       with a fixed seed, and from the run's own mean and covariance, so what
       is drawn is that belief and nothing else. */
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
       in proportion to its weight. Collapsing it to one cloud would place the
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
      if (!belief) { pGeo.setDrawRange(0, 0); return "\u2014"; }

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
        var s = payload.states[i];
        trailPos[i * 3] = wx(s[0]);
        trailPos[i * 3 + 1] = 0.05;
        trailPos[i * 3 + 2] = wz(s[1]);
        var age = count > 1 ? i / (count - 1) : 1;
        trailCol[i * 3] = lerp(0.56, 0.91, age);
        trailCol[i * 3 + 1] = lerp(0.14, 0.31, age);
        trailCol[i * 3 + 2] = lerp(0.11, 0.24, age);
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
      trailGeo.setDrawRange(0, Math.max(0, count));
    }

    return {
      steps: payload.states.length,

      /* Framing scales with the world, because a trace decides how big the
         board is: the fixed distance that frames an 11x11 grid leaves a 5x5
         one as a smudge in the middle of the canvas. The constant term is the
         margin for the props, which do not scale with the grid — a lamp post
         is the same height on any board, so a small board needs proportionally
         more headroom, not less. */
      camera: {
        board: [0, extent * 1.29 + 2.15, extent * 1.18 + 1.95],
        top: [0, extent * 1.60 + 2.4, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var px = wx(sample.x), pz = wz(sample.y);

        // Heading from the direction of travel, so the body turns into the
        // corner rather than snapping between the four step directions.
        var next = payload.states[Math.min(sample.index + 1, payload.states.length - 1)];
        var current = payload.states[sample.index];
        var dx = next[0] - current[0], dy = next[1] - current[1];
        if (dx !== 0 || dy !== 0) {
          var target = Math.atan2(dy, dx);
          var diff = ((target - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          heading += diff * clamp(dt * 7, 0, 1);
        }
        rover.position.set(px, 0, pz);
        rover.rotation.y = -heading;

        var speed = playing ? 1 : 0;
        wheelSpin += dt * speed * 6.2;
        for (var w = 0; w < wheels.length; w++) wheels[w].rotation.y = wheelSpin;

        // Move the one shadow caster to the nearest lamp and take that much
        // light back off it, so the shadow follows without over-lighting.
        var nearest = null, nd = Infinity;
        for (var bl = 0; bl < beaconLights.length; bl++) {
          var L = beaconLights[bl];
          L.light.power = BEACON_LUMENS;
          var d2 = (L.x - px) * (L.x - px) + (L.z - pz) * (L.z - pz);
          if (d2 < nd) { nd = d2; nearest = L; }
        }
        if (nearest) {
          shadowLight.position.set(nearest.x, 1.15, nearest.z);
          shadowLight.target.position.set(px, 0.1, pz);
          shadowLight.target.updateMatrixWorld();
          var share = 0.62 * clamp(1 - (Math.sqrt(nd) - 1.5) / 6.0, 0.08, 1);
          shadowLight.power = BEACON_LUMENS * share;
          nearest.light.power = BEACON_LUMENS * (1 - share);
        }

        star.rotation.z = elapsed * 0.5;
        var pulse = 0.62 + Math.sin(elapsed * 2.1) * 0.22;
        for (var h = 0; h < hazardRims.length; h++) hazardRims[h].material.opacity = pulse;

        var beliefLabel = drawBelief(sample.index);
        drawTrail(sample);

        var step = trace.steps[sample.index] || {};
        return {
          follow: { x: px, z: pz, heading: heading },
          step: sample.index,
          action: step.action === null || step.action === undefined ? "—" : String(step.action),
          x: sample.x,
          y: sample.y,
          reward: step.reward,
          ret: running[sample.index],
          belief: beliefLabel
        };
      },

      setSenseRingsVisible: function (visible) {
        for (var i = 0; i < senseRings.length; i++) senseRings[i].visible = visible;
      }
    };
  }

  V.scenes["light_dark.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // scene's lamp power; it is not a knob to remove.
    exposure: 0.155
  };
})(window);
