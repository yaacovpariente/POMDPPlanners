/* SPDX-License-Identifier: MIT
 *
 * Safety Ant Velocity scene module.
 *
 * Builds the world from a trace's `payload.world` block and moves it from the
 * trace's recorded states, forces and beliefs. Nothing here is invented: there
 * is no fallback episode, no re-simulated physics and no synthetic belief. If
 * the player hands this module no trace, it draws nothing and says so.
 *
 * The one design decision worth keeping from the approved preview: the
 * constraint in this environment is on *speed*, not on position, so the safe
 * set is a disc in velocity space. It is drawn on the ground around the body
 * at one metre per m/s, with the velocity arrow inside it, and it travels with
 * the body because velocity space does. You watch the arrow reach the ring
 * rather than watching a number.
 *
 * The body pose is illustrative and the page says so. This environment's state
 * is four numbers — x, y, vx, vy. There are no joints, so the gait here is
 * driven by the recorded speed and is decoration.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  var COLORS = {
    lamp: 0xFFF1DC,
    ant: 0x1261C3,     // ant_sprite body base (18, 97, 195)
    antHi: 0x416C99,   // leg highlight (65, 108, 153)
    antDark: 0x091827  // leg core (9, 24, 39)
  };

  // Lumens. A service flood on an eight-metre mast, not a stadium rig, so the
  // pad falls off into the dark at its edges and the ant is the brightest
  // thing in frame.
  var FLOOD_LUMENS = 5000;
  var FLOOD_HEIGHT = 7.8;

  // One world unit per metre of position, and one world unit per m/s of
  // velocity. That equality is what lets the safe ring be a real circle on the
  // floor: at this scale the ring at 2.0 m/s has radius 2.
  var VSCALE = 1.0;

  // Cap on drawn belief particles. A trace may carry several hundred; past
  // this it is draw cost with no extra information at this scale.
  var MAX_DRAWN_PARTICLES = 320;

  /* The pad is poured concrete, built the way concrete_texture() in
     safety_ant_assets.py builds it: a mid-grey base, octaves of blotch, pits
     with a lit lip, hairline cracks and mineral grain. */
  function groundCanvas(seed) {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);

    g.fillStyle = "#7E7D77";   // the assets' 173-grey, darkened for night
    g.fillRect(0, 0, s, s);

    [[13, 0.05], [38, 0.035], [92, 0.022]].forEach(function (oct) {
      for (var i = 0; i < 900; i++) {
        var r = oct[0] * (0.4 + rnd() * 0.9);
        var v = rnd() > 0.5 ? 255 : 0;
        g.fillStyle = "rgba(" + v + "," + v + "," + v + "," + (oct[1] * 0.4).toFixed(3) + ")";
        g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
      }
    });

    // Pits: a dark hole with a pale lip, which the normal map turns into relief.
    for (var p = 0; p < 2600; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.0 + rnd() * 2.2;
      g.fillStyle = "rgba(60,60,57,0.38)";
      g.beginPath(); g.arc(px, py, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(186,184,176,0.20)";
      g.beginPath(); g.arc(px - pr * 0.25, py - pr * 0.3, pr * 0.7, 0, Math.PI * 2); g.fill();
    }

    g.lineWidth = 1.2;
    for (var c = 0; c < 70; c++) {
      var x = rnd() * s, y = rnd() * s;
      g.strokeStyle = "rgba(58,58,55,0.55)";
      g.beginPath(); g.moveTo(x, y);
      for (var k = 0; k < 6 + rnd() * 9; k++) {
        x += (rnd() - 0.5) * 42; y += 8 + rnd() * 34;
        g.lineTo(x, y);
      }
      g.stroke();
    }

    for (var j = 0; j < 4200; j++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(214,212,203,0.10)" : "rgba(0,0,0,0.16)";
      g.fillRect(rnd() * s, rnd() * s, 2, 2);
    }
    return cv;
  }

  /* Expansion joints, the way a poured pad is cast. They give the eye a scale
     reference while the ant travels. Painted onto colour only, so they do not
     emboss the concrete. */
  function paintJoints(cv, divisions) {
    var s = cv.width;
    var g = cv.getContext("2d");
    g.strokeStyle = "rgba(24,26,26,0.40)";
    g.lineWidth = 3;
    for (var k = 0; k <= divisions; k++) {
      var p = (k / divisions) * s;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.stroke();
      g.beginPath(); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
    }
  }

  /** A text sprite, shrunk to fit rather than clipped. */
  function labelSprite(text, colour, widthWorld) {
    var cv = document.createElement("canvas");
    cv.width = 640; cv.height = 128;
    var g = cv.getContext("2d");
    g.font = "600 54px system-ui, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillStyle = colour;
    var w = g.measureText(text).width;
    if (w > 600) g.font = "600 " + Math.floor(54 * 600 / w) + "px system-ui, sans-serif";
    g.fillText(text, 320, 68);
    var tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: tex, color: new THREE.Color(4.2, 4.2, 4.2),
      transparent: true, depthWrite: false, opacity: 0.95
    }));
    sprite.scale.set(widthWorld, widthWorld / 5, 1);
    sprite.userData = { canvas: cv, texture: tex, context: g, text: text };
    return sprite;
  }

  /** Repaint a label sprite in place, so the readout can follow the episode. */
  function repaintLabel(sprite, text, colour) {
    var d = sprite.userData;
    if (d.text === text && d.colour === colour) return;
    d.text = text; d.colour = colour;
    var g = d.context;
    g.clearRect(0, 0, 640, 128);
    g.font = "600 54px system-ui, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillStyle = colour;
    var w = g.measureText(text).width;
    if (w > 600) g.font = "600 " + Math.floor(54 * 600 / w) + "px system-ui, sans-serif";
    g.fillText(text, 320, 68);
    d.texture.needsUpdate = true;
  }

  function makeArrow(emissive, radius) {
    var group = new THREE.Group();
    var mat = new THREE.MeshBasicMaterial({ color: emissive });
    var shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 12), mat);
    shaft.position.y = 0.5;
    group.add(shaft);
    var head = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.6, radius * 6.5, 14), mat);
    group.add(head);
    group.userData = { shaft: shaft, head: head, radius: radius };
    return group;
  }

  /** Lay an arrow modelled up +Y flat along (dx, dz), starting at (x, y, z). */
  function aimArrow(group, x, y, z, dx, dz) {
    var len = Math.hypot(dx, dz);
    var d = group.userData;
    group.visible = len > 0.02;
    if (!group.visible) return;
    var body = Math.max(0.001, len - d.radius * 6.5);
    d.shaft.scale.y = body;
    d.shaft.position.y = body / 2;
    d.head.position.y = body + d.radius * 3.25;
    group.position.set(x, y, z);
    group.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(dx / len, 0, dz / len)
    );
  }

  function segment(from, to, r0, r1, mat) {
    var a = new THREE.Vector3().fromArray(from);
    var b = new THREE.Vector3().fromArray(to);
    var dir = new THREE.Vector3().subVectors(b, a);
    var len = dir.length();
    var mesh = new THREE.Mesh(new THREE.CylinderGeometry(r0, r1, len, 9), mat);
    mesh.position.copy(a).addScaledVector(dir, 0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    mesh.castShadow = true;
    return mesh;
  }

  /**
   * Build the Safety Ant Velocity world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json, payload kind safety_ant_velocity.v1.
   * @returns {Object} The scene module the player drives.
   */
  // The scene is one world built in one pass; splitting it into helpers that
  // each take a dozen already-derived locals would be longer and harder to read.
  // eslint-disable-next-line max-statements
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var states = payload.states;
    var scene = core.scene;
    var renderer = core.renderer;

    if (!states || !states.length) throw new Error("this trace has no recorded states");

    var SAFE_V = world.safe_velocity_threshold;
    var CRIT_V = world.critical_velocity_threshold;

    /* Framing. This world is unbounded — there is no grid whose middle the
       camera can sit over — so the scene is shifted to put the recorded
       trajectory's own centre at the origin. The board and top cameras then
       frame the episode that actually happened rather than a fixed window the
       ant may have walked straight out of. */
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    states.forEach(function (s) {
      if (s[0] < minX) minX = s[0];
      if (s[0] > maxX) maxX = s[0];
      if (s[1] < minY) minY = s[1];
      if (s[1] > maxY) maxY = s[1];
    });
    var centreX = (minX + maxX) / 2, centreY = (minY + maxY) / 2;
    // Half the framed region: the path, plus the critical ring, which travels
    // with the body and must not be cropped at either end of the run.
    var reach = Math.max(maxX - minX, maxY - minY) / 2 + CRIT_V * VSCALE + 1.2;

    // World +y maps to scene +z with no flip, which is the convention the
    // heading below depends on: rotation.y = -heading then points the body's
    // local +X along the recorded velocity.
    function wx(x) { return x - centreX; }
    function wz(y) { return y - centreY; }

    scene.background = new THREE.Color(0x05060A).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x070A0E, 0.017);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x24344C, 0x090808, 0.62));
    var moon = new THREE.DirectionalLight(0x8FA3C8, 0.25);
    moon.position.set(-8, 11, -5);
    scene.add(moon);
    core.buildNightEnvironment([
      [0.0, "#080E1C"], [0.45, "#121826"], [0.55, "#242A2E"], [1.0, "#060606"]
    ]);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);
    var glowTex = V.radialTexture(0.9, 0.3);

    // ---- pad -------------------------------------------------------------
    // Sized to hold the whole episode plus its rings, and kept inside the
    // core camera's 90-unit far plane.
    var PAD = clamp(Math.ceil(reach * 2 + 6), 20, 46);
    var gCanvas = groundCanvas(20260905);   // the seed the Python texture uses
    var groundNormal = V.normalMapFrom(renderer, gCanvas, 2.1);
    paintJoints(gCanvas, 3);
    var groundAlbedo = new THREE.CanvasTexture(gCanvas);
    groundAlbedo.wrapS = groundAlbedo.wrapT = THREE.RepeatWrapping;
    groundAlbedo.repeat.set(5, 5);
    groundAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    groundAlbedo.encoding = THREE.sRGBEncoding;   // colour data, not linear data
    groundNormal.wrapS = groundNormal.wrapT = THREE.RepeatWrapping;
    groundNormal.repeat.set(5, 5);

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(PAD, PAD),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.9, 0.9),
        roughness: 0.92, metalness: 0.0, envMapIntensity: 0.3
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    /* Ground carrying on past the pad. Without it the pad is a lit slab
       floating in a void, which is the single biggest tell that a night render
       is a render. The floods barely reach it, so it stays a suggestion. */
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(88, 88),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal, color: 0x2A2A26,
        roughness: 1.0, metalness: 0.0, envMapIntensity: 0.1
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.7;
    outer.receiveShadow = true;
    scene.add(outer);

    // The side of the pad, so the drop to the surrounding ground reads as a
    // structure rather than a seam.
    var plinth = new THREE.Mesh(
      new THREE.BoxGeometry(PAD + 0.9, 0.78, PAD + 0.9),
      new THREE.MeshStandardMaterial({ color: 0x4A4945, roughness: 0.9, metalness: 0.06 })
    );
    plinth.position.y = -0.56;
    plinth.castShadow = true;
    plinth.receiveShadow = true;
    scene.add(plinth);

    // ---- floodlights -----------------------------------------------------
    var floodLights = [];
    var mastMat = new THREE.MeshStandardMaterial({ color: 0x555049, roughness: 0.45, metalness: 0.8 });
    var housingMat = new THREE.MeshStandardMaterial({ color: 0x2E2B27, roughness: 0.5, metalness: 0.7 });
    var lensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(5.2, 4.6, 3.6) });
    var mastRing = PAD / 2 - 1.2;
    [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, -1.28], [0, 1.28]].forEach(function (unit) {
      var mx = unit[0] * mastRing, mz = unit[1] * mastRing;
      var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.16, 8.0, 12), mastMat);
      mast.position.set(mx, 4.0, mz);
      mast.castShadow = true;
      scene.add(mast);

      var foot = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.44, 0.24, 14), housingMat);
      foot.position.set(mx, 0.12, mz);
      foot.castShadow = true; foot.receiveShadow = true;
      scene.add(foot);

      // The head is tilted in towards the middle of the pad.
      var ang = Math.atan2(-mz, -mx);
      var head = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.42, 0.28), housingMat);
      head.position.set(mx + Math.cos(ang) * 0.22, FLOOD_HEIGHT + 0.1, mz + Math.sin(ang) * 0.22);
      head.rotation.y = -ang;
      head.rotation.z = 0.5;
      head.castShadow = true;
      scene.add(head);

      var lens = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.32), lensMat);
      lens.position.set(mx + Math.cos(ang) * 0.42, FLOOD_HEIGHT - 0.02, mz + Math.sin(ang) * 0.42);
      lens.lookAt(0, 0, 0);
      scene.add(lens);

      var flare = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xFFE6C0, transparent: true, opacity: 0.35,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      flare.scale.set(1.5, 1.5, 1);
      flare.position.set(mx + Math.cos(ang) * 0.42, FLOOD_HEIGHT, mz + Math.sin(ang) * 0.42);
      scene.add(flare);

      var light = new THREE.PointLight(COLORS.lamp, 1, PAD + 4, 2);
      light.power = FLOOD_LUMENS;
      light.position.set(mx, FLOOD_HEIGHT, mz);
      scene.add(light);
      floodLights.push({ light: light, x: mx, z: mz });
    });

    /* One shadow-casting lamp, not six. It is parked on whichever flood is
       nearest the ant and aimed at it, and that flood's own light is dimmed by
       the same amount, so the pad stays evenly lit while the ant always throws
       a shadow that points the right way. Six shadow-casting point lights
       would be thirty-six shadow passes a frame; this is one. Narrow cone plus
       normalBias, which is the fix for acne that a big negative bias is not. */
    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 34, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 1.2;
    shadowLight.shadow.camera.far = 34;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, FLOOD_HEIGHT, 0);
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    // ---- velocity rings ---------------------------------------------------
    /* The constraint is on speed, so the safe set is a disc in velocity space.
       It is drawn around the body at one metre per m/s: the tip of the green
       arrow is the velocity, and the only question the environment poses is
       which ring that tip is inside. The rings travel with the ant because
       velocity space does. */
    var ringGroup = new THREE.Group();
    scene.add(ringGroup);

    function flatRing(radius, width, colour, opacity) {
      var mesh = new THREE.Mesh(
        new THREE.RingGeometry(radius - width, radius + width, 128),
        new THREE.MeshBasicMaterial({
          color: colour, transparent: true, opacity: opacity,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      mesh.rotation.x = -Math.PI / 2;
      return mesh;
    }

    // Dashed, like the GIF renderer's dashed orange threshold circle.
    var safeRing = new THREE.Group();
    for (var d = 0; d < 36; d++) {
      var a0 = (d / 36) * Math.PI * 2;
      var seg = new THREE.Mesh(
        new THREE.RingGeometry(SAFE_V * VSCALE - 0.05, SAFE_V * VSCALE + 0.05, 6, 1, a0, 0.11),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(5.6, 3.4, 0.25), transparent: true, opacity: 0.95,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      seg.rotation.x = -Math.PI / 2;
      safeRing.add(seg);
    }
    safeRing.position.y = 0.035;
    ringGroup.add(safeRing);

    var critRing = flatRing(CRIT_V * VSCALE, 0.055, new THREE.Color(4.6, 0.75, 0.7), 0.85);
    critRing.position.y = 0.03;
    ringGroup.add(critRing);

    // A faint disc inside the safe ring: the region the agent is paid to fill.
    var safeDisc = new THREE.Mesh(
      new THREE.CircleGeometry(SAFE_V * VSCALE, 72),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(1.6, 1.0, 0.12), transparent: true, opacity: 0.028,
        blending: THREE.AdditiveBlending, depthWrite: false
      })
    );
    safeDisc.rotation.x = -Math.PI / 2;
    safeDisc.position.y = 0.02;
    ringGroup.add(safeDisc);

    // Labels, so neither ring is a silent circle. Both read the trace, so a
    // run with a different threshold is labelled with its own numbers.
    var safeLabel = labelSprite("SAFE " + SAFE_V.toFixed(1) + " m/s", "#F59E0B", 1.5);
    safeLabel.position.set(
      -0.71 * (SAFE_V * VSCALE + 0.55), 0.32, -0.71 * (SAFE_V * VSCALE + 0.55)
    );
    ringGroup.add(safeLabel);
    var critLabel = labelSprite("TERMINAL " + CRIT_V.toFixed(1) + " m/s", "#FF6F63", 1.8);
    critLabel.position.set(
      -0.71 * (CRIT_V * VSCALE + 0.55), 0.32, 0.71 * (CRIT_V * VSCALE + 0.55)
    );
    ringGroup.add(critLabel);

    /* The set the force direction was drawn from: a circle whose radius is the
       force magnitude the action commanded. Only the magnitude is the action —
       the environment draws the direction uniformly on (-pi, pi] — and this is
       what that uniformity looks like beside the one draw that happened. */
    var forceCircle = flatRing(1.0, 0.03, new THREE.Color(2.0, 0.7, 2.5), 0.7);
    forceCircle.position.y = 0.045;
    ringGroup.add(forceCircle);

    // Emissive colours deliberately exceed 1.0: the scene buffer is half-float,
    // so these bloom in the composite while painted surfaces do not.
    var velArrow = makeArrow(new THREE.Color(1.5, 7.2, 3.6), 0.055);
    scene.add(velArrow);
    var forceArrow = makeArrow(new THREE.Color(2.4, 0.8, 3.0), 0.042);
    scene.add(forceArrow);

    // ---- ant --------------------------------------------------------------
    /* Three glossy blue segments and six jointed legs, which is what
       ant_sprite() in safety_ant_assets.py draws, at the sprite's own colours.
       The body is modelled facing local +X. */
    var ant = new THREE.Group();
    ant.scale.setScalar(1.18);
    scene.add(ant);
    var body = new THREE.Group();
    ant.add(body);

    var shellMat = new THREE.MeshStandardMaterial({
      color: COLORS.ant, roughness: 0.22, metalness: 0.45, envMapIntensity: 1.1
    });
    var jointMat = new THREE.MeshStandardMaterial({
      color: COLORS.antDark, roughness: 0.5, metalness: 0.6
    });
    var limbMat = new THREE.MeshStandardMaterial({
      color: COLORS.antHi, roughness: 0.42, metalness: 0.55
    });

    // Abdomen, thorax, head — the sprite's three ellipses, in profile order.
    var abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.30, 26, 20), shellMat);
    abdomen.scale.set(1.0, 0.74, 0.82);
    abdomen.position.set(-0.40, 0.30, 0);
    abdomen.castShadow = true;
    body.add(abdomen);

    var thorax = new THREE.Mesh(new THREE.SphereGeometry(0.17, 22, 16), shellMat);
    thorax.scale.set(1.15, 0.85, 0.95);
    thorax.position.set(-0.02, 0.28, 0);
    thorax.castShadow = true;
    body.add(thorax);

    var waist = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.16, 10), jointMat);
    waist.rotation.z = Math.PI / 2;
    waist.position.set(-0.20, 0.28, 0);
    body.add(waist);

    var headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.19, 22, 16), shellMat);
    headMesh.scale.set(0.95, 0.88, 0.95);
    headMesh.position.set(0.30, 0.29, 0);
    headMesh.castShadow = true;
    body.add(headMesh);

    var neck = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.13, 10), jointMat);
    neck.rotation.z = Math.PI / 2;
    neck.position.set(0.15, 0.28, 0);
    body.add(neck);

    // Mandibles and the two antennae the sprite draws.
    [-1, 1].forEach(function (side) {
      var mandible = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.008, 0.16, 8), jointMat);
      mandible.position.set(0.46, 0.25, side * 0.06);
      mandible.rotation.z = -Math.PI / 2.4;
      mandible.rotation.y = side * 0.4;
      body.add(mandible);

      var lower = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.012, 0.22, 8), jointMat);
      lower.position.set(0.40, 0.42, side * 0.08);
      lower.rotation.z = -0.7;
      lower.rotation.x = -side * 0.3;
      body.add(lower);
      var upper = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.009, 0.20, 8), limbMat);
      upper.position.set(0.56, 0.52, side * 0.15);
      upper.rotation.z = -1.35;
      upper.rotation.x = -side * 0.4;
      body.add(upper);

      // Compound eyes: dark glass with a hot specular the bloom picks up.
      var eye = new THREE.Mesh(
        new THREE.SphereGeometry(0.055, 14, 12),
        new THREE.MeshStandardMaterial({
          color: 0x0A1526, roughness: 0.05, metalness: 0.2,
          emissive: 0x143A66, emissiveIntensity: 0.6
        })
      );
      eye.position.set(0.40, 0.33, side * 0.11);
      body.add(eye);
    });

    /* Six legs in the sprite's three pairs — front reaching forward, middle
       out, rear trailing. Each is a small group so the walk cycle can swing
       it; that cycle is decoration, because this environment has no joint
       state, only [x, y, vx, vy]. */
    var legs = [];
    [[0.20, 0.62], [0.00, 0.02], [-0.24, -0.60]].forEach(function (set, row) {
      [-1, 1].forEach(function (side) {
        // The leg is built with "outward" along +Z, then the hip is yawed so
        // +Z points out of the correct flank and a little fore or aft.
        var hip = new THREE.Group();
        hip.position.set(set[0], 0.26, side * 0.13);
        hip.rotation.y = (side > 0 ? 0 : Math.PI) + set[1] * side;
        body.add(hip);

        hip.add(segment([0, 0, 0], [0, 0.15, 0.27], 0.035, 0.026, jointMat));

        var knee = new THREE.Group();
        knee.position.set(0, 0.15, 0.27);
        hip.add(knee);
        knee.add(new THREE.Mesh(new THREE.SphereGeometry(0.036, 10, 8), limbMat));
        knee.add(segment([0, 0, 0], [0, -0.41, 0.17], 0.024, 0.011, limbMat));
        var foot = new THREE.Mesh(new THREE.SphereGeometry(0.022, 8, 6), jointMat);
        foot.position.set(0, -0.41, 0.17);
        knee.add(foot);

        legs.push({
          hip: hip, knee: knee, yaw0: hip.rotation.y, side: side,
          // Alternating tripod: the two sets that carry the body are half a
          // cycle apart, which is how a six-legged walk actually reads.
          phase: ((row + (side > 0 ? 1 : 0)) % 2) * Math.PI + row * 0.4
        });
      });
    });

    // Contact shadow: the shadow map handles the cast shadow, this is the dark
    // patch directly under the body that keeps the ant from looking pasted on.
    var blob = new THREE.Mesh(
      new THREE.PlaneGeometry(1.9, 1.5),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.45, depthWrite: false
      })
    );
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.012;
    ant.add(blob);

    // A small lamp on the ant, so it is never a silhouette from the chase view.
    var antGlow = new THREE.PointLight(0x9CC6FF, 1, 4.0, 2);
    antGlow.power = 27;
    antGlow.position.set(0.2, 0.55, 0);
    ant.add(antGlow);

    /* The speed readout, in the scene rather than in the page's HUD: the
       shared HUD has a slot for a position and none for a speed, and speed is
       the only quantity this environment rewards or penalises. */
    var speedLabel = labelSprite("0.00 m/s", "#62E19D", 1.9);
    speedLabel.position.set(0, 1.15, 0);
    ant.add(speedLabel);

    // ---- dust -------------------------------------------------------------
    var MOTES = 240;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(1357);
    for (var m = 0; m < MOTES; m++) {
      motePos[m * 3] = (moteSeed() - 0.5) * PAD;
      motePos[m * 3 + 1] = 0.15 + moteSeed() * 4.0;
      motePos[m * 3 + 2] = (moteSeed() - 0.5) * PAD;
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    var motes = new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.04, map: partTex, color: 0xFFE3BC, transparent: true, opacity: 0.35,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    }));
    motes.frustumCulled = false;
    scene.add(motes);

    // ---- belief -----------------------------------------------------------
    /* The state has two halves with different noise, so the belief is drawn in
       two places: on the ground where the position particles are, and at the
       velocity arrow's tip where the velocity particles are. Both come from
       the belief the run actually held — a cloud regenerated from the true
       state would look convincing and say nothing.

       The velocity cloud is coloured by each particle's own speed against the
       two thresholds, so a cloud straddling the orange ring reads as what it
       is: the agent cannot yet tell whether it is being penalised. */
    function cloudPoints(size) {
      var geo = new THREE.BufferGeometry();
      var pos = new Float32Array(MAX_DRAWN_PARTICLES * 3);
      var col = new Float32Array(MAX_DRAWN_PARTICLES * 3);
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      geo.setDrawRange(0, 0);
      var points = new THREE.Points(geo, new THREE.PointsMaterial({
        size: size, map: partTex, vertexColors: true, transparent: true,
        opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false,
        sizeAttenuation: true
      }));
      points.frustumCulled = false;
      scene.add(points);
      return { geo: geo, pos: pos, col: col, points: points };
    }
    var velCloud = cloudPoints(0.17);
    var posCloud = cloudPoints(0.075);

    // A Gaussian belief carries no particles, so it is sampled for display
    // only — with a fixed seed, and from the run's own mean and covariance, so
    // what is drawn is that belief and nothing else.
    var gaussianRnd = mulberry(20260920);
    var gaussianDraws = [];
    for (var gi = 0; gi < MAX_DRAWN_PARTICLES; gi++) {
      var u1 = Math.max(1e-6, gaussianRnd()), u2 = gaussianRnd();
      var mag = Math.sqrt(-2 * Math.log(u1));
      gaussianDraws.push([mag * Math.cos(2 * Math.PI * u2), mag * Math.sin(2 * Math.PI * u2)]);
    }

    // ---- trail ------------------------------------------------------------
    /* A ribbon, not a line: WebGL draws every line one pixel wide whatever
       linewidth says, and a one-pixel trail disappears under the bloom. */
    var TRAIL_W = 0.075;
    var trailPos = new Float32Array(states.length * 6);
    var trailCol = new Float32Array(states.length * 6);
    var trailIdx = new Uint16Array(Math.max(1, (states.length - 1) * 6));
    for (var q = 0; q < states.length - 1; q++) {
      var v0 = q * 2;
      trailIdx[q * 6] = v0; trailIdx[q * 6 + 1] = v0 + 1; trailIdx[q * 6 + 2] = v0 + 2;
      trailIdx[q * 6 + 3] = v0 + 1; trailIdx[q * 6 + 4] = v0 + 3; trailIdx[q * 6 + 5] = v0 + 2;
    }
    var trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3));
    trailGeo.setIndex(new THREE.BufferAttribute(trailIdx, 1));
    trailGeo.setDrawRange(0, 0);
    var trail = new THREE.Mesh(trailGeo, new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.95,
      blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
    }));
    trail.frustumCulled = false;
    scene.add(trail);

    core.linearize();

    // Running discounted return, on the same convention History uses.
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
      var n = states.length;
      var i0 = Math.floor(clamp(t, 0, n - 1));
      var i1 = Math.min(i0 + 1, n - 1);
      var f = clamp(t - i0, 0, 1);
      var a = states[i0], b = states[i1];
      return {
        index: i0,
        x: lerp(a[0], b[0], f), y: lerp(a[1], b[1], f),
        vx: lerp(a[2], b[2], f), vy: lerp(a[3], b[3], f)
      };
    }

    function writeCloud(cloud, slot, sceneX, sceneZ, height, colour) {
      cloud.pos[slot * 3] = sceneX;
      cloud.pos[slot * 3 + 1] = height;
      cloud.pos[slot * 3 + 2] = sceneZ;
      cloud.col[slot * 3] = colour[0];
      cloud.col[slot * 3 + 1] = colour[1];
      cloud.col[slot * 3 + 2] = colour[2];
    }

    // Above 1.0 so the cloud blooms; the three bands are the visualizer's own
    // GREEN, ORANGE and RED.
    var HOT = 5.0;
    function velocityColour(speed) {
      if (speed > CRIT_V) return [0.86 * HOT, 0.15 * HOT, 0.15 * HOT];
      if (speed > SAFE_V) return [0.96 * HOT, 0.62 * HOT, 0.04 * HOT];
      return [0.38 * HOT, 0.88 * HOT, 0.62 * HOT];
    }
    var POSITION_COLOUR = [0.19 * 4.0, 0.61 * 4.0, 1.00 * 4.0];

    function commit(cloud, count) {
      cloud.geo.attributes.position.needsUpdate = true;
      cloud.geo.attributes.color.needsUpdate = true;
      cloud.geo.setDrawRange(0, count);
    }

    function hideBelief() {
      velCloud.geo.setDrawRange(0, 0);
      posCloud.geo.setDrawRange(0, 0);
    }

    /* One particle of this environment's belief is the whole state — position
       and velocity — so each one is written into both clouds: its position on
       the ground, its velocity at the body, one metre per m/s, exactly where
       the green arrow's tip would be if that particle were the truth. */
    function drawParticleCloud(particles, count, bodyX, bodyZ) {
      var unsafe = 0;
      for (var p = 0; p < count; p++) {
        var particle = particles[p];
        var speed = Math.hypot(particle[2], particle[3]);
        if (speed > SAFE_V) unsafe++;
        writeCloud(
          velCloud, p,
          bodyX + particle[2] * VSCALE, bodyZ + particle[3] * VSCALE,
          0.22 + (p % 5) * 0.015, velocityColour(speed)
        );
        writeCloud(posCloud, p, wx(particle[0]), wz(particle[1]), 0.07, POSITION_COLOUR);
      }
      commit(velCloud, count);
      commit(posCloud, count);
      return unsafe;
    }

    function drawParticles(belief, bodyX, bodyZ) {
      var count = Math.min(belief.particles.length, MAX_DRAWN_PARTICLES);
      if (count && (!Array.isArray(belief.particles[0]) || belief.particles[0].length < 4)) {
        // A particle that is not a four-number state. This scene plots
        // positions and velocities, so it says so rather than plotting
        // whatever the first two entries happen to be.
        hideBelief();
        return belief.num_particles + " particles of an unexpected shape";
      }
      var unsafe = drawParticleCloud(belief.particles, count, bodyX, bodyZ);
      var label = belief.num_particles +
        (belief.weighted === false ? " uniform particles" : " particles");
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " drawn)";
      }
      return label + " · " + Math.round(100 * unsafe / count) + "% over " +
        SAFE_V.toFixed(1) + " m/s";
    }

    /** Lower-triangular Cholesky of a 2x2, so a drawn cloud has the real shape. */
    function cholesky2(a, b, c) {
      var l11 = Math.sqrt(Math.max(a, 1e-9));
      var l21 = b / l11;
      var l22 = Math.sqrt(Math.max(c - l21 * l21, 1e-9));
      return [l11, l21, l22];
    }

    /* A Gaussian over this state is drawn as its two 2-D marginals: the
       position block and the velocity block. A marginal of a Gaussian is a
       Gaussian, so both clouds are exactly what the belief says about those
       coordinates. The position/velocity cross-covariance is not drawn,
       because nothing on screen could show it. */
    function drawGaussianComponent(mean, covariance, slot, howMany, bodyX, bodyZ) {
      var lp = cholesky2(covariance[0][0], covariance[1][0], covariance[1][1]);
      var lv = cholesky2(covariance[2][2], covariance[3][2], covariance[3][3]);
      var unsafe = 0;
      for (var q = 0; q < howMany; q++) {
        var z1 = gaussianDraws[q][0], z2 = gaussianDraws[q][1];
        var px = mean[0] + lp[0] * z1;
        var py = mean[1] + lp[1] * z1 + lp[2] * z2;
        var vx = mean[2] + lv[0] * z1;
        var vy = mean[3] + lv[1] * z1 + lv[2] * z2;
        var speed = Math.hypot(vx, vy);
        if (speed > SAFE_V) unsafe++;
        writeCloud(
          velCloud, slot + q, bodyX + vx * VSCALE, bodyZ + vy * VSCALE,
          0.22 + ((slot + q) % 5) * 0.015, velocityColour(speed)
        );
        writeCloud(posCloud, slot + q, wx(px), wz(py), 0.07, POSITION_COLOUR);
      }
      return unsafe;
    }

    function drawGaussian(belief, bodyX, bodyZ) {
      var count = MAX_DRAWN_PARTICLES;
      var unsafe = drawGaussianComponent(belief.mean, belief.covariance, 0, count, bodyX, bodyZ);
      commit(velCloud, count);
      commit(posCloud, count);
      return "Gaussian · " + Math.round(100 * unsafe / count) + "% over " +
        SAFE_V.toFixed(1) + " m/s";
    }

    /* A mixture is drawn as its components, each given a share of the points
       in proportion to its weight. Collapsing it to one cloud would place the
       belief's mass between its modes, where the belief says nothing is. */
    function drawGaussianMixture(belief, bodyX, bodyZ) {
      var slot = 0, unsafe = 0;
      for (var k = 0; k < belief.components.length && slot < MAX_DRAWN_PARTICLES; k++) {
        var component = belief.components[k];
        var share = Math.max(1, Math.round(component.weight * MAX_DRAWN_PARTICLES));
        share = Math.min(share, MAX_DRAWN_PARTICLES - slot, gaussianDraws.length);
        unsafe += drawGaussianComponent(
          component.mean, component.covariance, slot, share, bodyX, bodyZ
        );
        slot += share;
      }
      commit(velCloud, slot);
      commit(posCloud, slot);
      return belief.components.length + "-component mixture · " +
        Math.round(100 * unsafe / Math.max(1, slot)) + "% over " + SAFE_V.toFixed(1) + " m/s";
    }

    function drawBelief(index, bodyX, bodyZ) {
      var belief = payload.beliefs[index];
      if (!belief) { hideBelief(); return "—"; }

      if (belief.kind === "particles") return drawParticles(belief, bodyX, bodyZ);
      if (belief.kind === "gaussian") return drawGaussian(belief, bodyX, bodyZ);
      if (belief.kind === "gaussian_mixture") return drawGaussianMixture(belief, bodyX, bodyZ);
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner;
        // it is not one episode's belief, and merging its members would show a
        // cloud that was never anyone's belief.
        hideBelief();
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }

      // A belief class core has no payload for yet. Named, so the gap is
      // diagnosable, and drawn as nothing, so it is not invented.
      hideBelief();
      return "not recorded (" + (belief.belief_class || belief.kind) + ")";
    }

    function drawTrail(t) {
      var count = Math.max(2, Math.min(states.length, Math.floor(t) + 2));
      var prevX = null, prevZ = null;
      for (var i = 0; i < count; i++) {
        var sample = sampleAt(Math.min(i, t));
        var sx = wx(sample.x), sz = wz(sample.y);
        var nx = 0, nz = 0;
        if (prevX !== null) { nx = -(sz - prevZ); nz = sx - prevX; }
        var nl = Math.hypot(nx, nz) || 1;
        nx = nx / nl * TRAIL_W; nz = nz / nl * TRAIL_W;
        var a = i * 6;
        trailPos[a] = sx + nx; trailPos[a + 1] = 0.05; trailPos[a + 2] = sz + nz;
        trailPos[a + 3] = sx - nx; trailPos[a + 4] = 0.05; trailPos[a + 5] = sz - nz;
        // The visualizer's BLUE, hot where the path is newest.
        var k = lerp(0.35, 2.4, count > 1 ? i / (count - 1) : 1);
        for (var c = 0; c < 2; c++) {
          trailCol[a + c * 3] = 0.19 * k;
          trailCol[a + c * 3 + 1] = 0.61 * k;
          trailCol[a + c * 3 + 2] = 1.00 * k;
        }
        prevX = sx; prevZ = sz;
      }
      // The first sample has no previous point to take a normal from, so it
      // borrows the second's, which keeps the ribbon from pinching to a point.
      if (count > 1) {
        var ox = trailPos[6] - trailPos[9], oz = trailPos[8] - trailPos[11];
        var bx = (trailPos[0] + trailPos[3]) / 2, bz = (trailPos[2] + trailPos[5]) / 2;
        trailPos[0] = bx + ox / 2; trailPos[2] = bz + oz / 2;
        trailPos[3] = bx - ox / 2; trailPos[5] = bz - oz / 2;
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
      trailGeo.setDrawRange(0, (count - 1) * 6);
    }

    var heading = 0;
    var gait = 0;
    var forceScales = world.force_scales || [];

    return {
      steps: states.length,

      /* Framing scales with the episode, because the trace decides how far the
         ant went: a fixed distance that frames a twelve-metre run leaves a
         two-metre one as a smudge. The constant term is headroom for the
         masts, which are the same height whatever the run did. */
      camera: {
        board: [0, reach * 1.35 + 5.2, reach * 1.25 + 5.0],
        top: [0, reach * 2.1 + 5.5, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var px = wx(sample.x), pz = wz(sample.y);
        var speed = Math.hypot(sample.vx, sample.vy);

        /* Heading from the recorded velocity, so the body faces the way it is
           actually moving; at a standstill it keeps the heading it had. World
           +y is scene +z, so rotation.y = -heading points local +X — the head
           end — along (vx, vy). */
        if (speed > 0.05) {
          var target = Math.atan2(sample.vy, sample.vx);
          var diff = ((target - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          heading += diff * clamp(dt * 6, 0, 1);
        }
        ant.position.set(px, 0, pz);
        ant.rotation.y = -heading;

        // Gait: stride frequency and amplitude scale with the recorded speed.
        // Illustrative — this environment has no joint state.
        gait += dt * (playing ? 1 : 0) * (1.2 + speed * 2.6);
        var amp = clamp(speed * 0.16, 0.05, 0.34);
        for (var li = 0; li < legs.length; li++) {
          var leg = legs[li];
          var phase = gait * 3.0 + leg.phase;
          leg.hip.rotation.y = leg.yaw0 + Math.sin(phase) * amp * leg.side;
          leg.knee.rotation.x = -0.12 + Math.max(0, Math.cos(phase)) * amp * 1.1;
        }
        body.position.y = Math.sin(gait * 6.0) * 0.012 * clamp(speed, 0, 1);
        body.rotation.z = Math.sin(gait * 3.0) * 0.03 * clamp(speed, 0, 1);

        // The rings travel with the ant: velocity space is centred on the body.
        ringGroup.position.set(px, 0, pz);
        var over = speed > SAFE_V;
        for (var si = 0; si < safeRing.children.length; si++) {
          safeRing.children[si].material.opacity = over ? 0.55 : 0.95;
        }
        critRing.material.opacity = over ? 0.95 : 0.55;
        safeDisc.material.opacity = over ? 0.010 : 0.028;

        // Velocity arrow: from the body, one metre per m/s.
        aimArrow(velArrow, px, 0.34, pz, sample.vx * VSCALE, sample.vy * VSCALE);

        /* Applied force. The magnitude is the action; the direction was drawn
           uniformly by the environment and recovered in the exporter from the
           recorded transition. The faint circle is the whole set that draw
           came from, at this step's magnitude. */
        var force = payload.applied_forces ? payload.applied_forces[sample.index] : null;
        var magnitude = force ? Math.hypot(force[0], force[1]) : 0;
        forceCircle.visible = magnitude > 0.01;
        if (forceCircle.visible) forceCircle.scale.set(magnitude, magnitude, 1);
        if (force) {
          aimArrow(forceArrow, px, 0.09, pz, force[0], force[1]);
        } else {
          forceArrow.visible = false;
        }

        // Move the one shadow-casting flood onto the nearest mast and take
        // that much power back off its own lamp.
        var nearest = null, nd = Infinity;
        for (var fl = 0; fl < floodLights.length; fl++) {
          var flood = floodLights[fl];
          flood.light.power = FLOOD_LUMENS;
          var d2 = (flood.x - px) * (flood.x - px) + (flood.z - pz) * (flood.z - pz);
          if (d2 < nd) { nd = d2; nearest = flood; }
        }
        if (nearest) {
          shadowLight.position.set(nearest.x, FLOOD_HEIGHT, nearest.z);
          shadowLight.target.position.set(px, 0.15, pz);
          shadowLight.target.updateMatrixWorld();
          var share = 0.58;
          shadowLight.power = FLOOD_LUMENS * share;
          nearest.light.power = FLOOD_LUMENS * (1 - share);
        }

        for (var mi = 0; mi < MOTES; mi++) {
          motePos[mi * 3] += Math.sin(elapsed * 0.3 + mi) * 0.002;
          motePos[mi * 3 + 1] += 0.004;
          if (motePos[mi * 3 + 1] > 4.2) motePos[mi * 3 + 1] = 0.15;
        }
        moteGeo.attributes.position.needsUpdate = true;

        var beliefLabel = drawBelief(sample.index, px, pz);
        drawTrail(t);

        repaintLabel(
          speedLabel,
          speed.toFixed(2) + " m/s" + (speed > CRIT_V ? " · terminal"
            : over ? " · violation" : ""),
          speed > CRIT_V ? "#FF6F63" : over ? "#F59E0B" : "#62E19D"
        );

        var step = trace.steps[sample.index] || {};
        var action = step.action;
        var actionLabel = action === null || action === undefined
          ? "—"
          : String(action);
        if (typeof action === "number" && forceScales[action] !== undefined) {
          actionLabel += " (force " +
            (forceScales[action] * world.max_force).toFixed(2) + " N)";
        }

        return {
          follow: { x: px, z: pz, heading: heading },
          step: sample.index,
          action: actionLabel,
          x: sample.x,
          y: sample.y,
          reward: step.reward,
          ret: running[sample.index],
          belief: beliefLabel
        };
      }
    };
  }

  V.scenes["safety_ant_velocity.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // scene's flood power; it is not a knob to remove.
    exposure: 0.165
  };
})(window);
