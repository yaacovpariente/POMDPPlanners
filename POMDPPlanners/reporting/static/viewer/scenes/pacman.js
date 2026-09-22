/* SPDX-License-Identifier: MIT
 *
 * PacMan scene module.
 *
 * Builds the maze from a trace's `payload.world` block and moves it from the
 * trace's recorded cells and beliefs. Nothing here is invented: there is no
 * fallback episode, no hand-placed route and no synthetic belief. If the
 * player hands this module no trace, it draws nothing and says so.
 *
 * The belief is over ghost positions, and it is drawn twice, in two
 * deliberately different registers:
 *
 *   * lit floor cells carrying the marginal mass, which is the same quantity
 *     and the same rule the 2-D GIF renderer paints as a red tile;
 *   * a particle swarm hovering well above the floor, one point per ghost per
 *     recorded particle, shaded by that particle's weight.
 *
 * Neither is ever drawn as a ghost body. A ghost body is the true state; the
 * red haze is what the run believed, and the two must never be confusable —
 * so the tiles sit flat on the floor, the swarm floats above head height, and
 * the ghosts themselves are solid, lit and eyed.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  // Taken from pacman_art.py, so the browser viewer and the GIF depict one
  // world: floor base (8,15,31), wall base (4,22,128), wall highlight
  // (134,193,255), pellet pool (93,130,255). The belief red is the 2-D
  // renderer's ghost-mass overlay.
  var COLORS = {
    wall: 0x123CB4,
    wallCap: 0x3A6EEE,
    pellet: 0x5D82FF,
    pac: 0xFFC21E,
    belief: 0xFF2A20,
    lamp: 0xFFE7C4,
    hazard: 0xFF4030
  };

  // The eight ghost hues pacman_art.ghost() cycles through, in its order, so
  // ghost 0 is the same ghost in both renderers.
  var GHOST_COLORS = [
    0xE8342B, 0x3BE06B, 0xFF9E2C, 0x2EC7E8,
    0xD06BFF, 0xFFE24A, 0xFF6FA8, 0x8FE0C0
  ];

  // Lumens. Four service lamps on posts, not a stadium: the far corners of
  // the maze have to stay dark, or the lighting says nothing.
  var LAMP_LUMENS = 820;

  // Cap on drawn belief points. A trace carries at most a few hundred
  // particles, each contributing one point per ghost; beyond this the extra
  // points are draw cost with no extra information at this scale.
  var MAX_DRAWN_POINTS = 1200;

  var PAC_R = 0.40;
  var WALL_H = 1.02;

  function groundCanvas(seed) {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#0B1225";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 900; i++) {
      var r = 12 + rnd() * 90;
      g.fillStyle = "rgba(" + (16 + rnd() * 26 | 0) + "," + (26 + rnd() * 30 | 0) + "," +
        (52 + rnd() * 44 | 0) + ",0.24)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
    }
    // Chips of stone: a lit cap and a dark side, which the normal map turns
    // into relief once the lamps hit it.
    for (var p = 0; p < 1500; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.4 + rnd() * 4.2;
      g.fillStyle = "rgba(4,6,14,0.6)";
      g.beginPath(); g.arc(px + pr * 0.35, py + pr * 0.35, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(120,146,196,0.26)";
      g.beginPath(); g.arc(px, py, pr * 0.85, 0, Math.PI * 2); g.fill();
    }
    for (var j = 0; j < 900; j++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(160,196,255,0.05)" : "rgba(0,0,0,0.26)";
      g.fillRect(rnd() * s, rnd() * s, 3, 3);
    }
    return cv;
  }

  /* The cell grid, drawn the way the 2-D renderer draws it: a dark seam
     between tiles with a cool inner edge. Painted onto colour only, after the
     normal map is derived, so it marks cells without embossing the stone. */
  function paintCells(cv, rows, cols) {
    var s = cv.width;
    var g = cv.getContext("2d");
    var cw = s / cols, ch = s / rows;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var x = c * cw, y = r * ch;
        g.strokeStyle = "rgba(2,4,8,0.85)";
        g.lineWidth = Math.min(cw, ch) * 0.05;
        g.strokeRect(x + cw * 0.02, y + ch * 0.02, cw * 0.96, ch * 0.96);
        g.strokeStyle = "rgba(80,103,134,0.30)";
        g.lineWidth = Math.min(cw, ch) * 0.022;
        g.strokeRect(x + cw * 0.07, y + ch * 0.07, cw * 0.86, ch * 0.86);
      }
    }
  }

  /**
   * Build the PacMan world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind pacman.v1.
   * @returns {Object} The scene module the player drives.
   */
  // The maze, its props, its lights, its two belief widgets and their update
  // are one world built from one payload; splitting it would mean passing a
  // dozen handles between functions that have no other caller.
  // eslint-disable-next-line max-statements
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var layout = payload.state_layout;
    var scene = core.scene;
    var renderer = core.renderer;

    var ROWS = world.maze_size[0], COLS = world.maze_size[1];
    var NUM_GHOSTS = world.num_ghosts;
    var halfR = (ROWS - 1) / 2, halfC = (COLS - 1) / 2;
    function wx(col) { return col - halfC; }   // grid col -> scene x
    function wz(row) { return row - halfR; }   // grid row -> scene z

    var wallSet = {};
    world.walls.forEach(function (w) { wallSet[w[0] + "," + w[1]] = true; });
    function isWall(r, c) { return wallSet[r + "," + c] === true; }

    scene.background = new THREE.Color(0x04050A).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x05070E, 0.028);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x1A2740, 0x05060C, 0.4));
    var moon = new THREE.DirectionalLight(0x8FA3C8, 0.25);
    moon.position.set(-6, 9, -4);
    scene.add(moon);
    // Without an environment map the chromed wall caps and the ghosts' eyes
    // have nothing to mirror, and PBR metal reflecting nothing reads as flat
    // grey plastic.
    core.buildNightEnvironment([
      [0.0, "#070C1C"], [0.45, "#101728"], [0.55, "#1A2038"], [1.0, "#050609"]
    ]);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);

    // ---- floor -----------------------------------------------------------
    var gCanvas = groundCanvas(90421);
    var groundNormal = V.normalMapFrom(renderer, gCanvas, 2.1);
    paintCells(gCanvas, ROWS, COLS);
    var groundAlbedo = new THREE.CanvasTexture(gCanvas);
    groundAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    groundAlbedo.encoding = THREE.sRGBEncoding;

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(COLS, ROWS),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.9, 0.9),
        roughness: 0.9, metalness: 0.05, envMapIntensity: 0.4
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    /* Ground carrying on past the maze. Without it the board is a lit slab in
       a void, which is the biggest tell that a night render is a render. It
       sits well below the floor rather than a few centimetres under it: two
       large near-coplanar surfaces stripe badly in a 16-bit depth buffer. */
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(80, 80),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.8, 0.8),
        color: 0x10141F, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.12
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.7;
    outer.receiveShadow = true;
    scene.add(outer);

    var plinth = new THREE.Mesh(
      new THREE.BoxGeometry(COLS + 0.9, 0.7, ROWS + 0.9),
      new THREE.MeshStandardMaterial({ color: 0x0A0E1A, roughness: 0.85, metalness: 0.2 })
    );
    plinth.position.y = -0.5;
    plinth.castShadow = true; plinth.receiveShadow = true;
    scene.add(plinth);

    // ---- walls -----------------------------------------------------------
    /* One raised block per blocked cell. The rim is the grid boundary made
       visible: the environment has no cells outside the maze, and a move that
       would leave it simply does not happen. */
    var wallMat = new THREE.MeshStandardMaterial({
      color: COLORS.wall, roughness: 0.3, metalness: 0.45, envMapIntensity: 1.1
    });
    var capMat = new THREE.MeshStandardMaterial({
      color: COLORS.wallCap, roughness: 0.2, metalness: 0.7, envMapIntensity: 1.4
    });
    // Emissive strip: deliberately above 1.0 so the bright pass blooms it.
    var stripMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.55, 1.35, 2.6) });
    var wallGeo = new THREE.BoxGeometry(0.92, WALL_H, 0.92);
    var capGeo = new THREE.BoxGeometry(0.97, 0.05, 0.97);
    var stripGeoX = new THREE.BoxGeometry(0.80, 0.022, 0.035);
    var stripGeoZ = new THREE.BoxGeometry(0.035, 0.022, 0.80);

    world.walls.forEach(function (w) {
      var x = wx(w[1]), z = wz(w[0]);
      var block = new THREE.Mesh(wallGeo, wallMat);
      block.position.set(x, WALL_H / 2, z);
      block.castShadow = true; block.receiveShadow = true;
      scene.add(block);
      var cap = new THREE.Mesh(capGeo, capMat);
      cap.position.set(x, WALL_H + 0.025, z);
      cap.castShadow = true; cap.receiveShadow = true;
      scene.add(cap);
      // Light strips along the top of each face: the neon an arcade maze is
      // known for, here as a real emitter the bloom pass picks up.
      [[0, 0.46], [0, -0.46]].forEach(function (o) {
        var s = new THREE.Mesh(stripGeoX, stripMat);
        s.position.set(x + o[0], WALL_H + 0.06, z + o[1]);
        scene.add(s);
      });
      [[0.46, 0], [-0.46, 0]].forEach(function (o) {
        var s = new THREE.Mesh(stripGeoZ, stripMat);
        s.position.set(x + o[0], WALL_H + 0.06, z + o[1]);
        scene.add(s);
      });
      var fill = new THREE.PointLight(0x3E7BFF, 1, 3.2, 2);
      fill.power = 46;
      fill.position.set(x, WALL_H + 0.22, z);
      scene.add(fill);
    });

    var rimMat = new THREE.MeshStandardMaterial({
      color: 0x0C1740, roughness: 0.45, metalness: 0.6, envMapIntensity: 0.8
    });
    var rimH = 0.5;
    [-(ROWS / 2 + 0.25), ROWS / 2 + 0.25].forEach(function (z) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(COLS + 1.0, rimH, 0.5), rimMat);
      m.position.set(0, rimH / 2, z);
      m.castShadow = true; m.receiveShadow = true;
      scene.add(m);
    });
    [-(COLS / 2 + 0.25), COLS / 2 + 0.25].forEach(function (x) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(0.5, rimH, ROWS + 1.0), rimMat);
      m.position.set(x, rimH / 2, 0);
      m.castShadow = true; m.receiveShadow = true;
      scene.add(m);
    });
    var rimStripMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.30, 0.78, 1.6) });
    [-(ROWS / 2 + 0.06), ROWS / 2 + 0.06].forEach(function (z) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(COLS, 0.03, 0.05), rimStripMat);
      m.position.set(0, rimH - 0.02, z);
      scene.add(m);
    });
    [-(COLS / 2 + 0.06), COLS / 2 + 0.06].forEach(function (x) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.03, ROWS), rimStripMat);
      m.position.set(x, rimH - 0.02, 0);
      scene.add(m);
    });

    // ---- hazard zones ----------------------------------------------------
    /* Drawn only when the run configured them. They are a probability field
       PacMan may walk into, not a wall, so they are see-through and the floor
       inside one takes the same light as the floor outside it. */
    var hazardRims = [];
    world.dangerous_areas.forEach(function (d) {
      var x = wx(d[1]), z = wz(d[0]);
      var radius = world.dangerous_area_radius;
      var disc = new THREE.Mesh(
        new THREE.CircleGeometry(radius, 48),
        new THREE.MeshBasicMaterial({
          color: COLORS.hazard, transparent: true, opacity: 0.10,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      disc.rotation.x = -Math.PI / 2;
      disc.position.set(x, 0.018, z);
      scene.add(disc);
      var rim = new THREE.Mesh(
        new THREE.RingGeometry(Math.max(0.02, radius - 0.05), radius + 0.05, 48),
        new THREE.MeshBasicMaterial({
          color: COLORS.hazard, transparent: true, opacity: 0.38,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        })
      );
      rim.rotation.x = -Math.PI / 2;
      rim.position.set(x, 0.024, z);
      scene.add(rim);
      hazardRims.push(rim);
    });

    // ---- corner lamps ----------------------------------------------------
    var lampLights = [];
    var lampPostMat = new THREE.MeshStandardMaterial({
      color: 0x4A4437, roughness: 0.4, metalness: 0.85
    });
    var lampLensMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(4.2, 3.6, 2.7) });
    [[-1, -1], [-1, 1], [1, -1], [1, 1]].forEach(function (s) {
      var x = s[0] * (COLS / 2 + 0.45), z = s[1] * (ROWS / 2 + 0.45);
      var post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.08, 2.6, 12), lampPostMat);
      post.position.set(x, 1.3, z); post.castShadow = true;
      scene.add(post);
      var head = new THREE.Mesh(new THREE.ConeGeometry(0.24, 0.30, 14, 1, false), lampPostMat);
      head.position.set(x, 2.66, z); head.rotation.x = Math.PI; head.castShadow = true;
      scene.add(head);
      var lens = new THREE.Mesh(new THREE.SphereGeometry(0.075, 14, 10), lampLensMat);
      lens.position.set(x, 2.58, z);
      scene.add(lens);
      var light = new THREE.PointLight(COLORS.lamp, 1, 13, 2);
      light.power = LAMP_LUMENS;
      light.position.set(x, 2.55, z);
      scene.add(light);
      lampLights.push({ light: light, x: x, z: z });
    });

    /* One shadow-casting lamp, not four. It is parked on whichever corner
       lamp is nearest PacMan and aimed at it, and that lamp's own output is
       reduced by the same amount, so the maze stays evenly lit while PacMan
       and the ghosts always throw a shadow pointing the right way. Narrow
       cone plus normalBias, which is the fix for acne a big negative bias is
       not. */
    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 16, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.5;
    shadowLight.shadow.camera.far = 18;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, 2.55, 0);
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    // ---- pellets ---------------------------------------------------------
    /* Every pellet the episode started with gets a node; which ones are
       showing is read per step from the trace, never inferred from where
       PacMan has been. */
    var pelletNodes = [];
    var pelletGeo = new THREE.SphereGeometry(0.115, 18, 14);
    var pelletMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.0, 3.2, 3.6) });
    var pelletRingGeo = new THREE.RingGeometry(0.30, 0.345, 40);
    world.initial_pellets.forEach(function (p) {
      var group = new THREE.Group();
      group.position.set(wx(p[1]), 0, wz(p[0]));
      scene.add(group);
      var bead = new THREE.Mesh(pelletGeo, pelletMat);
      bead.position.y = 0.30;
      group.add(bead);
      var light = new THREE.PointLight(COLORS.pellet, 1, 3.2, 2);
      light.power = 44;
      light.position.y = 0.32;
      group.add(light);
      var pool = new THREE.Sprite(new THREE.SpriteMaterial({
        map: poolTex, color: COLORS.pellet, transparent: true, opacity: 0.22,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      pool.scale.set(1.1, 1.1, 1);
      pool.position.y = 0.30;
      group.add(pool);
      var ring = new THREE.Mesh(pelletRingGeo, new THREE.MeshBasicMaterial({
        color: COLORS.pellet, transparent: true, opacity: 0.30,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.035;
      group.add(ring);
      pelletNodes.push({ group: group, key: p[0] + "," + p[1], bead: bead, light: light, base: 44 });
    });

    // ---- PacMan ----------------------------------------------------------
    /* Two jaws hinged at the centre, so the mouth really opens and closes
       instead of a texture flipping. The wedge is cut by a vertical plane
       through the heading, so the mouth opens sideways the way the arcade
       sprite does seen from above; a mouth hinged on a horizontal plane is
       almost edge-on from the raised camera and reads as a plain sphere. */
    var pacman = new THREE.Group();
    scene.add(pacman);
    var pacSkin = new THREE.MeshStandardMaterial({
      color: COLORS.pac, roughness: 0.28, metalness: 0.15,
      emissive: 0x6A4A00, emissiveIntensity: 0.55, envMapIntensity: 0.8
    });
    // The inside of the mouth is a cavity, so it stays much darker than the
    // skin or the open jaw reads as a pale disc instead of a hole.
    var mouthMat = new THREE.MeshStandardMaterial({
      color: 0x24150A, roughness: 0.85, metalness: 0.0, side: THREE.DoubleSide
    });
    var jawPos = new THREE.Group(), jawNeg = new THREE.Group();
    pacman.add(jawPos); pacman.add(jawNeg);
    var hemiGeo = new THREE.SphereGeometry(PAC_R, 40, 22, 0, Math.PI * 2, 0, Math.PI / 2);
    var pacCapGeo = new THREE.CircleGeometry(PAC_R, 40);
    var halfA = new THREE.Mesh(hemiGeo, pacSkin);
    halfA.rotation.x = Math.PI / 2; halfA.castShadow = true;
    jawPos.add(halfA);
    jawPos.add(new THREE.Mesh(pacCapGeo, mouthMat));
    var halfB = new THREE.Mesh(hemiGeo, pacSkin);
    halfB.rotation.x = -Math.PI / 2; halfB.castShadow = true;
    jawNeg.add(halfB);
    jawNeg.add(new THREE.Mesh(pacCapGeo, mouthMat));
    jawPos.position.y = PAC_R + 0.06;
    jawNeg.position.y = PAC_R + 0.06;

    var pacLight = new THREE.PointLight(0xFFC55A, 1, 3.4, 2);
    pacLight.power = 90;
    pacLight.position.y = PAC_R + 0.1;
    pacman.add(pacLight);
    var pacBlob = new THREE.Mesh(
      new THREE.PlaneGeometry(1.0, 1.0),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.45, depthWrite: false
      })
    );
    pacBlob.rotation.x = -Math.PI / 2;
    pacBlob.position.y = 0.012;
    pacman.add(pacBlob);

    // ---- ghosts ----------------------------------------------------------
    /* A real body: a domed hood over a skirt with scalloped feet, floating a
       little off the floor, with eyes that track PacMan. Solid, lit and
       shadow-casting — everything the belief haze is not. */
    function buildGhost(colorHex) {
      var g = new THREE.Group();
      var R = 0.34;
      var body = new THREE.MeshStandardMaterial({
        color: colorHex, roughness: 0.3, metalness: 0.1,
        emissive: colorHex, emissiveIntensity: 0.34, envMapIntensity: 0.7
      });
      var dome = new THREE.Mesh(
        new THREE.SphereGeometry(R, 28, 16, 0, Math.PI * 2, 0, Math.PI / 2), body
      );
      dome.position.y = 0.38; dome.castShadow = true;
      g.add(dome);
      var skirt = new THREE.Mesh(new THREE.CylinderGeometry(R, R, 0.38, 28, 1, false), body);
      skirt.position.y = 0.19; skirt.castShadow = true;
      g.add(skirt);
      for (var i = 0; i < 6; i++) {
        var a = (i / 6) * Math.PI * 2;
        var foot = new THREE.Mesh(new THREE.SphereGeometry(0.085, 12, 10), body);
        foot.position.set(Math.cos(a) * (R - 0.06), 0.03, Math.sin(a) * (R - 0.06));
        foot.castShadow = true;
        g.add(foot);
      }
      var eyes = new THREE.Group();
      eyes.position.y = 0.46;
      g.add(eyes);
      // Deliberately large and bright: at this scale the eyes are the only
      // thing separating a ghost from a coloured blob.
      var scleraMat = new THREE.MeshStandardMaterial({
        color: 0xF4F8FF, roughness: 0.1, metalness: 0.0,
        emissive: 0x9AB6E8, emissiveIntensity: 0.55, envMapIntensity: 1.6
      });
      var pupilMat = new THREE.MeshStandardMaterial({
        color: 0x0A1750, roughness: 0.08, metalness: 0.2, emissive: 0x060F30
      });
      [-0.135, 0.135].forEach(function (dz) {
        var sclera = new THREE.Mesh(new THREE.SphereGeometry(0.115, 18, 14), scleraMat);
        sclera.position.set(0.20, 0, dz);
        eyes.add(sclera);
        var pupil = new THREE.Mesh(new THREE.SphereGeometry(0.058, 12, 10), pupilMat);
        pupil.position.set(0.285, 0, dz);
        eyes.add(pupil);
      });
      var light = new THREE.PointLight(colorHex, 1, 2.8, 2);
      light.power = 62;
      light.position.y = 0.38;
      g.add(light);
      var blob = new THREE.Mesh(
        new THREE.PlaneGeometry(0.95, 0.95),
        new THREE.MeshBasicMaterial({
          map: poolTex, color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false
        })
      );
      blob.rotation.x = -Math.PI / 2;
      blob.position.y = 0.01;
      g.add(blob);
      scene.add(g);
      return { group: g, eyes: eyes, light: light };
    }
    var ghostNodes = [];
    for (var gi = 0; gi < NUM_GHOSTS; gi++) {
      ghostNodes.push(buildGhost(GHOST_COLORS[gi % GHOST_COLORS.length]));
    }

    // ---- belief, register one: lit floor cells ---------------------------
    /* One tile per free cell, its opacity the cell's share of the marginal
       ghost mass — the same rule the 2-D renderer uses for its red squares,
       so the two viewers agree about what the belief says. */
    var beliefTiles = [];
    var tileIndex = {};
    var tileGeo = new THREE.PlaneGeometry(0.92, 0.92);
    for (var r = 0; r < ROWS; r++) {
      for (var c = 0; c < COLS; c++) {
        if (isWall(r, c)) continue;
        var tile = new THREE.Mesh(tileGeo, new THREE.MeshBasicMaterial({
          color: COLORS.belief, transparent: true, opacity: 0,
          blending: THREE.AdditiveBlending, depthWrite: false
        }));
        tile.rotation.x = -Math.PI / 2;
        tile.position.set(wx(c), 0.028, wz(r));
        scene.add(tile);
        tileIndex[r + "," + c] = beliefTiles.length;
        beliefTiles.push(tile);
      }
    }

    // ---- belief, register two: the particle swarm ------------------------
    /* One point per ghost per recorded particle, hovering well above head
       height so it can never be mistaken for a body. The scatter within a
       cell is cosmetic and fixed-seed: a particle names a cell, not a
       sub-cell position, and spreading the points is how several hundred
       particles on one cell stay legible as mass rather than one dot. */
    var swarmCap = Math.min(MAX_DRAWN_POINTS, Math.max(1, maxParticleCount() * NUM_GHOSTS));
    var sPos = new Float32Array(swarmCap * 3);
    var sCol = new Float32Array(swarmCap * 3);
    var sGeo = new THREE.BufferGeometry();
    sGeo.setAttribute("position", new THREE.BufferAttribute(sPos, 3));
    sGeo.setAttribute("color", new THREE.BufferAttribute(sCol, 3));
    sGeo.setDrawRange(0, 0);
    var swarm = new THREE.Points(sGeo, new THREE.PointsMaterial({
      size: 0.22, map: partTex, vertexColors: true, transparent: true,
      opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false,
      sizeAttenuation: true
    }));
    scene.add(swarm);

    var jitter = [];
    var jitterRnd = mulberry(5150);
    for (var ji = 0; ji < swarmCap; ji++) {
      jitter.push([(jitterRnd() - 0.5) * 0.66, (jitterRnd() - 0.5) * 0.66, jitterRnd() * Math.PI * 2]);
    }

    function maxParticleCount() {
      var most = 1;
      payload.beliefs.forEach(function (b) {
        if (b.kind === "particles") most = Math.max(most, b.particles.length);
      });
      return most;
    }

    // ---- trail -----------------------------------------------------------
    /* One amber mark per visited cell, fading with age. A polyline would be a
       one-pixel thread at this camera distance; a mark on the floor reads as
       a route through a maze, which is what a discrete path is. */
    var trailMarks = [];
    var markGeo = new THREE.RingGeometry(0.17, 0.30, 24);
    payload.pacman_positions.forEach(function (p) {
      var mark = new THREE.Mesh(markGeo, new THREE.MeshBasicMaterial({
        color: 0xFFA83C, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      }));
      mark.rotation.x = -Math.PI / 2;
      mark.position.set(wx(p[1]), 0.042, wz(p[0]));
      scene.add(mark);
      trailMarks.push(mark);
    });

    core.linearize();

    // ---- episode bookkeeping ---------------------------------------------
    var running = [];
    var total = 0;
    for (var ri = 0; ri < trace.steps.length; ri++) {
      var reward = trace.steps[ri].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, ri);
      }
      running.push(total);
    }

    var heading = 0;
    var steps = payload.pacman_positions.length;

    // Grid steps are discrete, so the sample eases between cells rather than
    // teleporting: smoothstep keeps the motion crisp but not jerky.
    function ease(f) { return f * f * (3 - 2 * f); }

    function sampleAt(t) {
      var i0 = Math.floor(clamp(t, 0, steps - 1));
      var i1 = Math.min(i0 + 1, steps - 1);
      var raw = clamp(t - i0, 0, 1);
      var f = ease(raw);
      var a = payload.pacman_positions[i0], b = payload.pacman_positions[i1];
      var ga = payload.ghost_positions[i0], gb = payload.ghost_positions[i1];
      var ghosts = [];
      for (var g = 0; g < ghostNodes.length; g++) {
        var from = ga[g] || [0, 0], to = gb[g] || from;
        ghosts.push([lerp(from[0], to[0], f), lerp(from[1], to[1], f)]);
      }
      return {
        // The step a reader is "on" is the nearest recorded one, so the HUD
        // never shows a cell that is half way between two cells.
        index: raw < 0.5 ? i0 : i1,
        from: i0, frac: raw, moving: raw > 0.02 && raw < 0.98,
        row: lerp(a[0], b[0], f), col: lerp(a[1], b[1], f),
        ghosts: ghosts
      };
    }

    /* Ghost cells carried by one belief particle. A PacMan particle is a full
       state array, so the trace publishes the layout and this reads it rather
       than assuming the default maze's offsets. */
    function ghostCellsOf(particle) {
      if (!Array.isArray(particle)) return null;
      var needed = layout.ghosts_start + 2 * NUM_GHOSTS;
      if (particle.length < needed) return null;
      var cells = [];
      for (var g = 0; g < NUM_GHOSTS; g++) {
        cells.push([
          Math.round(particle[layout.ghosts_start + 2 * g]),
          Math.round(particle[layout.ghosts_start + 2 * g + 1])
        ]);
      }
      return cells;
    }

    function clearBelief() {
      for (var i = 0; i < beliefTiles.length; i++) beliefTiles[i].material.opacity = 0;
      sGeo.setDrawRange(0, 0);
    }

    /* Both registers are written in one pass over the particles: the tiles
       accumulate each particle's weight into its ghost cells, the swarm gets
       one point per ghost per particle. They are the same numbers read twice,
       which is the point — if they ever disagreed, one of them would be
       decoration. */
    var massBuf = new Float32Array(beliefTiles.length);
    function drawParticleBelief(belief, elapsed) {
      var mass = massBuf;
      mass.fill(0);
      var count = Math.min(belief.particles.length, belief.weights.length);
      var maxWeight = 0;
      var slot = 0;
      var spatial = 0;

      for (var p = 0; p < count; p++) {
        var cells = ghostCellsOf(belief.particles[p]);
        if (!cells) continue;
        spatial++;
        // An unweighted belief is uniform by construction, so shading it by
        // weight would imply structure the belief does not have.
        var weight = belief.weighted === false ? 1 / count : belief.weights[p];
        if (weight > maxWeight) maxWeight = weight;
        for (var g = 0; g < cells.length; g++) {
          var row = cells[g][0], col = cells[g][1];
          var at = tileIndex[row + "," + col];
          if (at !== undefined) mass[at] += weight;
          if (slot < swarmCap) {
            var jx = jitter[slot][0], jz = jitter[slot][1], ph = jitter[slot][2];
            sPos[slot * 3] = wx(col) + jx;
            // Well clear of a ghost's 0.7-unit dome, and gently adrift, so the
            // swarm reads as uncertainty rather than as objects.
            sPos[slot * 3 + 1] = 1.30 + (slot % 6) * 0.055 + Math.sin(elapsed * 1.1 + ph) * 0.05;
            sPos[slot * 3 + 2] = wz(row) + jz;
            slot++;
          }
        }
      }

      if (!spatial) {
        clearBelief();
        return belief.num_particles + " particles, not PacMan states";
      }

      // Shade the swarm by each point's share of the heaviest particle, in
      // the same red the tiles use. Above 1.0 so the mode blooms.
      var written = 0;
      var hot = 3.0;
      for (var q = 0; q < count && written < slot; q++) {
        var cells2 = ghostCellsOf(belief.particles[q]);
        if (!cells2) continue;
        var w = belief.weighted === false ? 1 : (maxWeight > 0 ? belief.weights[q] / maxWeight : 0);
        for (var g2 = 0; g2 < cells2.length && written < slot; g2++) {
          sCol[written * 3] = lerp(0.55, 1.0, w) * hot;
          sCol[written * 3 + 1] = lerp(0.07, 0.24, w) * hot;
          sCol[written * 3 + 2] = lerp(0.05, 0.14, w) * hot;
          written++;
        }
      }
      sGeo.attributes.position.needsUpdate = true;
      sGeo.attributes.color.needsUpdate = true;
      sGeo.setDrawRange(0, slot);

      // Tiles, normalised by the heaviest cell, exactly as the GIF renderer
      // normalises its overlay.
      var peak = 0, lit = 0;
      for (var m = 0; m < mass.length; m++) if (mass[m] > peak) peak = mass[m];
      for (var k = 0; k < beliefTiles.length; k++) {
        var share = peak > 0 ? mass[k] / peak : 0;
        // A floor, so cells holding a rounding crumb of mass stay dark.
        beliefTiles[k].material.opacity = share < 0.07 ? 0 : 0.06 + share * 0.34;
        if (share >= 0.07) lit++;
      }

      var label = belief.num_particles +
        (belief.weighted === false ? " uniform particles" : " particles") +
        " over " + lit + (lit === 1 ? " cell" : " cells");
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " drawn)";
      }
      return label;
    }

    function drawBelief(index, elapsed) {
      var belief = payload.beliefs[index];
      if (!belief) { clearBelief(); return "—"; }
      if (belief.kind === "particles") return drawParticleBelief(belief, elapsed);
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner;
        // it is not one episode's belief, and merging its members would show
        // a cloud that was never anyone's belief.
        clearBelief();
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }
      // PacMan's state is a grid cell, so a Gaussian over it would be a
      // belief class nobody has fitted to this environment. Named rather than
      // guessed at, so the gap is diagnosable.
      clearBelief();
      return "not drawn (" + (belief.belief_class || belief.kind) + ")";
    }

    function drawTrail(index) {
      for (var i = 0; i < trailMarks.length; i++) {
        if (i > index) { trailMarks[i].material.opacity = 0; continue; }
        var age = index === 0 ? 1 : i / index;
        trailMarks[i].material.opacity = 0.10 + age * 0.34;
      }
    }

    function actionLabel(step) {
      if (step.action === null || step.action === undefined) return "—";
      var name = world.action_names[step.action];
      return name === undefined ? String(step.action) : name;
    }

    return {
      steps: steps,

      /* Framing scales with the maze, because a trace decides how big the
         board is: a distance that frames a 7x7 grid leaves a 15x15 one
         spilling off the canvas. The constant term is the margin for the
         props, which do not scale with the grid — a lamp post is the same
         height on any board, so a small board needs proportionally more
         headroom, not less. */
      camera: {
        board: [0, Math.max(ROWS, COLS) * 0.86 + 2.4, Math.max(ROWS, COLS) * 0.80 + 2.2],
        top: [0, Math.max(ROWS, COLS) * 1.18 + 2.2, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       *
       * The `follow` block the shared rig's chase camera reads is PacMan's,
       * not a ghost's. This environment has several agents and only one of
       * them is the one being planned for; chasing a ghost would put the
       * camera behind something the policy does not control.
       *
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var px = wx(sample.col), pz = wz(sample.row);

        // Heading from the direction of travel, turned into rather than
        // snapped between the four step directions.
        var a = payload.pacman_positions[sample.from];
        var b = payload.pacman_positions[Math.min(sample.from + 1, steps - 1)];
        var dx = b[1] - a[1], dz = b[0] - a[0];
        if (dx !== 0 || dz !== 0) {
          var target = Math.atan2(dz, dx);
          var diff = ((target - heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          heading += diff * clamp(dt * 9, 0, 1);
        }
        pacman.position.set(px, 0, pz);
        pacman.rotation.y = -heading;

        // Chomp. The jaws only work while PacMan is crossing a cell; at rest
        // the mouth idles half open.
        var chew = sample.moving && playing
          ? Math.abs(Math.sin(elapsed * 9.0))
          : 0.6 + Math.sin(elapsed * 2.2) * 0.2;
        var mouth = 0.16 + chew * 0.62;
        jawPos.rotation.y = -mouth;
        jawNeg.rotation.y = mouth;

        // Pellets: showing exactly those the recorded state still has.
        var alive = {};
        (payload.pellets[sample.index] || []).forEach(function (p) {
          alive[p[0] + "," + p[1]] = true;
        });
        for (var pn = 0; pn < pelletNodes.length; pn++) {
          var node = pelletNodes[pn];
          var showing = alive[node.key] === true;
          node.group.visible = showing;
          node.light.power = showing ? node.base : 0;
          if (showing) node.bead.position.y = 0.30 + Math.sin(elapsed * 2.4 + pn) * 0.03;
        }

        // Ghosts: float, bob, and look at PacMan.
        for (var g = 0; g < ghostNodes.length; g++) {
          var gp = sample.ghosts[g];
          var gx = wx(gp[1]), gz = wz(gp[0]);
          ghostNodes[g].group.position.set(
            gx, 0.06 + Math.sin(elapsed * 2.0 + g * 1.7) * 0.035, gz
          );
          /* The body turns toward PacMan, but only three-quarters of the way:
             a ghost facing exactly away from the camera loses its eyes, and
             the eyes are what make it a ghost rather than a coloured blob. */
          var toPac = Math.atan2(pz - gz, px - gx);
          var toCam = Math.atan2(core.camera.position.z - gz, core.camera.position.x - gx);
          var blend = ((toPac - toCam + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
          ghostNodes[g].group.rotation.y = -(toCam + blend * 0.7);
          ghostNodes[g].eyes.rotation.z = clamp(Math.sin(elapsed * 1.3 + g) * 0.12, -0.2, 0.2);
        }

        // Move the one shadow caster to the nearest lamp and take that much
        // light back off it, so the shadow follows without over-lighting.
        var nearest = null, nd = Infinity;
        for (var li = 0; li < lampLights.length; li++) {
          var L = lampLights[li];
          L.light.power = LAMP_LUMENS;
          var d2 = (L.x - px) * (L.x - px) + (L.z - pz) * (L.z - pz);
          if (d2 < nd) { nd = d2; nearest = L; }
        }
        if (nearest) {
          shadowLight.position.set(nearest.x, 2.55, nearest.z);
          shadowLight.target.position.set(px, 0.1, pz);
          shadowLight.target.updateMatrixWorld();
          var share = 0.6 * clamp(1 - (Math.sqrt(nd) - 2.0) / 7.0, 0.1, 1);
          shadowLight.power = LAMP_LUMENS * share;
          nearest.light.power = LAMP_LUMENS * (1 - share);
        }

        var pulse = 0.62 + Math.sin(elapsed * 2.1) * 0.22;
        for (var h = 0; h < hazardRims.length; h++) hazardRims[h].material.opacity = pulse;

        var beliefLabel = drawBelief(sample.index, elapsed);
        drawTrail(sample.index);

        var step = trace.steps[sample.index] || {};
        var cell = payload.pacman_positions[sample.index];
        return {
          follow: { x: px, z: pz, heading: heading },
          step: sample.index,
          action: actionLabel(step),
          // The HUD's x/y are the recorded cell: column across, row down,
          // which is how every PacMan coordinate in the package reads.
          x: cell[1],
          y: cell[0],
          reward: step.reward,
          ret: running[sample.index],
          belief: beliefLabel
        };
      },

      setBeliefTilesVisible: function (visible) {
        for (var i = 0; i < beliefTiles.length; i++) beliefTiles[i].visible = visible;
      },

      setBeliefSwarmVisible: function (visible) { swarm.visible = visible; }
    };
  }

  V.scenes["pacman.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for
    // this scene's lamp power; it is not a knob to remove.
    exposure: 0.175
  };
})(window);
