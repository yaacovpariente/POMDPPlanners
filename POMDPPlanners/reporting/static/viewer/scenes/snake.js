/* SPDX-License-Identifier: MIT
 *
 * Snake scene module.
 *
 * Builds the pit from a trace's `payload.world` block and moves it from the
 * trace's recorded bodies, apples and beliefs. Nothing here is invented: there
 * is no fallback episode and no synthetic belief. If the player hands this
 * module no trace, it draws nothing and says so.
 *
 * Three things in here are the environment rather than decoration:
 *
 *  1. The snake is one swept surface, not a chain of cubes. A frame is built
 *     at every sample along the body's own path and a superelliptical section
 *     laid around it, so it shades as one animal with no joints between cells.
 *     Growth needs no special case: the body slides along a single polyline
 *     whose tail index simply stops advancing on the step that ate.
 *  2. The belief is the agent's posterior over the one hidden cell, drawn as an
 *     amber field on the board and again as a labelled inset, with its entropy
 *     in bits. It is read out of the belief core serialised, never regenerated
 *     from the true apple.
 *  3. The three turns are marked where they would land when the environment's
 *     own transition says they are fatal. That is knowledge the agent has --
 *     the body and the walls are observed exactly -- so showing it credits the
 *     agent with nothing it did not have.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* Straight out of snake_visualizer.py, so the viewer and the GIF are the
     same world in two media. */
  var COLORS = {
    tileDark: 0x182C45,
    tileLight: 0x1E3651,
    wallEdge: 0x44608A,
    amber: 0xFFB02E,
    apple: 0xC8322C,
    appleDark: 0x8D1F1B,
    lamp: 0xFFE4B0
  };

  // Lumens. Four lanterns on tall posts around a stone pit, which is a lot more
  // light than Light-Dark's service lamps; the composite stops down to match.
  var LANTERN_LUMENS = 21000;

  /* How a per-cell probability becomes brightness. The GIF scales each cell
     against the step's own peak, because a flat posterior in a flat matplotlib
     heatmap is otherwise invisible. On a lit board that law backfires: under
     the uniform prior every cell sits at the peak, so the whole floor floods
     amber and "the agent knows nothing" comes out as the loudest frame of the
     episode. Here brightness is the probability itself, raised to a power, so
     the uniform prior is a faint even wash and a collapsed posterior is one
     hot cell. The ordering between cells is identical; what changes is that
     the overall level now carries how sure the agent is. */
  var BELIEF_MAX_ALPHA = 0.58;
  var BELIEF_GAMMA = 0.35;

  function shade(probability) {
    return probability > 0 ? Math.pow(probability, BELIEF_GAMMA) : 0;
  }

  var QUADRANTS = ["NE", "NW", "SE", "SW"];
  var ACTIONS = ["turn left", "go straight", "turn right"];
  // Matches DIRECTIONS in snake_pomdp.py: north is -row, and the list runs
  // clockwise, so a right turn is +1 and a left turn is -1.
  var DIRECTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  var TERMINATION = ["running", "hit the wall", "hit itself", "starved", "reached target length"];

  function smooth(t) { return t * t * (3 - 2 * t); }

  function stoneCanvas(seed) {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#101C2E";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 900; i++) {
      var r = 14 + rnd() * 110;
      g.fillStyle = "rgba(" + (22 + rnd() * 34 | 0) + "," + (40 + rnd() * 40 | 0) + "," +
        (62 + rnd() * 46 | 0) + ",0.22)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
    }
    // Pits: a dark underside and a lit cap, which the normal map turns into
    // relief so every hollow catches the lanterns from the side they are on.
    for (var p = 0; p < 1500; p++) {
      var px = rnd() * s, py = rnd() * s, pr = 1.4 + rnd() * 4.2;
      g.fillStyle = "rgba(4,9,16,0.55)";
      g.beginPath(); g.arc(px + pr * 0.35, py + pr * 0.35, pr, 0, Math.PI * 2); g.fill();
      g.fillStyle = "rgba(122,152,190,0.22)";
      g.beginPath(); g.arc(px, py, pr * 0.85, 0, Math.PI * 2); g.fill();
    }
    for (var j = 0; j < 900; j++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(180,210,255,0.045)" : "rgba(0,0,0,0.22)";
      g.fillRect(rnd() * s, rnd() * s, 3, 3);
    }
    return cv;
  }

  /* The skin, painted once. u runs around the section (0.25 is the spine, 0.75
     the belly) and v runs along the body from tail to head, so the renderer's
     green-head-to-cyan-tail gradient bakes straight into the map and stays put
     while the body grows. */
  function snakeSkin() {
    var W = 512, H = 1024;
    var cv = document.createElement("canvas");
    cv.width = W; cv.height = H;
    var g = cv.getContext("2d");
    var rnd = mulberry(8891);

    function mix(a, b, t) {
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    }
    var tailDorsal = [16, 74, 82], headDorsal = [40, 84, 36];
    var tailVent = [86, 132, 130], headVent = [126, 140, 92];

    var img = g.createImageData(W, H);
    var d = img.data;
    for (var y = 0; y < H; y++) {
      // CanvasTexture uploads with flipY, so uv v = 0 samples the LAST canvas
      // row. The tail is therefore painted at the bottom.
      var v = 1 - y / (H - 1);
      var dor = mix(tailDorsal, headDorsal, v);
      var ven = mix(tailVent, headVent, v);
      for (var x = 0; x < W; x++) {
        // +1 on the spine, -1 on the belly.
        var back = Math.cos(2 * Math.PI * (x / W - 0.25));
        var col = mix(ven, dor, Math.pow((back + 1) / 2, 1.5));
        var i = (y * W + x) * 4;
        d[i] = col[0]; d[i + 1] = col[1]; d[i + 2] = col[2]; d[i + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);

    // Dark saddle blotches down the spine, fading out onto the flanks. Two
    // offset lobes rather than one ring: a blotch that stops at the flank is
    // what stopped this reading as a segmented grub.
    g.globalCompositeOperation = "multiply";
    for (var s = 0; s < 15; s++) {
      var cy = (s + 0.5) / 15 * H + (rnd() - 0.5) * 12;
      var hgt = H / 15 * (0.30 + rnd() * 0.16);
      var grad = g.createLinearGradient(0, 0, W, 0);
      grad.addColorStop(0.00, "rgba(190,204,196,1)");
      grad.addColorStop(0.14, "rgba(96,112,102,1)");
      grad.addColorStop(0.25, "rgba(62,80,70,1)");
      grad.addColorStop(0.36, "rgba(96,112,102,1)");
      grad.addColorStop(0.50, "rgba(198,210,202,1)");
      grad.addColorStop(1.00, "rgba(255,255,255,1)");
      g.fillStyle = grad;
      g.beginPath();
      g.ellipse(W * 0.25, cy, W * (0.16 + rnd() * 0.05), hgt, 0, 0, Math.PI * 2);
      g.fill();
      g.beginPath();
      g.ellipse(W * (0.19 + rnd() * 0.12), cy + hgt * (rnd() - 0.5), W * 0.09, hgt * 0.7,
        0, 0, Math.PI * 2);
      g.fill();
    }
    g.globalCompositeOperation = "source-over";

    // Scales: rows across u, staggered half a scale each row, each a rounded
    // lozenge with a lit upper lip and a dark lower edge.
    var cols = 40, rows = 74;
    var sw = W / cols, sh = H / rows;
    for (var ry = 0; ry < rows; ry++) {
      for (var rx = 0; rx < cols + 1; rx++) {
        var ox = (rx + (ry % 2 ? 0.5 : 0)) * sw;
        var oy = ry * sh;
        var jitter = 0.86 + rnd() * 0.28;
        g.beginPath();
        g.ellipse(ox, oy + sh * 0.5, sw * 0.56 * jitter, sh * 0.62 * jitter, 0, 0, Math.PI * 2);
        g.strokeStyle = "rgba(6,14,12,0.22)";
        g.lineWidth = 1.2;
        g.stroke();
        g.beginPath();
        g.ellipse(ox, oy + sh * 0.40, sw * 0.40 * jitter, sh * 0.34 * jitter, 0, 0, Math.PI * 2);
        g.fillStyle = "rgba(235,255,248," + (0.035 + rnd() * 0.05).toFixed(3) + ")";
        g.fill();
      }
    }

    // A dark band through the eye line and a pale jaw, over the head's share of
    // v only. The eye sits about 25 degrees above the flank: u = 0.07 one side,
    // u = 0.43 the other.
    var headTop = H * 0.16;
    [0.07, 0.43].forEach(function (uu) {
      var gr = g.createLinearGradient(0, 0, 0, headTop);
      gr.addColorStop(0.0, "rgba(18,30,20,0.72)");
      gr.addColorStop(0.7, "rgba(18,30,20,0.30)");
      gr.addColorStop(1.0, "rgba(18,30,20,0)");
      g.fillStyle = gr;
      g.fillRect(uu * W - W * 0.028, 0, W * 0.056, headTop);
    });
    var jaw = g.createLinearGradient(0, 0, 0, headTop * 0.8);
    jaw.addColorStop(0.0, "rgba(226,232,180,0.34)");
    jaw.addColorStop(1.0, "rgba(226,232,180,0)");
    g.fillStyle = jaw;
    g.fillRect(W * 0.60, 0, W * 0.30, headTop * 0.8);

    // Belly scutes: wide plates across the underside, not lozenges.
    for (var by = 0; by < 86; by++) {
      var yy = by / 86 * H;
      g.fillStyle = "rgba(255,255,255,0.05)";
      g.fillRect(W * 0.62, yy, W * 0.26, H / 86 * 0.5);
      g.fillStyle = "rgba(0,0,0,0.10)";
      g.fillRect(W * 0.62, yy + H / 86 * 0.5, W * 0.26, H / 86 * 0.5);
    }
    return cv;
  }

  /** A floating text plate, used for the apple's label and the outcome. */
  function labelSprite(text, rgb, width) {
    var cv = document.createElement("canvas");
    cv.width = 512; cv.height = 128;
    var g = cv.getContext("2d");
    g.fillStyle = "rgba(7,20,38,0.86)";
    g.strokeStyle = "rgba(" + rgb + ",0.95)";
    g.lineWidth = 4;
    g.beginPath();
    if (g.roundRect) g.roundRect(6, 24, 500, 80, 10); else g.rect(6, 24, 500, 80);
    g.fill(); g.stroke();
    g.fillStyle = "rgb(" + rgb + ")";
    g.font = "600 40px ui-monospace, SFMono-Regular, Menlo, monospace";
    g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(text, 256, 65);
    var tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: tex, transparent: true, depthWrite: false, depthTest: false
    }));
    sprite.scale.set(width, width / 4, 1);
    sprite.renderOrder = 10;
    return sprite;
  }

  /**
   * Build the Snake world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind snake.v1.
   * @returns {Object} The scene module the player drives.
   */
  /* One environment's whole world, built once. Splitting it would mean passing
     twenty meshes and their update rules between helpers used exactly here. */
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var layout = payload.state_layout;
    var scene = core.scene;
    var renderer = core.renderer;
    var maxAniso = renderer.capabilities.getMaxAnisotropy();

    var GRID = world.grid_size;
    var CELLS = GRID * GRID;
    var WINDOW_R = world.window_radius;
    var HALF = (GRID - 1) / 2;
    // Row 0 is the far edge and column 0 the left one, which is the GIF's
    // orientation; keeping them the same means a reader can hold one picture.
    function wx(col) { return col - HALF; }
    function wz(row) { return row - HALF; }

    var reduceMotion = global.matchMedia
      ? global.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;

    scene.background = new THREE.Color(0x040A14).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x05101F, 0.022);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x2A4A7C, 0x090C12, 1.5));
    var moon = new THREE.DirectionalLight(0x9FB3D8, 0.75);
    moon.position.set(-6, 9, -4);
    scene.add(moon);
    core.buildNightEnvironment([
      [0.0, "#08131F"], [0.45, "#0E1B2B"], [0.55, "#221E14"], [1.0, "#050607"]
    ]);

    var poolTex = V.radialTexture(0.85, 0.42);
    var glowTex = V.radialTexture(0.9, 0.3);

    var stoneCv = stoneCanvas(20260920);
    var stoneNormal = V.normalMapFrom(renderer, stoneCv, 2.1);
    stoneNormal.anisotropy = maxAniso;
    var stoneAlbedo = new THREE.CanvasTexture(stoneCv);
    stoneAlbedo.anisotropy = maxAniso;
    stoneAlbedo.encoding = THREE.sRGBEncoding;

    /* One tile per playable cell, as a real slab with a grout gap, so the grid
       is geometry rather than a painted line. Instanced: one draw call, with a
       per-cell colour for the checker. */
    var tileMat = new THREE.MeshStandardMaterial({
      map: stoneAlbedo, normalMap: stoneNormal,
      normalScale: new THREE.Vector2(0.9, 0.9),
      roughness: 0.82, metalness: 0.04, envMapIntensity: 0.4
    });
    var tiles = new THREE.InstancedMesh(new THREE.BoxGeometry(0.94, 0.12, 0.94), tileMat, CELLS);
    tiles.receiveShadow = true;
    var tmpM = new THREE.Matrix4(), tmpC = new THREE.Color();
    for (var r = 0; r < GRID; r++) {
      for (var c = 0; c < GRID; c++) {
        tmpM.makeTranslation(wx(c), -0.06, wz(r));
        tiles.setMatrixAt(r * GRID + c, tmpM);
        // Scaled down: under this much lamp power the raw hex comes out pale
        // grey, and the point of taking the palette from the GIF is to look
        // like it.
        tiles.setColorAt(r * GRID + c, tmpC
          .setHex((r + c) % 2 ? COLORS.tileLight : COLORS.tileDark)
          .convertSRGBToLinear().multiplyScalar(0.60));
      }
    }
    tiles.instanceMatrix.needsUpdate = true;
    if (tiles.instanceColor) tiles.instanceColor.needsUpdate = true;
    scene.add(tiles);

    /* The wall is not a cell of the state -- it sits just outside the playable
       area -- so it is built as a ring of blocks one cell out, exactly where a
       head that leaves the grid ends up. */
    var wallMat = new THREE.MeshStandardMaterial({
      map: stoneAlbedo, normalMap: stoneNormal,
      normalScale: new THREE.Vector2(1.0, 1.0),
      roughness: 0.88, metalness: 0.05, envMapIntensity: 0.3
    });
    var ring = [];
    for (var rr = -1; rr <= GRID; rr++) {
      for (var cc = -1; cc <= GRID; cc++) {
        if (rr >= 0 && rr < GRID && cc >= 0 && cc < GRID) continue;
        ring.push([rr, cc]);
      }
    }
    var walls = new THREE.InstancedMesh(new THREE.BoxGeometry(0.98, 1, 0.98), wallMat, ring.length);
    walls.castShadow = true; walls.receiveShadow = true;
    var wallRnd = mulberry(4242);
    for (var wi = 0; wi < ring.length; wi++) {
      var h = 0.46 + wallRnd() * 0.16;
      tmpM.makeScale(1, h, 1);
      tmpM.setPosition(wx(ring[wi][1]), h / 2 - 0.02, wz(ring[wi][0]));
      walls.setMatrixAt(wi, tmpM);
      walls.setColorAt(wi, tmpC.setHex(COLORS.wallEdge).convertSRGBToLinear()
        .multiplyScalar((0.72 + wallRnd() * 0.5) * 0.90));
    }
    walls.instanceMatrix.needsUpdate = true;
    if (walls.instanceColor) walls.instanceColor.needsUpdate = true;
    scene.add(walls);

    /* Ground carrying on past the wall, so the pit is seated in a place rather
       than floating. Set well below the tiles rather than beside them: the
       render target's depth buffer is 16-bit and two near-coplanar planes
       stripe. */
    var outerAlbedo = stoneAlbedo.clone();
    outerAlbedo.needsUpdate = true;
    outerAlbedo.wrapS = outerAlbedo.wrapT = THREE.RepeatWrapping;
    outerAlbedo.repeat.set(8, 8);
    var outerNormal = stoneNormal.clone();
    outerNormal.needsUpdate = true;
    outerNormal.wrapS = outerNormal.wrapT = THREE.RepeatWrapping;
    outerNormal.repeat.set(8, 8);
    var outer = new THREE.Mesh(
      new THREE.PlaneGeometry(90, 90),
      new THREE.MeshStandardMaterial({
        map: outerAlbedo, normalMap: outerNormal,
        normalScale: new THREE.Vector2(0.8, 0.8),
        color: 0x6B7F99, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.1
      })
    );
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.14;
    outer.receiveShadow = true;
    scene.add(outer);

    /* ------------------------------------------------------------- belief
       Two readings of the same numbers. The floor sheet is the per-cell
       posterior at cell resolution with nearest sampling, so a cell is a cell
       and the ramp stays honest; the glow points give it height, so a spike is
       visible from the board camera without reading a colour ramp. */
    var beliefCv = document.createElement("canvas");
    beliefCv.width = beliefCv.height = GRID;
    var beliefCtx = beliefCv.getContext("2d");
    var beliefImg = beliefCtx.createImageData(GRID, GRID);
    var beliefTex = new THREE.CanvasTexture(beliefCv);
    beliefTex.magFilter = THREE.NearestFilter;
    beliefTex.minFilter = THREE.LinearFilter;
    beliefTex.generateMipmaps = false;
    var beliefSheet = new THREE.Mesh(
      new THREE.PlaneGeometry(GRID, GRID),
      new THREE.MeshBasicMaterial({
        map: beliefTex, transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending,
        // Held a little under full strength: the sheet is additive over lit
        // stone, and at 1.0 a hot cell blooms past the apple standing on it.
        opacity: 0.88
      })
    );
    beliefSheet.rotation.x = -Math.PI / 2;
    beliefSheet.position.y = 0.035;
    beliefSheet.renderOrder = 2;
    scene.add(beliefSheet);

    /* One soft amber glow per cell. Points rather than geometry: a solid column
       reads as an object standing on the board, and a posterior is not an
       object -- as columns this came out as a field of traffic cones. */
    var glowPos = new Float32Array(CELLS * 3);
    var glowCol = new Float32Array(CELLS * 3);
    for (var bc = 0; bc < CELLS; bc++) {
      glowPos[bc * 3] = wx(bc % GRID);
      glowPos[bc * 3 + 1] = 0.30;
      glowPos[bc * 3 + 2] = wz(Math.floor(bc / GRID));
    }
    var glowGeo = new THREE.BufferGeometry();
    glowGeo.setAttribute("position", new THREE.BufferAttribute(glowPos, 3));
    glowGeo.setAttribute("color", new THREE.BufferAttribute(glowCol, 3));
    var beliefCols = new THREE.Points(glowGeo, new THREE.PointsMaterial({
      size: 1.55, map: glowTex, vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.55
    }));
    beliefCols.renderOrder = 3;
    scene.add(beliefCols);

    // A weak amber light at the belief's mode, so the posterior spills a little
    // real light onto the stone instead of floating over it.
    var beliefLight = new THREE.PointLight(COLORS.amber, 1, 6.5, 2);
    beliefLight.power = 90;
    beliefLight.position.set(0, 0.5, 0);
    scene.add(beliefLight);

    /* --------------------------------------------------- the vision window
       A cyan cage one cell tall around the clipped window, plus a pool on the
       stone under it. Clipped to the grid exactly as window_cells() clips it,
       so its area shrinks against a wall the way the sensor's does. */
    var winGroup = new THREE.Group();
    var winEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({
        color: new THREE.Color(0.55, 3.4, 3.7), transparent: true, opacity: 0.9
      })
    );
    winGroup.add(winEdges);
    var winPool = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: new THREE.Color(0.10, 0.62, 0.70),
        transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending, opacity: 0.30
      })
    );
    winPool.rotation.x = -Math.PI / 2;
    winPool.position.y = 0.03;
    winPool.renderOrder = 1;
    winGroup.add(winPool);
    scene.add(winGroup);

    var senseLight = new THREE.SpotLight(0x8FF0FA, 1, 11, 0.74, 0.85, 2);
    senseLight.power = 520;
    scene.add(senseLight);
    scene.add(senseLight.target);

    /* ----------------------------------------------------------- lanterns */
    var lanterns = [];
    var lampGlass = new THREE.MeshBasicMaterial({ color: new THREE.Color(4.2, 3.4, 2.3) });
    var brass = new THREE.MeshStandardMaterial({
      color: 0x8A6E3E, roughness: 0.34, metalness: 0.9
    });
    [
      [-2.2, -2.2], [-2.2, GRID + 1.2], [GRID + 1.2, -2.2], [GRID + 1.2, GRID + 1.2]
    ].forEach(function (cell) {
      var group = new THREE.Group();
      var post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.11, 3.7, 12), brass);
      post.position.y = 1.85; post.castShadow = true;
      group.add(post);
      var cage = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.02, 8, 18), brass);
      cage.rotation.x = Math.PI / 2; cage.position.y = 3.78;
      group.add(cage);
      var glass = new THREE.Mesh(new THREE.SphereGeometry(0.145, 16, 12), lampGlass);
      glass.position.y = 3.78;
      group.add(glass);
      var halo = new THREE.Mesh(
        new THREE.PlaneGeometry(1.5, 1.5),
        new THREE.MeshBasicMaterial({
          map: glowTex, color: new THREE.Color(1.5, 1.15, 0.6),
          transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
        })
      );
      halo.position.y = 3.78;
      group.add(halo);
      var light = new THREE.PointLight(COLORS.lamp, 1, 60, 2);
      light.power = LANTERN_LUMENS;
      light.position.y = 3.78;
      group.add(light);
      group.position.set(wx(cell[1]), 0.3, wz(cell[0]));
      scene.add(group);
      lanterns.push({ light: light, x: group.position.x, z: group.position.z });
    });

    /* One shadow caster, moved to the lantern nearest the snake and paid for
       out of that lantern's own output, so the snake has a shadow without one
       corner of the pit going bright. Narrow cone plus normalBias, which is the
       fix for acne that a big negative bias is not. */
    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 30, 0.52, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(1024, 1024);
    shadowLight.shadow.camera.near = 0.5;
    shadowLight.shadow.camera.far = 30;
    shadowLight.shadow.normalBias = 0.035;
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    /* ---------------------------------------------------------- the snake
       M sections of N segments each, written in place every frame, so a body
       that grows or slides costs no allocation. */
    var skinCv = snakeSkin();
    var skinTex = new THREE.CanvasTexture(skinCv);
    skinTex.encoding = THREE.sRGBEncoding;
    skinTex.anisotropy = maxAniso;
    skinTex.wrapS = THREE.RepeatWrapping;
    var skinNormal = V.normalMapFrom(renderer, skinCv, 1.5);
    skinNormal.anisotropy = maxAniso;
    skinNormal.wrapS = THREE.RepeatWrapping;

    var M = 96, N = 22;
    var VCOUNT = M * (N + 1) + 2;          // two more for the cap centres
    var TAIL_CAP = M * (N + 1), NOSE_CAP = TAIL_CAP + 1;
    var bodyPos = new Float32Array(VCOUNT * 3);
    var bodyUv = new Float32Array(VCOUNT * 2);
    var bodyIdx = [];
    for (var bi = 0; bi < M - 1; bi++) {
      for (var bj = 0; bj < N; bj++) {
        var q0 = bi * (N + 1) + bj, q1 = q0 + 1, q2 = q0 + (N + 1), q3 = q2 + 1;
        bodyIdx.push(q0, q1, q3, q0, q3, q2);
      }
    }
    // Fans closing each end, wound so they face outwards: reversed, a nose
    // renders as a dark crater.
    for (var cj = 0; cj < N; cj++) {
      bodyIdx.push(cj, TAIL_CAP, cj + 1);
      bodyIdx.push((M - 1) * (N + 1) + cj, (M - 1) * (N + 1) + cj + 1, NOSE_CAP);
    }
    var bodyGeo = new THREE.BufferGeometry();
    bodyGeo.setAttribute("position", new THREE.BufferAttribute(bodyPos, 3));
    bodyGeo.setAttribute("uv", new THREE.BufferAttribute(bodyUv, 2));
    bodyGeo.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(VCOUNT * 3), 3));
    bodyGeo.setIndex(bodyIdx);
    bodyUv[TAIL_CAP * 2] = 0.5; bodyUv[TAIL_CAP * 2 + 1] = 0.0;
    bodyUv[NOSE_CAP * 2] = 0.5; bodyUv[NOSE_CAP * 2 + 1] = 1.0;
    for (var ui = 0; ui < M; ui++) {
      for (var uj = 0; uj <= N; uj++) {
        var uo = (ui * (N + 1) + uj) * 2;
        bodyUv[uo] = uj / N;
        bodyUv[uo + 1] = ui / (M - 1);
      }
    }
    var snakeMesh = new THREE.Mesh(bodyGeo, new THREE.MeshStandardMaterial({
      map: skinTex, normalMap: skinNormal,
      normalScale: new THREE.Vector2(1.25, 1.25),
      roughness: 0.64, metalness: 0.0, envMapIntensity: 0.22
    }));
    snakeMesh.castShadow = true;
    snakeMesh.receiveShadow = true;
    scene.add(snakeMesh);

    /* The section profile, tail (s = 0) to snout (s = 1). Control points read
       off a real colubrid: a needle tail, the thickest part about a third back
       from the head, a narrowed neck, then the head flaring wider than the neck
       and flattening into a wedge. */
    var PROFILE = [
      // s,   halfWidth, topScale, botScale, superellipse exponent
      [0.000, 0.012, 0.95, 0.95, 2.0],
      [0.090, 0.070, 0.95, 0.90, 2.1],
      [0.300, 0.134, 0.96, 0.84, 2.3],
      [0.560, 0.158, 0.98, 0.86, 2.4],
      [0.720, 0.150, 0.98, 0.86, 2.4],
      [0.815, 0.118, 0.98, 0.92, 2.2],   // neck
      [0.900, 0.186, 0.80, 0.62, 3.0],   // skull: wide, flat, square-shouldered
      [0.950, 0.182, 0.74, 0.58, 3.1],
      [0.980, 0.136, 0.66, 0.52, 2.8],
      [1.000, 0.055, 0.58, 0.46, 2.5]    // rounded snout
    ];
    function profileAt(s) {
      var i = 0;
      while (i < PROFILE.length - 2 && s > PROFILE[i + 1][0]) i++;
      var a = PROFILE[i], b = PROFILE[i + 1];
      var t = smooth(clamp((s - a[0]) / Math.max(b[0] - a[0], 1e-6), 0, 1));
      return {
        hw: lerp(a[1], b[1], t), top: lerp(a[2], b[2], t),
        bot: lerp(a[3], b[3], t), exp: lerp(a[4], b[4], t)
      };
    }

    /* One polyline the whole body always lies along. Index 0 is the tail at
       step 0; every later step appends the new head cell. Because eating keeps
       the tail where it is, the tail index simply stops advancing on that step
       -- so the same curve carries growth with no special case. */
    var bodies = payload.bodies;
    var path = [];
    var first = bodies[0];
    for (var pi = first.length - 1; pi >= 0; pi--) path.push(first[pi]);
    for (var pt = 1; pt < bodies.length; pt++) path.push(bodies[pt][0]);
    var headIdx = [], tailIdx = [];
    for (var hi = 0; hi < bodies.length; hi++) {
      headIdx.push(first.length - 1 + hi);
      tailIdx.push(first.length - 1 + hi - (payload.lengths[hi] - 1));
    }
    var points = path.map(function (cell) {
      return new THREE.Vector3(wx(cell[1]), 0, wz(cell[0]));
    });
    // A two-point chain has no curvature for Catmull-Rom to work with; the
    // shortest reachable body is three cells, so this only guards a trace that
    // should not exist.
    if (points.length < 3) points.push(points[points.length - 1].clone());
    var curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.5);
    var pathN = points.length;

    // Scratch vectors, allocated once: this runs M times a frame.
    var sP = [], sT = [];
    for (var si = 0; si < M; si++) { sP.push(new THREE.Vector3()); sT.push(new THREE.Vector3()); }
    var vRight = new THREE.Vector3(), vUp = new THREE.Vector3(), vTmp = new THREE.Vector3();
    var UPREF = new THREE.Vector3(0, 1, 0);
    var headP = new THREE.Vector3(), headT = new THREE.Vector3(),
      headR = new THREE.Vector3(), headU = new THREE.Vector3();

    function buildBody(tailAt, headAt, elapsed, moving) {
      var i, s;
      for (i = 0; i < M; i++) {
        s = i / (M - 1);
        curve.getPoint(clamp(lerp(tailAt, headAt, s) / (pathN - 1), 0, 1), sP[i]);
        /* A ride height that ramps off the tail tip and then holds flat, plus a
           lift under the head, the way a snake carries it when hunting. Both
           ends of the lift have zero slope, so the head's tangent stays
           horizontal and the head stays level. */
        sP[i].y = 0.022 + 0.124 * smooth(clamp(s / 0.24, 0, 1))
          + 0.055 * smooth(clamp((s - 0.68) / 0.32, 0, 1));
      }
      for (i = 0; i < M; i++) {
        sT[i].subVectors(sP[Math.min(M - 1, i + 1)], sP[Math.max(0, i - 1)]);
        if (sT[i].lengthSq() < 1e-12) sT[i].set(0, 0, 1);
        sT[i].normalize();
      }
      if (!reduceMotion) {
        // The lateral ripple has to be added across the tangent, or it just
        // shifts the whole animal sideways.
        var amp = moving ? 0.052 : 0.016;
        for (i = 0; i < M; i++) {
          s = i / (M - 1);
          // Dies away at both ends: the head leads, the tail tip is thin.
          var w = Math.sin(Math.PI * clamp(s * 1.06, 0, 1))
            * (1 - clamp((s - 0.78) / 0.22, 0, 1) * 0.85);
          vRight.crossVectors(UPREF, sT[i]);
          if (vRight.lengthSq() < 1e-8) vRight.set(1, 0, 0);
          vRight.normalize();
          sP[i].addScaledVector(vRight, Math.sin(s * 9.0 - elapsed * 4.2) * amp * w);
        }
        for (i = 0; i < M; i++) {
          sT[i].subVectors(sP[Math.min(M - 1, i + 1)], sP[Math.max(0, i - 1)]);
          if (sT[i].lengthSq() < 1e-12) sT[i].set(0, 0, 1);
          sT[i].normalize();
        }
      }

      for (i = 0; i < M; i++) {
        s = i / (M - 1);
        var p = profileAt(s);
        vRight.crossVectors(UPREF, sT[i]);
        if (vRight.lengthSq() < 1e-8) vRight.set(1, 0, 0);
        vRight.normalize();
        vUp.crossVectors(sT[i], vRight).normalize();
        var k = 2 / p.exp;
        for (var j = 0; j <= N; j++) {
          var ang = (j / N) * Math.PI * 2;
          var cs = Math.cos(ang), sn = Math.sin(ang);
          var cc = (cs < 0 ? -1 : 1) * Math.pow(Math.abs(cs), k);
          var ss = (sn < 0 ? -1 : 1) * Math.pow(Math.abs(sn), k);
          vTmp.copy(sP[i])
            .addScaledVector(vRight, p.hw * cc)
            .addScaledVector(vUp, p.hw * (sn >= 0 ? p.top : p.bot) * ss);
          var o = (i * (N + 1) + j) * 3;
          bodyPos[o] = vTmp.x; bodyPos[o + 1] = vTmp.y; bodyPos[o + 2] = vTmp.z;
        }
        if (i === 0) {
          vTmp.copy(sP[0]).addScaledVector(sT[0], -p.hw * 0.7);
          bodyPos[TAIL_CAP * 3] = vTmp.x;
          bodyPos[TAIL_CAP * 3 + 1] = vTmp.y;
          bodyPos[TAIL_CAP * 3 + 2] = vTmp.z;
        }
        if (i === M - 1) {
          headP.copy(sP[i]); headT.copy(sT[i]); headR.copy(vRight); headU.copy(vUp);
          // The snout is rounded, so its cap centre sits forward of the ring.
          vTmp.copy(sP[i]).addScaledVector(sT[i], p.hw * 1.15);
          bodyPos[NOSE_CAP * 3] = vTmp.x;
          bodyPos[NOSE_CAP * 3 + 1] = vTmp.y;
          bodyPos[NOSE_CAP * 3 + 2] = vTmp.z;
        }
      }
      bodyGeo.attributes.position.needsUpdate = true;
      bodyGeo.computeVertexNormals();
      /* The ring is duplicated at j = 0 and j = N so the UV can wrap, which
         leaves the seam's two halves with different normals and a crease down
         the flank. Averaging them closes it. */
      var nrm = bodyGeo.attributes.normal.array;
      for (i = 0; i < M; i++) {
        var a3 = (i * (N + 1)) * 3, b3 = (i * (N + 1) + N) * 3;
        var nx = nrm[a3] + nrm[b3], ny = nrm[a3 + 1] + nrm[b3 + 1], nz = nrm[a3 + 2] + nrm[b3 + 2];
        var len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        nrm[a3] = nrm[b3] = nx / len;
        nrm[a3 + 1] = nrm[b3 + 1] = ny / len;
        nrm[a3 + 2] = nrm[b3 + 2] = nz / len;
      }
      bodyGeo.attributes.normal.needsUpdate = true;
      bodyGeo.computeBoundingSphere();
    }

    /* Head furniture: eyes with slit pupils and a tongue, parented to nothing
       and placed from the head's own frame every update, so they cannot drift
       out of alignment with a body that is rebuilt from scratch. */
    var eyeMat = new THREE.MeshStandardMaterial({
      color: 0x1A1206, roughness: 0.06, metalness: 0.0, envMapIntensity: 2.0,
      emissive: 0xFFC24A, emissiveIntensity: 0.30
    });
    var pupilMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.02, 0.02, 0.02) });
    var eyes = [], pupils = [];
    for (var ei = 0; ei < 2; ei++) {
      var eye = new THREE.Mesh(new THREE.SphereGeometry(0.034, 14, 12), eyeMat);
      scene.add(eye); eyes.push(eye);
      var pupil = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.034, 0.010), pupilMat);
      scene.add(pupil); pupils.push(pupil);
    }
    var tongueMat = new THREE.MeshStandardMaterial({
      color: 0xA8243A, roughness: 0.35, metalness: 0.0, emissive: 0x2A0508
    });
    var tongue = new THREE.Group();
    var tongueStem = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.010, 0.11, 6), tongueMat);
    tongueStem.rotation.x = Math.PI / 2;
    tongueStem.position.z = 0.055;
    tongue.add(tongueStem);
    [-1, 1].forEach(function (sgn) {
      var fork = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.007, 0.07, 5), tongueMat);
      fork.rotation.x = Math.PI / 2;
      fork.rotation.y = sgn * 0.42;
      fork.position.set(sgn * 0.014, 0, 0.142);
      tongue.add(fork);
    });
    scene.add(tongue);

    /* ---------------------------------------------------------- the apple
       The hidden cell, drawn for the reader only. A red pin rises from it and a
       tag says so, because a viewer that shows the answer without saying it is
       the answer is worse than one that hides it. */
    var appleGroup = new THREE.Group();
    var appleBody = new THREE.Mesh(
      new THREE.SphereGeometry(0.24, 26, 20),
      new THREE.MeshStandardMaterial({
        color: COLORS.apple, roughness: 0.44, metalness: 0.0, envMapIntensity: 0.4
      })
    );
    appleBody.scale.set(1, 0.92, 1);
    appleBody.position.y = 0.24;
    appleBody.castShadow = true;
    appleGroup.add(appleBody);
    // The dimple at the stem, which is most of what makes a red ball an apple.
    var dimple = new THREE.Mesh(
      new THREE.SphereGeometry(0.070, 14, 10),
      new THREE.MeshStandardMaterial({ color: COLORS.appleDark, roughness: 0.5 })
    );
    dimple.position.y = 0.425;
    dimple.scale.set(1, 0.45, 1);
    appleGroup.add(dimple);
    var stem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.014, 0.02, 0.15, 6),
      new THREE.MeshStandardMaterial({ color: 0x4A3520, roughness: 0.85 })
    );
    stem.position.y = 0.50;
    stem.rotation.z = 0.2;
    appleGroup.add(stem);
    var leaf = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 10, 8),
      new THREE.MeshStandardMaterial({ color: 0x3E7A2A, roughness: 0.7, side: THREE.DoubleSide })
    );
    leaf.scale.set(1.5, 0.12, 0.7);
    leaf.position.set(0.09, 0.53, 0);
    leaf.rotation.z = -0.35;
    appleGroup.add(leaf);
    var pin = new THREE.Mesh(
      new THREE.CylinderGeometry(0.008, 0.008, 1.5, 6),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(1.4, 0.24, 0.22), transparent: true, opacity: 0.55
      })
    );
    pin.position.y = 1.05;
    appleGroup.add(pin);
    var appleTag = labelSprite("HIDDEN FROM THE AGENT", "236,120,112", 2.5);
    appleTag.position.y = 1.85;
    appleGroup.add(appleTag);
    // A cyan reticle that snaps round the apple on the step the window fired.
    var reticle = new THREE.Mesh(
      new THREE.RingGeometry(0.34, 0.40, 40),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.5, 3.2, 3.5), transparent: true,
        side: THREE.DoubleSide, depthWrite: false
      })
    );
    reticle.rotation.x = -Math.PI / 2;
    reticle.position.y = 0.05;
    appleGroup.add(reticle);
    scene.add(appleGroup);

    /* -------------------------------------------------------- fatal turns
       One marker per action, over the cell that turn would step into. Wall and
       body get different colours because they are different mistakes: the wall
       is the beginner's, the body is the one that ends a long game. */
    var dangerMarks = [];
    for (var dm = 0; dm < 3; dm++) {
      var group = new THREE.Group();
      var mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(2.4, 0.4, 0.5), transparent: true,
        opacity: 0.85, depthWrite: false
      });
      for (var kb = 0; kb < 2; kb++) {
        var bar = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.03, 0.09), mat);
        bar.rotation.y = kb ? -Math.PI / 4 : Math.PI / 4;
        group.add(bar);
      }
      group.position.y = 0.09;
      group.visible = false;
      group.userData.mat = mat;
      scene.add(group);
      dangerMarks.push(group);
    }

    /* ------------------------------------------------------------ outcome */
    var outcomeTag = labelSprite("", "255,255,255", 3.4);
    outcomeTag.visible = false;
    scene.add(outcomeTag);
    var flash = new THREE.PointLight(0xFF3A46, 1, 14, 2);
    flash.power = 0;
    scene.add(flash);

    /* ---------------------------------------------------------- dust motes */
    var MOTES = 240;
    var motePos = new Float32Array(MOTES * 3);
    var moteRnd = mulberry(1357);
    for (var mi = 0; mi < MOTES; mi++) {
      motePos[mi * 3] = (moteRnd() - 0.5) * (GRID + 3);
      motePos[mi * 3 + 1] = moteRnd() * 2.3 + 0.08;
      motePos[mi * 3 + 2] = (moteRnd() - 0.5) * (GRID + 3);
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    scene.add(new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.035, map: glowTex, color: new THREE.Color(0.8, 0.95, 1.2),
      transparent: true, opacity: 0.34, depthWrite: false,
      blending: THREE.AdditiveBlending
    })));

    core.linearize();

    /* ------------------------------------------------------ belief readout
       The page's HUD is shared across every environment and has one belief
       slot, which is not enough for a 2-D posterior. So this scene adds its own
       panel inside the viewer: the same numbers as the amber field, at cell
       resolution, beside the reading they came from and the verdict on each
       turn. It is created here rather than in the site's HTML because a scene
       module owns its own presentation and edits no shared file. */
    var root = document.getElementById("viewer");
    var panel = document.createElement("div");
    panel.className = "snake-belief";
    panel.style.cssText = [
      "position:absolute", "top:10px", "right:10px", "width:190px",
      "padding:8px 10px", "border-radius:6px", "background:rgba(7,20,38,.72)",
      "color:#e8f4ff", "font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace",
      "pointer-events:none"
    ].join(";");
    var mapSize = Math.max(96, Math.min(168, GRID * 12));
    panel.innerHTML =
      '<div style="color:#7fa8cc;letter-spacing:.04em">APPLE POSTERIOR</div>' +
      '<canvas width="' + mapSize + '" height="' + mapSize + '" ' +
      'style="width:100%;height:auto;margin:5px 0;border:1px solid #1d4570;' +
      'image-rendering:pixelated;display:block"></canvas>' +
      '<div data-role="entropy" style="color:#FFB02E"></div>' +
      '<div data-role="reading" style="color:#2fe3f2"></div>' +
      '<div data-role="hunger" style="color:#7fa8cc"></div>' +
      '<div data-role="turns" style="margin-top:4px"></div>';
    root.appendChild(panel);
    var mapCanvas = panel.querySelector("canvas");
    var mapCtx = mapCanvas.getContext("2d");
    var entropyEl = panel.querySelector('[data-role="entropy"]');
    var readingEl = panel.querySelector('[data-role="reading"]');
    var hungerEl = panel.querySelector('[data-role="hunger"]');
    var turnsEl = panel.querySelector('[data-role="turns"]');

    /* The posterior over the apple, read out of the belief core serialised.
       Every belief this environment ships is a particle belief, and its
       particles are whole Snake states, so the apple cell is read from the
       slots the payload's own state_layout names -- never from the true apple,
       which is exactly what a belief drawn from the answer would be. */
    var posterior = new Float64Array(CELLS);
    function readPosterior(belief) {
      posterior.fill(0);
      if (!belief || belief.kind !== "particles" || !belief.particles.length) return null;
      var particles = belief.particles, weights = belief.weights;
      var mass = 0;
      for (var i = 0; i < particles.length; i++) {
        var particle = particles[i];
        if (!Array.isArray(particle)) return null;
        var row = Math.round(particle[layout.food_row]);
        var col = Math.round(particle[layout.food_col]);
        // A won episode's states carry no apple; its slots hold the empty
        // marker, and there is nothing left to be uncertain about.
        if (row < 0 || col < 0 || row >= GRID || col >= GRID) continue;
        var w = weights[i] === undefined ? 1 / particles.length : weights[i];
        posterior[row * GRID + col] += w;
        mass += w;
      }
      if (mass <= 0) return null;
      for (var k = 0; k < CELLS; k++) posterior[k] /= mass;
      return belief;
    }

    function entropyBits() {
      var h = 0;
      for (var i = 0; i < CELLS; i++) {
        var p = posterior[i];
        if (p > 1e-12) h -= p * Math.log2(p);
      }
      return h;
    }

    /** Paint the board's amber field, and return the mode's flat cell index. */
    function paintField(peak, live) {
      var data = beliefImg.data;
      var best = -1, bestP = 0;
      for (var i = 0; i < CELLS; i++) {
        var rel = shade(posterior[i]);
        // Additive sheet: the alpha channel does the work, and the colour is
        // the visualizer's amber, so a faint cell does not wash out to white.
        data[i * 4] = 255; data[i * 4 + 1] = 176; data[i * 4 + 2] = 46;
        data[i * 4 + 3] = live ? Math.round(rel * BELIEF_MAX_ALPHA * 255) : 0;
        if (posterior[i] > bestP) { bestP = posterior[i]; best = i; }
        var glow = live ? rel * 1.25 : 0;
        glowCol[i * 3] = glow;
        glowCol[i * 3 + 1] = glow * 0.60;
        glowCol[i * 3 + 2] = glow * 0.13;
      }
      beliefCtx.putImageData(beliefImg, 0, 0);
      beliefTex.needsUpdate = true;
      glowGeo.attributes.color.needsUpdate = true;
      if (best >= 0 && live) {
        beliefLight.position.set(wx(best % GRID), 0.45, wz(Math.floor(best / GRID)));
        beliefLight.power = 60 + 130 * Math.pow(peak, 0.4);
      } else {
        beliefLight.power = 0;
      }
      return best;
    }

    /* The panel's own picture of the same numbers. The board is the truth and
       this is what the agent has; the two are meant to look different. */
    function paintInset(head, food, live) {
      var s = mapCanvas.width / GRID;
      mapCtx.fillStyle = "#07121F";
      mapCtx.fillRect(0, 0, mapCanvas.width, mapCanvas.height);
      if (live) {
        for (var i = 0; i < CELLS; i++) {
          var a = shade(posterior[i]) * 0.95;
          if (a > 0.004) {
            mapCtx.fillStyle = "rgba(255,176,46," + a.toFixed(3) + ")";
            mapCtx.fillRect((i % GRID) * s, Math.floor(i / GRID) * s, s, s);
          }
        }
      }
      // The window, clipped exactly as window_cells() clips it.
      var r0 = Math.max(head[0] - WINDOW_R, 0), c0 = Math.max(head[1] - WINDOW_R, 0);
      var r1 = Math.min(head[0] + WINDOW_R, GRID - 1), c1 = Math.min(head[1] + WINDOW_R, GRID - 1);
      if (r1 >= r0 && c1 >= c0) {
        mapCtx.strokeStyle = "rgba(47,227,242,0.9)";
        mapCtx.lineWidth = 1;
        mapCtx.strokeRect(c0 * s + 0.5, r0 * s + 0.5, (c1 - c0 + 1) * s - 1, (r1 - r0 + 1) * s - 1);
      }
      if (head[0] >= 0 && head[0] < GRID && head[1] >= 0 && head[1] < GRID) {
        mapCtx.fillStyle = "#59EB54";
        mapCtx.fillRect(head[1] * s + 2, head[0] * s + 2, s - 4, s - 4);
      }
      if (food) {
        mapCtx.strokeStyle = "#ff4d5a";
        mapCtx.lineWidth = 2;
        mapCtx.strokeRect(food[1] * s + 1, food[0] * s + 1, s - 2, s - 2);
      }
    }

    function headingOf(body) {
      if (body.length < 2) return [0, 1];
      return [body[0][0] - body[1][0], body[0][1] - body[1][1]];
    }
    function turnedHeading(heading, action) {
      var index = 0;
      for (var i = 0; i < 4; i++) {
        if (DIRECTIONS[i][0] === heading[0] && DIRECTIONS[i][1] === heading[1]) index = i;
      }
      if (action === 0) index = (index + 3) % 4;
      if (action === 2) index = (index + 1) % 4;
      return DIRECTIONS[index];
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

    function sampleAt(t) {
      var n = bodies.length;
      var i0 = Math.floor(clamp(t, 0, n - 1));
      var i1 = Math.min(i0 + 1, n - 1);
      var f = clamp(t - i0, 0, 1);
      return {
        index: i0,
        head: lerp(headIdx[i0], headIdx[i1], f),
        tail: lerp(tailIdx[i0], tailIdx[i1], f)
      };
    }

    var taggedTermination = -1;

    return {
      steps: bodies.length,

      /* Framing scales with the board, because a trace decides how big it is:
         the distance that frames a 12x12 grid leaves a 6x6 one as a smudge in
         the middle of the canvas. The constant term is headroom for the props,
         which do not scale with the grid -- a lantern post is the same height
         on any board. */
      camera: {
        board: [0, GRID * 1.05 + 3.8, GRID * 1.03 + 3.3],
        top: [0, GRID * 1.40 + 4.7, 0.01]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var index = sample.index;
        var step = trace.steps[index] || {};
        var body = bodies[index];
        var head = body[0];
        var food = payload.foods[index];
        var termination = payload.terminations[index];
        var live = termination === 0;

        buildBody(sample.tail, sample.head, elapsed, playing && step.action !== null);

        // Head furniture, placed from the head's own frame, never a guess.
        var flick = reduceMotion ? 0 : Math.max(0, Math.sin(elapsed * 2.3) * 3 - 2.1);
        for (var e = 0; e < 2; e++) {
          var sgn = e === 0 ? 1 : -1;
          vTmp.copy(headP)
            .addScaledVector(headT, -0.150)
            .addScaledVector(headR, sgn * 0.150)
            .addScaledVector(headU, 0.072);
          eyes[e].position.copy(vTmp);
          pupils[e].position.copy(vTmp)
            .addScaledVector(headR, sgn * 0.026)
            .addScaledVector(headT, 0.002);
          pupils[e].lookAt(vTmp.clone().addScaledVector(headR, sgn * 2));
        }
        tongue.position.copy(headP).addScaledVector(headT, 0.045).addScaledVector(headU, -0.028);
        tongue.scale.setScalar(0.55 + flick * 0.55);
        // Object3D.lookAt points +Z at the target -- cameras and lights are the
        // ones that look down -Z -- and the tongue is modelled along +Z.
        tongue.lookAt(vTmp.copy(headP).addScaledVector(headT, 1));
        tongue.visible = live;

        // The vision window, clipped to the grid as the sensor clips it.
        var r0 = Math.max(head[0] - WINDOW_R, 0), c0 = Math.max(head[1] - WINDOW_R, 0);
        var r1 = Math.min(head[0] + WINDOW_R, GRID - 1);
        var c1 = Math.min(head[1] + WINDOW_R, GRID - 1);
        var haveWindow = r1 >= r0 && c1 >= c0;
        winGroup.visible = haveWindow;
        if (haveWindow) {
          var w = c1 - c0 + 1, d = r1 - r0 + 1;
          var cx = wx((c0 + c1) / 2), cz = wz((r0 + r1) / 2);
          winEdges.scale.set(w, 0.9, d);
          winEdges.position.set(cx, 0.42, cz);
          winPool.scale.set(w * 1.5, d * 1.5, 1);
          winPool.position.set(cx, 0.03, cz);
          winEdges.material.opacity = 0.55
            + (reduceMotion ? 0.2 : Math.sin(elapsed * 2.4) * 0.18 + 0.18);
        }
        senseLight.position.copy(headP).setY(2.6);
        senseLight.target.position.set(wx(head[1]), 0, wz(head[0]));
        senseLight.target.updateMatrixWorld();

        // The belief, and the entropy that is the whole story of this episode.
        var belief = readPosterior(payload.beliefs[index]);
        var peak = 0;
        for (var pk = 0; pk < CELLS; pk++) if (posterior[pk] > peak) peak = posterior[pk];
        var drawn = !!belief;
        paintField(peak, drawn);
        beliefSheet.visible = drawn;
        beliefCols.visible = drawn;
        var bits = drawn ? entropyBits() : 0;

        // The apple.
        appleGroup.visible = !!food;
        if (food) {
          appleGroup.position.set(wx(food[1]), reduceMotion ? 0 : Math.sin(elapsed * 1.5) * 0.015,
            wz(food[0]));
          appleBody.rotation.y = elapsed * 0.4;
          var sighted = !!payload.sightings[index];
          reticle.visible = sighted;
          reticle.scale.setScalar(sighted ? 1 + Math.sin(elapsed * 5) * 0.05 : 1);
        }

        // The fatal turns, from the environment's own transition outcome.
        var turnOutcomes = payload.turn_outcomes[index];
        var heading = headingOf(body);
        for (var a = 0; a < 3; a++) {
          var mark = dangerMarks[a];
          var fatal = turnOutcomes && (turnOutcomes[a] === 1 || turnOutcomes[a] === 2);
          mark.visible = !!fatal;
          if (!fatal) continue;
          var th = turnedHeading(heading, a);
          mark.position.set(wx(head[1] + th[1]), turnOutcomes[a] === 1 ? 0.74 : 0.12,
            wz(head[0] + th[0]));
          var hot = 0.7 + (reduceMotion ? 0.2 : Math.sin(elapsed * 4 + a) * 0.2 + 0.2);
          if (turnOutcomes[a] === 1) mark.userData.mat.color.setRGB(2.2 * hot, 0.8 * hot, 0.3 * hot);
          else mark.userData.mat.color.setRGB(2.6 * hot, 0.25 * hot, 0.35 * hot);
        }

        // The outcome.
        outcomeTag.visible = !live;
        if (!live) {
          outcomeTag.position.set(headP.x, 1.7, headP.z);
          if (taggedTermination !== termination) {
            taggedTermination = termination;
            var rgb = termination === 4 ? "120,235,140"
              : termination === 3 ? "245,196,81" : "255,120,128";
            var replacement = labelSprite(TERMINATION[termination].toUpperCase(), rgb, 3.6);
            outcomeTag.material.map.dispose();
            outcomeTag.material.map = replacement.material.map;
            outcomeTag.material.needsUpdate = true;
          }
          flash.position.set(headP.x, 0.8, headP.z);
          flash.color.setHex(termination === 4 ? 0x46E07A : 0xFF3A46);
          flash.power = 700 * (0.5 + 0.5 * Math.abs(Math.sin(elapsed * 3)));
        } else {
          flash.power = 0;
        }

        // Lanterns: one shadow caster, on the nearest one, paid for out of it.
        var nearest = null, nd = Infinity;
        for (var li = 0; li < lanterns.length; li++) {
          var lamp = lanterns[li];
          lamp.light.power = LANTERN_LUMENS;
          var dd = (lamp.x - headP.x) * (lamp.x - headP.x) + (lamp.z - headP.z) * (lamp.z - headP.z);
          if (dd < nd) { nd = dd; nearest = lamp; }
        }
        if (nearest) {
          shadowLight.position.set(nearest.x, 3.78, nearest.z);
          shadowLight.target.position.set(headP.x, 0.15, headP.z);
          shadowLight.target.updateMatrixWorld();
          // The spot covers a narrow cone, so it takes the share of the lamp's
          // output that points at the snake and leaves the rest to the lamp.
          var share = 0.22 * clamp(1 - (Math.sqrt(nd) - 2.0) / 14.0, 0.1, 1);
          shadowLight.power = LANTERN_LUMENS * share;
          nearest.light.power = LANTERN_LUMENS * (1 - share);
        }

        if (!reduceMotion) {
          for (var m = 0; m < MOTES; m++) {
            motePos[m * 3] += Math.sin(elapsed * 0.3 + m) * 0.0014;
            motePos[m * 3 + 1] += 0.0018;
            if (motePos[m * 3 + 1] > 2.4) motePos[m * 3 + 1] = 0.1;
          }
          moteGeo.attributes.position.needsUpdate = true;
        }

        // The panel.
        paintInset(head, food, drawn);
        var beliefLabel;
        if (!drawn) {
          beliefLabel = payload.beliefs[index]
            ? "not drawn (" + (payload.beliefs[index].belief_class ||
              payload.beliefs[index].kind) + ")"
            : "—";
          entropyEl.textContent = live ? beliefLabel : "episode over";
        } else {
          beliefLabel = bits.toFixed(2) + " bits, best cell "
            + (peak * 100).toFixed(1) + "%";
          entropyEl.textContent = "entropy " + bits.toFixed(2) + " bits · best "
            + (peak * 100).toFixed(1) + "%";
        }
        var scent = payload.scents[index];
        var seen = payload.sightings[index];
        readingEl.textContent = scent === null || scent === undefined
          ? "no reading yet"
          : "scent " + QUADRANTS[scent] + (seen ? " · SIGHTED " + seen[0] + "," + seen[1]
            : " · not sighted");
        hungerEl.textContent = "length " + payload.lengths[index] + "/" + world.target_length
          + " · hunger " + payload.steps_since_food[index] + "/" + world.starvation_limit;
        var rows = "";
        for (var ta = 0; ta < 3; ta++) {
          var verdict = "—", colour = "#7fa8cc";
          if (turnOutcomes) {
            if (turnOutcomes[ta] === 1) { verdict = "WALL"; colour = "#f5c451"; }
            else if (turnOutcomes[ta] === 2) { verdict = "OWN BODY"; colour = "#ff4d5a"; }
            else { verdict = "safe"; }
          }
          rows += '<div style="color:' + colour + '">' + ACTIONS[ta] + ": " + verdict
            + (step.action === ta ? " ◀" : "") + "</div>";
        }
        turnsEl.innerHTML = rows;

        return {
          follow: { x: headP.x, z: headP.z, heading: Math.atan2(headT.z, headT.x) },
          step: index,
          action: step.action === null || step.action === undefined
            ? "—" : ACTIONS[step.action],
          // The board is a grid, so the head has a row and a column, not an x
          // and a y. The continuous defaults are kept filled in behind it.
          pos: "head row " + head[0] + "  col " + head[1],
          x: head[1],
          y: head[0],
          reward: step.reward,
          ret: running[index],
          belief: beliefLabel
        };
      }
    };
  }

  V.scenes["snake.v1"] = {
    build: build,
    // Four lanterns at 21000 lm each: real lumens blow out instantly, so the
    // camera stops down further than Light-Dark's. Tuned for this lamp power;
    // it is not a knob to remove.
    exposure: 0.118
  };
})(window);
