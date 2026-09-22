/* SPDX-License-Identifier: MIT
 *
 * Racetrack scene module.
 *
 * Builds the circuit from a trace's `payload.world.lanes` — highway-env's own
 * lane parameters, shipped with the episode — and moves every vehicle from the
 * states the episode recorded. Nothing is re-simulated and nothing is inferred
 * from the planner's curvature model, which is a different curve and is
 * allowed to be wrong. A trace that carries no map draws no road and says so,
 * rather than laying this circuit under an episode that was run on another.
 *
 * What this scene is *for* is the gap between two things the payload keeps
 * apart:
 *
 *  - the other vehicles, drawn where the world actually put them;
 *  - the detections, drawn where the ego's sensor reported something.
 *
 * A car inside the range ring with no detection on it is a car the planner
 * could not see, and that is the whole subject of this environment. So the two
 * are never merged, the range ring is drawn at the run's own
 * `max_detection_range_m`, and an unseen car is left plainly unmarked rather
 * than quietly given a marker.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  var COLORS = {
    ego: 0xC6432A,
    traffic: 0x3F7FBF,
    detection: 0x2EBE4A,
    ring: 0x2EBE4A,
    lamp: 0xFFE4B4
  };

  /* Lumens for the mast lights around the circuit. A circuit is 130 m across
     and a lamp obeys the inverse square, so this is an order of magnitude
     above the lamps in the smaller scenes and still leaves the far infield
     dark. It was 2600 first, which lit the grass and left the asphalt reading
     as a hole in the ground; 26000 fixed that and turned the night into an
     overcast afternoon. This is the value between them. */
  var MAST_LUMENS = 9000;

  // Metres per world unit. The circuit is about 130 m across, which at 1:1
  // would put every camera outside the renderer's far plane.
  var METRES_PER_UNIT = 9.0;

  // Cap on drawn belief particles per layer. A trace may carry several
  // hundred; beyond this it is draw cost with no added information.
  var MAX_DRAWN_PARTICLES = 420;

  // highway-env's LineType, as the lane parameters carry it. A continuous
  // edge is the outside of the drivable surface — the place a real circuit
  // puts a kerb and, behind it, a barrier.
  var LINE_NONE = 0, LINE_STRIPED = 1, LINE_CONTINUOUS = 2;

  // Kerb geometry, in metres. A real rumble strip is about half a metre wide
  // and painted in blocks a little under a metre long; these are those.
  var KERB_WIDTH_M = 0.55;
  var KERB_BLOCK_M = 0.9;
  var KERB_RISE_M = 0.06;

  // Barrier geometry, in metres: an Armco rail on posts, set back from the
  // kerb by a run-off margin.
  var BARRIER_OFFSET_M = 1.6;
  var BARRIER_HEIGHT_M = 0.75;

  /** Asphalt, painted once and shared by every lane. */
  function asphaltCanvas(seed) {
    var s = 512;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#34383E";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 9000; i++) {
      g.fillStyle = "rgba(" + (62 + rnd() * 44 | 0) + "," + (65 + rnd() * 42 | 0) + "," +
        (70 + rnd() * 42 | 0) + ",0.55)";
      g.fillRect(rnd() * s, rnd() * s, 1 + rnd() * 2, 1 + rnd() * 2);
    }
    // Patches: a resurfaced circuit is not one flat grey, and the seams are
    // most of what stops a large road reading as a painted plane.
    for (var p = 0; p < 26; p++) {
      g.fillStyle = "rgba(" + (30 + rnd() * 26 | 0) + "," + (32 + rnd() * 24 | 0) + "," +
        (36 + rnd() * 24 | 0) + ",0.5)";
      g.beginPath();
      g.ellipse(rnd() * s, rnd() * s, 20 + rnd() * 70, 14 + rnd() * 50, rnd() * 3, 0, 6.3);
      g.fill();
    }
    return cv;
  }

  /** Grass and gravel for the ground the circuit sits on. */
  function groundCanvas(seed) {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(seed);
    g.fillStyle = "#0D1107";
    g.fillRect(0, 0, s, s);
    for (var i = 0; i < 1400; i++) {
      var r = 14 + rnd() * 120;
      g.fillStyle = "rgba(" + (18 + rnd() * 22 | 0) + "," + (26 + rnd() * 26 | 0) + "," +
        (16 + rnd() * 18 | 0) + ",0.3)";
      g.beginPath(); g.arc(rnd() * s, rnd() * s, r, 0, Math.PI * 2); g.fill();
    }
    for (var t = 0; t < 12000; t++) {
      g.fillStyle = "rgba(" + (26 + rnd() * 30 | 0) + "," + (36 + rnd() * 32 | 0) + "," +
        (20 + rnd() * 20 | 0) + ",0.4)";
      g.fillRect(rnd() * s, rnd() * s, 1, 1 + rnd() * 3);
    }
    return cv;
  }

  /**
   * Build the Racetrack world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind racetrack.v1.
   * @returns {Object} The scene module the player drives.
   */
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;
    var lanes = world.lanes || [];

    /* The circuit's own extent decides the scene's centre and the cameras'
       standoff. Taken from the map when there is one and from the driven path
       when there is not, so a trace with no map still frames its episode. */
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    function grow(x, y) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    lanes.forEach(function (lane) {
      lane.left.forEach(function (p) { grow(p[0], p[1]); });
      lane.right.forEach(function (p) { grow(p[0], p[1]); });
    });
    payload.ego.forEach(function (e) { grow(e.x, e.y); });
    if (!isFinite(minX)) { minX = -10; maxX = 10; minY = -10; maxY = 10; }
    var midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;

    /* Metres to world units. highway-env's plane is (x, y); the renderer's
       ground plane is (x, z), so y becomes z and a heading measured
       anticlockwise in that plane becomes a clockwise rotation about Y. */
    function wx(x) { return (x - midX) / METRES_PER_UNIT; }
    function wz(y) { return (y - midY) / METRES_PER_UNIT; }
    function wu(metres) { return metres / METRES_PER_UNIT; }

    var extent = Math.max(maxX - minX, maxY - minY) / METRES_PER_UNIT;

    scene.background = new THREE.Color(0x05070C).convertSRGBToLinear();
    scene.fog = new THREE.FogExp2(0x070A10, 0.012);
    scene.fog.color.convertSRGBToLinear();

    scene.add(new THREE.HemisphereLight(0x22304A, 0x0A0C08, 0.5));
    var moon = new THREE.DirectionalLight(0x8FA3C8, 0.3);
    moon.position.set(-10, 16, -6);
    scene.add(moon);
    core.buildNightEnvironment();

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);

    // Ground.
    var gCanvas = groundCanvas(20260922);
    var groundNormal = V.normalMapFrom(renderer, gCanvas, 1.6);
    var groundAlbedo = new THREE.CanvasTexture(gCanvas);
    groundAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    groundAlbedo.encoding = THREE.sRGBEncoding;
    groundAlbedo.wrapS = groundAlbedo.wrapT = THREE.RepeatWrapping;
    groundAlbedo.repeat.set(10, 10);
    groundNormal.wrapS = groundNormal.wrapT = THREE.RepeatWrapping;
    groundNormal.repeat.set(10, 10);
    var ground = new THREE.Mesh(
      new THREE.PlaneGeometry(extent * 3.4, extent * 3.4),
      new THREE.MeshStandardMaterial({
        map: groundAlbedo, normalMap: groundNormal,
        normalScale: new THREE.Vector2(0.7, 0.7),
        roughness: 1.0, metalness: 0.0, envMapIntensity: 0.1
      })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    /* The road. One mesh per lane, built pair by pair between the lane's two
       edges — which the payload samples at matching points precisely so this
       can be done without guessing which point on one edge faces which on the
       other. It is lifted a few millimetres above the ground and biased at
       raster time, because the render target's depth buffer is 16-bit and two
       near-coplanar surfaces would otherwise z-fight. */
    var asphaltCv = asphaltCanvas(1312);
    var asphaltAlbedo = new THREE.CanvasTexture(asphaltCv);
    asphaltAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    asphaltAlbedo.encoding = THREE.sRGBEncoding;
    asphaltAlbedo.wrapS = asphaltAlbedo.wrapT = THREE.RepeatWrapping;
    var asphaltNormal = V.normalMapFrom(renderer, asphaltCv, 0.9);
    asphaltNormal.wrapS = asphaltNormal.wrapT = THREE.RepeatWrapping;
    var roadMat = new THREE.MeshStandardMaterial({
      map: asphaltAlbedo, normalMap: asphaltNormal,
      normalScale: new THREE.Vector2(0.55, 0.55),
      roughness: 0.82, metalness: 0.05, envMapIntensity: 0.3,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2
    });

    var ROAD_Y = 0.006;

    /* Outward normals along one edge of a lane.
     *
     * "Outward" is away from the lane's other edge, which the payload gives
     * point for point — so this needs no curvature, no winding order and no
     * assumption about which way the circuit runs. That matters: this track
     * has segments that run clockwise and segments that run anticlockwise, and
     * anything derived from winding would put the kerbs on the infield for
     * half the lap.
     */
    function outwardNormals(edge, other) {
      var normals = [];
      for (var i = 0; i < edge.length; i++) {
        var ox = edge[i][0] - other[i][0];
        var oy = edge[i][1] - other[i][1];
        var len = Math.hypot(ox, oy) || 1;
        normals.push([ox / len, oy / len]);
      }
      return normals;
    }

    /* A kerb: the red and white rumble strip along the outside of the track.
     *
     * This is the single thing that makes a grey ribbon read as a circuit
     * rather than a road, and it is why the map looked like neither. The
     * blocks are laid by distance travelled along the edge, not per sample, so
     * they stay the same length through a hairpin and down a straight instead
     * of stretching with the sampling. */
    function buildKerb(edge, normals) {
      var positions = [];
      var colors = [];
      var indices = [];
      var travelled = 0;
      for (var i = 0; i < edge.length; i++) {
        if (i > 0) {
          travelled += Math.hypot(edge[i][0] - edge[i - 1][0], edge[i][1] - edge[i - 1][1]);
        }
        var inner = edge[i];
        var outer = [
          edge[i][0] + normals[i][0] * KERB_WIDTH_M,
          edge[i][1] + normals[i][1] * KERB_WIDTH_M
        ];
        positions.push(wx(inner[0]), ROAD_Y + 0.002, wz(inner[1]));
        positions.push(wx(outer[0]), ROAD_Y + wu(KERB_RISE_M), wz(outer[1]));
        // Red and white in alternating blocks, the colours above 1.0 on the
        // white so a kerb under a mast light flares the way paint does.
        var white = Math.floor(travelled / KERB_BLOCK_M) % 2 === 0;
        var c = white ? [0.95, 0.93, 0.90] : [0.78, 0.12, 0.10];
        colors.push(c[0], c[1], c[2], c[0], c[1], c[2]);
        if (i > 0) {
          var a = (i - 1) * 2, b = a + 1, cc = a + 2, dd = a + 3;
          indices.push(a, b, cc, b, dd, cc);
        }
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
      geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(colors), 3));
      geo.setIndex(indices);
      geo.computeVertexNormals();
      var mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.55, metalness: 0.0, envMapIntensity: 0.4,
        side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3
      }));
      mesh.receiveShadow = true;
      scene.add(mesh);
    }

    /* The barrier behind the kerb: an Armco rail on posts, set back across a
       run-off margin. It is what closes the circuit visually — without it the
       tarmac just stops and the eye reads the edge as unfinished. */
    var railMat = new THREE.MeshStandardMaterial({
      color: 0xB9BDC2, roughness: 0.42, metalness: 0.85, envMapIntensity: 0.6
    });
    var postMat = new THREE.MeshStandardMaterial({
      color: 0x4C5054, roughness: 0.6, metalness: 0.7
    });

    function buildBarrier(edge, normals) {
      var points = [];
      for (var i = 0; i < edge.length; i++) {
        points.push(new THREE.Vector3(
          wx(edge[i][0] + normals[i][0] * BARRIER_OFFSET_M),
          wu(BARRIER_HEIGHT_M),
          wz(edge[i][1] + normals[i][1] * BARRIER_OFFSET_M)
        ));
      }
      if (points.length < 2) return;
      var curve = new THREE.CatmullRomCurve3(points);
      var rail = new THREE.Mesh(
        new THREE.TubeGeometry(curve, Math.max(8, points.length), wu(0.16), 4, false),
        railMat
      );
      rail.castShadow = true;
      rail.receiveShadow = true;
      scene.add(rail);
      // Posts every few metres, which is what gives the rail a scale and a
      // shadow rather than leaving it a floating ribbon.
      for (var q = 0; q < points.length; q += 6) {
        var post = new THREE.Mesh(
          new THREE.BoxGeometry(wu(0.12), wu(BARRIER_HEIGHT_M), wu(0.12)), postMat
        );
        post.position.set(points[q].x, wu(BARRIER_HEIGHT_M) / 2, points[q].z);
        post.castShadow = true;
        scene.add(post);
      }
    }
    lanes.forEach(function (lane) {
      var n = Math.min(lane.left.length, lane.right.length);
      if (n < 2) return;
      var positions = new Float32Array(n * 2 * 3);
      var uvs = new Float32Array(n * 2 * 2);
      var along = 0;
      for (var i = 0; i < n; i++) {
        if (i > 0) {
          along += Math.hypot(lane.left[i][0] - lane.left[i - 1][0],
            lane.left[i][1] - lane.left[i - 1][1]);
        }
        positions[i * 6] = wx(lane.left[i][0]);
        positions[i * 6 + 1] = ROAD_Y;
        positions[i * 6 + 2] = wz(lane.left[i][1]);
        positions[i * 6 + 3] = wx(lane.right[i][0]);
        positions[i * 6 + 4] = ROAD_Y;
        positions[i * 6 + 5] = wz(lane.right[i][1]);
        // Tile the asphalt by real metres, so the grain is the same size on a
        // long straight and a tight hairpin.
        uvs[i * 4] = along / 6; uvs[i * 4 + 1] = 0;
        uvs[i * 4 + 2] = along / 6; uvs[i * 4 + 3] = 1;
      }
      var indices = [];
      for (var q = 0; q < n - 1; q++) {
        var a = q * 2, b = q * 2 + 1, c = q * 2 + 2, d = q * 2 + 3;
        indices.push(a, b, c, b, d, c);
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
      geo.setIndex(indices);
      geo.computeVertexNormals();
      var mesh = new THREE.Mesh(geo, roadMat);
      mesh.receiveShadow = true;
      scene.add(mesh);

      /* Rubber. A circuit that has been driven on is darker where the cars
         go, and the band is off-centre through a corner because that is where
         the line is. Drawn from the lane's own edges at a fixed blend, not
         from this episode's trajectory — it is the track's history, not this
         planner's, and reading one as the other would be a lie about the run. */
      var rubber = new THREE.Mesh(geo.clone(), new THREE.MeshBasicMaterial({
        color: 0x0B0C0E, transparent: true, opacity: 0.32, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3
      }));
      rubber.scale.set(0.999, 1, 0.999);
      rubber.position.y = 0.001;
      scene.add(rubber);

      /* Kerbs and barriers, on every edge the lane parameters call continuous.
         A continuous edge is the outside of the drivable surface; a striped
         one is the line between this lane and its neighbour, and putting a
         kerb down the middle of a two-lane track would be nonsense. */
      [lane.left, lane.right].forEach(function (edge, side) {
        if (lane.line_types[side] !== LINE_CONTINUOUS) return;
        var normals = outwardNormals(edge, side === 0 ? lane.right : lane.left);
        buildKerb(edge, normals);
        buildBarrier(edge, normals);
      });

      /* Markings, at the type highway-env gives each edge. A striped edge is
         drawn striped and a continuous one solid, because on this circuit that
         distinction is which side the car may cross. */
      [lane.left, lane.right].forEach(function (edge, side) {
        var type = lane.line_types[side];
        if (type === LINE_NONE) return;
        var pts = edge.map(function (p) {
          return new THREE.Vector3(wx(p[0]), ROAD_Y + 0.004, wz(p[1]));
        });
        var lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
        var line;
        if (type === LINE_STRIPED) {
          line = new THREE.Line(lineGeo, new THREE.LineDashedMaterial({
            color: 0xE8E2C8, transparent: true, opacity: 0.55,
            dashSize: wu(3), gapSize: wu(3)
          }));
          line.computeLineDistances();
        } else {
          line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({
            color: 0xF2EDD6, transparent: true, opacity: 0.7
          }));
        }
        scene.add(line);
      });
    });

    /* Mast lights around the circuit, placed on its bounding box rather than
       on the road, so they never stand in a lane. One shadow caster follows
       the ego; sixteen shadow-casting masts would cost more than the rest of
       the frame and look no different. */
    var mastMat = new THREE.MeshStandardMaterial({
      color: 0x6B6455, roughness: 0.45, metalness: 0.8
    });
    var headMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(4.0, 3.6, 2.9) });
    var margin = extent * 0.95;
    [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, -1], [0, 1]].forEach(function (corner) {
      var x = corner[0] * margin, z = corner[1] * margin;
      var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.12, 6.4, 10), mastMat);
      mast.position.set(x, 3.2, z);
      mast.castShadow = true;
      scene.add(mast);
      var head = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.18, 0.34), headMat);
      head.position.set(x, 6.5, z);
      scene.add(head);
      var light = new THREE.PointLight(COLORS.lamp, 1, extent * 2.0, 2);
      light.power = MAST_LUMENS;
      light.position.set(x, 6.45, z);
      scene.add(light);
    });

    var shadowLight = new THREE.SpotLight(COLORS.lamp, 1, 22, 0.55, 0.6, 2);
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.5;
    shadowLight.shadow.camera.far = 24;
    shadowLight.shadow.bias = -0.0008;
    shadowLight.shadow.normalBias = 0.03;
    scene.add(shadowLight);
    scene.add(shadowLight.target);

    /* Start/finish. Laid across the first segment, over both its lanes,
       because that is where this circuit's lane graph begins and so where the
       arclength an episode reports is measured from. A chequered band is the
       one marking that says "circuit" on its own. */
    if (lanes.length >= 2) {
      var startLanes = [lanes[0], lanes[1]];
      var squares = 8;
      startLanes.forEach(function (lane) {
        if (lane.left.length < 4) return;
        var i = 2;
        var lx = lane.left[i][0], ly = lane.left[i][1];
        var rx = lane.right[i][0], ry = lane.right[i][1];
        // Across the track, and along it. The second row of the chequer has to
        // step ALONG the circuit; stepping along a world axis instead lays the
        // two rows side by side and the band comes out crossing the track
        // sideways, which is what the first attempt did.
        var acrossX = rx - lx, acrossY = ry - ly;
        var tangentX = lane.left[i + 1][0] - lx, tangentY = lane.left[i + 1][1] - ly;
        var tangentLen = Math.hypot(tangentX, tangentY) || 1;
        tangentX /= tangentLen; tangentY /= tangentLen;
        var rowDepth = 0.6;
        var angle = Math.atan2(acrossY, acrossX);
        for (var q = 0; q < squares; q++) {
          for (var row = 0; row < 2; row++) {
            var t0 = q / squares, t1 = (q + 1) / squares;
            var white = (q + row) % 2 === 0;
            var ax = lx + acrossX * t0, ay = ly + acrossY * t0;
            var bx = lx + acrossX * t1, by = ly + acrossY * t1;
            var cx = (ax + bx) / 2 + tangentX * rowDepth * (row - 0.5);
            var cy = (ay + by) / 2 + tangentY * rowDepth * (row - 0.5);
            var tile = new THREE.Mesh(
              new THREE.PlaneGeometry(
                Math.hypot(bx - ax, by - ay) / METRES_PER_UNIT, wu(rowDepth)
              ),
              new THREE.MeshStandardMaterial({
                color: white ? 0xF2F0EA : 0x17181A,
                roughness: 0.7, metalness: 0.0,
                polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4
              })
            );
            tile.rotation.x = -Math.PI / 2;
            tile.rotation.z = -angle;
            tile.position.set(wx(cx), ROAD_Y + 0.004, wz(cy));
            scene.add(tile);
          }
        }
      });
    }

    var carLength = wu(world.vehicle_length_m);
    var carWidth = wu(world.vehicle_width_m);

    /** One vehicle body, in the given paint. */
    function makeVehicle(colour, withLights) {
      var group = new THREE.Group();
      var paint = new THREE.MeshStandardMaterial({
        color: colour, roughness: 0.35, metalness: 0.5
      });
      var body = new THREE.Mesh(
        V.roundedBox(carLength, carLength * 0.16, carWidth, carLength * 0.035), paint
      );
      body.position.y = carLength * 0.12;
      body.castShadow = true;
      group.add(body);
      var cabin = new THREE.Mesh(
        V.roundedBox(carLength * 0.44, carLength * 0.13, carWidth * 0.82, carLength * 0.03),
        new THREE.MeshStandardMaterial({
          color: 0x22354F, roughness: 0.12, metalness: 0.4,
          emissive: 0x0E1A2E, emissiveIntensity: 0.6
        })
      );
      cabin.position.set(-carLength * 0.04, carLength * 0.25, 0);
      cabin.castShadow = true;
      group.add(cabin);
      var wheelGeo = new THREE.CylinderGeometry(
        carLength * 0.09, carLength * 0.09, carWidth * 0.16, 12
      );
      var wheelMat = new THREE.MeshStandardMaterial({
        color: 0x141211, roughness: 0.9, metalness: 0.1
      });
      [[0.3, 0.46], [0.3, -0.46], [-0.3, 0.46], [-0.3, -0.46]].forEach(function (w) {
        var wheel = new THREE.Mesh(wheelGeo, wheelMat);
        wheel.position.set(carLength * w[0], carLength * 0.09, carWidth * w[1]);
        wheel.rotation.x = Math.PI / 2;
        wheel.castShadow = true;
        group.add(wheel);
      });
      if (withLights) {
        var bulbMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.2, 3.0, 2.5) });
        [[0.48, 0.3], [0.48, -0.3]].forEach(function (h) {
          var bulb = new THREE.Mesh(new THREE.SphereGeometry(carLength * 0.045, 8, 8), bulbMat);
          bulb.position.set(carLength * h[0], carLength * 0.16, carWidth * h[1]);
          group.add(bulb);
        });
      }
      var blob = new THREE.Mesh(
        new THREE.PlaneGeometry(carLength * 1.7, carWidth * 2.4),
        new THREE.MeshBasicMaterial({
          map: poolTex, color: 0x000000, transparent: true, opacity: 0.42, depthWrite: false
        })
      );
      blob.rotation.x = -Math.PI / 2;
      blob.position.y = 0.012;
      group.add(blob);
      return group;
    }

    var ego = makeVehicle(COLORS.ego, true);
    scene.add(ego);

    var headlight = new THREE.SpotLight(0xFFF1D6, 1, wu(60), 0.42, 0.6, 2);
    headlight.power = 1400;
    headlight.position.set(carLength * 0.45, carLength * 0.2, 0);
    var headTarget = new THREE.Object3D();
    headTarget.position.set(wu(45), -carLength * 0.1, 0);
    ego.add(headlight);
    ego.add(headTarget);
    headlight.target = headTarget;

    /* One body per agent slot, built once and hidden when its slot is empty.
       Creating and destroying meshes per step would stutter, and a slot that
       is empty must show nothing rather than a car parked at the origin. */
    var traffic = [];
    for (var slot = 0; slot < world.max_tracked_agents; slot++) {
      var other = makeVehicle(COLORS.traffic, false);
      other.visible = false;
      scene.add(other);
      traffic.push(other);
    }

    /* Detection markers: where the sensor said something was. One per slot,
       hidden when the sensor reported fewer returns than there are slots — a
       marker left behind would claim a reading the episode never received. */
    var markers = [];
    var markerGeo = new THREE.TorusGeometry(carLength * 0.75, carLength * 0.05, 8, 24);
    var markerMat = new THREE.MeshBasicMaterial({
      color: COLORS.detection, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false
    });
    for (var m = 0; m < world.max_tracked_agents; m++) {
      var marker = new THREE.Mesh(markerGeo, markerMat);
      marker.rotation.x = -Math.PI / 2;
      marker.visible = false;
      scene.add(marker);
      markers.push(marker);
    }

    /* The range gate, drawn at the run's own number. This ring is the dial the
       whole environment turns on: traffic inside it may be reported, traffic
       outside it never is. Drawing it is what lets a reader tell "the planner
       missed that car" from "the planner could not have seen that car". */
    var rangeRing = new THREE.Mesh(
      new THREE.RingGeometry(
        Math.max(0.01, wu(world.max_detection_range_m) - 0.02),
        wu(world.max_detection_range_m) + 0.02,
        96
      ),
      new THREE.MeshBasicMaterial({
        color: COLORS.ring, transparent: true, opacity: 0.22,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
      })
    );
    rangeRing.rotation.x = -Math.PI / 2;
    rangeRing.position.y = 0.02;
    scene.add(rangeRing);

    /* Belief cloud. The particles are whole state vectors, so there are two
       things in them worth drawing and they are not the same thing:

         - where the filter thinks the *ego* is, which on this world is nearly
           observed and so is usually a tight knot;
         - where it thinks the *traffic* is, which is the hidden state and the
           reason the environment exists.

       They are drawn as one buffer in two colours rather than two objects,
       because a reader compares them and a second Points object with its own
       depth sorting makes that comparison harder, not easier. */
    var maxParticles = 1;
    payload.beliefs.forEach(function (b) {
      if (b.kind === "particles") maxParticles = Math.max(maxParticles, b.particles.length);
    });
    maxParticles = Math.min(maxParticles, MAX_DRAWN_PARTICLES);
    var slotsPerParticle = 1 + world.max_tracked_agents;
    var capacity = maxParticles * slotsPerParticle;
    var pPos = new Float32Array(capacity * 3);
    var pCol = new Float32Array(capacity * 3);
    var pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    pGeo.setAttribute("color", new THREE.BufferAttribute(pCol, 3));
    pGeo.setDrawRange(0, 0);
    var particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
      size: 0.12, map: partTex, vertexColors: true, transparent: true,
      opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false,
      sizeAttenuation: true
    }));
    scene.add(particles);

    // Trail: the ego's recorded path, drawn up to the current step.
    var trailPos = new Float32Array(payload.ego.length * 3);
    var trailCol = new Float32Array(payload.ego.length * 3);
    var trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3));
    trailGeo.setDrawRange(0, 0);
    var trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.9
    }));
    scene.add(trail);

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

    var egoAt = { x: 0, z: 0, heading: 0 };

    /** Shortest signed difference between two angles, for interpolation. */
    function angleDelta(from, to) {
      return ((to - from + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    }

    function sampleAt(t) {
      var n = payload.ego.length;
      var i0 = Math.floor(clamp(t, 0, n - 1));
      var i1 = Math.min(i0 + 1, n - 1);
      var f = clamp(t - i0, 0, 1);
      var a = payload.ego[i0], b = payload.ego[i1];
      return {
        index: i0,
        x: lerp(a.x, b.x, f),
        y: lerp(a.y, b.y, f),
        // Interpolated the short way round, so a car crossing the wrap point
        // does not spin through a full turn between two steps.
        heading: a.heading + angleDelta(a.heading, b.heading) * f,
        speed: lerp(a.speed, b.speed, f),
        lane_offset: lerp(a.lane_offset, b.lane_offset, f)
      };
    }

    /** Place one body at a world pose, in metres and radians. */
    function place(object, x, y, heading) {
      object.position.set(wx(x), 0, wz(y));
      // The plane's y axis becomes the renderer's z, which reverses the sense
      // of a rotation in it.
      object.rotation.y = -heading;
      object.visible = true;
    }

    function writeParticle(slot, x, y, colour) {
      pPos[slot * 3] = wx(x);
      pPos[slot * 3 + 1] = 0.05;
      pPos[slot * 3 + 2] = wz(y);
      pCol[slot * 3] = colour[0];
      pCol[slot * 3 + 1] = colour[1];
      pCol[slot * 3 + 2] = colour[2];
    }

    /* A particle is a whole state vector. The ego's own slots come first, then
       one block per tracked agent, holding that agent *relative to that
       particle's ego* — so each particle's traffic has to be rotated by that
       particle's own heading, not by the true one. Using the true heading
       would draw a belief that is partly the answer. */
    var EGO_X = 0, EGO_Y = 1, EGO_HEADING = 2;
    var AGENT_PRESENT = 0, AGENT_REL_X = 1, AGENT_REL_Y = 2;

    function drawParticles(belief) {
      var count = Math.min(belief.particles.length, maxParticles);
      if (count && !Array.isArray(belief.particles[0])) {
        pGeo.setDrawRange(0, 0);
        return belief.num_particles + " particles of an unreadable shape";
      }
      var maxWeight = 0;
      for (var w = 0; w < belief.weights.length; w++) {
        if (belief.weights[w] > maxWeight) maxWeight = belief.weights[w];
      }
      var written = 0;
      var tracked = 0;
      for (var p = 0; p < count; p++) {
        var particle = belief.particles[p];
        if (particle.length < world.ego_state_width) continue;
        var rel = belief.weighted === false
          ? 1
          : (maxWeight > 0 ? belief.weights[p] / maxWeight : 0);
        // The ego knot: cool and dim, because it is the part of the belief
        // that is nearly known and should not dominate the frame.
        writeParticle(written++, particle[EGO_X], particle[EGO_Y],
          [lerp(0.25, 0.5, rel) * 1.6, lerp(0.35, 0.7, rel) * 1.6, lerp(0.6, 1.0, rel) * 1.6]);

        var cos = Math.cos(particle[EGO_HEADING]), sin = Math.sin(particle[EGO_HEADING]);
        for (var a = 0; a < world.max_tracked_agents; a++) {
          var base = world.ego_state_width + a * world.agent_slot_width;
          if (particle[base + AGENT_PRESENT] <= 0) continue;
          var rx = particle[base + AGENT_REL_X], ry = particle[base + AGENT_REL_Y];
          writeParticle(
            written++,
            particle[EGO_X] + rx * cos - ry * sin,
            particle[EGO_Y] + rx * sin + ry * cos,
            // Amber where the mass is, deep red in the tail. Above 1.0 so the
            // heavy end blooms.
            [lerp(0.86, 1.0, rel) * 3.4, lerp(0.16, 0.84, rel) * 3.4, lerp(0.13, 0.18, rel) * 3.4]
          );
          tracked++;
        }
      }
      pGeo.attributes.position.needsUpdate = true;
      pGeo.attributes.color.needsUpdate = true;
      pGeo.setDrawRange(0, written);

      var label = belief.num_particles +
        (belief.weighted === false ? " uniform particles" : " particles");
      label += tracked ? ", " + tracked + " tracked-vehicle points" : ", no vehicle in any particle";
      if (belief.num_written < belief.num_particles) {
        label += " (heaviest " + belief.num_written + " drawn)";
      }
      return label;
    }

    function drawBelief(index) {
      var belief = payload.beliefs[index];
      if (!belief) { pGeo.setDrawRange(0, 0); return "—"; }
      if (belief.kind === "particles") return drawParticles(belief);
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner;
        // merging its members would show a cloud that was never anyone's.
        pGeo.setDrawRange(0, 0);
        return "batch of " + belief.batch_size + " beliefs, not drawn";
      }
      /* A Gaussian over a twelve-dimensional driving state has no honest
         two-dimensional picture, and this environment has never been run with
         one. Named rather than approximated. */
      pGeo.setDrawRange(0, 0);
      return "not drawn (" + (belief.belief_class || belief.kind) + ")";
    }

    function drawTrail(sample) {
      var count = sample.index + 1;
      for (var i = 0; i < count; i++) {
        var e = payload.ego[i];
        trailPos[i * 3] = wx(e.x);
        trailPos[i * 3 + 1] = 0.03;
        trailPos[i * 3 + 2] = wz(e.y);
        var age = count > 1 ? i / (count - 1) : 1;
        trailCol[i * 3] = lerp(0.5, 0.95, age);
        trailCol[i * 3 + 1] = lerp(0.14, 0.34, age);
        trailCol[i * 3 + 2] = lerp(0.1, 0.22, age);
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.attributes.color.needsUpdate = true;
      trailGeo.setDrawRange(0, Math.max(0, count));
    }

    /** How this step's action reads as the command it actually was. */
    function actionLabel(action) {
      if (action === null || action === undefined) return "—";
      var preset = world.action_presets[action];
      if (!preset) return String(action);
      return action + " (accel " + preset[0].toFixed(2) +
        ", steer " + preset[1].toFixed(2) + ")";
    }

    return {
      steps: payload.ego.length,

      /* Replay at the speed it was driven. The player's default is 1.6 steps a
         second, which suits a world where a step is one decision and has no
         duration; here a step is one control period, so at the default the car
         crawls round at about a third of the 8 m/s it actually held. The
         payload carries the controller's frequency for exactly this. */
      stepsPerSecond: world.policy_frequency_hz || undefined,

      camera: {
        // The circuit is wide and flat, so both fixed cameras look down at it
        // from well outside; the standoff is the same lens arithmetic as every
        // other scene, scaled by the map's own extent.
        board: [0, extent * 0.50 + 1.8, extent * 0.62 + 2.2],
        top: [0, extent * 0.88 + 2.4, 0.01],

        /* Chase is this scene's own. The rig's version stands a fixed 2.9
           units behind the followed point, which on a circuit this size puts
           the camera two car lengths back and the road filling the frame. The
           shot that reads here is further back and higher, looking at the
           corner the car is entering. */
        modes: {
          chase: function (ctx) {
            var back = Math.cos(egoAt.heading), side = Math.sin(egoAt.heading);
            ctx.pos.lerp(new ctx.THREE.Vector3(
              egoAt.x - back * wu(26), wu(12), egoAt.z + side * wu(26)
            ), clamp(ctx.dt * 3.0, 0, 1));
            ctx.look.lerp(new ctx.THREE.Vector3(
              egoAt.x + back * wu(14), 0.1, egoAt.z - side * wu(14)
            ), clamp(ctx.dt * 4.0, 0, 1));
          }
        }
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        place(ego, sample.x, sample.y, sample.heading);
        egoAt.x = wx(sample.x);
        egoAt.z = wz(sample.y);
        egoAt.heading = sample.heading;

        rangeRing.position.set(egoAt.x, 0.02, egoAt.z);

        shadowLight.position.set(egoAt.x, wu(28), egoAt.z + wu(10));
        shadowLight.target.position.set(egoAt.x, 0, egoAt.z);
        shadowLight.target.updateMatrixWorld();

        // Traffic, at the positions the world recorded — not interpolated
        // between steps, because a vehicle that left the state's slots has no
        // "between" to be in.
        var others = payload.agents[sample.index] || [];
        for (var v = 0; v < traffic.length; v++) traffic[v].visible = false;
        for (var o = 0; o < others.length && o < traffic.length; o++) {
          var car = others[o];
          /* Heading from the vehicle's own world velocity, which the payload
             reports absolute rather than relative to the ego — a relative one
             would spin the car whenever the two matched speeds.

             A car that is genuinely stopped has no direction of travel to
             read, so it keeps the heading it last had rather than snapping to
             the +x that `atan2(0, 0)` returns. */
          var speed = Math.hypot(car.vx, car.vy);
          var heading = speed > 1e-3
            ? Math.atan2(car.vy, car.vx)
            : -traffic[o].rotation.y;
          place(traffic[o], car.x, car.y, heading);
        }

        // Detections, at the positions the sensor reported. Unlabeled by
        // construction, so they are drawn as marks on the ground and never
        // attached to a particular vehicle.
        var seen = payload.detections[sample.index];
        for (var k = 0; k < markers.length; k++) markers[k].visible = false;
        if (seen) {
          for (var d = 0; d < seen.length && d < markers.length; d++) {
            markers[d].position.set(wx(seen[d].x), 0.03, wz(seen[d].y));
            markers[d].visible = true;
          }
        }

        var beliefLabel = drawBelief(sample.index);
        drawTrail(sample);

        var step = trace.steps[sample.index] || {};
        var detectionLabel = seen === null || seen === undefined
          ? "no reading"
          : seen.length + " of " + others.length + " seen";

        return {
          follow: { x: egoAt.x, z: egoAt.z, heading: sample.heading },
          step: sample.index,
          action: actionLabel(step.action),
          // Metres and metres per second, named: this world's numbers are
          // physical and "x 31.20  y 5.00" would hide that.
          pos: sample.x.toFixed(1) + ", " + sample.y.toFixed(1) + " m  ·  " +
            sample.speed.toFixed(1) + " m/s  ·  " + detectionLabel,
          x: sample.x,
          y: sample.y,
          reward: step.reward,
          ret: running[sample.index],
          belief: beliefLabel
        };
      }
    };
  }

  V.scenes["racetrack.v1"] = {
    build: build,
    // Real lumens blow out instantly, so the camera stops down. Tuned for this
    // scene's mast power; it is not a knob to remove.
    exposure: 0.145
  };
})(window);
