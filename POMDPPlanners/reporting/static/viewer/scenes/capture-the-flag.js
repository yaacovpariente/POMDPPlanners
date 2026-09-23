/* SPDX-License-Identifier: MIT
 *
 * CaptureTheFlag scene module.
 *
 * Builds the field from a trace's `payload.world` block and moves it from the
 * trace's recorded states and beliefs. Nothing here is invented: there is no
 * fallback episode, no hand-placed waypoint and no synthetic belief. If the
 * player hands this module no trace, it draws nothing and says so.
 *
 * Two things make this scene different from the others.
 *
 * It is the only daylight world in the set, and daylight is harder than night:
 * a night scene hides its ground, while a lit field shows every shortcut. The
 * sun sits at twenty-one degrees, which buys long raking shadows -- the only
 * thing that makes the terrain's relief visible -- and costs about a stop and
 * a half, because a surface facing straight up then receives only sin(21 deg)
 * of the sun's irradiance. The ground is a desaturated olive ramp with
 * thresholded dry regions rather than one flat green, canopies grow at the
 * limb tips with real gaps in them, and the roofs are weathered tile with the
 * team colour moved to the pennants, where a saturated accent reads better
 * than it would on a saturated roof.
 *
 * And it is multi-agent: two blue players and two red ones by default, with
 * the team sizes read from the trace. Two hidden variables, of two different
 * shapes, are drawn two different ways -- the red players' positions as
 * per-cell markers, because they are spatial, and the banner's cell as
 * labelled bars over four candidates, because that belief is a choice among
 * named cells and a cloud would imply a position it does not have.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* Palettes lifted from capture_the_flag_assets.py, darkest first, so the
     viewer and the GIF depict one world in one set of colours. */
  var BLUE_P = [0x182C64, 0x2854AC, 0x548AE8, 0xA2C6FF];
  var RED_P = [0x681A18, 0xA8322A, 0xD45C4C, 0xFAAC98];
  var SKIN_P = [0xA8744E, 0xD6A87C, 0xF0CEA8];
  var WOOD_P = [0x3A281A, 0x5C4028, 0x805E3A, 0xA68254];
  var STONE_P = [0x303036, 0x4E4E58, 0x70707C, 0x9898A6];
  var LEAF_GREEN = [0x2A3A20, 0x384C28, 0x486034, 0x5A7440];
  var LEAF_DRY = [0x5A4A28, 0x766034, 0x927846, 0xAC9058];
  var BELIEF_GOLD = 0xECBC5C;
  // One hue per red player, as the GIF's overlay uses.
  var HUES = [0xD45C4C, 0xB05CC8, 0x6CC8D4, 0xE0A94C];

  var GRASS = ["#2C3220", "#363D25", "#404629", "#4A502E",
               "#545833", "#5E6139", "#6B6C42", "#7A7650"];
  var DRY = ["#81784E", "#8D8455", "#988E5E", "#A2966A"];
  var EARTH = ["#4E3F2C", "#5C4C34", "#6C5B40", "#7C6B4C", "#8C7B5A"];

  // Per-player action ids, from capture_the_flag_pomdp_utils.
  var ACTION_NAMES = ["north", "east", "south", "west", "stay", "SCAN"];
  var ACTION_SCAN = 5;

  // How many cells of one red player's marginal are drawn. Past a dozen the
  // remaining cells carry a percent apiece and cost a draw call each.
  var MAX_BELIEF_CELLS = 12;
  // Fixed unit-normal draws, used only to jitter nothing -- see marginalOf.
  var WATER_Y = -0.24;

  function smooth(e0, e1, x) {
    var t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  /* Sobel over a blurred copy of a painted height pass. The core has its own
     normalMapFrom, which blurs harder and reads luminance; this one keeps the
     1.2px blur and the single channel the terrain's strengths were tuned
     against, and those strengths are what make the field look mown rather
     than printed. */
  function normalMapFrom(renderer, cv, strength) {
    var s = cv.width;
    var soft = document.createElement("canvas");
    soft.width = soft.height = s;
    var sctx = soft.getContext("2d");
    sctx.filter = "blur(1.2px)";
    sctx.drawImage(cv, 0, 0);
    sctx.filter = "none";
    var data = sctx.getImageData(0, 0, s, s).data;
    var dst = document.createElement("canvas");
    dst.width = dst.height = s;
    var dctx = dst.getContext("2d");
    var img = dctx.createImageData(s, s);
    var o = img.data;
    function h(x, y) {
      x = (x + s) % s; y = (y + s) % s;
      return data[((y * s + x) << 2)] / 255;
    }
    for (var y = 0; y < s; y++) {
      for (var x = 0; x < s; x++) {
        var nx = -(h(x + 1, y) - h(x - 1, y)) * strength;
        var ny = -(h(x, y + 1) - h(x, y - 1)) * strength;
        var len = Math.sqrt(nx * nx + ny * ny + 1);
        var i = (y * s + x) * 4;
        o[i] = (nx / len * 0.5 + 0.5) * 255;
        o[i + 1] = (ny / len * 0.5 + 0.5) * 255;
        o[i + 2] = (1 / len * 0.5 + 0.5) * 255;
        o[i + 3] = 255;
      }
    }
    dctx.putImageData(img, 0, 0);
    var tex = new THREE.CanvasTexture(dst);
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  }

  /**
   * Build the CaptureTheFlag world from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with kind capture_the_flag.v1.
   * @returns {Object} The scene module the player drives.
   */
  // The world is one scene: terrain, water, props, units and two belief
  // layers, each needing the field geometry the one above it established.
  // Splitting it would mean threading a dozen closures through helpers that
  // have exactly one caller.
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var layout = payload.state_layout || {};
    var scene = core.scene;
    var renderer = core.renderer;

    var GW = world.grid_size[0], GH = world.grid_size[1];
    var MIDLINE = world.midline;
    var HX = (GW - 1) / 2, HZ = (GH - 1) / 2;
    function wx(gx) { return gx - HX; }
    function wz(gy) { return gy - HZ; }

    var N_STEPS = payload.blue_cells.length;
    var N_BLUE = world.n_blue, N_RED = world.n_red;

    /* The channel's centreline, in grid x, for a grid y. This is
       capture_the_flag_assets.river_axis: the water, the banks and the bridge
       are all drawn from it, so the crossing cannot drift off the river.
       It is a copy of the GIF's art, not of the environment -- the field's
       dynamics know nothing about a river -- so if that function is ever
       reshaped, this is the line that has to move with it. Every tree is
       nudged clear of the axis below, so no tree stands in water under either
       shape. */
    function riverAxis(gy) {
      return MIDLINE + 0.22 * Math.sin(gy * 0.85 + 0.6) + 0.09 * Math.sin(gy * 2.1);
    }

    /* Ground height in world units: a shallow roll over the whole valley plus
       the carved channel. Every prop is planted with this, so nothing floats. */
    function groundY(gx, gy) {
      var roll = Math.sin(gx * 0.55 + 1.3) * 0.035 + Math.sin(gy * 0.72 - 0.4) * 0.03
        + Math.sin((gx + gy) * 0.31) * 0.025;
      var d = Math.abs(gx - riverAxis(gy));
      return roll - 0.62 * smooth(0.92, 0.26, d);
    }

    function place(obj, gx, gy, lift) {
      obj.position.set(wx(gx), groundY(gx, gy) + (lift || 0), wz(gy));
    }

    /* A longer lens than a game camera, and a far plane that reaches the ring
       of hills and the sky dome. The core's defaults are set for the night
       scenes, which have neither. near stays where the core put it: the render
       target's depth buffer is 16-bit. */
    core.camera.fov = 34;
    core.camera.far = 180;
    core.camera.updateProjectionMatrix();

    /* The rest of the prototype's post tuning. The core's defaults are set for
       the night scenes, where a lamp is one of a handful of bright things in a
       dark frame and can afford to bloom hard. A lit field is bright
       everywhere, so the same bloom turns the grass milky and the grain, sized
       for a dark image, reads as dirt on the lens in a daylight one. */
    core.composite.uniforms.bloomStrength.value = 0.26;
    core.composite.uniforms.grain.value = 0.013;
    core.composite.uniforms.aberration.value = 0.0014;

    scene.background = new THREE.Color(0xA8B6AE).convertSRGBToLinear();
    // Aerial perspective: thick enough that the hills sit behind real air,
    // thin enough that the field itself stays crisp.
    scene.fog = new THREE.FogExp2(0xA8B6AE, 0.0152);
    scene.fog.color.convertSRGBToLinear();

    /* One sun direction for the whole scene. The directional light, the sky
       texture's own sun disc and therefore every reflection read from this, so
       the key light, the specular streak on the river and the sky a viewer can
       see all agree with each other.

       Twenty-one degrees above the horizon. Long raking shadows do more for an
       outdoor scene than any amount of texture work: they give the frame a
       time of day, they plant every object on the ground, and they are the
       only thing that makes the terrain's relief visible at all. The cost is
       that a surface facing straight up receives only sin(21 deg) of the sun's
       irradiance, so the intensity is raised to match. */
    var SUN_ELEVATION = 21 * Math.PI / 180;
    var SUN_DIR = new THREE.Vector3(
      -Math.cos(SUN_ELEVATION) * 0.86,
      Math.sin(SUN_ELEVATION),
      -Math.cos(SUN_ELEVATION) * 0.51
    ).normalize();

    var sun = new THREE.DirectionalLight(0xFFD49A, 8.4);
    sun.position.copy(SUN_DIR).multiplyScalar(30);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    // Wide enough to cover every bit of ground the board camera sees. Outside
    // the frustum the shadow lookup clamps to the edge texel, which painted a
    // phantom shadow across the near corner of the field.
    var shadowSpan = Math.max(GW, GH) * 2.4 + 4;
    sun.shadow.camera.left = -shadowSpan;
    sun.shadow.camera.right = shadowSpan;
    sun.shadow.camera.top = shadowSpan;
    sun.shadow.camera.bottom = -shadowSpan;
    sun.shadow.camera.near = 4;
    sun.shadow.camera.far = 80;
    // A real shadow edge is soft. PCF cannot soften it with distance, but a
    // wide radius gets the rest, and a razor edge on grass is one of the
    // loudest "this is a render" signals there is.
    sun.shadow.radius = 2.6;
    // normalBias, not a big negative bias: a narrow offset along the surface
    // normal is what stops acne on a rolling, normal-mapped field.
    sun.shadow.normalBias = 0.026;
    sun.shadow.bias = -0.0003;
    scene.add(sun);
    scene.add(sun.target);
    sun.target.position.set(0, 0, 0);
    sun.target.updateMatrixWorld();

    /* Outdoors the sky is a light, not a backdrop. Most of the fill comes from
       the environment map below; this only tops it up, with the horizon's blue
       above and the grass's green beneath, so shadowed sides go blue and the
       undersides of canopies go green instead of both going black. */
    scene.add(new THREE.HemisphereLight(0x8FB4E0, 0x484226, 0.42));

    /* The sky texture: a horizon-to-zenith gradient with real cloud banks and
       the sun's own disc painted where SUN_DIR points, run through PMREM.
       Every metal, every wet surface and every shadowed face is lit off this,
       which is where most of the realism in a daylight render comes from. The
       core's own environment builder paints a night sky, so this scene builds
       its own rather than asking for one it would have to undo. */
    var skyTex;
    (function buildSky() {
      var W = 1024, H = 512;
      var cv = document.createElement("canvas");
      cv.width = W; cv.height = H;
      var g = cv.getContext("2d");

      // Zenith to horizon to ground haze. The band at the horizon is
      // deliberately pale and warm: that is where the air is thickest.
      var grad = g.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0.00, "#1E56AE");
      grad.addColorStop(0.18, "#3272C2");
      grad.addColorStop(0.34, "#5E9AD2");
      grad.addColorStop(0.44, "#9CBBD4");
      grad.addColorStop(0.487, "#D4CEBE");
      grad.addColorStop(0.50, "#DCCEAE");
      grad.addColorStop(0.52, "#96A484");
      grad.addColorStop(0.70, "#62744A");
      grad.addColorStop(1.00, "#3A4830");
      g.fillStyle = grad;
      g.fillRect(0, 0, W, H);

      // three's equirect lookup puts the zenith at v = 1 and a canvas texture
      // is flipped, so row 0 is the zenith. Solve for the sun's pixel.
      var sunU = Math.atan2(SUN_DIR.z, SUN_DIR.x) / (Math.PI * 2) + 0.5;
      var sunV = Math.asin(clamp(SUN_DIR.y, -1, 1)) / Math.PI + 0.5;
      var sx = sunU * W, sy = (1 - sunV) * H;

      // Cloud banks, thicker and lower near the horizon, thinning towards the
      // zenith. They give the reflections structure to pick up.
      var rnd = mulberry(4180);
      for (var i = 0; i < 260; i++) {
        var cy = rnd() * 0.47 * H;
        var band = 1 - cy / (0.46 * H);
        var cx = rnd() * W;
        var rx = 22 + rnd() * 120 * (0.4 + band);
        var ry = rx * (0.10 + rnd() * 0.16);
        // Clouds near the sun catch its light; the rest stay cool and flat.
        var toSun = 1 - clamp(
          Math.hypot(((cx - sx + W * 1.5) % W) - W * 0.5, cy - sy) / 300, 0, 1
        );
        var warm = 0.35 + 0.65 * toSun;
        var a = (0.05 + rnd() * 0.20) * (0.45 + band);
        var cg = g.createRadialGradient(cx, cy, 0, cx, cy, rx);
        cg.addColorStop(0, "rgba(" + (240 + 15 * warm | 0) + "," + (238 + 14 * warm | 0) +
          "," + (232 + 10 * warm | 0) + "," + a.toFixed(3) + ")");
        cg.addColorStop(1, "rgba(215,222,230,0)");
        g.save();
        g.translate(cx, cy);
        g.scale(1, ry / rx);
        g.translate(-cx, -cy);
        g.fillStyle = cg;
        g.beginPath(); g.arc(cx, cy, rx, 0, Math.PI * 2); g.fill();
        g.restore();
      }

      // The sun: a wide halo, then the disc. Bright, but nowhere near the real
      // thing -- the directional light supplies the key, and a sky this hot in
      // the environment map would flood every shadow.
      for (var h = 0; h < 3; h++) {
        var hr = [190, 92, 34][h];
        var ha = [0.16, 0.30, 0.55][h];
        var hg = g.createRadialGradient(sx, sy, 0, sx, sy, hr);
        hg.addColorStop(0, "rgba(255,246,222," + ha + ")");
        hg.addColorStop(1, "rgba(255,240,210,0)");
        g.fillStyle = hg;
        g.beginPath(); g.arc(sx, sy, hr, 0, Math.PI * 2); g.fill();
      }
      g.fillStyle = "#FFFBF0";
      g.beginPath(); g.arc(sx, sy, 11, 0, Math.PI * 2); g.fill();

      skyTex = new THREE.CanvasTexture(cv);
      skyTex.mapping = THREE.EquirectangularReflectionMapping;
      skyTex.encoding = THREE.sRGBEncoding;
      var pmrem = new THREE.PMREMGenerator(renderer);
      pmrem.compileEquirectangularShader();
      scene.environment = pmrem.fromEquirectangular(skyTex).texture;
      pmrem.dispose();
    })();

    // The visible dome. Basic material, so it is a backdrop and not a second
    // light -- the environment map already does the lighting.
    var dome = new THREE.Mesh(
      new THREE.SphereGeometry(86, 48, 28),
      new THREE.MeshBasicMaterial({
        map: skyTex, side: THREE.BackSide, fog: false, depthWrite: false
      })
    );
    scene.add(dome);

    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);

    /* ----------------------------------------------------------- terrain
       Painted with drawing operations rather than per-pixel noise. That is not
       a shortcut: at a hundred pixels per grid cell a per-pixel loop is too
       slow to run on page load, and noise gives mottling where grass needs
       strokes. The recipe is the environment's own -- ground_plane's grass and
       earth ramps, its worn trails between the bases and the crossing, its
       whisper of territory tint either side of the midline, and its channel
       with shallows and foam.

       Three maps come out of it: albedo, a normal map derived from a
       separately painted height pass, and a roughness map. Grass is not
       uniformly rough -- trampled earth scatters, fresh blades catch a sheen --
       and two scales of variation in all three is most of what stops a large
       surface reading as one flat colour. */
    var TERRAIN = Math.max(30, Math.max(GW, GH) * 3.2);
    var GX0 = HX - TERRAIN / 2, GX1 = HX + TERRAIN / 2;
    var GY0 = HZ - TERRAIN / 2, GY1 = HZ + TERRAIN / 2;

    var BLUE_BASE = world.blue_base, RED_BASE = world.red_base;
    var BLUE_FLAG = world.blue_flag_cell;
    var CANDIDATES = world.red_flag_candidates;

    // The worn trails: base -> crossing -> base, as _worn_paths builds them.
    var CROSSING = riverAxis(BLUE_BASE[1]);
    var PATHS = [
      [BLUE_BASE[0], BLUE_BASE[1], CROSSING - 0.6, BLUE_BASE[1]],
      [CROSSING + 0.6, BLUE_BASE[1], RED_BASE[0], RED_BASE[1]]
    ];

    function paintTerrain(S, mode) {
      var cv = document.createElement("canvas");
      cv.width = cv.height = S;
      var g = cv.getContext("2d");
      var PPU = S / TERRAIN;
      var rnd = mulberry(20260919 + (mode === "height" ? 7 : mode === "rough" ? 13 : 0));
      function px(gx) { return (gx - GX0) * PPU; }
      function py(gy) { return (gy - GY0) * PPU; }

      // 1. Base coat.
      g.fillStyle = mode === "albedo" ? "#4A502C" : (mode === "height" ? "#7E7E7E" : "#D2D2D2");
      g.fillRect(0, 0, S, S);

      // 2. Blades. Short directional strokes, batched by colour so a hundred
      //    thousand of them still paint in well under a second. This is the
      //    detail that survives the close camera; without it the field is
      //    smooth paint the moment anything gets near it.
      var lanes = mode === "albedo" ? GRASS
        : ["#5A5A5A", "#6E6E6E", "#828282", "#969696", "#AAAAAA", "#BEBEBE", "#D2D2D2", "#E6E6E6"];
      var perLane = Math.round(96000 / lanes.length * (S / 3072) * (S / 3072) * 2.4);
      g.lineCap = "round";
      for (var L = 0; L < lanes.length; L++) {
        g.strokeStyle = lanes[L];
        g.globalAlpha = mode === "rough" ? 0.10 : 0.30 + (L / lanes.length) * 0.38;
        g.lineWidth = Math.max(1.8, PPU * (0.030 + rnd() * 0.026));
        g.beginPath();
        for (var b = 0; b < perLane; b++) {
          var bx = rnd() * S, by = rnd() * S;
          // Blades lean with a slow flow field, the way a field lies after wind.
          var ang = -1.35 + Math.sin(bx * 0.0016) * 0.5 + Math.cos(by * 0.0021) * 0.5
            + (rnd() - 0.5) * 1.5;
          var len = PPU * (0.10 + rnd() * 0.16);
          g.moveTo(bx, by);
          g.lineTo(bx + Math.cos(ang) * len, by + Math.sin(ang) * len);
        }
        g.stroke();
      }
      g.globalAlpha = 1;

      /* 3. Two scales of smooth variation, laid OVER the blades. Under them
         the strokes average out and the field goes back to one flat green,
         which is the whole thing this pass exists to prevent.

         Drawn as upscaled low-resolution noise rather than as a scatter of
         radial gradients: gradients leave visible discs wherever one lands
         stronger than its neighbours, and a field dappled with circles is
         worse than no variation at all. */
      function noiseLayer(cells, alpha, op) {
        var small = document.createElement("canvas");
        small.width = small.height = cells;
        var sg = small.getContext("2d");
        var id = sg.createImageData(cells, cells);
        for (var n = 0; n < cells * cells; n++) {
          // Biased dark: the bright half of a meadow is the exception.
          var v = 64 + Math.pow(rnd(), 0.78) * 122;
          id.data[n * 4] = id.data[n * 4 + 1] = id.data[n * 4 + 2] = v;
          id.data[n * 4 + 3] = 255;
        }
        sg.putImageData(id, 0, 0);
        // Two upscales through an intermediate: one jump from 64 to 3072 keeps
        // the square grid of the source visible even with smoothing on.
        var mid = document.createElement("canvas");
        mid.width = mid.height = cells * 8;
        var mg = mid.getContext("2d");
        mg.imageSmoothingEnabled = true;
        mg.imageSmoothingQuality = "high";
        mg.drawImage(small, 0, 0, mid.width, mid.height);
        g.save();
        g.globalCompositeOperation = op;
        g.globalAlpha = alpha;
        g.imageSmoothingEnabled = true;
        g.imageSmoothingQuality = "high";
        g.drawImage(mid, 0, 0, S, S);
        g.restore();
      }
      noiseLayer(20, 0.78, "overlay");    // cloud-shadow scale, a cell or two
      noiseLayer(62, 0.52, "overlay");    // half a cell
      noiseLayer(150, 0.20, "overlay");   // tuft scale

      /* 3b. Dry ground. A masked tint rather than another noise overlay:
         overlay shifts value and leaves hue alone, and what a real field has is
         whole regions that are a different colour, not the same green darker.
         The mask is thresholded, so dry ground has a boundary and the rest of
         the field stays untouched. */
      function dryLayer(cells, threshold, palette, alpha) {
        var small = document.createElement("canvas");
        small.width = small.height = cells;
        var sg = small.getContext("2d");
        var id = sg.createImageData(cells, cells);
        for (var n = 0; n < cells * cells; n++) {
          var t = rnd();
          var c = new THREE.Color(palette[Math.floor(rnd() * palette.length)]);
          id.data[n * 4] = c.r * 255;
          id.data[n * 4 + 1] = c.g * 255;
          id.data[n * 4 + 2] = c.b * 255;
          id.data[n * 4 + 3] = Math.pow(clamp((t - threshold) / (1 - threshold), 0, 1), 1.4) * 255;
        }
        sg.putImageData(id, 0, 0);
        var mid = document.createElement("canvas");
        mid.width = mid.height = cells * 8;
        var mg = mid.getContext("2d");
        mg.imageSmoothingEnabled = true;
        mg.imageSmoothingQuality = "high";
        mg.drawImage(small, 0, 0, mid.width, mid.height);
        g.save();
        g.globalAlpha = alpha;
        g.imageSmoothingEnabled = true;
        g.imageSmoothingQuality = "high";
        g.drawImage(mid, 0, 0, S, S);
        g.restore();
      }
      if (mode === "albedo") {
        dryLayer(11, 0.70, DRY, 0.50);                       // whole dry stretches
        dryLayer(26, 0.74, DRY, 0.34);                       // burnt patches
        dryLayer(30, 0.78, ["#6E6048", "#5E5440"], 0.34);    // bare brown scuff

        // Some blades belong to the dry ground, so a straw patch is made of
        // straw rather than of green grass under a tan wash.
        g.lineCap = "round";
        for (var dl = 0; dl < 3; dl++) {
          g.strokeStyle = DRY[dl];
          g.globalAlpha = 0.20;
          g.lineWidth = Math.max(1.8, PPU * 0.034);
          g.beginPath();
          for (var db = 0; db < 9000; db++) {
            var dbx = rnd() * S, dby = rnd() * S;
            var dang = -1.35 + Math.sin(dbx * 0.0016) * 0.5 + Math.cos(dby * 0.0021) * 0.5
              + (rnd() - 0.5) * 1.5;
            var dlen = PPU * (0.10 + rnd() * 0.16);
            g.moveTo(dbx, dby);
            g.lineTo(dbx + Math.cos(dang) * dlen, dby + Math.sin(dang) * dlen);
          }
          g.stroke();
        }
        g.globalAlpha = 1;

        // 4. A hue drift on top of the value drift, so the far end of the
        //    field is not simply a darker version of the near end.
        for (var m = 0; m < 900; m++) {
          var mx = rnd() * S, my = rnd() * S, mr2 = PPU * (0.5 + rnd() * 2.6);
          g.globalAlpha = 0.05 + rnd() * 0.10;
          var pick = (rnd() < 0.22 ? DRY : GRASS)[Math.floor(rnd() * 4)];
          var hg2 = g.createRadialGradient(mx, my, 0, mx, my, mr2);
          hg2.addColorStop(0, pick);
          hg2.addColorStop(1, "rgba(80,88,52,0)");
          g.fillStyle = hg2;
          g.beginPath(); g.arc(mx, my, mr2, 0, Math.PI * 2); g.fill();
        }
        g.globalAlpha = 1;
      }

      // 5. Dry stalks and bare speckle, and on the albedo a dusting of
      //    wildflower heads -- the near-field 3D scatter carries the rest.
      for (var d = 0; d < 11000 * (S / 3072); d++) {
        var dx = rnd() * S, dy = rnd() * S;
        var p2 = rnd();
        if (mode === "albedo") {
          g.fillStyle = p2 < 0.46 ? "#8D8455"
            : (p2 < 0.80 ? EARTH[1]
              : (p2 < 0.94 ? "#A2966A" : (p2 < 0.975 ? "#CFC9A8" : "#B8A8C4")));
        } else {
          var dv = mode === "height" ? (p2 < 0.5 ? 60 : 180) : (p2 < 0.5 ? 120 : 230);
          g.fillStyle = "rgb(" + dv + "," + dv + "," + dv + ")";
        }
        g.globalAlpha = 0.18 + rnd() * 0.5;
        g.fillRect(dx, dy, Math.max(1, PPU * 0.018), Math.max(1, PPU * 0.018));
      }
      g.globalAlpha = 1;

      // 6. Territory tint. A whisper, not a second biome: saturated blue and
      //    red are reserved for the units, the pennants and the banners.
      if (mode === "albedo") {
        g.globalAlpha = 0.038;
        g.fillStyle = "#C28040";
        g.fillRect(px(MIDLINE), 0, S - px(MIDLINE), S);
        g.fillStyle = "#3C74C0";
        g.fillRect(0, 0, px(MIDLINE), S);
        g.globalAlpha = 1;
      }

      // 7. The worn trails. Several passes of decreasing width so the edge
      //    feathers into the grass instead of stopping at a line, plus scuffed
      //    patches where feet actually land.
      for (var w = 0; w < 5; w++) {
        g.lineWidth = PPU * (0.84 - w * 0.13);
        g.lineCap = "round";
        g.globalAlpha = 0.11 + w * 0.05;
        g.strokeStyle = mode === "albedo" ? EARTH[1 + (w % 3)]
          : (mode === "height" ? "#5E5E5E" : "#EFEFEF");
        PATHS.forEach(function (seg) {
          g.beginPath();
          // A worn path wanders; a ruled one reads as a road marking.
          var steps = 26;
          for (var t = 0; t <= steps; t++) {
            var f = t / steps;
            var gx = lerp(seg[0], seg[2], f), gy = lerp(seg[1], seg[3], f);
            gy += Math.sin(f * 7.1 + seg[0]) * 0.13 + Math.sin(f * 3.2) * 0.09;
            if (t === 0) g.moveTo(px(gx), py(gy)); else g.lineTo(px(gx), py(gy));
          }
          g.stroke();
        });
      }
      // Scuffs: bare earth where a unit stands often. These are what make a
      // cell legible without ruling a lattice over the grass.
      [BLUE_BASE, RED_BASE, BLUE_FLAG].concat(CANDIDATES).forEach(function (cell) {
        for (var k = 0; k < 16; k++) {
          var a2 = rnd() * 6.28, rr = rnd() * 0.42;
          var sx2 = px(cell[0] + Math.cos(a2) * rr), sy2 = py(cell[1] + Math.sin(a2) * rr);
          g.globalAlpha = 0.07 + rnd() * 0.16;
          g.fillStyle = mode === "albedo" ? EARTH[1 + (k % 4)]
            : (mode === "height" ? "#666666" : "#EAEAEA");
          g.beginPath();
          g.ellipse(sx2, sy2, PPU * (0.09 + rnd() * 0.17), PPU * (0.06 + rnd() * 0.12),
            rnd() * 3.14, 0, Math.PI * 2);
          g.fill();
        }
      });
      g.globalAlpha = 1;

      // 8. The channel: wet sand on the banks, a foam line on the break, and a
      //    dark gravel bed. The water is its own surface, well above this.
      var rows = 260;
      for (var pass = 0; pass < 3; pass++) {
        g.beginPath();
        for (var r2 = 0; r2 <= rows; r2++) {
          var gy2 = lerp(GY0, GY1, r2 / rows);
          var ax2 = riverAxis(gy2);
          if (r2 === 0) g.moveTo(px(ax2), py(gy2)); else g.lineTo(px(ax2), py(gy2));
        }
        if (pass === 0) {        // damp sand shoulder
          g.lineWidth = PPU * 1.12; g.globalAlpha = 0.55;
          g.strokeStyle = mode === "albedo" ? "#75664A"
            : (mode === "height" ? "#8A8A8A" : "#C6C6C6");
        } else if (pass === 1) { // foam on the break
          g.lineWidth = PPU * 0.92; g.globalAlpha = 0.30;
          g.strokeStyle = mode === "albedo" ? "#C4E2E8"
            : (mode === "height" ? "#9C9C9C" : "#8C8C8C");
        } else {                 // the bed itself
          g.lineWidth = PPU * 0.98; g.globalAlpha = 1;
          g.strokeStyle = mode === "albedo" ? "#243C44"
            : (mode === "height" ? "#3A3A3A" : "#5E5E5E");
        }
        g.stroke();
      }
      // Gravel in the bed, so the shallows have something to read through.
      for (var gv = 0; gv < 5200 * (S / 3072); gv++) {
        var gy3 = lerp(GY0, GY1, rnd());
        var off = (rnd() - 0.5) * 0.9;
        g.globalAlpha = 0.10 + rnd() * 0.3;
        g.fillStyle = mode === "albedo" ? (rnd() < 0.5 ? "#3A5460" : "#4E6A74") : "rgb(90,90,90)";
        g.beginPath();
        g.arc(px(riverAxis(gy3) + off), py(gy3), PPU * (0.012 + rnd() * 0.030), 0, Math.PI * 2);
        g.fill();
      }
      g.globalAlpha = 1;
      return cv;
    }

    var terrainAlbedo = new THREE.CanvasTexture(paintTerrain(3072, "albedo"));
    terrainAlbedo.anisotropy = renderer.capabilities.getMaxAnisotropy();
    terrainAlbedo.encoding = THREE.sRGBEncoding;   // colour data, not linear data
    var terrainNormal = normalMapFrom(renderer, paintTerrain(2048, "height"), 3.8);
    var terrainRough = new THREE.CanvasTexture(paintTerrain(1024, "rough"));
    terrainRough.anisotropy = 4;

    var terrainGeo = new THREE.PlaneGeometry(TERRAIN, TERRAIN, 300, 300);
    (function displace() {
      var pos = terrainGeo.attributes.position;
      for (var i = 0; i < pos.count; i++) {
        // The plane is built in its own XY and rotated flat afterwards, so its
        // y is the world z. Convert both to grid space before sampling.
        var gx = pos.getX(i) + HX;
        var gy = -pos.getY(i) + HZ;
        var y = groundY(gx, gy);
        // The rim falls away, so the mesh's edge is a slope disappearing
        // behind the hills rather than a cliff hanging in the haze.
        var rim = Math.max(Math.abs(pos.getX(i)), Math.abs(pos.getY(i))) / (TERRAIN / 2);
        y -= smooth(0.78, 1.0, rim) * 2.6;
        pos.setZ(i, y);
      }
      terrainGeo.computeVertexNormals();
    })();

    var terrain = new THREE.Mesh(terrainGeo, new THREE.MeshStandardMaterial({
      map: terrainAlbedo,
      normalMap: terrainNormal,
      normalScale: new THREE.Vector2(1.75, 1.75),
      roughnessMap: terrainRough,
      roughness: 1.0,
      metalness: 0.0,
      envMapIntensity: 0.50
    }));
    terrain.rotation.x = -Math.PI / 2;
    terrain.receiveShadow = true;
    scene.add(terrain);

    /* A ring of hills beyond the terrain's rim. Two jobs: they hide where the
       mesh stops, and they give the valley somewhere to be. They sit deep in
       the haze, so they stay a silhouette rather than a second lit surface --
       which is what aerial perspective does to a real hillside a hundred
       metres off. */
    (function hills() {
      var rnd = mulberry(7301);
      for (var i = 0; i < 40; i++) {
        var a = (i / 40) * Math.PI * 2 + rnd() * 0.16;
        var d = TERRAIN / 2 + rnd() * 20;
        var rad = 4.5 + rnd() * 9;
        var hill = new THREE.Mesh(new THREE.SphereGeometry(rad, 9, 6),
          new THREE.MeshStandardMaterial({
            color: new THREE.Color().setHSL(
              0.17 + rnd() * 0.07, 0.11 + rnd() * 0.08, 0.26 + rnd() * 0.1
            ),
            roughness: 1.0, metalness: 0.0, flatShading: true
          }));
        hill.position.set(Math.cos(a) * d, -rad * (0.56 + rnd() * 0.26), Math.sin(a) * d);
        hill.scale.y = 0.34 + rnd() * 0.26;
        hill.rotation.y = rnd() * 6.28;
        scene.add(hill);
      }
    })();

    /* ------------------------------------------------------------- water
       A ribbon that follows the river axis, not a sheet across the map: a big
       flat plane would have to sit within a few centimetres of the field to
       look right, and two large near-parallel surfaces stripe under a 16-bit
       depth buffer. The channel is cut 62cm deep and the water sits 24cm down,
       so nothing here is anywhere near coplanar. The foam line is painted into
       the terrain, not floated above the water, for the same reason.

       What makes it read as water is the normal map: two scales of ripple
       scrolling at different speeds, catching the sun disc the sky paints. A
       tinted plane never looks wet. */
    var rippleTex;
    (function buildRipples() {
      var S = 512;
      var cv = document.createElement("canvas");
      cv.width = cv.height = S;
      var g = cv.getContext("2d");
      g.fillStyle = "#808080";
      g.fillRect(0, 0, S, S);
      var rnd = mulberry(6611);
      // Long swells first, then chop on top of them.
      for (var scale = 0; scale < 2; scale++) {
        var n = scale === 0 ? 70 : 340;
        var rad = scale === 0 ? 90 : 26;
        for (var i = 0; i < n; i++) {
          var cx = rnd() * S, cy = rnd() * S, r = rad * (0.5 + rnd());
          var gr = g.createRadialGradient(cx, cy, 0, cx, cy, r);
          var lit = rnd() < 0.5;
          gr.addColorStop(0, lit ? "rgba(190,190,255,0.30)" : "rgba(40,40,190,0.30)");
          gr.addColorStop(0.5, lit ? "rgba(40,40,190,0.18)" : "rgba(190,190,255,0.18)");
          gr.addColorStop(1, "rgba(128,128,255,0)");
          g.fillStyle = gr;
          g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fill();
        }
      }
      rippleTex = new THREE.CanvasTexture(cv);
      rippleTex.wrapS = rippleTex.wrapT = THREE.RepeatWrapping;
      rippleTex.repeat.set(2, 26);
    })();

    var waterMat = new THREE.MeshStandardMaterial({
      color: 0x27637F,
      roughness: 0.09,
      metalness: 0.02,
      normalMap: rippleTex,
      normalScale: new THREE.Vector2(0.42, 0.42),
      envMapIntensity: 1.35
    });
    (function buildWater() {
      var y0 = GY0 - 1, y1 = GY1 + 1, N = 260, HALFW = 0.58;
      var verts = [], uvs = [], idx = [];
      for (var i = 0; i <= N; i++) {
        var gy = lerp(y0, y1, i / N);
        var ax = riverAxis(gy);
        verts.push(wx(ax - HALFW), 0, wz(gy));
        verts.push(wx(ax + HALFW), 0, wz(gy));
        uvs.push(0, i / N, 1, i / N);
        if (i < N) {
          var b = i * 2;
          idx.push(b, b + 1, b + 2, b + 1, b + 3, b + 2);
        }
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
      geo.setIndex(idx);
      geo.computeVertexNormals();
      var mesh = new THREE.Mesh(geo, waterMat);
      mesh.position.y = WATER_Y;
      mesh.receiveShadow = true;
      scene.add(mesh);
    })();

    /* -------------------------------------------------------------- props */
    var stoneMat = STONE_P.map(function (c, i) {
      return new THREE.MeshStandardMaterial({
        color: c, roughness: 0.92 - i * 0.02, metalness: 0.04 + i * 0.01
      });
    });
    var woodMat = WOOD_P.map(function (c, i) {
      return new THREE.MeshStandardMaterial({ color: c, roughness: 0.95 - i * 0.02, metalness: 0 });
    });
    var skinMat = new THREE.MeshStandardMaterial({
      color: SKIN_P[1], roughness: 0.75, metalness: 0.0
    });
    var steelMat = new THREE.MeshStandardMaterial({
      color: 0xA9AEBA, roughness: 0.28, metalness: 0.92
    });
    var goldMat = new THREE.MeshStandardMaterial({
      color: 0xE4C46C, roughness: 0.26, metalness: 0.95
    });
    function teamMat(blue, shade, extra) {
      var p = blue ? BLUE_P : RED_P;
      var o = { color: p[shade], roughness: 0.68, metalness: 0.06 };
      if (extra) { for (var k in extra) { o[k] = extra[k]; } }
      return new THREE.MeshStandardMaterial(o);
    }

    var swayers = [];

    /* -- trees. Four builds, not one shape repeated: a tapered trunk that
       bends, two or three real branches, and a canopy of clusters placed at
       the limb tips so the silhouette is ragged. A uniform tree stamped seven
       times is as loud a "game board" signal as a grid. Green on blue's half,
       autumn on red's, the same rule _draw_one uses when it passes
       `dry = cell_x > midline`. */
    function bentTrunk(r0, r1, h, bend, mat) {
      var geo = new THREE.CylinderGeometry(r0, r1, h, 9, 6);
      var pos = geo.attributes.position;
      for (var i = 0; i < pos.count; i++) {
        var y = pos.getY(i);
        var f = (y + h / 2) / h;                       // 0 at the root
        // Bend and swell: a trunk is neither straight nor a perfect circle.
        pos.setX(i, pos.getX(i) * (1 + 0.22 * Math.sin(f * 5.3)) + bend * f * f * h);
        pos.setZ(i, pos.getZ(i) * (1 + 0.18 * Math.cos(f * 4.1 + 1.2)));
      }
      geo.computeVertexNormals();
      var m = new THREE.Mesh(geo, mat);
      m.castShadow = true;
      m.receiveShadow = true;
      return m;
    }

    function buildTree(seed, dry, species) {
      var rnd = mulberry(seed);
      var leaves = dry ? LEAF_DRY : LEAF_GREEN;
      var g = new THREE.Group();

      var bark = new THREE.MeshStandardMaterial({
        color: new THREE.Color(WOOD_P[1]).multiplyScalar(0.86 + rnd() * 0.3),
        roughness: 0.97, metalness: 0.0, flatShading: true
      });

      // Species 0/1 are broad and low, 2 is tall and narrow, 3 is a scrubby
      // multi-stem. Height, crown shape and cluster count all follow from it.
      var tall = species === 2;
      var scrub = species === 3;
      var h = scrub ? 0.46 : (tall ? 1.02 : 0.68 + rnd() * 0.22);
      var trunk = bentTrunk(
        tall ? 0.038 : 0.052, scrub ? 0.07 : 0.10, h, (rnd() - 0.5) * 0.22, bark
      );
      trunk.position.y = h / 2;
      g.add(trunk);

      // Roots flaring into the ground: a cylinder that just stops at the soil
      // is one of the tells that a tree was placed rather than grown.
      for (var r = 0; r < 5; r++) {
        var ra = (r / 5) * 6.28 + rnd() * 0.6;
        var root = new THREE.Mesh(new THREE.ConeGeometry(0.045 + rnd() * 0.03, 0.20, 5), bark);
        root.position.set(Math.cos(ra) * 0.085, 0.05, Math.sin(ra) * 0.085);
        root.rotation.set(Math.cos(ra) * 0.9, 0, -Math.sin(ra) * 0.9);
        root.castShadow = true;
        g.add(root);
      }

      var canopy = new THREE.Group();
      canopy.position.y = h * (scrub ? 0.55 : 0.80);
      g.add(canopy);

      var nBranch = scrub ? 3 : (tall ? 3 : 2);
      for (var b = 0; b < nBranch; b++) {
        var ba = rnd() * 6.28, bl = 0.22 + rnd() * 0.24;
        var br = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.030, bl, 6), bark);
        br.position.set(
          Math.cos(ba) * bl * 0.35, -h * 0.12 + rnd() * h * 0.2, Math.sin(ba) * bl * 0.35
        );
        br.rotation.set(Math.cos(ba) * 1.15, 0, -Math.sin(ba) * 1.15);
        br.castShadow = true;
        canopy.add(br);
      }

      /* Foliage grows at the ENDS of branches, not as a shell wrapped round
         the trunk, and that is the whole difference between a tree and a
         broccoli floret. Limbs are generated first; clusters are then hung in
         twos and threes near each limb's tip, with nothing filling the middle.
         The gaps between limbs are gaps a viewer can see sky through, and they
         are also the holes that break up the shadow this tree throws. */
      var limbs = [];
      var nLimb = scrub ? 4 : (tall ? 6 : 7);
      for (var li = 0; li < nLimb; li++) {
        var la = (li / nLimb) * 6.28 + rnd() * 0.9;
        var lreach = (scrub ? 0.20 : (tall ? 0.20 : 0.32)) * (0.55 + rnd() * 0.85);
        var lrise = (scrub ? 0.10 : (tall ? 0.30 : 0.16)) * (0.3 + rnd() * 1.4) - 0.04;
        limbs.push([Math.cos(la) * lreach, lrise, Math.sin(la) * lreach]);
        var limb = new THREE.Mesh(
          new THREE.CylinderGeometry(0.010, 0.026, lreach * 1.9, 5), bark
        );
        limb.position.set(
          Math.cos(la) * lreach * 0.5, lrise * 0.5 - 0.02, Math.sin(la) * lreach * 0.5
        );
        limb.lookAt(new THREE.Vector3(Math.cos(la) * lreach, lrise, Math.sin(la) * lreach));
        limb.rotateX(Math.PI / 2);
        limb.castShadow = true;
        canopy.add(limb);
      }

      var nClust = limbs.length * (scrub ? 3 : 7);
      for (var c = 0; c < nClust; c++) {
        var limbEnd = limbs[c % limbs.length];
        // Bunched at the tip, scattered a little back along the limb.
        var back = Math.pow(rnd(), 1.9) * 0.52;
        var spread = scrub ? 0.10 : 0.17;
        var cxp = limbEnd[0] * (1 - back) + (rnd() - 0.5) * spread;
        var cyp = limbEnd[1] * (1 - back) + (rnd() - 0.5) * spread * 0.9;
        var czp = limbEnd[2] * (1 - back) + (rnd() - 0.5) * spread;
        var up = clamp(0.5 + cyp * 2.0, 0, 1);
        var outlier = rnd() < 0.22;
        var size = (outlier ? 0.062 + rnd() * 0.045 : 0.108 + rnd() * 0.080) * (scrub ? 0.85 : 1);
        // Two shades within one cluster read as light passing through.
        var shade = up > 0.50 ? leaves[2 + (rnd() < 0.4 ? 1 : 0)] : leaves[rnd() < 0.35 ? 1 : 2];
        var blob = new THREE.Mesh(
          new THREE.IcosahedronGeometry(size, 1),
          new THREE.MeshStandardMaterial({
            color: new THREE.Color(shade).multiplyScalar(0.92 + rnd() * 0.22),
            roughness: 0.93, metalness: 0.0, flatShading: true
          })
        );
        blob.position.set(cxp * (outlier ? 1.35 : 1), cyp, czp * (outlier ? 1.35 : 1));
        // Flattened and stretched: a leaf mass hangs, it is not a sphere.
        blob.scale.set(1 + rnd() * 0.6, 0.62 + rnd() * 0.36, 1 + rnd() * 0.6);
        blob.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
        blob.castShadow = true;
        blob.receiveShadow = true;
        canopy.add(blob);
      }

      g.userData.canopy = canopy;
      g.userData.phase = rnd() * 6.28;
      return g;
    }

    /* The dark patch a canopy puts on the ground. A shadow map alone leaves a
       prop looking pasted on, because it misses the ambient the prop occludes;
       this is the cheap stand-in for that occlusion. Offset in the depth test
       rather than lifted, so it never fights the field it lies on. */
    function contactBlob(radius, alpha) {
      var m = new THREE.Mesh(new THREE.PlaneGeometry(radius * 2, radius * 2),
        new THREE.MeshBasicMaterial({
          map: poolTex, color: 0x0C1408, transparent: true, opacity: alpha,
          depthWrite: false, polygonOffset: true,
          polygonOffsetFactor: -4, polygonOffsetUnits: -4
        }));
      m.rotation.x = -Math.PI / 2;
      m.position.y = 0.03;
      return m;
    }

    (function placeTrees() {
      var rnd = mulberry(4711);
      world.trees.forEach(function (cell, ti) {
        var gx = cell[0], gy = cell[1];
        // A tree standing in the channel would be a tree standing in the
        // river: nudge it to the near bank, keeping the cell it occupies.
        var ax = riverAxis(gy);
        if (Math.abs(gx - ax) < 0.95) gx = gx <= ax ? ax - 0.98 : ax + 0.98;
        gx += (rnd() - 0.5) * 0.18;
        var gyj = gy + (rnd() - 0.5) * 0.18;

        var tree = buildTree(cell[0] * 31 + cell[1] * 17 + 7, cell[0] > MIDLINE, ti % 4);
        place(tree, gx, gyj, 0);
        tree.rotation.y = rnd() * 6.28;
        var sc = 0.80 + rnd() * 0.34;
        tree.scale.set(sc, sc * (0.9 + rnd() * 0.25), sc);
        scene.add(tree);
        swayers.push(tree.userData);

        // Broken shade, not one soft disc. A gappy canopy does not put an even
        // pool of dark on the ground, and the low sun throws it well to one side.
        for (var sb = 0; sb < 4; sb++) {
          var sblob = contactBlob((0.30 + rnd() * 0.26) * sc, 0.30 + rnd() * 0.16);
          place(sblob, gx + 1.05 + (rnd() - 0.5) * 0.7, gyj + 0.62 + (rnd() - 0.5) * 0.7, 0.03);
          scene.add(sblob);
        }

        // Saplings at the foot, so a tree is a place and not an object
        // standing on a lawn.
        for (var k = 0; k < 3; k++) {
          var sa = rnd() * 6.28, sd = 0.32 + rnd() * 0.3;
          var sap = buildTree(cell[0] * 91 + k * 13, cell[0] > MIDLINE, 3);
          place(sap, gx + Math.cos(sa) * sd, gyj + Math.sin(sa) * sd, 0);
          var ss = 0.26 + rnd() * 0.22;
          sap.scale.set(ss, ss, ss);
          sap.rotation.y = rnd() * 6.28;
          scene.add(sap);
        }
      });
    })();

    /* -- ground cover. Instanced, because close range wants thousands of items
       and thousands of draw calls would not run. Clumped rather than
       sprinkled: real grass grows in patches around what shelters it, and an
       even sprinkle is as artificial as bare ground. Nothing lands on a cell
       an actor uses. */
    (function scatter() {
      var rnd = mulberry(20260919);
      var busy = [];
      world.trees.forEach(function (c) { busy.push(c); });
      busy.push(BLUE_BASE, RED_BASE, BLUE_FLAG);
      CANDIDATES.forEach(function (c) { busy.push(c); });
      payload.blue_cells.concat(payload.red_cells).forEach(function (row) {
        row.forEach(function (c) { busy.push(c); });
      });

      function clearOf(gx, gy, r) {
        for (var i = 0; i < busy.length; i++) {
          if (Math.hypot(gx - busy[i][0], gy - busy[i][1]) < r) return false;
        }
        return true;
      }

      // A clump of blades: six tapered triangles fanning out of one root. Flat
      // shaded, no texture, but it is grass-shaped, which a cone is not.
      function bladeClump() {
        var v = [], n = 6;
        for (var b = 0; b < n; b++) {
          var a = (b / n) * 6.28 + rnd() * 0.6;
          var lean = 0.022 + rnd() * 0.042;
          var hgt = 0.052 + rnd() * 0.055;
          var wdt = 0.010;
          var tx = Math.cos(a) * lean, tz = Math.sin(a) * lean;
          var px2 = Math.cos(a + 1.57) * wdt, pz2 = Math.sin(a + 1.57) * wdt;
          v.push(-px2, 0, -pz2, px2, 0, pz2, tx, hgt, tz);
          v.push(px2, 0, pz2, -px2, 0, -pz2, tx, hgt, tz);   // both faces
        }
        var geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
        geo.computeVertexNormals();
        return geo;
      }

      function instanced(geo, mat, count) {
        var im = new THREE.InstancedMesh(geo, mat, count);
        im.castShadow = true;
        im.receiveShadow = true;
        im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        scene.add(im);
        return im;
      }

      var dummy = new THREE.Object3D();
      // The nearest cells are where a viewer's eye goes to decide whether the
      // ground is real, so the field carries several times the cover of the
      // outfield, which the haze swallows anyway.
      var TUFTS = 4600, STONES = 520, FLOWERS = 520, LITTER = 1500;
      var tufts = instanced(bladeClump(), new THREE.MeshStandardMaterial({
        color: 0xFFFFFF, roughness: 0.95, metalness: 0.0, side: THREE.DoubleSide
      }), TUFTS);
      var stones = instanced(new THREE.DodecahedronGeometry(0.085, 0),
        new THREE.MeshStandardMaterial({
          color: 0xFFFFFF, roughness: 0.88, metalness: 0.04, flatShading: true
        }), STONES);
      var flowers = instanced(new THREE.IcosahedronGeometry(0.026, 0),
        new THREE.MeshStandardMaterial({ color: 0xFFFFFF, roughness: 0.7, metalness: 0.0 }),
        FLOWERS);
      // Leaf litter and twigs: flat, dull, and the cheapest thing that stops
      // close ground reading as a smooth surface with grass standing on it.
      var litter = instanced(new THREE.PlaneGeometry(0.075, 0.048),
        new THREE.MeshStandardMaterial({
          color: 0xFFFFFF, roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide
        }), LITTER);
      var litterCols = [0x6E5E3C, 0x7C6A44, 0x8A7A52, 0x5E5034, 0x8E8258];

      // Clump centres, so the cover reads as patches rather than as static.
      var clumps = [];
      for (var cc = 0; cc < 300; cc++) {
        if (cc % 3) {
          clumps.push([lerp(-2.5, GW + 1.5, rnd()), lerp(-2.5, GH + 1.5, rnd()),
            0.28 + rnd() * 0.95]);
        } else {
          clumps.push([lerp(GX0 + 2, GX1 - 2, rnd()), lerp(GY0 + 2, GY1 - 2, rnd()),
            0.35 + rnd() * 1.6]);
        }
      }

      function sample(nearClump) {
        if (nearClump && rnd() < 0.72) {
          var c = clumps[Math.floor(rnd() * clumps.length)];
          var a = rnd() * 6.28, d = Math.pow(rnd(), 0.6) * c[2];
          return [c[0] + Math.cos(a) * d, c[1] + Math.sin(a) * d];
        }
        return [lerp(GX0 + 1, GX1 - 1, rnd()), lerp(GY0 + 1, GY1 - 1, rnd())];
      }

      var grassCols = [0x4E5432, 0x59603A, 0x646B42, 0x71764C, 0x80835A, 0x8E9066, 0x9C9A72];
      var stoneCols = [0x736C62, 0x837C70, 0x8E877A, 0x605A52];
      var flowerCols = [0xD8D2B8, 0xD6C476, 0xC4B8C8, 0xCFCFC2, 0xC8AFB4];
      var counts = { t: 0, s: 0, f: 0, l: 0 };
      var guard = 0;

      while ((counts.t < TUFTS || counts.s < STONES || counts.f < FLOWERS || counts.l < LITTER)
        && guard++ < 190000) {
        var want = counts.t < TUFTS ? "t"
          : (counts.f < FLOWERS ? "f" : (counts.l < LITTER ? "l" : "s"));
        var pt = sample(want !== "s");
        var gx = pt[0], gy = pt[1];
        if (gx < GX0 + 0.5 || gx > GX1 - 0.5 || gy < GY0 + 0.5 || gy > GY1 - 0.5) continue;
        if (Math.abs(gx - riverAxis(gy)) < 0.66) continue;    // not in the water
        var inField = gx > -0.8 && gx < GW - 0.2 && gy > -0.8 && gy < GH - 0.2;
        if (inField && !clearOf(gx, gy, want === "t" ? 0.40 : 0.46)) continue;
        // The playing surface stays readable; the cover thickens away from it.
        if (!inField && want === "t" && rnd() > 0.34) continue;
        if (!inField && rnd() > 0.5) continue;

        dummy.position.set(wx(gx), groundY(gx, gy), wz(gy));
        dummy.rotation.set(0, rnd() * 6.28, 0);
        if (want === "t") {
          var ts = 0.62 + rnd() * 0.75;
          dummy.scale.set(ts, ts * (0.55 + rnd() * 0.75), ts);
          dummy.updateMatrix();
          tufts.setMatrixAt(counts.t, dummy.matrix);
          tufts.setColorAt(counts.t, new THREE.Color(
            grassCols[Math.floor(rnd() * grassCols.length)]
          ).convertSRGBToLinear());
          counts.t++;
        } else if (want === "f") {
          dummy.position.y += 0.10 + rnd() * 0.08;
          var fs = 0.5 + rnd() * 0.55;
          dummy.scale.set(fs, fs, fs);
          dummy.updateMatrix();
          flowers.setMatrixAt(counts.f, dummy.matrix);
          flowers.setColorAt(counts.f, new THREE.Color(
            flowerCols[Math.floor(rnd() * flowerCols.length)]
          ).convertSRGBToLinear());
          counts.f++;
        } else if (want === "l") {
          dummy.rotation.set(-Math.PI / 2 + (rnd() - 0.5) * 0.5, 0, rnd() * 6.28);
          var ls = 0.6 + rnd() * 1.3;
          dummy.scale.set(ls, ls, ls);
          dummy.position.y += 0.012;
          dummy.updateMatrix();
          litter.setMatrixAt(counts.l, dummy.matrix);
          litter.setColorAt(counts.l, new THREE.Color(
            litterCols[Math.floor(rnd() * litterCols.length)]
          ).convertSRGBToLinear());
          counts.l++;
        } else {
          dummy.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3);
          var ss2 = 0.45 + rnd() * 1.1;
          dummy.scale.set(ss2, ss2 * (0.5 + rnd() * 0.4), ss2);
          dummy.position.y -= 0.02;
          dummy.updateMatrix();
          stones.setMatrixAt(counts.s, dummy.matrix);
          stones.setColorAt(counts.s, new THREE.Color(
            stoneCols[Math.floor(rnd() * stoneCols.length)]
          ).convertSRGBToLinear());
          counts.s++;
        }
      }
      [tufts, stones, flowers, litter].forEach(function (im) {
        im.count = im === tufts ? counts.t
          : (im === stones ? counts.s : (im === flowers ? counts.f : counts.l));
        im.instanceMatrix.needsUpdate = true;
        if (im.instanceColor) im.instanceColor.needsUpdate = true;
      });

      // Reeds on the banks, where the ground is wet and nothing else grows.
      var reeds = instanced(bladeClump(), new THREE.MeshStandardMaterial({
        color: 0x6E6A40, roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide
      }), 420);
      var rc = 0, rg = 0;
      while (rc < 420 && rg++ < 12000) {
        var ry2 = lerp(GY0 + 1, GY1 - 1, rnd());
        var side = rnd() < 0.5 ? -1 : 1;
        var rx2 = riverAxis(ry2) + side * (0.72 + rnd() * 0.34);
        dummy.position.set(wx(rx2), groundY(rx2, ry2), wz(ry2));
        dummy.rotation.set((rnd() - 0.5) * 0.3, rnd() * 6.28, (rnd() - 0.5) * 0.3);
        var rs = 0.9 + rnd() * 0.9;
        dummy.scale.set(rs * 0.6, rs * 2.2, rs * 0.6);
        dummy.updateMatrix();
        reeds.setMatrixAt(rc, dummy.matrix);
        rc++;
      }
      reeds.count = rc;
      reeds.instanceMatrix.needsUpdate = true;
    })();

    /* -- the bridge. Spans the channel on the blue base's row, drawn from the
       same axis the water is, which is how the GIF keeps it on the river. */
    (function bridge() {
      var row = BLUE_BASE[1];
      var ax = riverAxis(row);
      var deck = new THREE.Group();
      scene.add(deck);
      var span = 2.10, y = 0.055;
      for (var i = 0; i < 11; i++) {
        var gx = ax - span / 2 + span * (i / 10);
        var plank = new THREE.Mesh(
          new THREE.BoxGeometry(span / 11 * 0.88, 0.055, 0.78), woodMat[i % 2 ? 2 : 1]
        );
        plank.position.set(wx(gx), y, wz(row));
        plank.castShadow = true;
        plank.receiveShadow = true;
        deck.add(plank);
      }
      [-0.40, 0.40].forEach(function (off) {
        var rail = new THREE.Mesh(new THREE.BoxGeometry(span, 0.05, 0.05), woodMat[3]);
        rail.position.set(wx(ax), y + 0.30, wz(row + off));
        rail.castShadow = true;
        deck.add(rail);
        for (var k = -2; k <= 2; k++) {
          var post = new THREE.Mesh(
            new THREE.CylinderGeometry(0.035, 0.04, 0.34, 6), woodMat[1]
          );
          post.position.set(wx(ax + k * (span / 4.4)), y + 0.14, wz(row + off));
          post.castShadow = true;
          deck.add(post);
        }
      });
      [-0.55, 0.55].forEach(function (off) {
        var pile = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.6, 7), woodMat[0]);
        pile.position.set(wx(ax + off), y - 0.28, wz(row));
        pile.castShadow = true;
        deck.add(pile);
      });
    })();

    /* -- the keeps. One per base, stone with a team-coloured roof, the two
       colours keep_sprite paints. Plain boxes read as toy bricks, so the walls
       get a coursed-masonry albedo with its own normal and roughness maps,
       tapering buttresses, a tiled roof, a plank door, moss where the wall
       meets wet ground, and an occlusion patch underneath. */
    function masonryTextures() {
      var S = 512;
      function paint(mode) {
        var cv = document.createElement("canvas");
        cv.width = cv.height = S;
        var g = cv.getContext("2d");
        var rnd = mulberry(mode === "height" ? 313 : (mode === "rough" ? 919 : 171));
        g.fillStyle = mode === "albedo" ? "#54545E" : (mode === "height" ? "#3C3C3C" : "#C8C8C8");
        g.fillRect(0, 0, S, S);
        var rows = 13, rh = S / rows;
        for (var r = 0; r < rows; r++) {
          var offset = (r % 2) * 0.5;
          for (var c = -1; c < 6; c++) {
            var bw = S / 6, bx = (c + offset) * bw + 2, by = r * rh + 2;
            var t = rnd();
            if (mode === "albedo") {
              var lum = 0.70 + t * 0.55;
              g.fillStyle = "rgb(" + (86 * lum | 0) + "," + (86 * lum | 0) + "," +
                (96 * lum | 0) + ")";
            } else if (mode === "height") {
              var hv = 150 + t * 80 | 0;
              g.fillStyle = "rgb(" + hv + "," + hv + "," + hv + ")";
            } else {
              var rv = 180 + t * 60 | 0;
              g.fillStyle = "rgb(" + rv + "," + rv + "," + rv + ")";
            }
            g.fillRect(bx, by, bw - 4, rh - 4);
            // Pitting and a weathered top edge on every block.
            for (var k = 0; k < 14; k++) {
              g.globalAlpha = 0.06 + rnd() * 0.16;
              g.fillStyle = mode === "albedo" ? (rnd() < 0.5 ? "#2E2E36" : "#8E8E9A")
                : (rnd() < 0.5 ? "#606060" : "#B0B0B0");
              g.beginPath();
              g.arc(bx + rnd() * bw, by + rnd() * rh, 1 + rnd() * 4, 0, Math.PI * 2);
              g.fill();
            }
            g.globalAlpha = 1;
          }
        }
        return cv;
      }
      var alb = new THREE.CanvasTexture(paint("albedo"));
      alb.encoding = THREE.sRGBEncoding;
      var rough = new THREE.CanvasTexture(paint("rough"));
      [alb, rough].forEach(function (t) {
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.repeat.set(1.6, 1.4);
      });
      var nrm = normalMapFrom(renderer, paint("height"), 2.2);
      nrm.wrapS = nrm.wrapT = THREE.RepeatWrapping;
      nrm.repeat.set(1.6, 1.4);
      return { map: alb, normalMap: nrm, roughnessMap: rough };
    }
    var MASONRY = masonryTextures();

    function roofTexture(blue) {
      var S = 256;
      var cv = document.createElement("canvas");
      cv.width = cv.height = S;
      var g = cv.getContext("2d");
      var rnd = mulberry(blue ? 55 : 77);
      /* Weathered tile, not a team colour. A bright blue or red roof says
         "player one and player two" louder than anything else in the frame,
         and a saturated accent reads BETTER against a drab roof than against a
         saturated one -- so the team's colour moves to the banner, the pennant
         and the painted shield over the door, where a real fortification would
         carry it anyway. Each roof keeps a faint tint of its side, no more. */
      var base = new THREE.Color(blue ? BLUE_P[1] : RED_P[1]);
      base.lerp(new THREE.Color(0x6E6258), 0.80);
      var p = [
        base.clone().multiplyScalar(0.72),
        base.clone(),
        base.clone().multiplyScalar(1.22),
        base.clone().multiplyScalar(0.55)
      ].map(function (c) { return "#" + c.getHexString(); });
      g.fillStyle = p[1];
      g.fillRect(0, 0, S, S);
      // Overlapping courses of tile, each one slightly its own colour.
      for (var row = 0; row < 9; row++) {
        for (var col = 0; col < 10; col++) {
          var t = rnd();
          var c = new THREE.Color(p[t < 0.5 ? 1 : (t < 0.85 ? 2 : 0)])
            .multiplyScalar(0.74 + t * 0.52);
          g.fillStyle = "#" + c.getHexString();
          var x = col * (S / 10) + (row % 2) * (S / 20);
          var y = row * (S / 9);
          g.beginPath();
          g.moveTo(x, y + S / 9);
          g.lineTo(x, y + 4);
          g.arc(x + S / 20, y + 4, S / 20, Math.PI, 0);
          g.lineTo(x + S / 10, y + S / 9);
          g.closePath();
          g.fill();
          g.strokeStyle = "rgba(0,0,0,0.22)";
          g.lineWidth = 1;
          g.stroke();
        }
      }
      // Weathering: moss in the courses, soot and rain streaks down the pitch.
      for (var ms = 0; ms < 90; ms++) {
        g.globalAlpha = 0.05 + rnd() * 0.22;
        g.fillStyle = rnd() < 0.55 ? "#4C5230" : "#3A342C";
        g.beginPath();
        g.ellipse(rnd() * S, rnd() * S, 4 + rnd() * 26, 3 + rnd() * 16, rnd() * 3.14, 0,
          Math.PI * 2);
        g.fill();
      }
      g.globalAlpha = 0.12;
      g.lineWidth = 2;
      for (var st = 0; st < 60; st++) {
        var stx = rnd() * S;
        g.strokeStyle = rnd() < 0.5 ? "#2E2A24" : "#8A8478";
        g.beginPath();
        g.moveTo(stx, rnd() * S * 0.4);
        g.lineTo(stx + (rnd() - 0.5) * 8, S);
        g.stroke();
      }
      g.globalAlpha = 1;
      var tex = new THREE.CanvasTexture(cv);
      tex.encoding = THREE.sRGBEncoding;
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      tex.repeat.set(1.4, 1.4);
      return tex;
    }

    /* -- banners. A pole, a pennant and a gold finial, in the two team
       palettes banner_sprite uses. */
    function buildBanner(blue, scaleY) {
      var g = new THREE.Group();
      var p = blue ? BLUE_P : RED_P;
      var h = 0.78 * (scaleY || 1);
      var pole = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.022, h, 7),
        new THREE.MeshStandardMaterial({ color: 0xCEC6B4, roughness: 0.5, metalness: 0.3 }));
      pole.position.y = h / 2;
      pole.castShadow = true;
      g.add(pole);

      var cloth = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 0.26, 6, 3),
        new THREE.MeshStandardMaterial({
          color: p[1], roughness: 0.82, metalness: 0.0, side: THREE.DoubleSide
        }));
      cloth.position.set(0.21, h - 0.20, 0);
      cloth.castShadow = true;
      g.add(cloth);
      g.userData.cloth = cloth;

      var trim = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 0.07),
        new THREE.MeshStandardMaterial({
          color: p[2], roughness: 0.8, side: THREE.DoubleSide
        }));
      trim.position.set(0.21, h - 0.11, 0.002);
      g.add(trim);

      var finial = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 8), goldMat);
      finial.position.y = h + 0.02;
      g.add(finial);
      return g;
    }

    function buildKeep(blue, cell) {
      var g = new THREE.Group();
      // Off the cell, not on it: a unit spawning at its base has to be visible.
      var kx = cell[0] + (blue ? -0.66 : 0.66), ky = cell[1];
      place(g, kx, ky, 0);
      g.rotation.y = blue ? 0.16 : -0.16;
      scene.add(g);

      var wallMat = new THREE.MeshStandardMaterial({
        map: MASONRY.map, normalMap: MASONRY.normalMap, roughnessMap: MASONRY.roughnessMap,
        normalScale: new THREE.Vector2(1.1, 1.1),
        roughness: 1.0, metalness: 0.05, envMapIntensity: 0.7
      });
      var plinthMat = wallMat.clone();
      plinthMat.color = new THREE.Color(0x6E6E78);

      var plinth = new THREE.Mesh(new THREE.BoxGeometry(1.22, 0.16, 1.22), plinthMat);
      plinth.position.y = 0.08;
      plinth.castShadow = true; plinth.receiveShadow = true;
      g.add(plinth);

      var body = new THREE.Mesh(new THREE.BoxGeometry(0.90, 0.80, 0.90), wallMat);
      body.position.y = 0.55;
      body.castShadow = true; body.receiveShadow = true;
      g.add(body);

      // Buttresses. They give the silhouette corners to catch the low sun,
      // which is what stops a wall reading as a single flat panel.
      [[0.45, 0.45], [0.45, -0.45], [-0.45, 0.45], [-0.45, -0.45]].forEach(function (b) {
        var but = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.15, 0.86, 6), wallMat);
        but.position.set(b[0], 0.56, b[1]);
        but.castShadow = true; but.receiveShadow = true;
        g.add(but);
        var capB = new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.12, 6), wallMat);
        capB.position.set(b[0], 1.03, b[1]);
        capB.castShadow = true;
        g.add(capB);
      });

      var band = new THREE.Mesh(new THREE.BoxGeometry(0.98, 0.09, 0.98), plinthMat);
      band.position.y = 0.94;
      band.castShadow = true; band.receiveShadow = true;
      g.add(band);

      // Battlements, deliberately uneven: a perfect row of merlons is toy-like.
      var mr = mulberry(blue ? 21 : 42);
      for (var i = 0; i < 12; i++) {
        var side = Math.floor(i / 3), k = (i % 3) - 1;
        var mx = side === 0 ? k * 0.32 : (side === 1 ? 0.46 : (side === 2 ? k * 0.32 : -0.46));
        var mz = side === 0 ? -0.46 : (side === 1 ? k * 0.32 : (side === 2 ? 0.46 : k * 0.32));
        var mh = 0.13 + mr() * 0.07;
        var merlon = new THREE.Mesh(new THREE.BoxGeometry(0.17, mh, 0.17), wallMat);
        merlon.position.set(mx, 0.99 + mh / 2, mz);
        merlon.rotation.y = (mr() - 0.5) * 0.14;
        merlon.castShadow = true; merlon.receiveShadow = true;
        g.add(merlon);
      }

      var roof = new THREE.Mesh(new THREE.ConeGeometry(0.70, 0.60, 4, 3),
        new THREE.MeshStandardMaterial({
          map: roofTexture(blue), roughness: 0.78, metalness: 0.03, envMapIntensity: 0.6
        }));
      roof.position.y = 1.40;
      roof.rotation.y = Math.PI / 4;
      roof.castShadow = true; roof.receiveShadow = true;
      g.add(roof);

      var cap = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 10), goldMat);
      cap.position.y = 1.72;
      g.add(cap);

      // A pennant on the finial and a painted shield over the gate: the team's
      // colour, concentrated where it reads, against a drab roof.
      var pennant = buildBanner(blue, 0.58);
      pennant.position.set(0, 1.62, 0);
      pennant.rotation.y = blue ? 0.5 : -0.5;
      g.add(pennant);
      swayers.push({ canopy: pennant, phase: blue ? 1.1 : 4.3 });

      var crest = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.03, 20),
        teamMat(blue, 1, { roughness: 0.62 }));
      crest.rotation.x = Math.PI / 2;
      crest.position.set(0, 0.73, blue ? 0.47 : -0.47);
      crest.castShadow = true;
      g.add(crest);
      var crestRim = new THREE.Mesh(new THREE.TorusGeometry(0.148, 0.016, 8, 24), goldMat);
      crestRim.position.copy(crest.position);
      g.add(crestRim);
      var crestBoss = new THREE.Mesh(new THREE.SphereGeometry(0.040, 12, 10), goldMat);
      crestBoss.position.set(0, 0.73, blue ? 0.50 : -0.50);
      g.add(crestBoss);

      // A plank door, and arrow slits rather than windows.
      var doorSide = blue ? 0.455 : -0.455;
      for (var pl = 0; pl < 4; pl++) {
        var plank = new THREE.Mesh(new THREE.BoxGeometry(0.066, 0.42, 0.035), woodMat[pl % 2]);
        plank.position.set(-0.105 + pl * 0.07, 0.36, doorSide);
        plank.castShadow = true;
        g.add(plank);
      }
      var lintel = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.06, 0.07), plinthMat);
      lintel.position.set(0, 0.60, doorSide);
      lintel.castShadow = true;
      g.add(lintel);

      // Lit slits. Above 1.0 on purpose: the scene buffer is half-float, so
      // they can sit hot enough for the bright pass to find them.
      [[-0.26, 0.70], [0.26, 0.70]].forEach(function (w) {
        var slit = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.19, 0.03),
          new THREE.MeshBasicMaterial({ color: new THREE.Color(2.4, 1.6, 0.8) }));
        slit.position.set(w[0], w[1], blue ? 0.452 : -0.452);
        g.add(slit);
      });

      // Moss and fallen stone at the footing.
      var mossMat = new THREE.MeshStandardMaterial({
        color: 0x404A2A, roughness: 1.0, metalness: 0.0, flatShading: true
      });
      for (var ms = 0; ms < 22; ms++) {
        var ma = mr() * 6.28, md = 0.52 + mr() * 0.26;
        var clump = new THREE.Mesh(new THREE.IcosahedronGeometry(0.04 + mr() * 0.05, 0),
          mr() < 0.65 ? mossMat : stoneMat[1]);
        clump.position.set(Math.cos(ma) * md, 0.03 + mr() * 0.06, Math.sin(ma) * md);
        clump.scale.y = 0.5;
        clump.castShadow = true; clump.receiveShadow = true;
        g.add(clump);
      }

      var blob = contactBlob(1.15, 0.55);
      blob.position.set(0.18, 0.03, 0.14);
      g.add(blob);

      // A brazier by the gate. Real lumens with inverse-square decay, so in
      // daylight it is a warm accent on the wall and nothing more.
      var fire = new THREE.PointLight(0xFFB268, 1, 3.6, 2);
      fire.power = 44;
      fire.position.set(blue ? 0.62 : -0.62, 0.44, 0.52);
      g.add(fire);
      var ember = new THREE.Mesh(new THREE.SphereGeometry(0.055, 10, 8),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(2.6, 1.15, 0.38) }));
      ember.position.copy(fire.position);
      g.add(ember);
      return { group: g, fire: fire, ember: ember };
    }
    var blueKeep = buildKeep(true, BLUE_BASE);
    var redKeep = buildKeep(false, RED_BASE);

    var blueBanner = buildBanner(true, 1);
    place(blueBanner, BLUE_FLAG[0], BLUE_FLAG[1], 0);
    scene.add(blueBanner);

    // The red banner stands on whichever candidate the episode's own state
    // names. The cell is read per step rather than fixed, so an environment
    // whose flag is elsewhere is drawn where it is, not where a default says.
    var redBanner = buildBanner(false, 1);
    scene.add(redBanner);

    /* ----------------------------------------------------------- soldiers
       Low-poly, but with the parts the sprite has: greaves, a tunic in the
       team palette, a belt, a shoulder cape, a helmet, a spear and a shield. */
    function buildSoldier(blue) {
      var g = new THREE.Group();
      var p = blue ? BLUE_P : RED_P;
      var tunic = new THREE.MeshStandardMaterial({
        color: p[1], roughness: 0.72, metalness: 0.05
      });
      var tunicLit = new THREE.MeshStandardMaterial({
        color: p[2], roughness: 0.72, metalness: 0.05
      });
      var capeMat = new THREE.MeshStandardMaterial({
        color: p[0], roughness: 0.8, metalness: 0.03
      });

      var legs = new THREE.Group();
      g.add(legs);
      g.userData.legs = [];
      [-0.055, 0.055].forEach(function (off) {
        var leg = new THREE.Group();
        leg.position.set(0, 0.20, off);
        var shin = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.20, 0.075),
          new THREE.MeshStandardMaterial({ color: 0x4E4030, roughness: 0.9 }));
        shin.position.y = -0.10;
        shin.castShadow = true;
        leg.add(shin);
        var boot = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.045, 0.085),
          new THREE.MeshStandardMaterial({ color: 0x2C241C, roughness: 0.95 }));
        boot.position.set(0.012, -0.195, 0);
        boot.castShadow = true;
        leg.add(boot);
        legs.add(leg);
        g.userData.legs.push(leg);
      });

      var torso = new THREE.Mesh(new THREE.BoxGeometry(0.118, 0.21, 0.165), tunic);
      torso.position.y = 0.318;
      torso.castShadow = true; torso.receiveShadow = true;
      g.add(torso);

      // Shoulders wider than the waist, and a chest plate that catches the sun
      // separately from the tunic: a single box reads as a brick from anything
      // closer than the raised camera.
      var chest = new THREE.Mesh(new THREE.BoxGeometry(0.128, 0.105, 0.188), tunicLit);
      chest.position.y = 0.382;
      chest.castShadow = true;
      g.add(chest);

      [-0.105, 0.105].forEach(function (az) {
        var arm = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.155, 0.055), tunic);
        arm.position.set(0.006, 0.335, az);
        arm.rotation.x = az > 0 ? 0.12 : -0.12;
        arm.castShadow = true;
        g.add(arm);
        var hand = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.05, 0.045), skinMat);
        hand.position.set(0.012, 0.253, az * 1.05);
        hand.castShadow = true;
        g.add(hand);
      });

      var neck = new THREE.Mesh(new THREE.CylinderGeometry(0.030, 0.034, 0.045, 8), skinMat);
      neck.position.y = 0.442;
      g.add(neck);

      var belt = new THREE.Mesh(new THREE.BoxGeometry(0.165, 0.035, 0.215),
        new THREE.MeshStandardMaterial({ color: 0xCEB05C, roughness: 0.4, metalness: 0.6 }));
      belt.position.y = 0.225;
      belt.castShadow = true;
      g.add(belt);

      var cape = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.20, 0.19), capeMat);
      cape.position.set(-0.075, 0.30, 0);
      cape.castShadow = true;
      g.add(cape);

      var head = new THREE.Mesh(new THREE.SphereGeometry(0.052, 12, 10), skinMat);
      head.position.y = 0.487;
      head.scale.set(0.92, 1.05, 0.95);
      head.castShadow = true;
      g.add(head);

      // A dome with a low spike, not a party hat: the cone alone was most of
      // what made these figures read as toys at close range.
      var helm = new THREE.Mesh(
        new THREE.SphereGeometry(0.058, 14, 10, 0, Math.PI * 2, 0, Math.PI / 2), steelMat
      );
      helm.position.y = 0.503;
      helm.scale.set(1.0, 0.95, 1.02);
      helm.castShadow = true;
      g.add(helm);
      var spike = new THREE.Mesh(new THREE.ConeGeometry(0.026, 0.055, 8), steelMat);
      spike.position.y = 0.565;
      spike.castShadow = true;
      g.add(spike);
      [-0.052, 0.052].forEach(function (cz) {
        var cheek = new THREE.Mesh(new THREE.BoxGeometry(0.052, 0.055, 0.016), steelMat);
        cheek.position.set(0.006, 0.472, cz);
        cheek.castShadow = true;
        g.add(cheek);
      });
      /* The crest sweeps back, not up: with a swept crest, a visor and a nose
         guard, which way a figure faces is unmistakable in a still. Without a
         cue like this a mirrored heading is invisible, which is exactly how
         one of these viewers shipped with its agent walking backwards. */
      var plume = new THREE.Mesh(new THREE.BoxGeometry(0.085, 0.055, 0.028), tunicLit);
      plume.position.set(-0.035, 0.605, 0);
      plume.rotation.z = 0.35;
      plume.castShadow = true;
      g.add(plume);

      var visor = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.026, 0.078),
        new THREE.MeshStandardMaterial({ color: 0x14161C, roughness: 0.45, metalness: 0.5 }));
      visor.position.set(0.048, 0.492, 0);
      g.add(visor);

      var nose = new THREE.Mesh(new THREE.BoxGeometry(0.030, 0.075, 0.020), steelMat);
      nose.position.set(0.055, 0.470, 0);
      nose.castShadow = true;
      g.add(nose);

      // Spear: a haft and a steel head, carried on the outside arm.
      var spear = new THREE.Group();
      spear.position.set(0.016, 0.325, 0.128);
      g.add(spear);
      var haft = new THREE.Mesh(new THREE.CylinderGeometry(0.0085, 0.0095, 0.68, 6), woodMat[2]);
      haft.castShadow = true;
      spear.add(haft);
      var tip = new THREE.Mesh(new THREE.ConeGeometry(0.021, 0.075, 6), steelMat);
      tip.position.y = 0.375;
      tip.castShadow = true;
      spear.add(tip);
      spear.rotation.x = 0.10;

      // Round shield on the inside arm.
      var shield = new THREE.Mesh(new THREE.CylinderGeometry(0.082, 0.082, 0.018, 18), tunic);
      shield.rotation.z = Math.PI / 2;
      shield.position.set(0.006, 0.325, -0.140);
      shield.castShadow = true;
      g.add(shield);
      var rim = new THREE.Mesh(new THREE.TorusGeometry(0.081, 0.010, 8, 22), steelMat);
      rim.rotation.y = Math.PI / 2;
      rim.position.copy(shield.position);
      g.add(rim);
      var boss = new THREE.Mesh(new THREE.SphereGeometry(0.021, 10, 8), goldMat);
      boss.position.set(-0.006, 0.325, -0.140);
      g.add(boss);

      // Contact shadow. The shadow map does the cast shadow; this is the dark
      // patch under the boots that keeps the soldier from looking pasted on.
      var blob = new THREE.Mesh(new THREE.PlaneGeometry(0.60, 0.60),
        new THREE.MeshBasicMaterial({
          map: poolTex, color: 0x000000, transparent: true, opacity: 0.38, depthWrite: false
        }));
      blob.rotation.x = -Math.PI / 2;
      blob.position.y = 0.012;
      g.add(blob);
      return g;
    }

    var blueUnits = [], redUnits = [];
    for (var bi = 0; bi < N_BLUE; bi++) {
      blueUnits.push(buildSoldier(true));
      scene.add(blueUnits[bi]);
    }
    for (var ri2 = 0; ri2 < N_RED; ri2++) {
      redUnits.push(buildSoldier(false));
      scene.add(redUnits[ri2]);
    }

    // The banner a carrier wears on its back, one per blue player, because any
    // of them may be the carrier and the trace says which.
    var carriedBanners = blueUnits.map(function (unit) {
      var banner = buildBanner(false, 0.62);
      banner.visible = false;
      unit.add(banner);
      banner.position.set(-0.12, 0.30, 0);
      banner.rotation.z = -0.35;
      return banner;
    });

    // The selection ring the GIF draws under every blue player.
    var selRings = blueUnits.map(function (u) {
      var ring = new THREE.Mesh(new THREE.RingGeometry(0.20, 0.25, 30),
        new THREE.MeshBasicMaterial({
          color: 0x7EF688, transparent: true, opacity: 0.75,
          side: THREE.DoubleSide, depthWrite: false
        }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.018;
      u.add(ring);
      return ring;
    });

    // The scan ping: a ring that expands on the turn a player scans. It is the
    // only thing on screen that shows that player's detector widening from the
    // move half-distance to the scan one.
    var scanRings = blueUnits.map(function (u) {
      var ring = new THREE.Mesh(new THREE.RingGeometry(0.965, 1.00, 72),
        new THREE.MeshBasicMaterial({
          color: 0x9FE0F6, transparent: true, opacity: 0.0,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.04;
      u.add(ring);
      return ring;
    });
    var SCAN_REACH = Math.max(1, world.detector_half_distance_scan);

    /* ------------------------------------------------------------- belief
       Two layers, because the environment hides two different kinds of thing,
       and they are deliberately drawn in two different ways.

       A red player's position is spatial, so its marginal is drawn on the
       cells it puts mass on: a disc per cell, sized and faded by that cell's
       share, one hue per player. Tiny additive sprites vanish against lit
       grass; a disc on the cell does not, and it also says plainly that the
       belief is over cells rather than over a continuous position.

       The banner's cell is a choice among named candidates, so it gets a bar
       and a number on each one. A bar's height is a number you can compare
       across candidates at a glance, which a blob is not, and it does not
       imply the banner might be anywhere in between. */
    var beliefMarkers = [];
    for (var mj = 0; mj < N_RED; mj++) {
      var arr = [];
      for (var mk = 0; mk < MAX_BELIEF_CELLS; mk++) {
        var mg = new THREE.Group();
        var disc = new THREE.Mesh(new THREE.PlaneGeometry(1.18, 1.18),
          new THREE.MeshBasicMaterial({
            map: poolTex, color: HUES[mj % HUES.length], transparent: true,
            opacity: 0, depthWrite: false,
            polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3
          }));
        disc.rotation.x = -Math.PI / 2;
        mg.add(disc);
        var mring = new THREE.Mesh(new THREE.RingGeometry(0.31, 0.345, 30),
          new THREE.MeshBasicMaterial({
            color: 0xEADCE4, transparent: true, opacity: 0, depthWrite: false,
            polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3
          }));
        mring.rotation.x = -Math.PI / 2;
        mring.position.y = 0.004;
        mg.add(mring);
        mg.visible = false;
        scene.add(mg);
        arr.push({ group: mg, disc: disc, ring: mring });
      }
      beliefMarkers.push(arr);
    }

    // A faint ghost of each red player at its belief's mode, so the reader can
    // see what blue would act on without confusing it for the truth.
    var ghosts = [];
    for (var gj = 0; gj < N_RED; gj++) {
      var ghost = buildSoldier(false);
      (function (hue) {
        ghost.traverse(function (o) {
          if (!o.material) return;
          if (o.material.map === poolTex) { o.visible = false; return; }
          var m = o.material.clone();
          m.transparent = true;
          m.opacity = 0.30;
          m.depthWrite = false;
          if (m.color) m.color.setHex(hue);
          m.roughness = 1.0;
          m.metalness = 0.0;
          o.material = m;
          o.castShadow = false;
          o.receiveShadow = false;
        });
      })(HUES[gj % HUES.length]);
      ghost.visible = false;
      scene.add(ghost);
      ghosts.push(ghost);
    }

    function makeLabel() {
      var cv = document.createElement("canvas");
      cv.width = 192; cv.height = 96;
      var tex = new THREE.CanvasTexture(cv);
      tex.encoding = THREE.sRGBEncoding;
      var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, transparent: true, depthWrite: false, depthTest: false
      }));
      sprite.scale.set(0.46, 0.23, 1);
      sprite.userData = { cv: cv, tex: tex, last: "" };
      return sprite;
    }
    function setLabel(sprite, text, colour) {
      if (sprite.userData.last === text) return;
      sprite.userData.last = text;
      var g = sprite.userData.cv.getContext("2d");
      g.clearRect(0, 0, 192, 96);
      g.font = "600 56px 'IBM Plex Mono', monospace";
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.lineWidth = 9;
      g.strokeStyle = "rgba(10,12,10,0.85)";
      g.strokeText(text, 96, 50);
      g.fillStyle = colour;
      g.fillText(text, 96, 50);
      sprite.userData.tex.needsUpdate = true;
    }

    var candidateNodes = CANDIDATES.map(function (cell) {
      var g = new THREE.Group();
      place(g, cell[0], cell[1], 0);
      scene.add(g);

      var tile = new THREE.Mesh(new THREE.PlaneGeometry(0.90, 0.90),
        new THREE.MeshBasicMaterial({
          color: BELIEF_GOLD, transparent: true, opacity: 0.2,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        }));
      tile.rotation.x = -Math.PI / 2;
      tile.position.y = 0.03;
      g.add(tile);

      var edge = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.46, 4),
        new THREE.MeshBasicMaterial({
          color: 0xF4D478, transparent: true, opacity: 0.7,
          blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
        }));
      edge.rotation.x = -Math.PI / 2;
      edge.rotation.z = Math.PI / 4;
      edge.position.y = 0.05;
      g.add(edge);

      var bar = new THREE.Mesh(new THREE.BoxGeometry(0.068, 1, 0.068),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color(1.9, 1.30, 0.48), transparent: true, opacity: 0.72
        }));
      bar.position.set(-0.36, 0.5, -0.36);   // back corner, clear of the unit
      g.add(bar);

      var label = makeLabel();
      label.position.set(-0.36, 1.35, -0.36);
      g.add(label);
      return { group: g, tile: tile, edge: edge, bar: bar, label: label };
    });

    /* -------------------------------------------------------------- trails */
    function makeTrail(colour) {
      // Dots on the cells a player has stood on. A polyline drew a hard
      // hairline straight across the grass; footprints sit on the field
      // instead of over it.
      var pos = new Float32Array(Math.max(2, N_STEPS) * 3);
      var geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setDrawRange(0, 0);
      scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
        color: colour, map: partTex, size: 0.19, transparent: true, opacity: 0.55,
        depthWrite: false, sizeAttenuation: true
      })));
      return { geo: geo, pos: pos, max: Math.max(2, N_STEPS) };
    }
    var trails = blueUnits.map(function (unit, i) {
      return makeTrail(i === 0 ? 0x548AE8 : 0x7FA9EE);
    });

    core.linearize();

    /* ------------------------------------------------------------ episode */
    function laneOf(rows, index) {
      return rows.map(function (row) { return row[index]; });
    }
    var bluePaths = [], redPaths = [];
    for (var bp = 0; bp < N_BLUE; bp++) bluePaths.push(laneOf(payload.blue_cells, bp));
    for (var rp = 0; rp < N_RED; rp++) redPaths.push(laneOf(payload.red_cells, rp));

    var running = [];
    var total = 0;
    for (var si = 0; si < trace.steps.length; si++) {
      var rw = trace.steps[si].reward;
      if (rw !== null && rw !== undefined) total += rw * Math.pow(trace.discount_factor, si);
      running.push(total);
    }

    /* Cells are interpolated linearly, so a unit runs at constant speed
       between them instead of decelerating to a stop at each grid boundary
       (a smoothstep) or teleporting. A respawn IS a teleport, though: a
       tagged player is put back on its own base, so a slide across the map
       would draw a walk nobody took. */
    function cellAt(path, t) {
      var i0 = Math.floor(clamp(t, 0, N_STEPS - 1));
      var i1 = Math.min(i0 + 1, N_STEPS - 1);
      var f = clamp(t - i0, 0, 1);
      var a = path[i0], b = path[i1];
      if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) > 2.5) {
        return { x: f < 0.5 ? a[0] : b[0], y: f < 0.5 ? a[1] : b[1], jump: true };
      }
      return { x: lerp(a[0], b[0], f), y: lerp(a[1], b[1], f), jump: false };
    }
    /* Clamped at both ends, not just the top. The player's first frame can
       arrive with a negative t: its clock is seeded from performance.now()
       when the loop starts, and the first requestAnimationFrame timestamp is
       the time that frame began, which on a page that spent a second building
       this scene is earlier than that. One negative dt is enough to index a
       step of -1, and every field read from it is then undefined. */
    function turnOf(t) { return clamp(Math.round(t), 0, N_STEPS - 1); }

    /* `headings[i]` is a compass bearing: the world direction the unit
       travels, as atan2(dx, dz). The figures, though, are modelled facing
       their own local +x -- cape at the back, boots toed that way, shield and
       spear to either side -- and a group rotated by `rotation.y = bearing`
       aims its local +z, not its +x. Setting the two equal walked every
       soldier sideways. The quarter turn is the correction, applied in exactly
       one place so nothing can drift from it. */
    var MODEL_FRONT_OFFSET = -Math.PI / 2;
    var headings = [];
    var lastPos = [];
    var allUnits = blueUnits.concat(redUnits);
    var allPaths = bluePaths.concat(redPaths);
    for (var hi = 0; hi < allUnits.length; hi++) {
      headings.push(0);
      lastPos.push([allPaths[hi][0][0], allPaths[hi][0][1]]);
    }

    function faceUnit(unit, idx, gx, gy, dt) {
      var dx = gx - lastPos[idx][0], dz = gy - lastPos[idx][1];
      if (Math.abs(dx) + Math.abs(dz) > 1e-4) {
        var target = Math.atan2(dx, dz);
        var diff = ((target - headings[idx] + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
        headings[idx] += diff * clamp(dt * 9, 0, 1);
      }
      unit.rotation.y = headings[idx] + MODEL_FRONT_OFFSET;
    }

    /* --------------------------------------------------------- the belief
       A CaptureTheFlag particle is a whole state vector, so the marginals are
       taken the way the GIF's overlay takes them: sum each particle's weight
       onto the cell its red players stand on, and onto the candidate index its
       flag field names. Nothing is smoothed and nothing is invented -- a cell
       with no mass is a cell the run's own filter put no particles on. */
    function emptyBelief(note) {
      return { red: [], flag: [], note: note, drawn: false };
    }

    function marginalOf(belief) {
      if (!belief) return emptyBelief("—");
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner;
        // merging its members would show a cloud that was never anyone's.
        return emptyBelief("batch of " + belief.batch_size + " beliefs, not drawn");
      }
      if (belief.kind !== "particles") {
        // A Gaussian over a grid state, or a class core has no payload for
        // yet. Named, so the gap is diagnosable, and drawn as nothing, so it
        // is not invented.
        return emptyBelief("not drawn (" + (belief.belief_class || belief.kind) + ")");
      }
      var size = layout.size || 0;
      var particles = belief.particles || [];
      if (!particles.length || !Array.isArray(particles[0]) || particles[0].length !== size) {
        return emptyBelief(belief.num_particles + " particles, not state vectors");
      }
      var red = [], flag = [];
      var j;
      for (j = 0; j < N_RED; j++) red.push({});
      for (j = 0; j < CANDIDATES.length; j++) flag.push(0);
      for (var p = 0; p < particles.length; p++) {
        var w = belief.weights[p];
        var particle = particles[p];
        for (j = 0; j < N_RED; j++) {
          var key = particle[layout.red_pos + 2 * j] + ":" + particle[layout.red_pos + 2 * j + 1];
          red[j][key] = (red[j][key] || 0) + w;
        }
        var index = Math.round(particle[layout.flag_cell]);
        if (index >= 0 && index < flag.length) flag[index] += w;
      }
      var cells = red.map(function (table) {
        var out = [];
        for (var k in table) {
          if (!Object.prototype.hasOwnProperty.call(table, k)) continue;
          var parts = k.split(":");
          out.push({ x: +parts[0], y: +parts[1], w: table[k] });
        }
        out.sort(function (a, b) { return b.w - a.w; });
        return out;
      });
      var note = belief.num_particles + " particles";
      if (belief.num_written < belief.num_particles) {
        note += " (heaviest " + belief.num_written + " drawn)";
      }
      return { red: cells, flag: flag, note: note, drawn: true };
    }

    // One marginal per step, computed once: a filter's cloud does not change
    // between frames, and recomputing it every frame is the difference between
    // a smooth scene and a slideshow.
    var marginals = payload.beliefs.map(marginalOf);

    /* --------------------------------------------------------- the camera
       The core rig's chase mode follows ONE agent with one heading, and this
       world has four. So the scene picks the target itself -- whichever blue
       player carries the red banner, else blue player zero -- and hands the
       rig a virtual follow point placed behind that player, which pulls the
       rig's fixed 2.9-unit chase back to the 4.6 this field needs. Board and
       overhead stay the rig's own, framed from the field's size below. */
    var followTmp = { x: 0, z: 0, heading: 0 };

    function followPoint(index, px, pz) {
      var bearing = headings[index];
      // The rig works in (cos, sin); a bearing is (sin, cos).
      var dirX = Math.sin(bearing), dirZ = Math.cos(bearing);
      followTmp.x = px - dirX * 1.7;
      followTmp.z = pz - dirZ * 1.7;
      followTmp.heading = Math.atan2(dirZ, dirX);
      return followTmp;
    }

    var extent = Math.max(GW - 1, GH - 1);

    function updateUnits(t, dt, turn, nextTurn, playing) {
      for (var a = 0; a < allUnits.length; a++) {
        var unit = allUnits[a];
        var path = allPaths[a];
        var c = cellAt(path, t);
        var y = groundY(c.x, c.y);
        unit.position.set(wx(c.x), y, wz(c.y));
        faceUnit(unit, a, c.x, c.y, dt);
        lastPos[a] = [c.x, c.y];

        // Run cycle: the legs swing only while the unit is actually moving.
        var moving = !c.jump && (Math.abs(path[turn][0] - path[nextTurn][0])
          + Math.abs(path[turn][1] - path[nextTurn][1])) > 0;
        var swing = moving && playing ? Math.sin(t * Math.PI * 4) * 0.55 : 0;
        unit.userData.legs[0].rotation.x = swing;
        unit.userData.legs[1].rotation.x = -swing;
        unit.position.y = y + (moving && playing
          ? Math.abs(Math.sin(t * Math.PI * 4)) * 0.018 : 0);

        // Frozen after a tag: translucent, exactly as soldier_sprite fades it.
        var isBlue = a < N_BLUE;
        var counter = isBlue
          ? payload.freeze_blue[turn][a]
          : payload.freeze_red[turn][a - N_BLUE];
        var frozen = counter > 0;
        unit.traverse(function (o) {
          if (!o.material || !o.material.isMeshStandardMaterial) return;
          o.material.transparent = frozen;
          o.material.opacity = frozen ? 0.42 : 1.0;
        });
      }
    }

    var labelTmp = new THREE.Vector3();

    function updateBelief(turn, elapsed) {
      var m = marginals[turn] || emptyBelief("—");
      var j, k;
      for (j = 0; j < N_RED; j++) {
        var cells = m.red[j] || [];
        var peak = cells.length ? cells[0].w : 1;
        for (k = 0; k < MAX_BELIEF_CELLS; k++) {
          var marker = beliefMarkers[j][k];
          var cell = cells[k];
          var share = cell ? cell.w / peak : 0;
          if (!cell || share < 0.08) { marker.group.visible = false; continue; }
          marker.group.visible = true;
          // The players' clouds overlap, so each is nudged off the cell centre
          // in its own direction rather than stacked and hidden.
          var nudge = -0.13 + 0.26 * (N_RED > 1 ? j / (N_RED - 1) : 0.5);
          marker.group.position.set(
            wx(cell.x) + nudge, groundY(cell.x, cell.y) + 0.035, wz(cell.y) + nudge
          );
          var sc = 0.42 + 0.66 * share;
          marker.group.scale.set(sc, 1, sc);
          marker.disc.material.opacity = 0.44 + 0.80 * share;
          marker.ring.material.opacity = 0.05 + 0.16 * share;
        }
        var mode = (m.red[j] || [])[0];
        ghosts[j].visible = !!mode;
        if (mode) {
          ghosts[j].position.set(wx(mode.x), groundY(mode.x, mode.y) + 0.01, wz(mode.y));
          ghosts[j].rotation.y = redUnits[j].rotation.y;
        }
      }

      var best = -1, bestMass = 0;
      for (k = 0; k < candidateNodes.length; k++) {
        var mass = m.flag.length ? m.flag[k] : 0;
        var node = candidateNodes[k];
        node.bar.scale.y = Math.max(0.001, mass * 0.72);
        node.bar.position.y = mass * 0.36;
        node.bar.material.opacity = mass < 0.01 ? 0 : 0.88;
        node.tile.material.opacity = 0.09 + 0.36 * mass;
        node.edge.material.opacity = 0.35 + 0.80 * mass;
        node.label.position.y = mass * 0.72 + 0.26;
        setLabel(node.label, mass.toFixed(2), mass > 0.5 ? "#FFE9A8" : "#EBD6A0");
        node.label.material.opacity = m.drawn ? 1 : 0;
        // Sprites are world-sized, so a near camera turns a probability into a
        // billboard. Scale by distance to hold them steady on screen.
        node.label.getWorldPosition(labelTmp);
        var lk = clamp(core.camera.position.distanceTo(labelTmp) * 0.036, 0.30, 0.56);
        node.label.scale.set(lk * 2, lk, 1);
        if (mass > bestMass) { bestMass = mass; best = k; }
      }
      void elapsed;

      if (!m.drawn) return m.note;
      if (best < 0) return m.note;
      return "banner " + bestMass.toFixed(2) + " at (" + CANDIDATES[best][0] + ", " +
        CANDIDATES[best][1] + ") · " + m.note;
    }

    function updateTrails(t) {
      for (var tr = 0; tr < trails.length; tr++) {
        var trail = trails[tr];
        var path = bluePaths[tr];
        var upto = clamp(t, 0, N_STEPS - 1);
        var n = Math.max(2, Math.min(trail.max, Math.floor(upto) + 1));
        for (var i = 0; i < n; i++) {
          var c = path[Math.min(i, path.length - 1)];
          trail.pos[i * 3] = wx(c[0]);
          trail.pos[i * 3 + 1] = groundY(c[0], c[1]) + 0.07;
          trail.pos[i * 3 + 2] = wz(c[1]);
        }
        // The head follows the interpolated position, so the line ends at the
        // feet rather than at the last cell left behind.
        var f = clamp(upto - Math.floor(upto), 0, 1);
        var a = path[Math.min(Math.floor(upto), path.length - 1)];
        var b = path[Math.min(Math.floor(upto) + 1, path.length - 1)];
        var hx = lerp(a[0], b[0], f), hy = lerp(a[1], b[1], f);
        trail.pos[(n - 1) * 3] = wx(hx);
        trail.pos[(n - 1) * 3 + 1] = groundY(hx, hy) + 0.07;
        trail.pos[(n - 1) * 3 + 2] = wz(hy);
        trail.geo.attributes.position.needsUpdate = true;
        trail.geo.setDrawRange(0, n);
      }
    }

    function actionLabel(turn) {
      var actions = payload.player_actions[turn];
      if (!actions) return "—";
      return actions.map(function (a, i) {
        return "P" + (i + 1) + " " + (ACTION_NAMES[a] || a);
      }).join(" · ");
    }

    return {
      steps: N_STEPS,

      /* Framing scales with the world, because a trace decides how big the
         field is: the distance that frames a 9x7 field leaves a 5x3 one as a
         smudge in the middle of the canvas. The constant term is the margin
         for the props, which do not scale with the field -- a keep is the same
         height whatever the grid is, so a small field needs proportionally
         more headroom, not less. */
      camera: {
        board: [0, extent * 0.95 + 1.3, extent * 1.05 + 1.5],
        top: [0.01, extent * 1.15 + 1.6, 0.02]
      },

      /**
       * Advance the world to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var turn = turnOf(t);
        var nextTurn = Math.min(N_STEPS - 1, turn + 1);

        updateUnits(t, dt, turn, nextTurn, playing);

        // Banners. The red one stands on its cell until a blue player lifts
        // it, and then rides on that player's back -- the carrier's id comes
        // from the recorded state, so the banner is never on the wrong back.
        var carrier = payload.carrier_red_flag[turn];
        var flagCell = payload.red_flag_cell[turn];
        place(redBanner, flagCell[0], flagCell[1], 0);
        redBanner.visible = carrier === 0;
        for (var cb = 0; cb < carriedBanners.length; cb++) {
          carriedBanners[cb].visible = carrier === cb + 1;
        }
        if (playing) {
          var flap = Math.sin(elapsed * 3.1) * 0.10;
          blueBanner.userData.cloth.rotation.y = flap;
          redBanner.userData.cloth.rotation.y = flap * 0.8;
          carriedBanners.forEach(function (banner) {
            banner.userData.cloth.rotation.y = flap * 1.4;
          });
        }

        // Scan ping. It expands over the turn a player scans and fades out, so
        // the widened detector is visible rather than implied.
        var actions = payload.player_actions[turn];
        var phase = clamp(t - Math.floor(t), 0, 1);
        for (var s = 0; s < blueUnits.length; s++) {
          var scanning = !!actions && actions[s] === ACTION_SCAN;
          scanRings[s].material.opacity = scanning ? 0.30 * (1 - phase) * (1 - phase) : 0;
          var r = 0.7 + phase * (SCAN_REACH - 0.7);
          scanRings[s].scale.set(r, r, 1);
          selRings[s].material.opacity = 0.35 + Math.sin(elapsed * 2.4 + s) * 0.12;
        }

        var beliefLabel = updateBelief(turn, elapsed);
        updateTrails(t);

        // Braziers flicker, water drifts, canopies lean into the wind. The
        // wind is the cheapest thing on this page that makes the scene read as
        // alive rather than as a set.
        if (playing) {
          var flick = 0.82 + Math.sin(elapsed * 9.1) * 0.10 + Math.sin(elapsed * 3.7) * 0.08;
          blueKeep.fire.power = 44 * flick;
          redKeep.fire.power = 44 * (1.8 - flick);
          // Two ripple scales drifting at different speeds. One scrolling
          // layer reads as a conveyor belt; two beating against each other
          // read as flow.
          rippleTex.offset.set(Math.sin(elapsed * 0.07) * 0.05, -elapsed * 0.035);
          waterMat.normalScale.set(0.38 + Math.sin(elapsed * 0.9) * 0.05, 0.42);

          var gust = Math.sin(elapsed * 0.44) * 0.5 + 0.5;
          for (var sw = 0; sw < swayers.length; sw++) {
            var u = swayers[sw];
            var amp = 0.020 + 0.030 * gust;
            u.canopy.rotation.z = Math.sin(elapsed * 1.25 + u.phase) * amp;
            u.canopy.rotation.x = Math.cos(elapsed * 0.95 + u.phase * 1.7) * amp * 0.7;
          }
        }

        // The dome rides with the camera, so its far side never comes into
        // frame however far the orbit pulls back.
        dome.position.copy(core.camera.position);

        // Whoever carries the banner is what a viewer wants to watch; nobody
        // carrying it means the first player, who is the one with a trail.
        var target = carrier > 0 ? carrier - 1 : 0;
        var lead = blueUnits[target].position;

        var step = trace.steps[turn] || {};
        var cell = bluePaths[target][turn];
        return {
          follow: followPoint(target, lead.x, lead.z),
          step: turn,
          action: actionLabel(turn),
          x: cell[0],
          y: cell[1],
          reward: step.reward,
          ret: running[turn],
          belief: beliefLabel
        };
      }
    };
  }

  V.scenes["capture_the_flag.v1"] = {
    build: build,
    /* Daylight, not lamplight: the sun carries real intensity and the braziers
       real lumens, so the camera stops down the way a real one would. This one
       number sets the whole mood, and it is about a stop and a half above the
       night scenes' because a ground plane under a sun twenty-one degrees up
       receives only sin(21 deg) of its irradiance. */
    exposure: 0.445
  };
})(window);
