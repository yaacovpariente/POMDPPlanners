/* SPDX-License-Identifier: MIT
 *
 * Tiger scene module.
 *
 * A torch-lit stone chamber with two doors. The room is this viewer's
 * invention; everything that carries information comes from the trace — which
 * side the tiger was on, what the agent heard, which door it opened, and the
 * belief the run actually held.
 *
 * Two things about this environment drive the whole design:
 *
 *  1. The belief is ONE number over TWO hypotheses. It is drawn as two named
 *     bars that always sum to one, one over each door, never as a cloud in the
 *     room: there is nothing spatial to put a cloud on, and a cloud would imply
 *     a position the state space does not have.
 *  2. Opening a door does NOT end the episode. `is_terminal` is always false
 *     and the transition re-draws the tiger's side uniformly, so the belief the
 *     agent spent listens to earn is thrown away the moment it commits. That
 *     reset is the point of the environment, so the bars are held still during
 *     the swing and snap back to the recorded posterior at the very end of the
 *     step, where the reader can see it happen.
 *
 * There is no authored episode here. If the trace carries no belief this
 * module can read, it says so rather than drawing one.
 */
(function (global) {
  "use strict";

  var V = global.POMDPViewer;
  var THREE = global.THREE;
  var clamp = V.clamp, lerp = V.lerp, mulberry = V.mulberry;

  /* Palette: the constants tiger_visualizer.py draws its GIF with, so the page
     and the GIF are recognisably the same world. */
  var C = {
    text: 0xF6DDAB,
    gold: 0xE0B04C,
    red: 0xF17B6B,
    green: 0x99D695,
    track: 0x392F20
  };

  var DOOR_X = 2.05;            // door centres, +-
  var DOOR_W = 1.70;            // opening width
  var DOOR_H = 1.77;            // height of the straight jamb; arch sits on top
  var WALL_Z = -3.5;            // back wall centre
  var ROOM_W = 9.4, ROOM_D = 9.0, ROOM_H = 4.3;
  var HOLE_TOP = 2.92;
  // The cells behind the doors are boxes 3.2 m tall centred at y = 1.6, so
  // their floor is here. Everything that stands on it is driven off this
  // number rather than off a value tuned by eye.
  var CELL_FLOOR_Y = 1.6 - 3.2 / 2;

  var TORCH_LUMENS = 1150;

  /* How far in from its door each belief sign sits, as a fraction of the door
     offset. The page's HUD panel occupies the top-left of the stage, and at
     the prototype's 0.72 the left sign's number was behind it. */
  var SIGN_INSET = 0.52;

  /* The core rig looks at the origin in board, top and orbit modes, so the
     chamber is built in its own coordinates and then shifted, once, to put the
     point worth looking at — the middle of the door wall, at door height —
     where the rig expects it. Rebuilding the room around the origin instead
     would mean re-deriving every number the prototype settled on. */
  var LOOK_LOCAL = new THREE.Vector3(0, 1.30, WALL_Z + 0.4);
  var RIG_LOOK = new THREE.Vector3(0, 0.3, 0);
  var SHIFT = new THREE.Vector3().subVectors(RIG_LOOK, LOOK_LOCAL);

  function smooth(e0, e1, x) {
    var t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  }

  /* Sobel over a blurred copy of a canvas. The core has one of these, but with
     the blur fixed at 2 px; the fur relief needs almost none, or the hairs are
     averaged into a smooth sheen and the coat is the only polished surface in
     a room made of rough stone. So this one takes the blur radius. */
  function normalMapFrom(renderer, cv, strength, blurPx) {
    var s = cv.width;
    var soft = document.createElement("canvas");
    soft.width = soft.height = s;
    var sctx = soft.getContext("2d");
    sctx.filter = "blur(" + (blurPx === undefined ? 2 : blurPx) + "px)";
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
      var i = (y * s + x) * 4;
      return (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114) / 255;
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

  /* Flagstones. Irregular slabs with a dark mortar line and grit on top; the
     normal map is derived from this same canvas, so every slab edge catches the
     torches from the side the torch is actually on. */
  function floorCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(20260919);

    g.fillStyle = "#14100C";
    g.fillRect(0, 0, s, s);

    var rows = 9;
    var h = s / rows;
    for (var r = 0; r < rows; r++) {
      var x = -h * rnd();
      while (x < s) {
        var w = h * (0.8 + rnd() * 1.5);
        var tone = 46 + rnd() * 34;
        g.fillStyle = "rgb(" + (tone | 0) + "," + (tone * 0.85 | 0) + "," + (tone * 0.66 | 0) + ")";
        g.fillRect(x + 2.5, r * h + 2.5, w - 5, h - 5);
        // A lit top lip and a shaded bottom one: this is what the Sobel reads.
        g.fillStyle = "rgba(196,168,120,0.16)";
        g.fillRect(x + 2.5, r * h + 2.5, w - 5, 3);
        g.fillStyle = "rgba(0,0,0,0.42)";
        g.fillRect(x + 2.5, r * h + h - 6, w - 5, 3);
        x += w;
      }
    }
    for (var i = 0; i < 2600; i++) {
      g.fillStyle = rnd() > 0.5 ? "rgba(214,188,140,0.05)" : "rgba(0,0,0,0.20)";
      var pr = 1 + rnd() * 3;
      g.beginPath(); g.arc(rnd() * s, rnd() * s, pr, 0, Math.PI * 2); g.fill();
    }
    return cv;
  }

  /* Rough coursed stone. Same trick: bright top edge, dark bottom edge, so the
     relief comes out of the normal map rather than being painted into colour. */
  function wallCanvas() {
    var s = 1024;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(77123);

    g.fillStyle = "#0E0B08";
    g.fillRect(0, 0, s, s);

    var rows = 14;
    var h = s / rows;
    for (var r = 0; r < rows; r++) {
      var x = -h * 2 * rnd();
      while (x < s) {
        var w = h * (1.1 + rnd() * 2.2);
        var tone = 40 + rnd() * 36;
        g.fillStyle = "rgb(" + (tone | 0) + "," + (tone * 0.86 | 0) + "," + (tone * 0.68 | 0) + ")";
        g.fillRect(x + 3, r * h + 3, w - 6, h - 6);
        g.fillStyle = "rgba(190,164,118,0.13)";
        g.fillRect(x + 3, r * h + 3, w - 6, 3);
        g.fillStyle = "rgba(0,0,0,0.45)";
        g.fillRect(x + 3, r * h + h - 7, w - 6, 4);
        for (var p = 0; p < 22; p++) {
          g.fillStyle = rnd() > 0.5 ? "rgba(0,0,0,0.20)" : "rgba(200,176,132,0.06)";
          g.beginPath();
          g.arc(x + 6 + rnd() * (w - 12), r * h + 6 + rnd() * (h - 12), 1 + rnd() * 4, 0, Math.PI * 2);
          g.fill();
        }
        x += w;
      }
    }
    return cv;
  }

  /* Vertical oak planks with iron nail heads. */
  function doorCanvas() {
    var s = 512;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var rnd = mulberry(4242);
    g.fillStyle = "#6B4526";
    g.fillRect(0, 0, s, s);
    var planks = 7, pw = s / planks;
    for (var i = 0; i < planks; i++) {
      // Oak reads dark to the eye, but albedo is not brightness: painted this
      // dark the planks fall below the torchlight and the door renders black.
      var tone = 116 + rnd() * 42;
      g.fillStyle = "rgb(" + (tone | 0) + "," + (tone * 0.72 | 0) + "," + (tone * 0.54 | 0) + ")";
      g.fillRect(i * pw + 2, 0, pw - 4, s);
      for (var k = 0; k < 90; k++) {
        g.strokeStyle = rnd() > 0.5 ? "rgba(28,16,8,0.24)" : "rgba(226,186,132,0.12)";
        g.lineWidth = 1 + rnd() * 2;
        var gx = i * pw + 4 + rnd() * (pw - 8);
        g.beginPath();
        g.moveTo(gx, rnd() * s);
        g.lineTo(gx + (rnd() - 0.5) * 6, rnd() * s);
        g.stroke();
      }
      g.fillStyle = "rgba(0,0,0,0.62)";
      g.fillRect(i * pw, 0, 3, s);
    }
    return cv;
  }

  /* A flame is a soft, tapering column of light, not a solid cone: rendered as
     geometry it reads as a paper triangle. A teardrop painted once, used as an
     additive sprite, which also blooms correctly. */
  function flameTexture() {
    var w = 96, h = 192;
    var cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    var g = cv.getContext("2d");
    var img = g.createImageData(w, h);
    var d = img.data;
    for (var y = 0; y < h; y++) {
      var v = y / h;                       // 0 at the tip, 1 at the base
      var half = (0.07 + 0.30 * Math.pow(v, 0.65)) * w;
      for (var x = 0; x < w; x++) {
        var dx = Math.abs(x - w / 2) / Math.max(half, 0.001);
        // Gaussian across, not a hard cut: a flame has no silhouette edge, and
        // any boundary in the alpha reads instantly as a paper cone.
        var a = Math.exp(-dx * dx * 2.6);
        a *= Math.pow(v, 0.85) * (1 - Math.pow(Math.max(0, v - 0.80) / 0.20, 2));
        var i = (y * w + x) * 4;
        d[i] = 255; d[i + 1] = 255; d[i + 2] = 255;
        d[i + 3] = Math.min(255, a * 255) | 0;
      }
    }
    g.putImageData(img, 0, 0);
    return new THREE.CanvasTexture(cv);
  }

  /* ---------------------------------------------------------------- the cat
     The coat.

     The trunk is swept from cylinders' sections, not stretched spheres, and
     that is a coat decision rather than a modelling one: a sphere scaled long
     in z maps its meridians ALONG the body, so every stripe painted on it runs
     lengthwise. A swept section's u runs around the circumference, so a stripe
     is a stroke across the canvas and wraps the way a real one does.

     Canvas x is around the body, canvas y runs nose to tail. */
  function tigerCoat(renderer) {
    var W = 512, H = 512;
    var cv = document.createElement("canvas");
    cv.width = W; cv.height = H;
    var g = cv.getContext("2d");
    var rnd = mulberry(31337);

    g.fillStyle = "#5C2D0E";
    g.fillRect(0, 0, W, H);
    var spine = g.createLinearGradient(W * 0.36, 0, W * 0.5, 0);
    spine.addColorStop(0, "rgba(54,26,10,0)");
    spine.addColorStop(1, "rgba(54,26,10,0.5)");
    g.fillStyle = spine; g.fillRect(Math.floor(W * 0.36), 0, Math.ceil(W * 0.14), H);
    var spine2 = g.createLinearGradient(W * 0.64, 0, W * 0.5, 0);
    spine2.addColorStop(0, "rgba(54,26,10,0)");
    spine2.addColorStop(1, "rgba(54,26,10,0.5)");
    g.fillStyle = spine2; g.fillRect(Math.floor(W * 0.5), 0, Math.ceil(W * 0.14), H);

    /* One stripe: a stroke running around the barrel, wandering a little along
       the body, tapering to nothing at both ends so it dies out on the flank
       rather than stopping in a line. */
    function stripe(y0, w0, from, to, drift) {
      var segs = 22;
      g.strokeStyle = "#140C07";
      g.lineCap = "round";
      var px = from, py = y0;
      for (var k = 1; k <= segs; k++) {
        var t = k / segs;
        var nx = from + (to - from) * t;
        var ny = y0 + Math.sin(t * Math.PI) * drift + (rnd() - 0.5) * 3;
        g.lineWidth = Math.max(0.5, w0 * Math.sin(t * Math.PI) * (0.55 + rnd() * 0.5));
        g.beginPath(); g.moveTo(px, py); g.lineTo(nx, ny); g.stroke();
        px = nx; py = ny;
      }
    }

    for (var i = 0; i < 30; i++) {
      var y0 = -20 + (i / 30) * (H + 40) + (rnd() - 0.5) * 22;
      var w0 = 7 + rnd() * 15;
      var drift = (rnd() - 0.5) * 40;
      // Left flank and right flank are independent: stripes do not line up
      // across an animal's back.
      stripe(y0, w0, W * 0.10, W * 0.49, drift);
      stripe(y0 + (rnd() - 0.5) * 26, 7 + rnd() * 15, W * 0.90, W * 0.51, (rnd() - 0.5) * 40);
      if (rnd() < 0.34) stripe(y0 + 16 + rnd() * 14, w0 * 0.5, W * 0.16, W * 0.42, drift * 0.5);
    }

    // Pale belly and inner legs, painted last so it takes the stripe ends with
    // it: no stripe should survive onto the underside.
    [[0, 1], [W, -1]].forEach(function (e) {
      var bg = g.createLinearGradient(e[0], 0, e[0] + e[1] * W * 0.115, 0);
      bg.addColorStop(0, "rgba(174,156,126,0.92)");
      bg.addColorStop(0.5, "rgba(170,150,120,0.55)");
      bg.addColorStop(1, "rgba(168,148,118,0)");
      g.fillStyle = bg;
      g.fillRect(e[1] > 0 ? 0 : Math.floor(W * 0.885), 0, Math.ceil(W * 0.115), H);
    });

    // The hindquarters are darker in the coat as well as in the light: the eye
    // should find the head and lose the rest of the animal in the doorway.
    var aft = g.createLinearGradient(0, 0, 0, H * 0.64);
    aft.addColorStop(0, "rgba(8,5,3,0.80)");
    aft.addColorStop(0.45, "rgba(8,5,3,0.36)");
    aft.addColorStop(1, "rgba(8,5,3,0)");
    g.fillStyle = aft;
    g.fillRect(0, 0, W, Math.ceil(H * 0.64));

    for (var f = 0; f < 4200; f++) {
      g.strokeStyle = rnd() > 0.5 ? "rgba(0,0,0,0.12)" : "rgba(214,176,124,0.09)";
      g.lineWidth = 1;
      var fx = rnd() * W, fy = rnd() * H;
      g.beginPath(); g.moveTo(fx, fy);
      g.lineTo(fx + 3 + rnd() * 5, fy + (rnd() - 0.5) * 4);
      g.stroke();
    }

    var tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    /* A swept section starts at u = 0 pointing sideways and reaches the
       underside at u = 0.75, so the wrap is rolled a quarter turn to put the
       pale belly under the animal and the dark spine on top. */
    tex.offset.x = 0.25;
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  }

  /* Fur relief. The coat was the only smooth surface in a room made entirely of
     rough stone and rough wood, and that mismatch reads long before any
     proportion does. Fine directional streaks, turned into a normal map with
     almost no blur so the strokes survive as relief. */
  function furRelief(renderer, size, lengthScale) {
    var cv = document.createElement("canvas");
    cv.width = cv.height = size;
    var g = cv.getContext("2d");
    var rnd = mulberry(5309);
    g.fillStyle = "#808080";
    g.fillRect(0, 0, size, size);
    // Clumping first: broad soft ridges, so the fur is not uniform noise.
    for (var c = 0; c < 260; c++) {
      g.strokeStyle = rnd() > 0.5 ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.10)";
      g.lineWidth = 5 + rnd() * 9;
      var cx2 = rnd() * size, cy2 = rnd() * size;
      g.beginPath(); g.moveTo(cx2, cy2);
      g.lineTo(cx2 + (rnd() - 0.5) * 14, cy2 + (18 + rnd() * 40) * lengthScale);
      g.stroke();
    }
    for (var i = 0; i < 9000; i++) {
      var light = rnd() > 0.5;
      g.strokeStyle = light ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)";
      g.lineWidth = 1 + rnd() * 2.2;
      var x = rnd() * size, y = rnd() * size;
      var len = (10 + rnd() * 26) * lengthScale;
      g.beginPath(); g.moveTo(x, y);
      g.lineTo(x + (rnd() - 0.5) * 7, y + len);
      g.stroke();
    }
    var tex = normalMapFrom(renderer, cv, 1.5, 0.4);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }

  /* ------------------------------------------------------------- the mesh

     One swept surface per piece, not a bag of primitives. Given a chain of
     spine samples, each carrying its own half-width and half-height, this walks
     the chain, builds a frame at every sample, lays a section ring around it
     and stitches the rings into a tube. Normals come from the surface itself,
     so the whole piece shades as one object: no joint blobs, no butted caps.

     The section is a superellipse, so a piece can have square-ish shoulders (a
     skull) or be a plain ellipse (a limb) from the same code. Top and bottom
     half-heights are independent, which is what lets a chest be deep below the
     spine and a jaw hang below a muzzle.

     UVs fall out for free: u around the section, v along the chain by arc
     length, which is exactly what the painted stripes want. */
  function sweepGeometry(samples, radial, upRefIn, capStart, capEnd) {
    var M = samples.length, N = radial;
    var pos = [], uv = [], idx = [];
    var upRef = (upRefIn || new THREE.Vector3(0, 1, 0)).clone().normalize();
    var i;

    var tangents = [];
    for (i = 0; i < M; i++) {
      var a = samples[Math.max(0, i - 1)].p;
      var b = samples[Math.min(M - 1, i + 1)].p;
      var t = new THREE.Vector3().subVectors(b, a);
      if (t.lengthSq() < 1e-12) t.set(0, 0, 1);
      tangents.push(t.normalize());
    }
    var lens = [0];
    for (i = 1; i < M; i++) lens.push(lens[i - 1] + samples[i].p.distanceTo(samples[i - 1].p));
    var total = lens[M - 1] || 1;

    var right = new THREE.Vector3(), up = new THREE.Vector3(), v3 = new THREE.Vector3();
    for (i = 0; i < M; i++) {
      var sm = samples[i], T = tangents[i];
      right.crossVectors(upRef, T);
      if (right.lengthSq() < 1e-8) right.set(1, 0, 0);
      right.normalize();
      up.crossVectors(T, right).normalize();
      var e = sm.exp || 2.0, k = 2 / e;
      for (var j = 0; j <= N; j++) {
        var ang = (j / N) * Math.PI * 2;
        var c = Math.cos(ang), sn = Math.sin(ang);
        // Superellipse: e = 2 is an ellipse, higher is squarer-shouldered.
        var cc = (c < 0 ? -1 : 1) * Math.pow(Math.abs(c), k);
        var ss = (sn < 0 ? -1 : 1) * Math.pow(Math.abs(sn), k);
        var hh = sn >= 0 ? sm.hhTop : sm.hhBot;
        v3.copy(sm.p).addScaledVector(right, sm.hw * cc).addScaledVector(up, hh * ss);
        pos.push(v3.x, v3.y, v3.z);
        uv.push(j / N, lens[i] / total);
      }
    }
    for (i = 0; i < M - 1; i++) {
      for (var jj = 0; jj < N; jj++) {
        var a2 = i * (N + 1) + jj, b2 = a2 + 1, c2 = a2 + (N + 1), d2 = c2 + 1;
        idx.push(a2, b2, d2, a2, d2, c2);
      }
    }
    // Caps, so a nose or a paw closes instead of showing a hole.
    function cap(ringStart, T, sgn) {
      var sm = samples[sgn > 0 ? M - 1 : 0];
      var centre = sm.p.clone().addScaledVector(T, sgn * Math.max(sm.hw, sm.hhTop) * 0.55);
      var ci = pos.length / 3;
      pos.push(centre.x, centre.y, centre.z);
      uv.push(0.5, sgn > 0 ? 1 : 0);
      for (var j = 0; j < N; j++) {
        // Wound so the fan faces out. Reversed, the cap is a dark crater.
        if (sgn > 0) idx.push(ringStart + j, ringStart + j + 1, ci);
        else idx.push(ringStart + j, ci, ringStart + j + 1);
      }
    }
    if (capEnd) cap((M - 1) * (N + 1), tangents[M - 1], 1);
    if (capStart) cap(0, tangents[0], -1);

    var g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    /* The ring is duplicated at j = 0 and j = N so the UV can wrap, which
       leaves the two halves of the seam with different normals and a visible
       crease down the flank. Averaging them closes it. */
    var nrm = g.attributes.normal.array;
    for (i = 0; i < M; i++) {
      var a3 = (i * (N + 1)) * 3, b3 = (i * (N + 1) + N) * 3;
      var nx = nrm[a3] + nrm[b3], ny = nrm[a3 + 1] + nrm[b3 + 1], nz = nrm[a3 + 2] + nrm[b3 + 2];
      var L = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      nrm[a3] = nrm[b3] = nx / L;
      nrm[a3 + 1] = nrm[b3 + 1] = ny / L;
      nrm[a3 + 2] = nrm[b3 + 2] = nz / L;
    }
    g.attributes.normal.needsUpdate = true;
    return g;
  }

  // Chain rows are [z, centre y, half width, half height up, half height down].
  function chainZ(rows, x, exp) {
    return rows.map(function (r) {
      return {
        p: new THREE.Vector3(x || 0, r[1], r[0]),
        hw: r[2], hhTop: r[3], hhBot: r[4], exp: exp || 2.0
      };
    });
  }
  /* A limb's v runs along its length, so the coat's bands come out as rings
     round the leg, and at body density that is a zebra. Stretching v spaces
     them into the few oblique bars a tiger's leg actually carries. */
  function scaleV(geo, k) {
    var uv = geo.attributes.uv.array;
    for (var i = 1; i < uv.length; i += 2) uv[i] *= k;
    geo.attributes.uv.needsUpdate = true;
    return geo;
  }
  // Limb rows are [x, y, z, radius].
  function chainXYZ(rows, exp) {
    return rows.map(function (r) {
      return {
        p: new THREE.Vector3(r[0], r[1], r[2]),
        hw: r[3], hhTop: r[3], hhBot: r[3], exp: exp || 2.0
      };
    });
  }

  /* The animal, as a real surface.

     Proportions are a tiger's: about a metre at the shoulder and a little over
     two long. The spine carries the sag and rise that makes a big cat read as
     one — withers high, a dip at the loin, the haunch back up level with the
     withers — and the head hangs below the shoulder line, which is the line a
     stalking cat holds and a toy never does. */
  function buildTiger(renderer, nearSide) {
    var g3 = new THREE.Group();

    var fur = new THREE.MeshStandardMaterial({
      map: tigerCoat(renderer), normalMap: furRelief(renderer, 512, 1.0),
      normalScale: new THREE.Vector2(1.1, 1.1), color: 0xFFFFFF,
      // Fur is rough and dielectric and scatters almost nothing back. Any
      // gloss at all and it reads as plush.
      roughness: 1.0, metalness: 0.0, envMapIntensity: 0.02,
      transparent: true, opacity: 0
    });
    var pale = new THREE.MeshStandardMaterial({
      color: 0x6A5C46, roughness: 1.0, metalness: 0.0, envMapIntensity: 0.02,
      transparent: true, opacity: 0
    });
    var dark = new THREE.MeshStandardMaterial({
      color: 0x15100C, roughness: 0.95, metalness: 0.0, envMapIntensity: 0.02,
      transparent: true, opacity: 0
    });
    /* The eye IS the eyeshine. A separate bright sphere was swallowed three
       times over — by the eyeball, by the brow ridge, then by the eyeball
       again. A glossy dark ball that emits its own light cannot be occluded by
       anything but the skull, and reads from every angle. Emission sits just
       under the bloom threshold, so it is a wet eye catching the torches rather
       than a lamp. */
    var eyeball = new THREE.MeshStandardMaterial({
      color: 0x120A04, roughness: 0.08, metalness: 0.0, envMapIntensity: 1.8,
      emissive: 0xFFB43C, emissiveIntensity: 1.55
    });

    function add(geo, mat) {
      var m = new THREE.Mesh(geo, mat || fur);
      g3.add(m);
      return m;
    }

    /* ---- trunk: neck base through chest, loin and haunch to the tail ----
       z runs forward. Widths and heights move independently, which is the whole
       reason for sweeping sections rather than revolving a profile: a chest is
       deep and narrow, a haunch is broad and shallow. */
    var TRUNK = [
      [0.686, 0.800, 0.112, 0.108, 0.108],   // neck, carried UP to the head
      [0.588, 0.762, 0.138, 0.144, 0.140],
      [0.462, 0.708, 0.172, 0.186, 0.180],
      [0.322, 0.665, 0.208, 0.238, 0.225],   // withers, the high point
      [0.168, 0.630, 0.228, 0.262, 0.250],   // deep chest
      [0.014, 0.618, 0.228, 0.265, 0.258],
      [-0.140, 0.615, 0.210, 0.250, 0.235],
      [-0.294, 0.618, 0.176, 0.205, 0.190],  // loin: dip on top, tuck beneath
      [-0.434, 0.628, 0.170, 0.196, 0.186],
      [-0.574, 0.642, 0.204, 0.232, 0.212],
      [-0.714, 0.650, 0.228, 0.243, 0.222],  // haunch, back up to the withers
      [-0.854, 0.640, 0.196, 0.212, 0.196],
      [-0.980, 0.618, 0.140, 0.150, 0.142],
      [-1.078, 0.600, 0.090, 0.094, 0.090]   // tail base
    ];
    add(sweepGeometry(chainZ(TRUNK, 0, 2.15), 28, null, true, false));

    /* ---- head -------------------------------------------------------
       In its own group, pivoted at the neck, so it can be TURNED. The body is
       posed three-quarter to show its length; with the head welded into that
       pose the reveal shows the back of the skull and the muzzle end-on, which
       is a ring and a pale disc, not a face.

       Head-local coordinates: origin at the neck joint, +z forward along the
       muzzle. Short and broad, with a real stop behind the brow so the
       forehead does not run in one dome to the nose. */
    var headGroup = new THREE.Group();
    headGroup.position.set(0, 0.800, 0.686);
    g3.add(headGroup);
    function headBlob(r, sx, sy, sz, x, y, z, mat) {
      var m = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 12), mat || fur);
      m.scale.set(sx, sy, sz);
      m.position.set(x, y, z);
      headGroup.add(m);
      return m;
    }

    var HEAD = [
      [0.000, 0.000, 0.112, 0.110, 0.110],
      [0.058, 0.022, 0.150, 0.132, 0.128],
      [0.114, 0.032, 0.185, 0.150, 0.146],   // braincase, widest
      [0.162, 0.028, 0.182, 0.144, 0.158],   // cheeks
      [0.194, 0.016, 0.160, 0.134, 0.162],   // brow; jaw at its deepest
      /* The muzzle sits BELOW the eye line. Level with it, it stands in front
         of both eyes from any near-frontal view and hides them completely — a
         raycast from the door camera hit the skull, not the eye. On a cat the
         muzzle is below and between the eyes, and here that is a visibility
         requirement as much as an anatomical one. */
      [0.212, -0.020, 0.132, 0.084, 0.150],  // the STOP: the top drops here
      [0.244, -0.030, 0.124, 0.074, 0.132],  // muzzle holds its width
      [0.272, -0.036, 0.114, 0.068, 0.110],
      [0.290, -0.040, 0.092, 0.058, 0.088],
      [0.300, -0.042, 0.052, 0.040, 0.052]
    ];
    headGroup.add(new THREE.Mesh(sweepGeometry(chainZ(HEAD, 0, 2.7), 26, null, false, false), fur));

    /* ---- limbs: swept chains through the real joint path ----
       Each is one surface from shoulder to paw, starting inside the body so
       there is no join to see. A cylinder per bone with a ball over the joint
       is what reads as a doll. Heavy, too: a big cat's leg is close to as thick
       as its neck, and the thin tapers that look right on paper are stilts. */
    var FORE = [
      [0.180, 0.730, 0.245, 0.132],   // inside the shoulder
      [0.190, 0.580, 0.252, 0.142],
      [0.200, 0.470, 0.212, 0.116],   // elbow
      [0.206, 0.310, 0.238, 0.086],   // knee
      [0.210, 0.150, 0.256, 0.070],
      [0.212, 0.062, 0.272, 0.070],
      [0.212, 0.030, 0.300, 0.082]    // paw
    ];
    var HIND = [
      [0.172, 0.710, -0.640, 0.138],  // inside the haunch
      [0.186, 0.560, -0.648, 0.158],
      [0.200, 0.450, -0.580, 0.128],  // stifle
      [0.210, 0.290, -0.672, 0.088],  // hock
      [0.214, 0.140, -0.640, 0.072],
      [0.216, 0.058, -0.596, 0.072],
      [0.216, 0.030, -0.566, 0.084]   // paw
    ];
    var upZ = new THREE.Vector3(0, 0, 1);
    // Where each foot lands, kept so a contact shadow can follow it.
    var paws = [];
    [-1, 1].forEach(function (sx) {
      /* The near foreleg is put out in front, the far one stays under the
         chest. An outline that is one smooth continuous arc is the strongest
         primitive tell there is, and a limb crossing out of it breaks the arc
         for almost nothing. */
      var reach = sx === nearSide ? 0.16 : 0;
      add(scaleV(sweepGeometry(chainXYZ(FORE.map(function (r, i) {
        var t = i / (FORE.length - 1);
        return [sx * r[0], r[1], r[2] + reach * t * t, r[3]];
      })), 20, upZ, false, true), 0.60));
      add(scaleV(sweepGeometry(chainXYZ(HIND.map(function (r) {
        return [sx * r[0], r[1], r[2], r[3]];
      })), 20, upZ, false, true), 0.60));
      var fp = FORE[FORE.length - 1], hp = HIND[HIND.length - 1];
      paws.push(new THREE.Vector3(sx * fp[0], fp[1], fp[2] + reach));
      paws.push(new THREE.Vector3(sx * hp[0], hp[1], hp[2]));
    });

    /* ---- tail: out across the near flank and lifted, so it cuts the body's
       outline instead of trailing away behind where nothing can see it. ---- */
    var tail = [];
    for (var t = 0; t <= 9; t++) {
      var u = t / 9;
      tail.push([
        nearSide * (0.02 + 0.50 * Math.pow(u, 1.15)),
        0.600 + Math.sin(u * 2.4) * 0.26 - 0.09 * u * u,
        -1.078 - 0.26 * u,
        0.086 - 0.056 * u
      ]);
    }
    add(scaleV(sweepGeometry(chainXYZ(tail), 16, null, false, true), 0.55));

    /* ---- the face ---------------------------------------------------
       Features, not ratios. The proportions were right long before the face
       read as one, because a blunt head with no eyes, no nose and no mouth is
       a lump whatever its length-to-width is. */
    var eyes = [];

    // The muzzle's front, so it ends in a nose rather than an end cap.
    headBlob(0.055, 1.50, 1.25, 0.45, 0, -0.040, 0.294, fur);
    // The pad has to be NARROWER than the muzzle it sits on; wider, it stops
    // being a nose and becomes a dark oval in the middle of the face.
    headBlob(0.030, 1.28, 0.88, 0.55, 0, -0.030, 0.312, dark);
    headBlob(0.013, 1.0, 0.9, 0.7, -0.024, -0.038, 0.314, dark);
    headBlob(0.013, 1.0, 0.9, 0.7, 0.024, -0.038, 0.314, dark);

    // The mouth: a dark line under the muzzle, turning up at the corners.
    headBlob(0.054, 1.50, 0.15, 0.80, 0, -0.094, 0.264, dark);
    headBlob(0.026, 1.0, 0.22, 0.55, -0.074, -0.082, 0.230, dark);
    headBlob(0.026, 1.0, 0.22, 0.55, 0.074, -0.082, 0.230, dark);
    headBlob(0.034, 1.10, 0.50, 0.80, 0, -0.104, 0.226, pale);

    // Cheek ruffs. Proud of the skull, but only just: larger they read as a
    // bulldog's jowls rather than a cat's cheek.
    headBlob(0.058, 0.95, 1.00, 1.25, -0.136, -0.034, 0.150, fur);
    headBlob(0.058, 0.95, 1.00, 1.25, 0.136, -0.034, 0.150, fur);

    [-1, 1].forEach(function (sx) {
      /* A brow ridge over each eye, set up and BACK off it. Sitting level, this
         ridge enclosed the eyeshine entirely and hid it. */
      headBlob(0.048, 1.32, 0.40, 0.42, sx * 0.100, 0.138, 0.202, fur);
      /* A dark eye patch BEHIND the eyeball and flattened, so it frames the eye
         without ever standing in front of it. */
      headBlob(0.046, 1.20, 0.85, 0.22, sx * 0.100, 0.084, 0.232, dark);

      /* The eye's position was found by raycasting from the door camera across
         a grid of candidates, not by eye. The brow overhangs forward — the
         swept sections tilt with the spine where it drops into the muzzle — so
         anywhere behind z = 0.24 the forehead itself stands in front of the
         eye. This is the first position at which the camera's first hit is
         actually the eyeball. */
      var ball = new THREE.Mesh(new THREE.SphereGeometry(0.023, 14, 12), eyeball);
      ball.position.set(sx * 0.098, 0.082, 0.252);
      headGroup.add(ball);
      eyes.push(ball);

      /* Ears: small and rounded, but standing clear of the skull outline.
         Flush with it they vanish against a dark doorway, and the silhouette
         loses the one feature that says cat at any distance. */
      var ear = [
        [sx * 0.118, 0.118, 0.092, 0.044],
        [sx * 0.140, 0.168, 0.086, 0.060],
        [sx * 0.160, 0.208, 0.080, 0.054],
        [sx * 0.172, 0.236, 0.076, 0.026]
      ];
      var em = new THREE.Mesh(sweepGeometry(chainXYZ(ear), 14, upZ, false, true), fur);
      em.scale.z = 0.55;
      em.position.z = 0.086 * 0.45;
      headGroup.add(em);
      // The dark back of the ear, which is what makes it read at distance.
      headBlob(0.038, 0.75, 1.15, 0.35, sx * 0.152, 0.192, 0.066, dark);
    });

    g3.scale.setScalar(1.20);
    /* No global pitch. Pitching the whole body tilts the feet out of a common
       plane: measured, it lifted the forward paws 6 cm off the floor while the
       hind ones carried the contact. The spine carries the head-down line. */
    g3.rotation.x = 0;

    /* Grounding, measured rather than tuned. The limbs are swept chains that
       taper into the paw, so the lowest vertex is nowhere near the joint
       position the leg was laid out from. Taking the world bounding box after
       the pose is applied and offsetting by its floor gives contact by
       construction — measured at the old hand-set height, the feet were 7.7 cm
       THROUGH the floor. */
    g3.updateMatrixWorld(true);
    var bb = new THREE.Box3().setFromObject(g3);
    g3.userData = {
      fur: fur, pale: pale, dark: dark, eyes: eyes,
      groundOffset: -bb.min.y, paws: paws, headGroup: headGroup,
      // Head-local origin, for aiming a light at the face.
      faceLocal: new THREE.Vector3(0, 0.020, 0.230)
    };
    return g3;
  }

  /* The treasure: a chest and a heap of coins, emissive above 1.0 so the bloom
     pass picks them up the moment the door opens. */
  function buildHoard() {
    var g = new THREE.Group();
    var chest = new THREE.Mesh(
      new THREE.BoxGeometry(1.0, 0.5, 0.62),
      new THREE.MeshStandardMaterial({ color: 0x40260F, roughness: 0.75, metalness: 0.1 })
    );
    chest.position.set(0, 0.25, 0);
    g.add(chest);
    var lid = new THREE.Mesh(
      new THREE.CylinderGeometry(0.31, 0.31, 1.0, 16, 1, false, 0, Math.PI),
      new THREE.MeshStandardMaterial({ color: 0x4A2C12, roughness: 0.7, metalness: 0.15 })
    );
    lid.rotation.z = Math.PI / 2;
    lid.position.set(0, 0.5, 0);
    g.add(lid);
    var coinMat = new THREE.MeshStandardMaterial({
      color: 0xC79A3A, roughness: 0.28, metalness: 1.0,
      emissive: 0x4A3208, emissiveIntensity: 1.0, envMapIntensity: 1.4
    });
    var rnd = mulberry(5150);
    for (var i = 0; i < 46; i++) {
      var coin = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.014, 10), coinMat);
      var a = rnd() * Math.PI * 2, r = rnd() * 0.72;
      coin.position.set(Math.cos(a) * r, 0.01 + rnd() * 0.07, 0.42 + Math.sin(a) * r * 0.42);
      coin.rotation.set(rnd() * 0.5, rnd() * 3, rnd() * 0.5);
      g.add(coin);
    }
    return g;
  }

  /* --------------------------------------------------------- reading a belief

     The belief arrives in the shape core serialised it as, never as the class
     that produced it: every particle belief in the package — weighted,
     unweighted, incremental, vectorized — is "particles" here, and this widget
     reads them once rather than once per class.

     Tiger's particles are the state labels themselves, so the probability of a
     hypothesis is the mass its label carries. */
  function beliefReading(belief, states) {
    if (!belief) return { left: null, label: "—" };
    if (belief.kind !== "particles") {
      if (belief.kind === "particle_batch") {
        // A batch is several beliefs held together for a vectorized planner. It
        // is not one episode's belief, and merging its members would show a
        // distribution that was never anyone's.
        return { left: null, label: "batch of " + belief.batch_size + " beliefs, not drawn" };
      }
      return { left: null, label: "not recorded (" + (belief.belief_class || belief.kind) + ")" };
    }
    var mass = {}, total = 0, unknown = 0;
    for (var i = 0; i < belief.particles.length; i++) {
      var name = belief.particles[i];
      var w = belief.weighted === false ? 1 / belief.particles.length : belief.weights[i];
      if (states.indexOf(name) < 0) { unknown += w; continue; }
      mass[name] = (mass[name] || 0) + w;
      total += w;
    }
    if (total <= 0) {
      return { left: null, label: "no mass on either hypothesis" + (unknown ? " (off-support particles)" : "") };
    }
    var left = (mass[states[0]] || 0) / total;
    var label = states[0] + " " + left.toFixed(2) + " · " +
      states[1] + " " + (1 - left).toFixed(2) +
      " (" + belief.num_particles + (belief.weighted === false ? " uniform" : "") + " particles";
    if (belief.num_written < belief.num_particles) label += ", heaviest " + belief.num_written + " read";
    return { left: left, label: label + ")" };
  }

  /**
   * Build the Tiger chamber from one trace.
   *
   * @param {Object} core   A renderer core.
   * @param {Object} trace  A parsed trace.json with payload_kind tiger.v1.
   * @returns {Object} The scene module the player drives.
   */
  function build(core, trace) {
    var payload = trace.payload;
    var world = payload.world;
    var scene = core.scene;
    var renderer = core.renderer;
    var maxAniso = renderer.capabilities.getMaxAnisotropy();
    var reduceMotion = global.matchMedia
      ? global.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;

    var STATE_LEFT = world.states[0];     // "tiger_left"
    var STATE_RIGHT = world.states[1];    // "tiger_right"

    /* Every step, read once: what was done, what came back, where the tiger
       was, and what the belief was before and after. Nothing is recomputed
       from a model here — the posterior a step lands on is the NEXT step's
       recorded belief, which is what makes the reset after an open the run's
       own number rather than this file's arithmetic. */
    var readings = payload.beliefs.map(function (b) { return beliefReading(b, world.states); });
    var steps = payload.states.map(function (state, i) {
      var action = trace.steps[i] ? trace.steps[i].action : null;
      var opened = action === "open_left" ? 0 : action === "open_right" ? 1 : null;
      var heard = payload.observations[i] === "hear_left" ? 0
        : payload.observations[i] === "hear_right" ? 1 : -1;
      var after = readings[i + 1] && readings[i + 1].left !== null
        ? readings[i + 1].left : readings[i].left;
      return {
        action: action,
        observation: payload.observations[i],
        state: state,
        tigerLeft: state === STATE_LEFT,
        opened: opened,
        // Opening the tiger's own door is the -100. Which door hides it comes
        // from the state recorded BEFORE the action, never from the next one:
        // the transition has already re-drawn the side by then.
        hitTiger: opened === null ? false : (opened === 0) === (state === STATE_LEFT),
        listening: action === "listen",
        heard: action === "listen" ? heard : -1,
        before: readings[i].left,
        after: after,
        label: readings[i].label
      };
    });

    /* -------------------------------------------------------------- lighting */
    scene.background = new THREE.Color(0x050403).convertSRGBToLinear();
    // Smoke and dust hanging in a sealed room. Enough that the far wall sits
    // behind some air, little enough that both doors stay readable.
    scene.fog = new THREE.FogExp2(0x0B0705, 0.048);
    scene.fog.color.convertSRGBToLinear();

    // Almost nothing from the sky: this room has no windows. It exists only so
    // the deepest shadows are not pure black.
    scene.add(new THREE.HemisphereLight(0x2A1E14, 0x080604, 0.30));

    /* An environment map so the iron bands and the sconces have something to
       reflect; PBR metal with nothing to mirror reads as grey plastic. Here the
       "environment" is the room's own warm glow, not a sky. */
    core.buildNightEnvironment([
      [0.0, "#080604"], [0.55, "#221408"], [0.78, "#48250B"], [1.0, "#0C0806"]
    ]);

    /* The chamber is built in its own coordinates and shifted as one group, so
       the rig's look-at point lands on the middle of the door wall. */
    var room = new THREE.Group();
    room.position.copy(SHIFT);
    scene.add(room);

    var flameTex = flameTexture();
    var poolTex = V.radialTexture(0.85, 0.42);
    var partTex = V.radialTexture(0.95, 0.35);
    var glowTex = V.radialTexture(0.9, 0.3);

    /* ---------------------------------------------------------- the chamber */
    var floorCv = floorCanvas();
    var floorNormal = normalMapFrom(renderer, floorCv, 2.4);
    floorNormal.wrapS = floorNormal.wrapT = THREE.RepeatWrapping;
    floorNormal.repeat.set(2, 2);
    var floorAlbedo = new THREE.CanvasTexture(floorCv);
    floorAlbedo.anisotropy = maxAniso;
    floorAlbedo.encoding = THREE.sRGBEncoding;
    floorAlbedo.wrapS = floorAlbedo.wrapT = THREE.RepeatWrapping;
    floorAlbedo.repeat.set(2, 2);

    var wallCv = wallCanvas();
    var wallNormal = normalMapFrom(renderer, wallCv, 2.6);
    wallNormal.wrapS = wallNormal.wrapT = THREE.RepeatWrapping;
    var wallAlbedo = new THREE.CanvasTexture(wallCv);
    wallAlbedo.anisotropy = maxAniso;
    wallAlbedo.encoding = THREE.sRGBEncoding;
    wallAlbedo.wrapS = wallAlbedo.wrapT = THREE.RepeatWrapping;

    var stoneMat = new THREE.MeshStandardMaterial({
      map: wallAlbedo, normalMap: wallNormal,
      normalScale: new THREE.Vector2(1.1, 1.1),
      color: 0x9A8464, roughness: 0.94, metalness: 0.0, envMapIntensity: 0.25
    });
    // The arch surround is a paler, better-dressed stone than the rubble wall,
    // which is how the doors read as doors from across the room.
    var archMat = new THREE.MeshStandardMaterial({
      map: wallAlbedo, normalMap: wallNormal,
      normalScale: new THREE.Vector2(0.9, 0.9),
      color: 0xE6D8B8, roughness: 0.86, metalness: 0.0, envMapIntensity: 0.35
    });
    var ironMat = new THREE.MeshStandardMaterial({
      color: 0x24201C, roughness: 0.45, metalness: 0.9, envMapIntensity: 0.9
    });
    var bronzeMat = new THREE.MeshStandardMaterial({
      color: 0x6B5436, roughness: 0.4, metalness: 0.88, envMapIntensity: 0.9
    });

    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(ROOM_W, ROOM_D),
      new THREE.MeshStandardMaterial({
        map: floorAlbedo, normalMap: floorNormal,
        normalScale: new THREE.Vector2(1.0, 1.0),
        color: 0x8D7A5E, roughness: 0.82, metalness: 0.0, envMapIntensity: 0.3
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.z = WALL_Z + ROOM_D / 2 - 0.3;
    floor.receiveShadow = true;
    room.add(floor);

    function box(w, h, d, x, y, z, mat) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat || stoneMat);
      m.position.set(x, y, z);
      m.castShadow = true;
      m.receiveShadow = true;
      room.add(m);
      return m;
    }

    /* The back wall is four pieces, so the two door openings are real holes
       rather than a texture. Nothing here is coplanar with anything else within
       a few centimetres: the render target's depth buffer is 16-bit. */
    var HOLE_HW = DOOR_W / 2;
    var innerL = DOOR_X - HOLE_HW, innerR = DOOR_X + HOLE_HW;
    box(ROOM_W / 2 - innerR, HOLE_TOP, 0.4, -(innerR + ROOM_W / 2) / 2, HOLE_TOP / 2, WALL_Z);
    box(innerL * 2, HOLE_TOP, 0.4, 0, HOLE_TOP / 2, WALL_Z);
    box(ROOM_W / 2 - innerR, HOLE_TOP, 0.4, (innerR + ROOM_W / 2) / 2, HOLE_TOP / 2, WALL_Z);
    box(ROOM_W, ROOM_H - HOLE_TOP, 0.4, 0, (ROOM_H + HOLE_TOP) / 2, WALL_Z);

    // Side walls, a wall behind the camera so the orbit never shows the void,
    // and a ceiling that hides itself once the camera climbs above it.
    box(0.4, ROOM_H, ROOM_D, -ROOM_W / 2 - 0.2, ROOM_H / 2, WALL_Z + ROOM_D / 2 - 0.3);
    box(0.4, ROOM_H, ROOM_D, ROOM_W / 2 + 0.2, ROOM_H / 2, WALL_Z + ROOM_D / 2 - 0.3);
    box(ROOM_W + 0.8, ROOM_H, 0.4, 0, ROOM_H / 2, WALL_Z + ROOM_D - 0.1);
    var ceiling = box(ROOM_W + 0.8, 0.4, ROOM_D, 0, ROOM_H + 0.2, WALL_Z + ROOM_D / 2 - 0.3);
    ceiling.material = new THREE.MeshStandardMaterial({
      map: wallAlbedo, normalMap: wallNormal, color: 0x4A3E2E, roughness: 0.98, metalness: 0.0
    });
    var ceilingTop = room.position.y + ROOM_H;

    /* An arched opening: a rectangle with a half-round top. Used three times
       per door — for the stone reveal, the surround band and the leaf. */
    function archShape(w, h) {
      var s = new THREE.Shape();
      var hw = w / 2;
      s.moveTo(-hw, 0);
      s.lineTo(-hw, h);
      s.absarc(0, h, hw, Math.PI, 0, true);
      s.lineTo(hw, 0);
      s.closePath();
      return s;
    }
    function rectShape(w, h) {
      var s = new THREE.Shape();
      var hw = w / 2;
      s.moveTo(-hw, 0); s.lineTo(-hw, h); s.lineTo(hw, h); s.lineTo(hw, 0);
      s.closePath();
      return s;
    }

    var doorCv = doorCanvas();
    /* ExtrudeGeometry writes UVs in the shape's own units, not 0..1, so the
       door's UVs run about -0.81..0.81 across and 0..2.48 up. Clamped, that
       smears one column of planks over the whole leaf; repeated and scaled to
       the leaf's size, it tiles correctly. */
    var doorNormal = normalMapFrom(renderer, doorCv, 1.6);
    doorNormal.wrapS = doorNormal.wrapT = THREE.RepeatWrapping;
    doorNormal.repeat.set(0.62, 0.40);
    doorNormal.offset.set(0.5, 0);
    var doorAlbedo = new THREE.CanvasTexture(doorCv);
    doorAlbedo.anisotropy = maxAniso;
    doorAlbedo.encoding = THREE.sRGBEncoding;
    doorAlbedo.wrapS = doorAlbedo.wrapT = THREE.RepeatWrapping;
    doorAlbedo.repeat.set(0.62, 0.40);
    doorAlbedo.offset.set(0.5, 0);

    var oakMat = new THREE.MeshStandardMaterial({
      map: doorAlbedo, normalMap: doorNormal,
      normalScale: new THREE.Vector2(1.0, 1.0),
      color: 0xD6BE9E, roughness: 0.78, metalness: 0.02, envMapIntensity: 0.3
    });

    var doors = [];   // one entry per side: left = 0, right = 1

    [-1, 1].forEach(function (sign, idx) {
      var cx = sign * DOOR_X;

      /* The stone reveal closes the rectangular hole down to the arched
         opening. It sits flush in the wall and uses the wall's own texture, so
         it reads as part of the wall rather than as a separate object. */
      var revealShape = rectShape(DOOR_W + 0.58, HOLE_TOP + 0.02);
      revealShape.holes.push(archShape(DOOR_W, DOOR_H));
      var reveal = new THREE.Mesh(
        new THREE.ExtrudeGeometry(revealShape, { depth: 0.4, bevelEnabled: false }), stoneMat
      );
      reveal.position.set(cx, 0, WALL_Z - 0.2);
      reveal.castShadow = true;
      reveal.receiveShadow = true;
      room.add(reveal);

      /* The dressed arch band, standing 0.2 proud of the wall face. It is there
         to be lit from the side: the shadow it throws is the clearest cue that
         the torches are real lights and not a painted glow. */
      var bandShape = archShape(DOOR_W + 0.56, DOOR_H);
      bandShape.holes.push(archShape(DOOR_W + 0.02, DOOR_H));
      var band = new THREE.Mesh(
        new THREE.ExtrudeGeometry(bandShape, {
          depth: 0.2, bevelEnabled: true, bevelSize: 0.02,
          bevelThickness: 0.02, bevelSegments: 1
        }), archMat
      );
      band.position.set(cx, 0, WALL_Z + 0.2);
      band.castShadow = true;
      band.receiveShadow = true;
      room.add(band);

      // Jamb columns either side of the opening, with a capital under the arch.
      [-1, 1].forEach(function (js) {
        var col = new THREE.Mesh(
          new THREE.CylinderGeometry(0.16, 0.18, DOOR_H - 0.16, 12), archMat);
        col.position.set(cx + js * (DOOR_W / 2 + 0.26), (DOOR_H - 0.16) / 2, WALL_Z + 0.28);
        col.castShadow = true; col.receiveShadow = true;
        room.add(col);
        var cap = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.14, 0.34), archMat);
        cap.position.set(cx + js * (DOOR_W / 2 + 0.26), DOOR_H - 0.06, WALL_Z + 0.28);
        cap.castShadow = true; cap.receiveShadow = true;
        room.add(cap);
      });

      /* Behind each door, a sealed cell. Both are lit identically and faintly,
         so the sliver of light under the two doors gives nothing away: the
         whole point of the problem is that the doors are indistinguishable. */
      var cell = new THREE.Mesh(
        new THREE.BoxGeometry(2.6, 3.2, 2.6),
        new THREE.MeshStandardMaterial({
          map: wallAlbedo, normalMap: wallNormal,
          color: 0x3A2E22, roughness: 1.0, metalness: 0.0, side: THREE.BackSide
        })
      );
      cell.position.set(cx, 1.6, WALL_Z - 1.6);
      room.add(cell);

      /* A real floor inside the cell, in the chamber's own flagstone. The cell
         box has a bottom face, but it is unlit rubble and renders black, and an
         animal standing on a surface nobody can see reads as floating however
         exactly its feet are placed. */
      var cellFloor = new THREE.Mesh(
        new THREE.PlaneGeometry(2.5, 2.5),
        new THREE.MeshStandardMaterial({
          map: floorAlbedo, normalMap: floorNormal,
          normalScale: new THREE.Vector2(0.9, 0.9),
          color: 0x6E6049, roughness: 0.9, metalness: 0.0, envMapIntensity: 0.2
        })
      );
      cellFloor.rotation.x = -Math.PI / 2;
      cellFloor.position.set(cx, CELL_FLOOR_Y + 0.004, WALL_Z - 1.6);
      cellFloor.receiveShadow = true;
      room.add(cellFloor);
      var cellLight = new THREE.PointLight(0xFFC489, 1, 3.4, 2);
      cellLight.power = 26;                        // just enough to escape the gap
      cellLight.position.set(cx, 0.9, WALL_Z - 1.1);
      cellLight.userData.home = cellLight.position.clone();
      room.add(cellLight);

      /* The door leaf, hinged on its outer edge: the group sits at the hinge
         and the leaf hangs off it, which is what lets it swing. It is 6 cm
         short of the floor, which is the gap the light escapes through, and it
         hangs at the FRONT of the opening — set deeper, the 0.4 m stone reveal
         shadows it completely and the door renders black. */
      var LEAF_W = DOOR_W - 0.08;
      var GAP = 0.06;
      var hinge = new THREE.Group();
      hinge.position.set(cx + sign * (LEAF_W / 2), GAP, WALL_Z + 0.2);
      room.add(hinge);

      var leaf = new THREE.Mesh(
        new THREE.ExtrudeGeometry(archShape(LEAF_W, DOOR_H - GAP - 0.04),
          { depth: 0.11, bevelEnabled: false }), oakMat
      );
      leaf.position.set(-sign * (LEAF_W / 2), 0, -0.055);
      leaf.castShadow = true;
      leaf.receiveShadow = true;
      hinge.add(leaf);

      // Iron strap hinges, studs and a ring pull: the fittings are what stop a
      // flat plank panel from reading as a cupboard door.
      [0.42, 1.62].forEach(function (by) {
        var strap = new THREE.Mesh(new THREE.BoxGeometry(LEAF_W - 0.16, 0.13, 0.035), ironMat);
        strap.position.set(-sign * (LEAF_W / 2), by, 0.075);
        strap.castShadow = true;
        hinge.add(strap);
        for (var n = 0; n < 6; n++) {
          var stud = new THREE.Mesh(new THREE.SphereGeometry(0.028, 8, 6), ironMat);
          stud.position.set(-sign * (LEAF_W / 2) + (n - 2.5) * 0.24, by, 0.098);
          hinge.add(stud);
        }
      });
      var plate = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.26, 0.022), ironMat);
      plate.position.set(-sign * (LEAF_W / 2) - sign * 0.44, 1.14, 0.068);
      plate.castShadow = true;
      hinge.add(plate);
      var ring = new THREE.Mesh(new THREE.TorusGeometry(0.1, 0.019, 8, 18), ironMat);
      ring.position.set(-sign * (LEAF_W / 2) - sign * 0.44, 0.92, 0.076);
      ring.castShadow = true;
      hinge.add(ring);

      /* The contact shadow under the door: the ambient occlusion in the 6 cm
         gap, which a shadow map at this resolution cannot resolve on its own. */
      var underShadow = new THREE.Mesh(
        new THREE.PlaneGeometry(DOOR_W + 0.5, 0.9),
        new THREE.MeshBasicMaterial({
          map: poolTex, color: 0x000000, transparent: true, opacity: 0.72, depthWrite: false
        })
      );
      underShadow.rotation.x = -Math.PI / 2;
      underShadow.position.set(cx, 0.012, WALL_Z + 0.44);
      room.add(underShadow);

      // The thin warm line escaping under the door, on the side the camera is
      // on. Both doors leak identically: the moment one leaks more, the problem
      // is solved without listening.
      var slit = new THREE.Mesh(
        new THREE.PlaneGeometry(LEAF_W - 0.1, 0.5),
        new THREE.MeshBasicMaterial({
          map: poolTex, color: 0xFFB262, transparent: true, opacity: 0.5,
          blending: THREE.AdditiveBlending, depthWrite: false
        })
      );
      slit.rotation.x = -Math.PI / 2;
      slit.position.set(cx, 0.018, WALL_Z + 0.32);
      room.add(slit);

      /* The belief sign above the arch. The bar's length is the probability and
         the number is printed on it, so the reading is exact rather than a
         guess from a glow. Pulled in towards the middle of the lintel: at the
         door centres, and even at the prototype's inset, the left sign sat
         under the page's HUD panel and lost its number. */
      var signCv = document.createElement("canvas");
      signCv.width = 512; signCv.height = 160;
      var signTex = new THREE.CanvasTexture(signCv);
      signTex.encoding = THREE.sRGBEncoding;
      var beliefSign = new THREE.Mesh(
        new THREE.PlaneGeometry(1.9, 0.594),
        new THREE.MeshBasicMaterial({ map: signTex, transparent: true, depthWrite: false })
      );
      beliefSign.position.set(cx * SIGN_INSET, 3.22, WALL_Z + 0.26);
      room.add(beliefSign);

      // A stone tablet behind it, so the sign reads as carved into the lintel
      // rather than pasted over the render.
      var tablet = new THREE.Mesh(new THREE.BoxGeometry(2.14, 0.8, 0.14), archMat);
      tablet.position.set(cx * SIGN_INSET, 3.22, WALL_Z + 0.16);
      tablet.castShadow = true;
      tablet.receiveShadow = true;
      room.add(tablet);

      // Ripples on the floor for the listen cue. Four rings, staggered, so the
      // sound reads as a pulse travelling out rather than one expanding circle.
      var rings = [];
      for (var ri = 0; ri < 4; ri++) {
        var ringMesh = new THREE.Mesh(
          new THREE.RingGeometry(0.97, 1.0, 64),
          new THREE.MeshBasicMaterial({
            color: C.text, transparent: true, opacity: 0,
            blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
          })
        );
        ringMesh.rotation.x = -Math.PI / 2;
        ringMesh.position.set(cx, 0.03, WALL_Z + 0.5);
        room.add(ringMesh);
        rings.push(ringMesh);
      }

      // A rim of light down the jamb, brightened while that side is being heard.
      var cueLight = new THREE.PointLight(0xFFE6B4, 1, 5.0, 2);
      cueLight.power = 0;
      cueLight.position.set(cx, 1.5, WALL_Z + 0.9);
      room.add(cueLight);

      /* What is behind this door. Both are built, and which one is shown comes
         from the state the trace recorded for the step being played — so a
         later step in the same episode can show the other one, which is exactly
         what the re-draw after an open means. Each leaf swings out, so the gap
         opens on the centre-pier side; the tiger stands on that side, turned
         well off square, and the hindquarters run back into the cell where the
         light does not reach. */
      var tigerX = cx + sign * 0.34;
      var tigerYaw = -sign * 0.63;
      var tiger = buildTiger(renderer, sign);
      tiger.position.set(tigerX, 0, WALL_Z - 1.05);
      tiger.rotation.y = tigerYaw;
      tiger.visible = false;
      room.add(tiger);

      var hoard = buildHoard();
      hoard.position.set(cx, 0, WALL_Z - 1.05);
      hoard.visible = false;
      room.add(hoard);

      var prizeLight = new THREE.PointLight(0xFFD27A, 1, 6.0, 2);
      prizeLight.power = 0;
      prizeLight.position.set(cx, 1.1, WALL_Z - 0.9);
      room.add(prizeLight);

      /* A narrow rake for the tiger, in place of a point fill. A point light at
         this range finds every surface it can see and the animal reads as one
         smooth object; a tight cone across the shoulder, the cheek and the near
         foreleg leaves the flank to fall off into black and lets the shape stay
         a guess — which is the honest version of this environment. */
      var rakeLight = new THREE.SpotLight(0xFFCF9A, 1, 2.8, 0.30, 0.66, 2);
      rakeLight.power = 0;
      rakeLight.castShadow = false;
      rakeLight.position.set(cx - sign * 0.98, 2.16, WALL_Z - 0.06);
      room.add(rakeLight);
      rakeLight.target.position.set(cx - sign * 0.50, 0.58, WALL_Z - 1.05);
      room.add(rakeLight.target);

      /* One soft patch under each paw. The shadow map cannot resolve contact at
         this scale, and without it the eye has nothing to tell it the feet are
         touching — which is most of why the animal read as floating. */
      var pawShadows = [];
      for (var ps = 0; ps < 4; ps++) {
        var pm = new THREE.Mesh(
          new THREE.PlaneGeometry(0.58, 0.58),
          new THREE.MeshBasicMaterial({
            map: poolTex, color: 0x000000, transparent: true, opacity: 0, depthWrite: false
          })
        );
        pm.rotation.x = -Math.PI / 2;
        pm.position.set(cx, CELL_FLOOR_Y + 0.012, WALL_Z - 1.6);
        pm.visible = false;
        room.add(pm);
        pawShadows.push(pm);
      }

      doors.push({
        sign: beliefSign, signCv: signCv, signTex: signTex, signLast: -1,
        hinge: hinge, doorX: cx, sideSign: sign,
        rings: rings, cueLight: cueLight, slit: slit,
        tiger: tiger, tigerX: tigerX, tigerYaw: tigerYaw,
        hoard: hoard, prizeLight: prizeLight, rakeLight: rakeLight,
        pawShadows: pawShadows, cellLight: cellLight
      });
    });

    /* ------------------------------------------------------------------ fire
       Two wall torches. Real lumens, inverse-square, and an emissive flame far
       above 1.0 so the bloom pass has something genuinely hot to find. */
    var torches = [];
    [-1, 1].forEach(function (sign) {
      var tx = sign * 4.05, ty = 2.25, tz = WALL_Z + 0.35;

      var bracket = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.42), ironMat);
      bracket.position.set(tx, ty - 0.18, tz - 0.16);
      bracket.castShadow = true;
      room.add(bracket);

      var cup = new THREE.Mesh(
        new THREE.CylinderGeometry(0.17, 0.09, 0.3, 12, 1, true), bronzeMat);
      cup.position.set(tx, ty, tz);
      cup.castShadow = true;
      room.add(cup);

      var flame = new THREE.Sprite(new THREE.SpriteMaterial({
        map: flameTex, color: new THREE.Color(5.4, 2.6, 0.7),
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false
      }));
      flame.scale.set(0.34, 0.5, 1);
      flame.position.set(tx, ty + 0.3, tz);
      room.add(flame);

      var halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xFFB05A, transparent: true, opacity: 0.42,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      halo.scale.set(1.5, 1.5, 1);
      halo.position.set(tx, ty + 0.24, tz + 0.02);
      room.add(halo);

      var light = new THREE.PointLight(0xFFB870, 1, 11, 2);
      light.power = TORCH_LUMENS;
      light.position.set(tx, ty + 0.24, tz + 0.05);
      room.add(light);

      torches.push({
        light: light, flame: flame, halo: halo,
        x: tx, y: ty, z: tz, phase: sign > 0 ? 1.7 : 0
      });
    });

    /* One shadow-casting light, not four. It hangs where the agent stands and
       looks at the doors, so the leaves, the arch bands and the jamb columns
       throw shadows that agree with each other. Narrow cone plus normalBias,
       which is the fix for acne that a big negative bias is not. */
    var shadowLight = new THREE.SpotLight(0xFFC489, 1, 16, 0.58, 0.7, 2);
    shadowLight.power = 1500;
    shadowLight.castShadow = true;
    shadowLight.shadow.mapSize.set(2048, 2048);
    shadowLight.shadow.radius = 2;
    shadowLight.shadow.camera.near = 0.6;
    shadowLight.shadow.camera.far = 16;
    shadowLight.shadow.bias = -0.0006;
    shadowLight.shadow.normalBias = 0.035;
    shadowLight.position.set(0, 2.1, 2.3);
    room.add(shadowLight);
    shadowLight.target.position.set(0, 0.9, WALL_Z + 0.3);
    room.add(shadowLight.target);

    /* ----------------------------------------------------------- the listener
       The agent is not a character in this problem: it has no position and no
       body. What stands in for it is its brazier on the floor, which is also
       the foreground light and the near edge that stops the room from reading
       as a flat backdrop with two doors on it. */
    var lantern = new THREE.Group();
    lantern.position.set(0, 0, -0.35);
    lantern.scale.setScalar(0.9);
    room.add(lantern);

    var bowl = new THREE.Mesh(
      new THREE.CylinderGeometry(0.34, 0.16, 0.24, 18, 1, true),
      new THREE.MeshStandardMaterial({
        color: 0x2E2822, roughness: 0.5, metalness: 0.88,
        side: THREE.DoubleSide, envMapIntensity: 0.9
      })
    );
    bowl.position.y = 0.46;
    bowl.castShadow = true;
    lantern.add(bowl);
    var rim = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.028, 8, 22), ironMat);
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.58;
    rim.castShadow = true;
    lantern.add(rim);
    for (var lg = 0; lg < 3; lg++) {
      var la = (lg / 3) * Math.PI * 2 + 0.5;
      var leg = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.03, 0.42, 6), ironMat);
      leg.position.set(Math.cos(la) * 0.19, 0.2, Math.sin(la) * 0.19);
      leg.rotation.set(Math.sin(la) * 0.22, 0, -Math.cos(la) * 0.22);
      leg.castShadow = true;
      lantern.add(leg);
    }
    // Coals, emissive above 1.0 so the bloom pass finds them.
    var coalRnd = mulberry(8181);
    for (var cb = 0; cb < 14; cb++) {
      var coal = new THREE.Mesh(
        new THREE.SphereGeometry(0.035 + coalRnd() * 0.03, 7, 6),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(2.4 + coalRnd(), 0.8, 0.14) })
      );
      var ca = coalRnd() * Math.PI * 2, crr = coalRnd() * 0.22;
      coal.position.set(Math.cos(ca) * crr, 0.55, Math.sin(ca) * crr);
      lantern.add(coal);
    }
    var wick = new THREE.Sprite(new THREE.SpriteMaterial({
      map: flameTex, color: new THREE.Color(4.0, 2.0, 0.55),
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false
    }));
    wick.scale.set(0.5, 0.62, 1);
    wick.position.y = 0.86;
    lantern.add(wick);
    var brazierHalo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: 0xFF9A48, transparent: true, opacity: 0.4,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    brazierHalo.scale.set(1.5, 1.5, 1);
    brazierHalo.position.y = 0.7;
    lantern.add(brazierHalo);
    var lanternLight = new THREE.PointLight(0xFFAE68, 1, 7.5, 2);
    lanternLight.power = 520;
    lanternLight.position.y = 0.66;
    lantern.add(lanternLight);

    var lanternShadow = new THREE.Mesh(
      new THREE.PlaneGeometry(1.5, 1.5),
      new THREE.MeshBasicMaterial({
        map: poolTex, color: 0x000000, transparent: true, opacity: 0.5, depthWrite: false
      })
    );
    lanternShadow.rotation.x = -Math.PI / 2;
    lanternShadow.position.y = 0.012;
    lantern.add(lanternShadow);

    /* ------------------------------------------------------------------ dust
       Smoke drifting through the torchlight: what tells the eye the room has
       air in it, and the cheapest realism in the whole scene. */
    var MOTES = 320;
    var motePos = new Float32Array(MOTES * 3);
    var moteSeed = mulberry(1357);
    for (var m = 0; m < MOTES; m++) {
      motePos[m * 3] = (moteSeed() - 0.5) * ROOM_W;
      motePos[m * 3 + 1] = 0.1 + moteSeed() * (ROOM_H - 0.4);
      motePos[m * 3 + 2] = WALL_Z + 0.3 + moteSeed() * (ROOM_D - 1.2);
    }
    var moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    room.add(new THREE.Points(moteGeo, new THREE.PointsMaterial({
      size: 0.03, map: partTex, color: 0xFFDCAE, transparent: true, opacity: 0.30,
      blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    })));

    core.linearize();

    /* ------------------------------------------------------------ belief sign
       Drawn on a canvas rather than built from geometry, because the number has
       to be exact and readable at any distance. */
    function drawSign(door, label, value) {
      var g = door.signCv.getContext("2d");
      var known = value !== null;
      var lead = known && value >= 0.5;
      g.clearRect(0, 0, 512, 160);
      g.fillStyle = "rgba(20,16,13,0.88)";
      g.fillRect(0, 0, 512, 160);
      g.strokeStyle = "#E0B04C";
      g.lineWidth = 4;
      g.strokeRect(2, 2, 508, 156);

      g.font = "600 30px 'Chakra Petch', system-ui, sans-serif";
      g.fillStyle = "#A79372";
      g.textBaseline = "middle";
      g.fillText("TIGER " + label, 22, 38);

      g.font = "700 62px 'IBM Plex Mono', monospace";
      g.fillStyle = lead ? "#E0B04C" : "#F6DDAB";
      g.textAlign = "right";
      // A belief this page could not read is left blank rather than filled in
      // with the 0.50 a reader would take for the run's own number.
      g.fillText(known ? value.toFixed(2) : "—", 490, 44);
      g.textAlign = "left";

      // Track first, then the fill, so the empty part is still visible at
      // p = 0 and the two signs always read as halves of one distribution.
      g.fillStyle = "#392F20";
      g.fillRect(22, 92, 468, 40);
      if (known) {
        g.fillStyle = lead ? "#E0B04C" : "#8A6C33";
        g.fillRect(22, 92, Math.max(3, 468 * value), 40);
      }
      door.signTex.needsUpdate = true;
    }

    /* ---------------------------------------------------------------- sampling
       One sample of the episode at continuous step index t. Within a step the
       phases are explicit: the cue happens first, the belief moves only after
       it, and the door swings over the back half. Keeping those windows apart
       is what makes the update read as caused by the observation. */
    function sampleAt(t) {
      var n = steps.length;
      var tc = clamp(t, 0, n - 1);
      var i = Math.min(n - 1, Math.floor(tc));
      var f = clamp(tc - i, 0, 1);
      var s = steps[i];

      var cue = s.listening ? smooth(0.08, 0.26, f) * (1 - smooth(0.42, 0.66, f)) : 0;
      var cueT = s.listening ? clamp((f - 0.08) / 0.58, 0, 1) : 0;
      /* A listen moves the belief just after its cue lands. An open does not
         move it at all until the door is fully round: what happens then is not
         an update but the transition model re-drawing the tiger's side, so it
         belongs at the very end of the step. Watching the bars snap back to
         even there is the whole lesson of this environment. */
      var updated = s.opened !== null ? smooth(0.90, 1.0, f) : smooth(0.52, 0.86, f);
      // The leaf opens over the first half, so there is a real gap to see eyes
      // through well before the body fades up.
      var swing = s.opened !== null ? smooth(0.10, 0.70, f) : 0;

      var p = s.before === null || s.after === null
        ? (s.before === null ? s.after : s.before)
        : lerp(s.before, s.after, updated);

      return {
        i: i, f: f, step: s, cue: cue, cueT: cueT, swing: swing,
        pLeft: p,
        rewardBooked: f > 0.5
      };
    }

    /* The return the HUD shows, discounted the way the envelope's own
       `discounted_return` is, so the page and the file agree. */
    var running = [];
    var total = 0;
    for (var ri = 0; ri < trace.steps.length; ri++) {
      var reward = trace.steps[ri].reward;
      if (reward !== null && reward !== undefined) {
        total += reward * Math.pow(trace.discount_factor, ri);
      }
      running.push(total);
    }

    var tmpV = new THREE.Vector3();

    return {
      steps: steps.length,

      /* Framing in the shifted frame: the rig looks at the origin, which now
         sits in the middle of the door wall at door height. The board distance
         is the prototype's, which was set so both belief signs stay in frame on
         a 16:9 stage; the top view is high and pulled back over the floor, so
         the room reads as a plan with the two cells at the far side. */
      camera: {
        board: [0, 0.95, 7.9],
        top: [0, 6.6, 5.0]
      },

      /**
       * Advance the chamber to continuous step index t.
       * @returns {Object} HUD fields for the player to display.
       */
      update: function (t, dt, elapsed, playing) {
        var sample = sampleAt(t);
        var s = sample.step;
        var i;

        // Torch flicker: two torches out of phase, each a sum of two sines
        // rather than noise, so the light breathes instead of strobing.
        for (i = 0; i < torches.length; i++) {
          var T = torches[i];
          var fl = reduceMotion ? 1 :
            1 + Math.sin(elapsed * 8.3 + T.phase) * 0.07 + Math.sin(elapsed * 3.1 + T.phase * 2) * 0.05;
          T.light.power = TORCH_LUMENS * fl;
          T.flame.scale.set(0.34 * (0.94 + fl * 0.08), 0.5 * (0.86 + fl * 0.18), 1);
          T.flame.position.x = T.x + (reduceMotion ? 0 : Math.sin(elapsed * 6.2 + T.phase) * 0.012);
          T.halo.material.opacity = 0.34 + fl * 0.1;
        }
        lanternLight.power = 520 * (reduceMotion ? 1 : 1 + Math.sin(elapsed * 6.9) * 0.08);
        if (!reduceMotion) {
          var bf = 1 + Math.sin(elapsed * 7.7) * 0.1 + Math.sin(elapsed * 2.9) * 0.06;
          wick.scale.set(0.5 * (0.95 + bf * 0.05), 0.62 * (0.84 + bf * 0.18), 1);
        }

        // The listen cue. The room dips as the agent holds still to listen, and
        // a ripple runs out from the side it heard. The dip is what makes a
        // viewer stop and look, which is the reason to stage the action at all.
        var dip = sample.cue * 0.34;
        shadowLight.power = 900 * (1 - dip);
        lanternLight.power *= (1 - dip * 0.55);

        for (var di = 0; di < doors.length; di++) {
          var D = doors[di];
          var isHeard = s.heard === di;
          D.cueLight.power = isHeard ? 240 * sample.cue : 0;

          for (var rj = 0; rj < D.rings.length; rj++) {
            var R = D.rings[rj];
            if (!isHeard) { R.material.opacity = 0; continue; }
            var phase = clamp(sample.cueT * 1.5 - rj * 0.17, 0, 1);
            if (phase <= 0 || phase >= 1) { R.material.opacity = 0; continue; }
            var rad = 0.22 + phase * 2.6;
            R.scale.set(rad, rad, 1);
            R.material.opacity = 0.85 * (1 - phase) * (1 - phase);
          }

          // The belief signs: this hypothesis's share, redrawn only when it
          // actually moves.
          var pv = sample.pLeft === null ? null : (di === 0 ? sample.pLeft : 1 - sample.pLeft);
          var key = pv === null ? -1 : pv;
          if (Math.abs(key - D.signLast) > 0.004) {
            drawSign(D, di === 0 ? "LEFT" : "RIGHT", pv);
            D.signLast = key;
          }

          // The door swing, and what is behind it.
          var opening = s.opened === di ? sample.swing : 0;
          D.hinge.rotation.y = D.sideSign * opening * 1.15;
          D.slit.material.opacity = 0.34 * (1 - opening);

          // Which prize this door hides, for the step being played. The tiger
          // moves between episodes' steps because the model re-draws it, so
          // this is read per frame rather than fixed at build time.
          var isTiger = s.tigerLeft === (di === 0);
          D.tiger.visible = isTiger && opening > 0.02;
          D.hoard.visible = !isTiger && opening > 0.02;
          /* The treasure is lit from inside its cell, because gold in the dark
             is the point of it. The tiger is lit in slices instead. */
          if (isTiger) {
            // Aimed at the face, wherever the head has turned to: a rake parked
            // on a fixed point lights whatever happens to be there.
            var faceR = room.worldToLocal(
              D.tiger.userData.faceLocal.clone()
                .applyMatrix4(D.tiger.userData.headGroup.matrixWorld)
            );
            D.rakeLight.target.position.copy(faceR);
            D.rakeLight.target.updateMatrixWorld();
            D.rakeLight.position.set(
              faceR.x - D.sideSign * 0.52, faceR.y + 0.78, faceR.z + 0.62);
            D.rakeLight.distance = 2.6;
            D.rakeLight.angle = 0.46;
            D.rakeLight.power = opening * 150;
            D.prizeLight.power = 0;
          } else {
            D.rakeLight.power = 0;
            D.prizeLight.position.set(D.doorX, 1.1, WALL_Z - 0.9);
            D.prizeLight.distance = 6.0;
            D.prizeLight.power = opening * 420;
          }
          /* On a tiger door the cell light moves low and out to the pier side.
             Left where it is it sits INSIDE the animal once it steps forward,
             lights nothing, and leaves the floor black — and feet on a floor
             nobody can see read as floating however exactly they are placed. */
          if (isTiger) {
            D.cellLight.position.set(D.doorX - D.sideSign * 0.88, 0.52, WALL_Z - 0.78);
            D.cellLight.distance = 3.2;
            D.cellLight.power = 26 + opening * 95;
          } else {
            D.cellLight.position.copy(D.cellLight.userData.home);
            D.cellLight.distance = 3.4;
            D.cellLight.power = 26 + opening * 40;
          }

          if (D.tiger.visible) {
            /* Eyes first, then shape: the leaf has to be well clear before the
               coat resolves, or the eyes and the body arrive together and the
               beat is lost. */
            var shape = smooth(0.46, 0.94, opening);
            D.tiger.userData.fur.opacity = shape;
            D.tiger.userData.pale.opacity = shape;
            D.tiger.userData.dark.opacity = shape;
            /* It comes forward only as far as the reveal, never out into the
               room. The 0.4 m of stone the doorway is cut through then does the
               shading for free. */
            var lunge = smooth(0.30, 1.0, opening) * 0.48;
            D.tiger.position.z = WALL_Z - 1.78 + lunge;
            D.tiger.position.x = D.tigerX + (D.doorX - D.tigerX) * 0.22 * smooth(0.55, 1.0, opening);
            // Eyeshine is a reflection, so it wobbles as the torches do rather
            // than pulsing.
            for (var ei = 0; ei < D.tiger.userData.eyes.length; ei++) {
              var glint = reduceMotion ? 1 : 1 + Math.sin(elapsed * 3.3 + ei * 2.1) * 0.06;
              D.tiger.userData.eyes[ei].scale.setScalar(glint);
            }
            /* Height is driven off the measured bounding box, never set by
               hand. The breathing rise is added upward only, so the feet are on
               the floor at the bottom of it rather than through it. */
            var bob = reduceMotion ? 0 : (0.5 + 0.5 * Math.sin(elapsed * 1.6)) * 0.010;
            D.tiger.position.y = CELL_FLOOR_Y + D.tiger.userData.groundOffset + bob;
            D.tiger.rotation.y = D.tigerYaw + (reduceMotion ? 0 : Math.sin(elapsed * 0.7) * 0.05);
            /* The body stays three-quarter, because that is what shows its
               length. The HEAD turns back towards the doorway — most of the way
               back, not all of it: dead-on is the one angle at which a swept
               head shows only its end. */
            var hg = D.tiger.userData.headGroup;
            hg.rotation.y = -D.tigerYaw * 0.55 + (reduceMotion ? 0 : Math.sin(elapsed * 0.55) * 0.04);
            hg.rotation.x = -0.04;

            // A contact patch under each foot, following it wherever it lands.
            D.tiger.updateMatrixWorld(true);
            for (var pw = 0; pw < D.pawShadows.length; pw++) {
              var wp = room.worldToLocal(
                D.tiger.userData.paws[pw].clone().applyMatrix4(D.tiger.matrixWorld));
              var sh = D.pawShadows[pw];
              sh.visible = true;
              sh.position.set(wp.x, CELL_FLOOR_Y + 0.012, wp.z);
              sh.material.opacity = 0.78 * shape;
            }
          } else {
            for (var pw2 = 0; pw2 < D.pawShadows.length; pw2++) {
              D.pawShadows[pw2].visible = false;
            }
          }
        }

        if (!reduceMotion && playing) {
          for (var mi = 0; mi < MOTES; mi++) {
            motePos[mi * 3] += Math.sin(elapsed * 0.25 + mi) * 0.0013;
            motePos[mi * 3 + 1] += 0.0016;
            if (motePos[mi * 3 + 1] > ROOM_H - 0.3) motePos[mi * 3 + 1] = 0.1;
          }
          moteGeo.attributes.position.needsUpdate = true;
        }

        // The ceiling hides itself once the camera is above it. The scene is
        // not told which camera mode is on, and it does not need to be: being
        // outside the room is the condition that matters.
        ceiling.visible = core.camera.position.y < ceilingTop;

        /* Chase is the core rig's one agent-shaped mode, and Tiger has no
           agent: nothing in this problem has a position or a heading. What
           stands at a door is the attention of the episode, so the follow point
           is put just inside whichever door is in play — the one being opened,
           or the one being heard — and the heading points into it. The rig
           then places itself 2.9 m back along that heading, which is where the
           prototype's "at the door" camera stood. */
        var side = s.opened !== null ? (s.opened === 0 ? -1 : 1)
          : (s.heard === 0 ? -1 : s.heard === 1 ? 1 : -1);
        var back = s.opened !== null ? 1.1 : 0;
        /* Further back than the prototype's own door camera, because the rig
           aims 0.35 m off the floor rather than at door height: from 2.35 m the
           lens then filled the frame with the bottom half of one leaf. */
        var eye = tmpV.set(
          side * (DOOR_X - 0.40 - back * 0.4), 0, WALL_Z + 5.0 + back).add(SHIFT);
        var dx = side * DOOR_X + SHIFT.x - eye.x;
        var dz = WALL_Z + SHIFT.z - eye.z;
        var len = Math.hypot(dx, dz) || 1;
        var heading = Math.atan2(dz / len, dx / len);

        var reading = readings[sample.i] || { label: "—" };
        var envelope = trace.steps[sample.i] || {};
        var left = sample.pLeft;
        return {
          follow: {
            x: eye.x + (dx / len) * 2.9,
            z: eye.z + (dz / len) * 2.9,
            heading: heading
          },
          step: sample.i,
          action: envelope.action === null || envelope.action === undefined
            ? "—" : String(envelope.action),
          /* The shared HUD's position field. Tiger has no position, so it
             carries the one number the belief is: the two hypotheses' shares,
             which always sum to one. */
          x: left === null ? 0 : left,
          y: left === null ? 0 : 1 - left,
          reward: envelope.reward,
          ret: running[sample.i],
          belief: reading.label +
            (s.opened !== null && sample.f > 0.9
              ? " — door opened: " + (s.hitTiger ? "TIGER" : "TREASURE") +
                ", side re-drawn"
              : "")
        };
      }
    };
  }

  V.scenes["tiger.v1"] = {
    build: build,
    // The torches carry real lumens, so the camera stops down the way a real
    // one would for a night interior. This single number sets the mood; it is
    // not a knob to remove.
    exposure: 0.155
  };
})(window);
