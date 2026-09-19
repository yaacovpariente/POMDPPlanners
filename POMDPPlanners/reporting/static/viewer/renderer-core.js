/* SPDX-License-Identifier: MIT
 *
 * Shared renderer core for every POMDPPlanners episode viewer.
 *
 * Everything here is environment-agnostic: the physically-correct lighting,
 * the half-float bright/blur/ACES composite stack, the camera rig and the
 * playback clock. A scene module supplies the world and reads one sample per
 * frame; it never touches the pipeline. That split is what lets the other
 * fifteen environments reuse this file instead of forking it.
 *
 * Four things in here are tuned, not arbitrary, and each cost real time to
 * find (see VIEWER_BRIEF):
 *
 *  1. Exposure. The lamps carry real lumens, so the camera has to stop down.
 *     The composite's `exposure` uniform is the single number that sets the
 *     mood; a scene tunes it, it does not remove it.
 *  2. Colour space. The renderer is linear until the composite encodes it, so
 *     every sRGB hex must be converted on the way in. `linearize()` does that
 *     for standard materials and deliberately skips emitters, whose colours
 *     are above 1.0 and are radiance, not paint.
 *  3. Shadow acne. One narrow-coned shadow caster with `normalBias`, not a
 *     large negative `bias`.
 *  4. Depth precision. The scene render target's depth buffer is 16-bit, so
 *     `camera.near` stays at 0.6 and coplanar surfaces need `polygonOffset`.
 */
(function (global) {
  "use strict";

  var THREE = global.THREE;

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function lerp(a, b, t) { return a + (b - a) * t; }

  /** Deterministic PRNG, so a scene's scatter is identical on every load. */
  function mulberry(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  /** A soft radial sprite: light pools, belief particles, dust, contact shadows. */
  function radialTexture(inner, falloff) {
    var s = 128;
    var cv = document.createElement("canvas");
    cv.width = cv.height = s;
    var g = cv.getContext("2d");
    var grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grad.addColorStop(0, "rgba(255,255,255," + inner + ")");
    grad.addColorStop(falloff, "rgba(255,255,255," + (inner * 0.32) + ")");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, s, s);
    return new THREE.CanvasTexture(cv);
  }

  /** Sobel over a canvas's luminance. Turns a painted floor into ground. */
  function normalMapFrom(renderer, cv, strength) {
    var s = cv.width;
    var soft = document.createElement("canvas");
    soft.width = soft.height = s;
    var sctx = soft.getContext("2d");
    sctx.filter = "blur(2px)";
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

  var VERT = [
    "varying vec2 vUv;",
    "void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }"
  ].join("\n");

  var BRIGHT_FRAG = [
    "varying vec2 vUv;",
    "uniform sampler2D tex;",
    "uniform float threshold; uniform float knee;",
    "void main() {",
    "  vec3 c = texture2D(tex, vUv).rgb;",
    "  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));",
    // Soft knee, so a lamp edge ramps into the bloom instead of cutting.
    "  float w = clamp((l - threshold) / max(knee, 1e-4), 0.0, 1.0);",
    "  gl_FragColor = vec4(c * w * w, 1.0);",
    "}"
  ].join("\n");

  var BLUR_FRAG = [
    "varying vec2 vUv;",
    "uniform sampler2D tex; uniform vec2 dir; uniform vec2 texel;",
    "void main() {",
    "  vec3 sum = texture2D(tex, vUv).rgb * 0.2270270270;",
    "  vec2 o1 = dir * texel * 1.3846153846;",
    "  vec2 o2 = dir * texel * 3.2307692308;",
    "  sum += (texture2D(tex, vUv + o1).rgb + texture2D(tex, vUv - o1).rgb) * 0.3162162162;",
    "  sum += (texture2D(tex, vUv + o2).rgb + texture2D(tex, vUv - o2).rgb) * 0.0702702703;",
    "  gl_FragColor = vec4(sum, 1.0);",
    "}"
  ].join("\n");

  var COMPOSITE_FRAG = [
    "varying vec2 vUv;",
    "uniform sampler2D tex; uniform sampler2D bloom;",
    "uniform float bloomStrength; uniform float exposure; uniform float time;",
    "uniform float grain; uniform float aberration;",
    // Narkowicz's ACES fit: holds highlights without Reinhard's flat white
    // clip, and keeps warm lamps warm.
    "vec3 aces(vec3 x) {",
    "  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;",
    "  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);",
    "}",
    // A sin-based hash bands badly on software GL. This one does not.
    "float hash(vec2 p) {",
    "  vec3 p3 = fract(vec3(p.xyx) * 0.1031);",
    "  p3 += dot(p3, p3.yzx + 33.33);",
    "  return fract((p3.x + p3.y) * p3.z);",
    "}",
    "void main() {",
    "  vec2 d = vUv - 0.5;",
    "  float r2 = dot(d, d);",
    "  vec2 off = d * r2 * aberration * 4.0;",
    "  vec3 col;",
    "  col.r = texture2D(tex, vUv + off).r;",
    "  col.g = texture2D(tex, vUv).g;",
    "  col.b = texture2D(tex, vUv - off).b;",
    "  col += texture2D(bloom, vUv).rgb * bloomStrength;",
    "  col *= exposure;",
    "  col = aces(col);",
    "  col *= smoothstep(0.92, 0.18, r2);",
    "  float g = hash(vUv * 1024.0 + fract(time) * 137.0) - 0.5;",
    "  float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));",
    "  g *= 0.15 + 0.85 * smoothstep(0.0, 0.35, luma);",
    "  col += g * grain;",
    "  col = pow(max(col, 0.0), vec3(1.0 / 2.2));",
    "  gl_FragColor = vec4(col, 1.0);",
    "}"
  ].join("\n");

  /**
   * Build the renderer, the scene, the camera and the post stack.
   *
   * @param {Object} options
   * @param {HTMLCanvasElement} options.canvas  Canvas to render into.
   * @param {HTMLElement} options.stage         Element whose size the canvas follows.
   * @param {number} [options.exposure]         Composite exposure. Real lumens
   *   blow out instantly, so this is how the camera stops down.
   * @returns {Object|null} The core, or null when WebGL is unavailable.
   */
  function createCore(options) {
    var canvas = options.canvas;
    var stage = options.stage;
    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    } catch (e) {
      return null;
    }

    renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 1.75));
    // The scene renders linear into a float buffer; tone mapping, bloom and
    // grain all happen in the composite. three must not touch either on the way in.
    renderer.outputEncoding = THREE.LinearEncoding;
    renderer.toneMapping = THREE.NoToneMapping;
    // Real inverse-square falloff with lamp power in lumens. This is what
    // makes a dark scene dark: a 780 lm lamp cannot light the far corner.
    renderer.physicallyCorrectLights = true;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    var scene = new THREE.Scene();
    // A longer lens than a game camera: less edge stretch, which is most of
    // what makes a render read as a photograph. near stays at 0.6 because the
    // render target's depth buffer is only 16-bit.
    var camera = new THREE.PerspectiveCamera(36, 16 / 9, 0.6, 90);

    var quadScene = new THREE.Scene();
    var quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    var quadMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
    quadScene.add(quadMesh);

    var brightMat = new THREE.ShaderMaterial({
      uniforms: { tex: { value: null }, threshold: { value: 0.85 }, knee: { value: 0.45 } },
      vertexShader: VERT, fragmentShader: BRIGHT_FRAG
    });
    var blurMat = new THREE.ShaderMaterial({
      uniforms: {
        tex: { value: null },
        dir: { value: new THREE.Vector2(1, 0) },
        texel: { value: new THREE.Vector2(1, 1) }
      },
      vertexShader: VERT, fragmentShader: BLUR_FRAG
    });
    var compositeMat = new THREE.ShaderMaterial({
      uniforms: {
        tex: { value: null },
        bloom: { value: null },
        bloomStrength: { value: 0.55 },
        exposure: { value: options.exposure === undefined ? 0.155 : options.exposure },
        time: { value: 0 },
        grain: { value: 0.018 },
        aberration: { value: 0.0016 }
      },
      vertexShader: VERT, fragmentShader: COMPOSITE_FRAG
    });

    var sceneRT = null, brightRT = null, blurA = null, blurB = null;

    function makeTargets(w, h) {
      [sceneRT, brightRT, blurA, blurB].forEach(function (rt) { if (rt) rt.dispose(); });
      var opts = {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        // Lamps go far above 1.0 and the bright pass needs that range.
        type: THREE.HalfFloatType,
        encoding: THREE.LinearEncoding,
        depthBuffer: true,
        stencilBuffer: false
      };
      sceneRT = new THREE.WebGLRenderTarget(w, h, opts);
      var bw = Math.max(2, Math.floor(w / 2)), bh = Math.max(2, Math.floor(h / 2));
      var half = Object.assign({}, opts, { depthBuffer: false });
      brightRT = new THREE.WebGLRenderTarget(bw, bh, half);
      blurA = new THREE.WebGLRenderTarget(bw, bh, half);
      blurB = new THREE.WebGLRenderTarget(bw, bh, half);
      blurMat.uniforms.texel.value.set(1 / bw, 1 / bh);
    }

    function blit(material, target) {
      quadMesh.material = material;
      renderer.setRenderTarget(target || null);
      renderer.clear();
      renderer.render(quadScene, quadCam);
    }

    function renderComposite(elapsed) {
      renderer.setRenderTarget(sceneRT);
      renderer.clear();
      renderer.render(scene, camera);

      brightMat.uniforms.tex.value = sceneRT.texture;
      blit(brightMat, brightRT);

      var src = brightRT;
      for (var i = 0; i < 2; i++) {
        blurMat.uniforms.tex.value = src.texture;
        blurMat.uniforms.dir.value.set(1, 0);
        blit(blurMat, blurA);
        blurMat.uniforms.tex.value = blurA.texture;
        blurMat.uniforms.dir.value.set(0, 1);
        blit(blurMat, blurB);
        src = blurB;
      }

      compositeMat.uniforms.tex.value = sceneRT.texture;
      compositeMat.uniforms.bloom.value = blurB.texture;
      compositeMat.uniforms.time.value = elapsed;
      blit(compositeMat, null);
    }

    function resize() {
      var w = stage.clientWidth, h = stage.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      var pr = renderer.getPixelRatio();
      makeTargets(Math.floor(w * pr), Math.floor(h * pr));
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }

    /* Every PBR surface is written with an sRGB hex, which is how a person
       reads a colour, but the shader needs linear. Emitters are skipped on
       purpose: their colours are already above 1.0 and are radiance. */
    function linearize() {
      scene.traverse(function (obj) {
        var mats = obj.material ? (Array.isArray(obj.material) ? obj.material : [obj.material]) : [];
        mats.forEach(function (m) {
          if (!m || m.__linearized || !m.isMeshStandardMaterial) return;
          m.__linearized = true;
          if (m.color) m.color.convertSRGBToLinear();
          if (m.emissive) m.emissive.convertSRGBToLinear();
        });
      });
    }

    /* A night sky to reflect. Without an environment map, PBR metal has
       nothing to mirror and reads as flat grey plastic. */
    function buildNightEnvironment(stops) {
      var cv = document.createElement("canvas");
      cv.width = 256; cv.height = 128;
      var g = cv.getContext("2d");
      var grad = g.createLinearGradient(0, 0, 0, 128);
      (stops || [[0.0, "#0A1020"], [0.45, "#141A28"], [0.55, "#2A2418"], [1.0, "#060505"]])
        .forEach(function (s) { grad.addColorStop(s[0], s[1]); });
      g.fillStyle = grad;
      g.fillRect(0, 0, 256, 128);
      var rnd = mulberry(99);
      for (var i = 0; i < 260; i++) {
        g.fillStyle = "rgba(200,214,255," + (0.2 + rnd() * 0.7).toFixed(2) + ")";
        g.fillRect(rnd() * 256, rnd() * 58, 1, 1);
      }
      var tex = new THREE.CanvasTexture(cv);
      tex.mapping = THREE.EquirectangularReflectionMapping;
      tex.encoding = THREE.sRGBEncoding;
      var pmrem = new THREE.PMREMGenerator(renderer);
      pmrem.compileEquirectangularShader();
      scene.environment = pmrem.fromEquirectangular(tex).texture;
      pmrem.dispose();
      tex.dispose();
    }

    return {
      THREE: THREE,
      renderer: renderer,
      scene: scene,
      camera: camera,
      composite: compositeMat,
      renderComposite: renderComposite,
      resize: resize,
      linearize: linearize,
      buildNightEnvironment: buildNightEnvironment
    };
  }

  /**
   * The camera rig: Board / Top-down / Chase / Orbit, plus drag and wheel.
   *
   * @param {Object} core     A core from createCore.
   * @param {HTMLElement} canvas
   * @param {Object} [config] Board and top-down distances, in world units.
   */
  function createCameraRig(core, canvas, config) {
    var THREE = core.THREE;
    var cfg = config || {};
    var boardPos = cfg.board || [0, 12.9, 11.8];
    var topPos = cfg.top || [0, 16.0, 0.01];
    var state = { mode: "board", orbit: { theta: 0, phi: 0.83, dist: 15.6 } };
    var camPos = new THREE.Vector3(boardPos[0], boardPos[1], boardPos[2]);
    var camLook = new THREE.Vector3(0, 0, 0);
    var tmp = new THREE.Vector3();

    function boardCamera() {
      // The 36 mm lens needs more distance when the stage is taller than wide.
      var k = core.camera.aspect < 1 ? 1.34 : 1.0;
      return new THREE.Vector3(boardPos[0], boardPos[1] * k, boardPos[2] * k);
    }
    function topCamera() {
      var k = core.camera.aspect < 1 ? 1.3 : 1.0;
      return new THREE.Vector3(topPos[0], topPos[1] * k, topPos[2]);
    }

    var dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener("pointerdown", function (e) {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
      // Grabbing the scene is a request to orbit it.
      if (state.mode !== "orbit") rig.setMode("orbit");
    });
    canvas.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      state.orbit.theta -= (e.clientX - lastX) * 0.006;
      state.orbit.phi = clamp(state.orbit.phi + (e.clientY - lastY) * 0.005, 0.16, 1.45);
      lastX = e.clientX; lastY = e.clientY;
    });
    canvas.addEventListener("pointerup", function (e) {
      dragging = false;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener("wheel", function (e) {
      if (state.mode !== "orbit") return;
      e.preventDefault();
      state.orbit.dist = clamp(state.orbit.dist + e.deltaY * 0.012, 5, 26);
    }, { passive: false });

    var rig = {
      state: state,
      setMode: function (mode) {
        state.mode = mode;
        if (rig.onModeChange) rig.onModeChange(mode);
      },
      /**
       * Move the camera one frame.
       * @param {number} dt      Seconds since the last frame.
       * @param {Object} follow  {x, z, heading} of whatever chase follows.
       */
      update: function (dt, follow) {
        if (state.mode === "board") {
          camPos.lerp(boardCamera(), clamp(dt * 3, 0, 1));
          camLook.lerp(new THREE.Vector3(0, 0.3, 0), clamp(dt * 3, 0, 1));
        } else if (state.mode === "overhead") {
          camPos.lerp(topCamera(), clamp(dt * 3, 0, 1));
          camLook.lerp(new THREE.Vector3(0, 0, 0), clamp(dt * 3, 0, 1));
        } else if (state.mode === "chase" && follow) {
          tmp.set(Math.cos(follow.heading), 0, Math.sin(follow.heading));
          camPos.lerp(
            new THREE.Vector3(follow.x - tmp.x * 2.9, 1.75, follow.z - tmp.z * 2.9),
            clamp(dt * 3.2, 0, 1)
          );
          camLook.lerp(
            new THREE.Vector3(follow.x + tmp.x * 1.6, 0.35, follow.z + tmp.z * 1.6),
            clamp(dt * 4.5, 0, 1)
          );
        } else {
          var o = state.orbit;
          camPos.lerp(new THREE.Vector3(
            Math.sin(o.theta) * Math.cos(o.phi) * o.dist,
            Math.sin(o.phi) * o.dist,
            Math.cos(o.theta) * Math.cos(o.phi) * o.dist
          ), clamp(dt * 4, 0, 1));
          camLook.lerp(new THREE.Vector3(0, 0.4, 0), clamp(dt * 4, 0, 1));
        }
        core.camera.position.copy(camPos);
        core.camera.lookAt(camLook);
      }
    };
    return rig;
  }

  /**
   * The playback clock.
   *
   * It owns `t`, a continuous step index, and the transport state. It does not
   * own the scene: every frame it hands `t` back to `onFrame`. Note that it
   * always drives a requestAnimationFrame loop — a viewer that only renders on
   * demand looks perfect in a still and ships frozen.
   *
   * @param {Object} options
   * @param {number} options.steps        Number of steps in the trace.
   * @param {Function} options.onFrame    (t, dt, elapsed) => void
   * @param {number} [options.stepsPerSecond]
   */
  function createPlayer(options) {
    var reduceMotion = global.matchMedia
      ? global.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;
    var state = {
      t: 0,
      steps: options.steps,
      playing: !reduceMotion,
      speed: 1,
      rate: options.stepsPerSecond || 1.6
    };
    var last = (global.performance || Date).now();
    var running = false;

    function frame(now) {
      var dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      var elapsed = now / 1000;
      if (state.playing && state.steps > 1) {
        state.t += dt * state.rate * state.speed;
        if (state.t >= state.steps - 1) {
          state.t = state.steps - 1;
          state.playing = false;
          if (player.onEnd) player.onEnd();
        }
      }
      options.onFrame(state.t, dt, elapsed);
      global.requestAnimationFrame(frame);
    }

    var player = {
      state: state,
      play: function () {
        if (state.t >= state.steps - 1) state.t = 0;
        state.playing = true;
      },
      pause: function () { state.playing = false; },
      toggle: function () { if (state.playing) player.pause(); else player.play(); },
      seek: function (t) { state.t = clamp(t, 0, state.steps - 1); },
      setSpeed: function (speed) { state.speed = speed; },
      start: function () {
        if (running) return;
        running = true;
        last = (global.performance || Date).now();
        global.requestAnimationFrame(frame);
      },
      reduceMotion: reduceMotion
    };
    return player;
  }

  global.POMDPViewer = {
    clamp: clamp,
    lerp: lerp,
    mulberry: mulberry,
    radialTexture: radialTexture,
    normalMapFrom: normalMapFrom,
    createCore: createCore,
    createCameraRig: createCameraRig,
    createPlayer: createPlayer,
    scenes: {}
  };
})(window);
