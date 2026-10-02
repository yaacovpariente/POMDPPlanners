/* SPDX-License-Identifier: MIT
 *
 * What every figure builder on the site shares: the three looks, the SVG
 * helpers, number formatting, axis ticks, and the SVG and PNG downloads.
 *
 * One copy, so a figure from the planner comparison and one from a tuning
 * study look like they came from the same paper, and so "what is on screen
 * is what downloads" is implemented once.
 */
(function (global) {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  /* Three looks, because a figure goes to three places. Mono is what most
     journals want and prints safely in black and white; colour uses the
     Okabe–Ito palette, which stays distinguishable under every common form of
     colour blindness; slide is the same drawing on a dark ground for a talk.
     Whatever is on screen is what downloads. */
  var STYLES = {
    mono: {
      label: "Paper, mono",
      background: "#ffffff",
      ink: "#111111",
      muted: "#555555",
      grid: "#e2e2e2",
      bars: ["#4a4a4a", "#8c8c8c", "#2b2b2b", "#bdbdbd", "#6e6e6e"],
      error: "#111111"
    },
    colour: {
      label: "Paper, colour",
      background: "#ffffff",
      ink: "#111111",
      muted: "#555555",
      grid: "#e6e6e6",
      bars: ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"],
      error: "#222222"
    },
    slide: {
      label: "Slide, dark",
      background: "#14161c",
      ink: "#f4f2ee",
      muted: "#b3aea6",
      grid: "#2c303a",
      bars: ["#5BB8F5", "#FF9B54", "#5FD2A6", "#E888B8", "#F2C14E", "#9D8DF1"],
      error: "#f4f2ee"
    }
  };

  function niceTicks(low, high, count) {
    var span = high - low;
    if (span <= 0) return [low];
    var raw = span / count;
    var magnitude = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var step = magnitude;
    [1, 2, 2.5, 5, 10].some(function (factor) {
      if (magnitude * factor >= raw) { step = magnitude * factor; return true; }
      return false;
    });
    var ticks = [];
    for (var t = Math.ceil(low / step) * step; t <= high + step * 1e-9; t += step) {
      // Floating point leaves 0.30000000000000004 on an axis otherwise.
      ticks.push(Math.round(t / step) * step);
    }
    return ticks;
  }

  function node(name, attrs, text) {
    var element = document.createElementNS(SVG_NS, name);
    Object.keys(attrs).forEach(function (key) {
      element.setAttribute(key, attrs[key]);
    });
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function format(value) {
    var abs = Math.abs(value);
    if (abs === 0) return "0";
    if (abs < 0.001 || abs >= 100000) return value.toExponential(2);
    // Three significant figures, with the trailing zeros a paper does not want.
    return String(parseFloat(value.toPrecision(3)));
  }

  function humanize(name) {
    var words = name.replace(/_/g, " ").trim();
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  function fileName(base, extension) {
    var cleaned = (base || "chart").replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
    return (cleaned || "chart") + "." + extension;
  }

  function save(blob, name) {
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    // Revoked on the next turn of the event loop: revoking immediately races
    // the download in some browsers.
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function svgText(output) {
    var svg = output.querySelector("svg");
    if (!svg) return null;
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(svg);
  }

  function downloadSvg(output, name) {
    var text = svgText(output);
    if (!text) return;
    save(new Blob([text], { type: "image/svg+xml" }), name);
  }

  function downloadText(text, name, type) {
    save(new Blob([text], { type: type || "text/plain" }), name);
  }

  /* One CSV field: quoted when it holds a comma, a quote or a line break. */
  function csvField(value) {
    if (value === null || value === undefined) return "";
    var text = String(value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function downloadPng(output, background, name) {
    var svg = output.querySelector("svg");
    var text = svgText(output);
    if (!svg || !text) return;
    // Three times the drawing size: a figure placed at column width in a paper
    // is printed at about 300 dpi, and a screen-resolution PNG looks soft there.
    var scale = 3;
    var width = Number(svg.getAttribute("width"));
    var height = Number(svg.getAttribute("height"));
    var image = new Image();
    image.onload = function () {
      var canvas = document.createElement("canvas");
      canvas.width = width * scale;
      canvas.height = height * scale;
      var ctx = canvas.getContext("2d");
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(function (blob) {
        if (blob) save(blob, name);
      }, "image/png");
    };
    image.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(text)));
  }

  global.POMDPFigure = {
    SVG_NS: SVG_NS,
    STYLES: STYLES,
    niceTicks: niceTicks,
    node: node,
    format: format,
    humanize: humanize,
    fileName: fileName,
    downloadSvg: downloadSvg,
    downloadPng: downloadPng,
    downloadText: downloadText,
    csvField: csvField
  };
})(window);
