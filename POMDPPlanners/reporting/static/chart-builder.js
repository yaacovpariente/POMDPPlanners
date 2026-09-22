/* SPDX-License-Identifier: MIT
 *
 * The chart builder: one figure, drawn from the run's own metrics, with the
 * planners, the metric, the labels and the orientation chosen by the reader.
 *
 * The drawing follows a chosen style rather than the site's theme, because the
 * figure is going somewhere else — a journal page, a colour figure, a slide —
 * and what is on screen has to be what downloads.
 */
(function () {
  "use strict";

  var dataEl = document.getElementById("chart-data");
  var output = document.getElementById("chart-output");
  if (!dataEl || !output) return;

  var DATA = JSON.parse(dataEl.textContent);
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

  function style() {
    return STYLES[el.style.value] || STYLES.mono;
  }

  var el = {
    policies: document.getElementById("chart-policies"),
    metric: document.getElementById("chart-metric"),
    title: document.getElementById("chart-title"),
    yLabel: document.getElementById("chart-y"),
    xLabel: document.getElementById("chart-x"),
    orient: document.getElementById("chart-orient"),
    style: document.getElementById("chart-style"),
    errors: document.getElementById("chart-errors"),
    values: document.getElementById("chart-values"),
    svgButton: document.getElementById("chart-svg"),
    pngButton: document.getElementById("chart-png")
  };

  /* Metric names are shared across planners; a metric only one planner logged
     is still offered, because plotting that one planner is legitimate. */
  function metricNames() {
    var seen = {};
    DATA.policies.forEach(function (policy) {
      Object.keys(policy.metrics).forEach(function (name) {
        if (name.indexOf(DATA.ci.lower, name.length - DATA.ci.lower.length) !== -1) return;
        if (name.indexOf(DATA.ci.upper, name.length - DATA.ci.upper.length) !== -1) return;
        if (name.indexOf("_ci_width", name.length - 9) !== -1) return;
        seen[name] = true;
      });
    });
    return Object.keys(seen).sort();
  }

  function humanize(name) {
    var words = name.replace(/_/g, " ").trim();
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  /* Each planner's row in the controls: whether it is plotted, and what it is
     called in the figure. A run's planner names are identifiers — PFT_DPW_1s —
     and a paper wants "PFT-DPW (1 s)", so the label is the author's to write
     while the data stays keyed by the name the run logged. */
  function rows() {
    return Array.prototype.slice.call(el.policies.querySelectorAll("[data-policy]"));
  }

  function series() {
    var metric = el.metric.value;
    var found = [];
    rows().forEach(function (row) {
      var box = row.querySelector("input[type=checkbox]");
      if (!box.checked) return;
      var policy = DATA.policies.filter(function (p) { return p.name === box.value; })[0];
      if (!policy) return;
      var value = policy.metrics[metric];
      if (typeof value !== "number") return;
      var low = policy.metrics[metric + DATA.ci.lower];
      var high = policy.metrics[metric + DATA.ci.upper];
      var label = row.querySelector("input[type=text]").value.trim();
      found.push({
        name: label || policy.name,
        value: value,
        low: typeof low === "number" ? low : value,
        high: typeof high === "number" ? high : value
      });
    });
    return found;
  }

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

  function draw() {
    var rows = series();
    output.textContent = "";
    if (!rows.length) {
      var empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "Pick at least one planner that logged this metric.";
      output.appendChild(empty);
      return null;
    }

    var vertical = el.orient.value === "vertical";
    var withErrors = el.errors.checked;
    var withValues = el.values.checked;

    var spread = [];
    rows.forEach(function (row) {
      spread.push(row.value);
      if (withErrors) { spread.push(row.low); spread.push(row.high); }
    });
    spread.push(0);
    var low = Math.min.apply(null, spread);
    var high = Math.max.apply(null, spread);
    if (low === high) { low -= 1; high += 1; }
    // Enough headroom that a whisker, and the value printed past it, stay
    // inside the plot instead of landing on the axis line.
    var pad = (high - low) * 0.14;
    low -= pad;
    high += pad;

    var margin = vertical
      ? { left: 86, right: 32, top: 56, bottom: 80 }
      : { left: 190, right: 72, top: 56, bottom: 64 };
    // A slot per bar, before the plot area is known; the real band comes from
    // the plot area once the drawing has been sized.
    var slot = vertical ? 96 : 54;
    var band;
    var width = 620;
    var height = vertical ? 400 : margin.top + margin.bottom + slot * rows.length;
    if (vertical) {
      // Wide enough for the bars, and for the title, which is the reader's own
      // text: sized to the bars alone it was clipped at both ends.
      // A rough advance width for 18px Helvetica at 600 weight, with the
      // letter-spacing: too small and the title is clipped at both ends.
      var titleWidth = (el.title.value || "").length * 10.4 + 56;
      width = Math.max(420, margin.left + margin.right + slot * rows.length, titleWidth);
    }
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    // The bars share the plot area rather than each taking a fixed slot, so a
    // chart widened to fit its title spreads its bars instead of stranding
    // them against the value axis.
    band = (vertical ? plotW : plotH) / rows.length;

    var look = style();
    var svg = node("svg", {
      xmlns: SVG_NS,
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      "font-family": "Helvetica Neue, Helvetica, Arial, sans-serif"
    });
    svg.appendChild(node("rect", {
      x: 0, y: 0, width: width, height: height, fill: look.background
    }));

    function toValue(v) {
      var ratio = (v - low) / (high - low);
      return vertical
        ? margin.top + plotH - ratio * plotH
        : margin.left + ratio * plotW;
    }

    var ticks = niceTicks(low, high, 6);
    ticks.forEach(function (tick) {
      var p = toValue(tick);
      if (vertical) {
        svg.appendChild(node("line", {
          x1: margin.left, y1: p, x2: margin.left + plotW, y2: p,
          stroke: look.grid, "stroke-width": 1, "stroke-dasharray": "3 4"
        }));
        svg.appendChild(node("text", {
          x: margin.left - 12, y: p + 4, "text-anchor": "end",
          "font-size": 12.5, fill: look.muted
        }, format(tick)));
      } else {
        svg.appendChild(node("line", {
          x1: p, y1: margin.top, x2: p, y2: margin.top + plotH,
          stroke: look.grid, "stroke-width": 1, "stroke-dasharray": "3 4"
        }));
        svg.appendChild(node("text", {
          x: p, y: margin.top + plotH + 22, "text-anchor": "middle",
          "font-size": 12.5, fill: look.muted
        }, format(tick)));
      }
    });

    var zero = toValue(0);
    rows.forEach(function (row, index) {
      var centre = (vertical ? margin.left : margin.top) + band * index + band / 2;
      var thickness = Math.min(vertical ? 54 : 26, band * 0.55);
      var at = toValue(row.value);

      var fill = look.bars[index % look.bars.length];
      if (vertical) {
        svg.appendChild(node("rect", {
          x: centre - thickness / 2,
          y: Math.min(zero, at),
          width: thickness,
          height: Math.max(1, Math.abs(at - zero)),
          rx: 3,
          fill: fill
        }));
      } else {
        svg.appendChild(node("rect", {
          x: Math.min(zero, at),
          y: centre - thickness / 2,
          width: Math.max(1, Math.abs(at - zero)),
          height: thickness,
          rx: 3,
          fill: fill
        }));
      }

      if (withErrors && row.high > row.low) {
        var a = toValue(row.low);
        var b = toValue(row.high);
        var cap = 6;
        if (vertical) {
          svg.appendChild(node("line", { x1: centre, y1: a, x2: centre, y2: b, stroke: look.error, "stroke-width": 1.4 }));
          svg.appendChild(node("line", { x1: centre - cap, y1: a, x2: centre + cap, y2: a, stroke: look.error, "stroke-width": 1.4 }));
          svg.appendChild(node("line", { x1: centre - cap, y1: b, x2: centre + cap, y2: b, stroke: look.error, "stroke-width": 1.4 }));
        } else {
          svg.appendChild(node("line", { x1: a, y1: centre, x2: b, y2: centre, stroke: look.error, "stroke-width": 1.4 }));
          svg.appendChild(node("line", { x1: a, y1: centre - cap, x2: a, y2: centre + cap, stroke: look.error, "stroke-width": 1.4 }));
          svg.appendChild(node("line", { x1: b, y1: centre - cap, x2: b, y2: centre + cap, stroke: look.error, "stroke-width": 1.4 }));
        }
      }

      if (withValues) {
        // Placed past everything the bar draws, on the side away from zero, so
        // it never lands on the bar or on its interval whisker.
        var reach = row.value;
        if (withErrors && row.high > row.low) {
          reach = row.value >= 0 ? Math.max(row.high, row.value) : Math.min(row.low, row.value);
        }
        var edge = toValue(reach);
        var text = format(row.value);
        if (vertical) {
          svg.appendChild(node("text", {
            x: centre, y: row.value >= 0 ? edge - 9 : edge + 19,
            "text-anchor": "middle", "font-size": 13, "font-weight": "600", fill: look.ink
          }, text));
        } else {
          svg.appendChild(node("text", {
            x: row.value >= 0 ? edge + 9 : edge - 9, y: centre + 4,
            "text-anchor": row.value >= 0 ? "start" : "end",
            "font-size": 13, "font-weight": "600", fill: look.ink
          }, text));
        }
      }

      // The planner's name, along the category axis.
      if (vertical) {
        svg.appendChild(node("text", {
          x: centre, y: margin.top + plotH + 25, "text-anchor": "middle",
          "font-size": 13.5, fill: look.ink
        }, row.name));
      } else {
        svg.appendChild(node("text", {
          x: margin.left - 14, y: centre + 4, "text-anchor": "end",
          "font-size": 13.5, fill: look.ink
        }, row.name));
      }
    });

    // Axis lines last, so they sit over the grid.
    svg.appendChild(node("line", {
      x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH,
      stroke: look.ink, "stroke-width": 1, opacity: 0.55
    }));
    svg.appendChild(node("line", {
      x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH,
      stroke: look.ink, "stroke-width": 1, opacity: 0.55
    }));

    if (el.title.value) {
      svg.appendChild(node("text", {
        x: width / 2, y: 32, "text-anchor": "middle",
        "font-size": 18, "font-weight": "600", "letter-spacing": "0.2",
        fill: look.ink
      }, el.title.value));
    }
    var valueLabel = el.yLabel.value;
    var categoryLabel = el.xLabel.value;
    if (vertical) {
      if (valueLabel) {
        svg.appendChild(node("text", {
          x: 18, y: margin.top + plotH / 2, "text-anchor": "middle", "font-size": 14,
          fill: look.muted, transform: "rotate(-90 18 " + (margin.top + plotH / 2) + ")"
        }, valueLabel));
      }
      if (categoryLabel) {
        svg.appendChild(node("text", {
          x: margin.left + plotW / 2, y: height - 16, "text-anchor": "middle",
          "font-size": 14, fill: look.muted
        }, categoryLabel));
      }
    } else {
      if (valueLabel) {
        svg.appendChild(node("text", {
          x: margin.left + plotW / 2, y: height - 14, "text-anchor": "middle",
          "font-size": 14, fill: look.muted
        }, valueLabel));
      }
      if (categoryLabel) {
        svg.appendChild(node("text", {
          x: 18, y: margin.top + plotH / 2, "text-anchor": "middle", "font-size": 14,
          fill: look.muted, transform: "rotate(-90 18 " + (margin.top + plotH / 2) + ")"
        }, categoryLabel));
      }
    }

    output.style.background = look.background;
    output.appendChild(svg);
    return svg;
  }

  function fileName(extension) {
    var base = (el.title.value || el.metric.value || "chart")
      .replace(/[^A-Za-z0-9_-]+/g, "_")
      .replace(/^_+|_+$/g, "");
    return (base || "chart") + "." + extension;
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

  function svgText() {
    var svg = output.querySelector("svg");
    if (!svg) return null;
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(svg);
  }

  el.svgButton.addEventListener("click", function () {
    var text = svgText();
    if (!text) return;
    save(new Blob([text], { type: "image/svg+xml" }), fileName("svg"));
  });

  el.pngButton.addEventListener("click", function () {
    var svg = output.querySelector("svg");
    var text = svgText();
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
      ctx.fillStyle = style().background;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(function (blob) {
        if (blob) save(blob, fileName("png"));
      }, "image/png");
    };
    image.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(text)));
  });

  // Build the controls, then draw whenever any of them changes.
  DATA.policies.forEach(function (policy) {
    var row = document.createElement("div");
    row.className = "policy-row";
    row.setAttribute("data-policy", policy.name);

    var toggle = document.createElement("label");
    toggle.className = "check";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.value = policy.name;
    box.checked = true;
    toggle.appendChild(box);
    toggle.appendChild(document.createTextNode(" " + policy.name));

    var rename = document.createElement("input");
    rename.type = "text";
    rename.value = policy.name;
    rename.autocomplete = "off";
    rename.setAttribute("aria-label", "Label for " + policy.name + " in the figure");

    row.appendChild(toggle);
    row.appendChild(rename);
    el.policies.appendChild(row);
  });

  var names = metricNames();
  names.forEach(function (name) {
    var option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    el.metric.appendChild(option);
  });
  if (names.indexOf("average_return") !== -1) el.metric.value = "average_return";

  function syncLabels() {
    el.yLabel.value = humanize(el.metric.value);
    el.title.value = humanize(el.metric.value) + " on " + DATA.environment;
  }
  syncLabels();
  el.xLabel.value = "Planner";

  el.metric.addEventListener("change", function () { syncLabels(); draw(); });
  document.getElementById("chart-form").addEventListener("input", draw);
  document.getElementById("chart-form").addEventListener("change", draw);

  draw();
})();
