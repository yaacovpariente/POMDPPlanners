/* SPDX-License-Identifier: MIT
 *
 * The chart builder: one figure, drawn from the run's own metrics, with the
 * planners, the metric, the labels and the orientation chosen by the reader.
 *
 * The drawing is deliberately plain — black on white, one grey for the error
 * bars, no gradients or shadows — because the output is meant to go into a
 * paper, where the page is white and the figure is printed. It does not follow
 * the site's theme for the same reason: what you download is what you saw.
 */
(function () {
  "use strict";

  var dataEl = document.getElementById("chart-data");
  var output = document.getElementById("chart-output");
  if (!dataEl || !output) return;

  var DATA = JSON.parse(dataEl.textContent);
  var SVG_NS = "http://www.w3.org/2000/svg";

  var el = {
    policies: document.getElementById("chart-policies"),
    metric: document.getElementById("chart-metric"),
    title: document.getElementById("chart-title"),
    yLabel: document.getElementById("chart-y"),
    xLabel: document.getElementById("chart-x"),
    orient: document.getElementById("chart-orient"),
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

  function chosenPolicies() {
    return Array.prototype.slice
      .call(el.policies.querySelectorAll("input:checked"))
      .map(function (box) { return box.value; });
  }

  function series() {
    var metric = el.metric.value;
    var chosen = chosenPolicies();
    var rows = [];
    DATA.policies.forEach(function (policy) {
      if (chosen.indexOf(policy.name) === -1) return;
      var value = policy.metrics[metric];
      if (typeof value !== "number") return;
      var low = policy.metrics[metric + DATA.ci.lower];
      var high = policy.metrics[metric + DATA.ci.upper];
      rows.push({
        name: policy.name,
        value: value,
        low: typeof low === "number" ? low : value,
        high: typeof high === "number" ? high : value
      });
    });
    return rows;
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
      var titleWidth = (el.title.value || "").length * 8.6 + 48;
      width = Math.max(420, margin.left + margin.right + slot * rows.length, titleWidth);
    }
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    // The bars share the plot area rather than each taking a fixed slot, so a
    // chart widened to fit its title spreads its bars instead of stranding
    // them against the value axis.
    band = (vertical ? plotW : plotH) / rows.length;

    var svg = node("svg", {
      xmlns: SVG_NS,
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      "font-family": "Helvetica, Arial, sans-serif"
    });
    svg.appendChild(node("rect", { x: 0, y: 0, width: width, height: height, fill: "#ffffff" }));

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
          stroke: "#d9d9d9", "stroke-width": 1
        }));
        svg.appendChild(node("text", {
          x: margin.left - 10, y: p + 4, "text-anchor": "end",
          "font-size": 13, fill: "#333333"
        }, format(tick)));
      } else {
        svg.appendChild(node("line", {
          x1: p, y1: margin.top, x2: p, y2: margin.top + plotH,
          stroke: "#d9d9d9", "stroke-width": 1
        }));
        svg.appendChild(node("text", {
          x: p, y: margin.top + plotH + 22, "text-anchor": "middle",
          "font-size": 13, fill: "#333333"
        }, format(tick)));
      }
    });

    var zero = toValue(0);
    rows.forEach(function (row, index) {
      var centre = (vertical ? margin.left : margin.top) + band * index + band / 2;
      var thickness = Math.min(vertical ? 54 : 26, band * 0.55);
      var at = toValue(row.value);

      if (vertical) {
        svg.appendChild(node("rect", {
          x: centre - thickness / 2,
          y: Math.min(zero, at),
          width: thickness,
          height: Math.max(1, Math.abs(at - zero)),
          fill: "#4a4a4a"
        }));
      } else {
        svg.appendChild(node("rect", {
          x: Math.min(zero, at),
          y: centre - thickness / 2,
          width: Math.max(1, Math.abs(at - zero)),
          height: thickness,
          fill: "#4a4a4a"
        }));
      }

      if (withErrors && row.high > row.low) {
        var a = toValue(row.low);
        var b = toValue(row.high);
        var cap = 6;
        if (vertical) {
          svg.appendChild(node("line", { x1: centre, y1: a, x2: centre, y2: b, stroke: "#111111", "stroke-width": 1.4 }));
          svg.appendChild(node("line", { x1: centre - cap, y1: a, x2: centre + cap, y2: a, stroke: "#111111", "stroke-width": 1.4 }));
          svg.appendChild(node("line", { x1: centre - cap, y1: b, x2: centre + cap, y2: b, stroke: "#111111", "stroke-width": 1.4 }));
        } else {
          svg.appendChild(node("line", { x1: a, y1: centre, x2: b, y2: centre, stroke: "#111111", "stroke-width": 1.4 }));
          svg.appendChild(node("line", { x1: a, y1: centre - cap, x2: a, y2: centre + cap, stroke: "#111111", "stroke-width": 1.4 }));
          svg.appendChild(node("line", { x1: b, y1: centre - cap, x2: b, y2: centre + cap, stroke: "#111111", "stroke-width": 1.4 }));
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
            "text-anchor": "middle", "font-size": 13, fill: "#111111"
          }, text));
        } else {
          svg.appendChild(node("text", {
            x: row.value >= 0 ? edge + 9 : edge - 9, y: centre + 4,
            "text-anchor": row.value >= 0 ? "start" : "end",
            "font-size": 13, fill: "#111111"
          }, text));
        }
      }

      // The planner's name, along the category axis.
      if (vertical) {
        svg.appendChild(node("text", {
          x: centre, y: margin.top + plotH + 24, "text-anchor": "middle",
          "font-size": 13, fill: "#111111"
        }, row.name));
      } else {
        svg.appendChild(node("text", {
          x: margin.left - 12, y: centre + 4, "text-anchor": "end",
          "font-size": 13, fill: "#111111"
        }, row.name));
      }
    });

    // Axis lines last, so they sit over the grid.
    svg.appendChild(node("line", {
      x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH,
      stroke: "#111111", "stroke-width": 1.2
    }));
    svg.appendChild(node("line", {
      x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH,
      stroke: "#111111", "stroke-width": 1.2
    }));

    if (el.title.value) {
      svg.appendChild(node("text", {
        x: width / 2, y: 30, "text-anchor": "middle",
        "font-size": 17, "font-weight": "bold", fill: "#111111"
      }, el.title.value));
    }
    var valueLabel = el.yLabel.value;
    var categoryLabel = el.xLabel.value;
    if (vertical) {
      if (valueLabel) {
        svg.appendChild(node("text", {
          x: 18, y: margin.top + plotH / 2, "text-anchor": "middle", "font-size": 14,
          fill: "#111111", transform: "rotate(-90 18 " + (margin.top + plotH / 2) + ")"
        }, valueLabel));
      }
      if (categoryLabel) {
        svg.appendChild(node("text", {
          x: margin.left + plotW / 2, y: height - 16, "text-anchor": "middle",
          "font-size": 14, fill: "#111111"
        }, categoryLabel));
      }
    } else {
      if (valueLabel) {
        svg.appendChild(node("text", {
          x: margin.left + plotW / 2, y: height - 14, "text-anchor": "middle",
          "font-size": 14, fill: "#111111"
        }, valueLabel));
      }
      if (categoryLabel) {
        svg.appendChild(node("text", {
          x: 18, y: margin.top + plotH / 2, "text-anchor": "middle", "font-size": 14,
          fill: "#111111", transform: "rotate(-90 18 " + (margin.top + plotH / 2) + ")"
        }, categoryLabel));
      }
    }

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
      ctx.fillStyle = "#ffffff";
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
    var label = document.createElement("label");
    label.className = "check";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.value = policy.name;
    box.checked = true;
    label.appendChild(box);
    label.appendChild(document.createTextNode(" " + policy.name));
    el.policies.appendChild(label);
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
