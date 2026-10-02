/* SPDX-License-Identifier: MIT
 *
 * The chart builder: one figure, drawn from the run's own metrics, with the
 * planners, the metric, the labels and the orientation chosen by the reader.
 *
 * The drawing follows a chosen style rather than the site's theme, because the
 * figure is going somewhere else — a journal page, a colour figure, a slide —
 * and what is on screen has to be what downloads.
 *
 * A builder mounts into its own container and finds its controls by role
 * within it, never by page-wide id, so a page can hold several -- one per
 * environment on a study page -- without one driving another's figure.
 */
(function (global) {
  "use strict";

  // The looks, helpers and downloads every figure builder shares.
  var kit = global.POMDPFigure;
  var STYLES = kit.STYLES;
  var niceTicks = kit.niceTicks;
  var node = kit.node;
  var format = kit.format;
  var humanize = kit.humanize;
  var SVG_NS = kit.SVG_NS;

  /* Mount one builder into ``root``. Returns ``{select}``, which ticks the
     given planners and picks a metric, for a page that wants the builder to
     start from what the reader was looking at. */
  function mount(root) {
    var dataEl = root.querySelector('script[data-role="data"]');
    var output = root.querySelector('[data-role="output"]');
    if (!dataEl || !output) return null;
    var DATA = JSON.parse(dataEl.textContent);

    function role(name) {
      return root.querySelector('[data-role="' + name + '"]');
    }

    function style() {
      return STYLES[el.style.value] || STYLES.mono;
    }

    var el = {
      policies: role("policies"),
      metric: role("metric"),
      title: role("title"),
      yLabel: role("y"),
      xLabel: role("x"),
      orient: role("orient"),
      style: role("style"),
      errors: role("errors"),
      values: role("values"),
      svgButton: role("svg"),
      pngButton: role("png"),
      form: role("form")
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
      return kit.fileName(el.title.value || el.metric.value || "chart", extension);
    }

    el.svgButton.addEventListener("click", function () {
      kit.downloadSvg(output, fileName("svg"));
    });

    el.pngButton.addEventListener("click", function () {
      kit.downloadPng(output, style().background, fileName("png"));
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

    /* Start from a given set of planners and a metric; anything not given is
       left as it is. */
    function select(planners, metric) {
      if (planners && planners.length) {
        rows().forEach(function (row) {
          row.querySelector("input[type=checkbox]").checked =
            planners.indexOf(row.getAttribute("data-policy")) !== -1;
        });
      }
      if (metric && names.indexOf(metric) !== -1 && el.metric.value !== metric) {
        el.metric.value = metric;
        syncLabels();
      }
      draw();
    }

    syncLabels();
    el.xLabel.value = "Planner";

    el.metric.addEventListener("change", function () { syncLabels(); draw(); });
    el.form.addEventListener("input", draw);
    el.form.addEventListener("change", draw);

    draw();
    return { select: select };
  }

  global.POMDPChartBuilder = { mount: mount };

  /* Every builder on the page mounts itself. A standalone builder page reads
     its starting point from the address: ?planners=a,b ticks only those
     planners and ?metric=x picks the metric. */
  document.querySelectorAll("[data-chart-builder]").forEach(function (root) {
    var builder = mount(root);
    if (!builder) return;
    root.chartBuilder = builder;
    if (root.hasAttribute("data-read-query")) {
      var query = new URLSearchParams(global.location.search);
      builder.select(
        query.get("planners") ? query.get("planners").split(",") : null,
        query.get("metric")
      );
    }
  });
})(window);
