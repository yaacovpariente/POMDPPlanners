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
      form: role("form"),
      mode: role("mode"),
      field: role("field"),
      kind: role("kind"),
      bins: role("bins"),
      layout: role("layout"),
      csvButton: role("csv")
    };
    var episodesUrl = root.getAttribute("data-episodes-url");
    // Per-episode records, fetched the first time that mode is chosen.
    var EPISODES = null;

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
      if (el.mode && el.mode.value === "episodes") return drawEpisodes();
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

    /* ---- Per episode: the distribution behind a logged mean. ---- */

    function quantile(sorted, q) {
      if (!sorted.length) return NaN;
      var position = (sorted.length - 1) * q;
      var low = Math.floor(position);
      var high = Math.ceil(position);
      return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
    }

    /* The ticked planners' values of the chosen per-episode field. */
    function episodeSeries() {
      var field = el.field.value;
      var found = [];
      rows().forEach(function (row) {
        var box = row.querySelector("input[type=checkbox]");
        if (!box.checked || !EPISODES) return;
        var records = EPISODES.policies[box.value] || [];
        var values = records.map(function (r) { return r[field]; }).filter(function (v) {
          return typeof v === "number" && isFinite(v);
        });
        if (!values.length) return;
        var label = row.querySelector("input[type=text]").value.trim();
        found.push({ name: label || box.value, values: values });
      });
      return found;
    }

    function frame(width, height) {
      var look = style();
      var svg = node("svg", {
        xmlns: SVG_NS,
        viewBox: "0 0 " + width + " " + height,
        width: width,
        height: height,
        "font-family": "Helvetica Neue, Helvetica, Arial, sans-serif"
      });
      svg.appendChild(node("rect", { x: 0, y: 0, width: width, height: height, fill: look.background }));
      return svg;
    }

    function drawEpisodes() {
      output.textContent = "";
      if (!EPISODES) {
        output.textContent = "Loading the episodes…";
        return null;
      }
      var series = episodeSeries();
      if (!series.length) {
        var empty = document.createElement("p");
        empty.className = "empty";
        empty.textContent = "Pick at least one planner with episodes that recorded this value.";
        output.appendChild(empty);
        return null;
      }
      var look = style();
      var kind = el.kind.value;
      var width = 640;
      var height = 420;
      var margin = { left: 86, right: 28, top: 58, bottom: 74 };
      var plotW = width - margin.left - margin.right;
      var plotH = height - margin.top - margin.bottom;
      var svg = frame(width, height);
      var pooled = [];
      series.forEach(function (s) { pooled = pooled.concat(s.values); });
      var lo = Math.min.apply(null, pooled);
      var hi = Math.max.apply(null, pooled);
      if (lo === hi) { lo -= 1; hi += 1; }

      function hGrid(yLow, yHigh, toY, labels) {
        kit.niceTicks(yLow, yHigh, 6).forEach(function (tick) {
          if (tick < yLow - 1e-9 || tick > yHigh + 1e-9) return;
          var y = toY(tick);
          svg.appendChild(node("line", {
            x1: margin.left, y1: y, x2: margin.left + plotW, y2: y,
            stroke: look.grid, "stroke-width": 1, "stroke-dasharray": "3 4"
          }));
          if (labels !== false) {
            svg.appendChild(node("text", {
              x: margin.left - 12, y: y + 4, "text-anchor": "end", "font-size": 12.5, fill: look.muted
            }, format(tick)));
          }
        });
      }

      function xTicks(xLow, xHigh, toX) {
        kit.niceTicks(xLow, xHigh, 6).forEach(function (tick) {
          if (tick < xLow - 1e-9 || tick > xHigh + 1e-9) return;
          svg.appendChild(node("text", {
            x: toX(tick), y: margin.top + plotH + 22, "text-anchor": "middle",
            "font-size": 12.5, fill: look.muted
          }, format(tick)));
        });
      }

      var legend = false;
      if (kind === "histogram") {
        var bins = Math.max(1, Math.min(100, parseInt(el.bins.value, 10) || 8));
        var step = (hi - lo) / bins;
        var counts = series.map(function (s) {
          var c = [];
          for (var i = 0; i < bins; i++) c.push(0);
          s.values.forEach(function (v) {
            c[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / step)))] += 1;
          });
          return c;
        });
        var tallest = Math.max.apply(null, counts.map(function (c) { return Math.max.apply(null, c); }));
        var toY = function (v) { return margin.top + plotH - (v / (tallest * 1.1)) * plotH; };
        var toX = function (v) { return margin.left + (v - lo) / (hi - lo) * plotW; };
        hGrid(0, tallest * 1.1, toY);
        var side = el.layout.value === "side";
        counts.forEach(function (c, index) {
          c.forEach(function (count, bin) {
            if (!count) return;
            var x0 = toX(lo + bin * step);
            var w = toX(lo + (bin + 1) * step) - x0;
            var x = side ? x0 + (w / series.length) * index : x0;
            var barW = side ? w / series.length : w;
            svg.appendChild(node("rect", {
              x: x + 1, y: toY(count), width: Math.max(1, barW - 2),
              height: toY(0) - toY(count), fill: look.bars[index % look.bars.length],
              "fill-opacity": side ? 1 : 0.6
            }));
          });
        });
        xTicks(lo, hi, toX);
        legend = true;
      } else if (kind === "ecdf") {
        var toYE = function (v) { return margin.top + plotH - v * plotH; };
        var toXE = function (v) { return margin.left + (v - lo) / (hi - lo) * plotW; };
        hGrid(0, 1, toYE);
        series.forEach(function (s, index) {
          var sorted = s.values.slice().sort(function (a, b) { return a - b; });
          var points = [toXE(lo) + "," + toYE(0)];
          sorted.forEach(function (v, i) {
            points.push(toXE(v) + "," + toYE(i / sorted.length));
            points.push(toXE(v) + "," + toYE((i + 1) / sorted.length));
          });
          points.push(toXE(hi) + "," + toYE(1));
          svg.appendChild(node("polyline", {
            points: points.join(" "), fill: "none",
            stroke: look.bars[index % look.bars.length], "stroke-width": 2.2
          }));
        });
        xTicks(lo, hi, toXE);
        legend = true;
      } else {
        // Box and strip: one column per planner, the value up the side.
        var pad = (hi - lo) * 0.08;
        var yLow = lo - pad;
        var yHigh = hi + pad;
        var toYB = function (v) { return margin.top + plotH - (v - yLow) / (yHigh - yLow) * plotH; };
        hGrid(yLow, yHigh, toYB);
        var band = plotW / series.length;
        series.forEach(function (s, index) {
          var centre = margin.left + band * index + band / 2;
          var colour = look.bars[index % look.bars.length];
          var sorted = s.values.slice().sort(function (a, b) { return a - b; });
          if (kind === "box") {
            var q1 = quantile(sorted, 0.25);
            var q2 = quantile(sorted, 0.5);
            var q3 = quantile(sorted, 0.75);
            var reach = 1.5 * (q3 - q1);
            var inside = sorted.filter(function (v) { return v >= q1 - reach && v <= q3 + reach; });
            var lowW = inside.length ? inside[0] : q1;
            var highW = inside.length ? inside[inside.length - 1] : q3;
            var half = Math.min(40, band * 0.28);
            svg.appendChild(node("line", { x1: centre, y1: toYB(lowW), x2: centre, y2: toYB(q1), stroke: look.ink, "stroke-width": 1.4 }));
            svg.appendChild(node("line", { x1: centre, y1: toYB(q3), x2: centre, y2: toYB(highW), stroke: look.ink, "stroke-width": 1.4 }));
            [lowW, highW].forEach(function (w) {
              svg.appendChild(node("line", { x1: centre - half / 2, y1: toYB(w), x2: centre + half / 2, y2: toYB(w), stroke: look.ink, "stroke-width": 1.4 }));
            });
            svg.appendChild(node("rect", {
              x: centre - half, y: toYB(q3), width: half * 2, height: Math.max(1, toYB(q1) - toYB(q3)),
              fill: colour, "fill-opacity": 0.85, stroke: look.ink, "stroke-width": 1.2, rx: 2
            }));
            svg.appendChild(node("line", { x1: centre - half, y1: toYB(q2), x2: centre + half, y2: toYB(q2), stroke: look.background, "stroke-width": 2.4 }));
            sorted.forEach(function (v) {
              if (v < lowW || v > highW) {
                svg.appendChild(node("circle", { cx: centre, cy: toYB(v), r: 3.5, fill: "none", stroke: look.ink, "stroke-width": 1.2 }));
              }
            });
          } else {
            var spread = Math.min(30, band * 0.25);
            s.values.forEach(function (v, i) {
              // A fixed spread rather than random jitter, so the figure that
              // downloads is the figure on screen, every time.
              var offset = s.values.length > 1 ? (i / (s.values.length - 1) - 0.5) * 2 * spread : 0;
              svg.appendChild(node("circle", {
                cx: centre + offset, cy: toYB(v), r: 4.5, fill: colour, "fill-opacity": 0.8,
                stroke: look.background, "stroke-width": 1
              }));
            });
            var mean = s.values.reduce(function (a, b) { return a + b; }, 0) / s.values.length;
            svg.appendChild(node("line", {
              x1: centre - spread - 8, y1: toYB(mean), x2: centre + spread + 8, y2: toYB(mean),
              stroke: look.ink, "stroke-width": 2
            }));
          }
          svg.appendChild(node("text", {
            x: centre, y: margin.top + plotH + 25, "text-anchor": "middle", "font-size": 13.5, fill: look.ink
          }, s.name + " (" + s.values.length + ")"));
        });
      }

      svg.appendChild(node("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH, stroke: look.ink, "stroke-width": 1, opacity: 0.55 }));
      svg.appendChild(node("line", { x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH, stroke: look.ink, "stroke-width": 1, opacity: 0.55 }));

      if (legend) {
        var x = margin.left + plotW;
        for (var i = series.length - 1; i >= 0; i--) {
          var text = series[i].name + " (" + series[i].values.length + ")";
          x -= text.length * 6.8;
          svg.appendChild(node("text", { x: x, y: margin.top - 10, "font-size": 12, fill: look.ink }, text));
          svg.appendChild(node("rect", { x: x - 16, y: margin.top - 20, width: 11, height: 11, fill: look.bars[i % look.bars.length] }));
          x -= 28;
        }
      }
      if (el.title.value) {
        svg.appendChild(node("text", {
          x: width / 2, y: 30, "text-anchor": "middle", "font-size": 18, "font-weight": "600",
          "letter-spacing": "0.2", fill: look.ink
        }, el.title.value));
      }
      if (el.yLabel.value) {
        svg.appendChild(node("text", {
          x: 18, y: margin.top + plotH / 2, "text-anchor": "middle", "font-size": 14,
          fill: look.muted, transform: "rotate(-90 18 " + (margin.top + plotH / 2) + ")"
        }, el.yLabel.value));
      }
      if (el.xLabel.value) {
        svg.appendChild(node("text", {
          x: margin.left + plotW / 2, y: height - 16, "text-anchor": "middle",
          "font-size": 14, fill: look.muted
        }, el.xLabel.value));
      }
      output.style.background = look.background;
      output.appendChild(svg);
      return svg;
    }

    /* The two text fields keep their places but change meaning with the
       mode: per episode they are the plot's vertical and horizontal axes. */
    function caption(input, text) {
      var label = input.closest("label");
      if (label && label.firstChild && label.firstChild.nodeType === 3) {
        label.firstChild.nodeValue = text + " ";
      }
    }

    function showControls() {
      var episodesMode = el.mode && el.mode.value === "episodes";
      root.querySelectorAll("[data-mode]").forEach(function (label) {
        label.hidden = label.getAttribute("data-mode") !== (episodesMode ? "episodes" : "aggregate");
      });
      root.querySelectorAll("[data-kind]").forEach(function (label) {
        if (episodesMode) label.hidden = label.getAttribute("data-kind") !== el.kind.value;
      });
      caption(el.yLabel, episodesMode ? "Vertical axis" : "Value axis");
      caption(el.xLabel, episodesMode ? "Horizontal axis" : "Planner axis");
    }

    function syncEpisodeLabels() {
      var field = humanize(el.field.value || "value");
      var kind = el.kind.value;
      el.title.value = field + " per episode on " + DATA.environment;
      if (kind === "histogram") { el.xLabel.value = field; el.yLabel.value = "Frequency"; }
      else if (kind === "ecdf") { el.xLabel.value = field; el.yLabel.value = "Fraction of episodes at or below"; }
      else { el.xLabel.value = "Planner"; el.yLabel.value = field; }
    }

    function loadEpisodes() {
      if (EPISODES || !episodesUrl) return;
      fetch(episodesUrl, { cache: "no-store" })
        .then(function (response) {
          if (!response.ok) throw new Error("HTTP " + response.status);
          return response.json();
        })
        .then(function (data) {
          EPISODES = data;
          var fields = {};
          Object.keys(data.policies).forEach(function (name) {
            data.policies[name].forEach(function (record) {
              Object.keys(record).forEach(function (key) {
                if (key !== "episode" && typeof record[key] === "number") fields[key] = true;
              });
            });
          });
          var names = Object.keys(fields).sort();
          var lead = ["discounted_return", "return", "steps"].filter(function (n) { return fields[n]; });
          names = lead.concat(names.filter(function (n) { return lead.indexOf(n) === -1; }));
          el.field.textContent = "";
          names.forEach(function (name) {
            var option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            el.field.appendChild(option);
          });
          syncEpisodeLabels();
          draw();
        })
        .catch(function () {
          output.textContent = "Could not load this run's episodes.";
        });
    }

    function csv() {
      var lines = [];
      var csvField = kit.csvField;
      var ticked = rows().filter(function (row) {
        return row.querySelector("input[type=checkbox]").checked;
      }).map(function (row) { return row.getAttribute("data-policy"); });
      if (el.mode && el.mode.value === "episodes" && EPISODES) {
        var columns = ["episode", "return", "discounted_return", "steps", "ended"];
        ticked.forEach(function (name) {
          (EPISODES.policies[name] || []).forEach(function (record) {
            Object.keys(record).forEach(function (key) {
              if (columns.indexOf(key) === -1) columns.push(key);
            });
          });
        });
        lines.push(["environment", "planner"].concat(columns).map(csvField).join(","));
        ticked.forEach(function (name) {
          (EPISODES.policies[name] || []).forEach(function (record) {
            lines.push([DATA.environment, name].concat(columns.map(function (c) {
              return record[c];
            })).map(csvField).join(","));
          });
        });
      } else {
        var metric = el.metric.value;
        lines.push(["environment", "planner", "metric", "value", "ci_lower", "ci_upper"].join(","));
        DATA.policies.forEach(function (policy) {
          if (ticked.indexOf(policy.name) === -1) return;
          lines.push([DATA.environment, policy.name, metric, policy.metrics[metric],
            policy.metrics[metric + DATA.ci.lower], policy.metrics[metric + DATA.ci.upper]]
            .map(csvField).join(","));
        });
      }
      return lines.join("\n") + "\n";
    }

    function fileName(extension) {
      return kit.fileName(el.title.value || el.metric.value || "chart", extension);
    }

    if (el.csvButton) {
      el.csvButton.addEventListener("click", function () {
        var base = el.mode && el.mode.value === "episodes"
          ? DATA.environment + "_episodes_" + (el.field.value || "")
          : (el.title.value || el.metric.value || "chart");
        kit.downloadText(csv(), kit.fileName(base, "csv"), "text/csv");
      });
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
    if (el.mode) {
      el.mode.addEventListener("change", function (event) {
        event.stopPropagation();
        showControls();
        if (el.mode.value === "episodes") {
          if (EPISODES) syncEpisodeLabels();
          loadEpisodes();
        } else {
          syncLabels();
          el.xLabel.value = "Planner";
        }
        draw();
      });
      [el.field, el.kind].forEach(function (control) {
        control.addEventListener("change", function (event) {
          event.stopPropagation();
          showControls();
          syncEpisodeLabels();
          draw();
        });
      });
      showControls();
    }
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
