/* SPDX-License-Identifier: MIT
 *
 * The diagnostics chart builder: one figure of a tuning study, from its trial
 * records, with the chart, the labels and the look chosen by the reader.
 *
 * The planner comparison's builder draws bars; a tuning study is trials, so
 * this draws points: an objective or a parameter over the trials, an
 * objective against a parameter, the Pareto front, the trial durations. The
 * looks, the text fields and the downloads are the same as the comparison's
 * (figure-kit.js), and what is on screen is what downloads.
 */
(function () {
  "use strict";

  var dataEl = document.getElementById("tuning-chart-data");
  var output = document.getElementById("tc-output");
  if (!dataEl || !output || !window.POMDPFigure) return;

  var DATA = JSON.parse(dataEl.textContent);
  var kit = window.POMDPFigure;
  var node = kit.node;
  var format = kit.format;

  var el = {
    kind: document.getElementById("tc-kind"),
    objective: document.getElementById("tc-objective"),
    parameter: document.getElementById("tc-parameter"),
    title: document.getElementById("tc-title"),
    y: document.getElementById("tc-y"),
    x: document.getElementById("tc-x"),
    style: document.getElementById("tc-style"),
    bestLine: document.getElementById("tc-best-line"),
    stop: document.getElementById("tc-stop"),
    highlight: document.getElementById("tc-highlight"),
    legend: document.getElementById("tc-legend"),
    form: document.getElementById("tuning-chart-form")
  };

  function look() {
    return kit.STYLES[el.style.value] || kit.STYLES.mono;
  }

  var completed = DATA.trials.filter(function (t) { return t.state === "COMPLETE"; });
  var pareto = {};
  DATA.pareto.forEach(function (n) { pareto[n] = true; });

  function kindOf(trial) {
    if (trial.number === DATA.best) return "best";
    if (pareto[trial.number]) return "pareto";
    return "trial";
  }

  function direction(name) {
    var found = DATA.objectives.filter(function (o) { return o.name === name; })[0];
    return found ? found.direction : "maximize";
  }

  function parameter(name) {
    return DATA.parameters.filter(function (p) { return p.name === name; })[0] || { name: name };
  }

  /* A categorical parameter, or one whose sampled values are not all
     numbers, is drawn on a named axis. */
  function categories(name, values) {
    var spec = parameter(name);
    var numeric = values.every(function (v) { return typeof v === "number"; });
    if (!spec.choices && numeric) return null;
    var names = (spec.choices || []).map(String);
    values.forEach(function (v) {
      if (names.indexOf(String(v)) === -1) names.push(String(v));
    });
    return names;
  }

  /* What one chart kind plots: points, an optional line, axis labels and
     categorical axes. */
  function figure() {
    var kind = el.kind.value;
    var objective = el.objective.value;
    var param = el.parameter.value;
    var result = { points: [], line: null, xCats: null, yCats: null, xInteger: false };

    if (kind === "objective-history") {
      var best = null;
      var line = [];
      var maximize = direction(objective) !== "minimize";
      completed.forEach(function (t) {
        var v = t.objectives[objective];
        if (typeof v !== "number") return;
        best = best === null ? v : (maximize ? Math.max(best, v) : Math.min(best, v));
        line.push([t.number, best]);
        result.points.push({ x: t.number, y: v, kind: kindOf(t) });
      });
      result.line = line;
      result.xInteger = true;
    } else if (kind === "parameter-slice") {
      var values = completed.map(function (t) { return t.params[param]; }).filter(function (v) {
        return v !== undefined;
      });
      result.xCats = categories(param, values);
      completed.forEach(function (t) {
        var v = t.objectives[objective];
        var p = t.params[param];
        if (typeof v !== "number" || p === undefined) return;
        var x = result.xCats ? result.xCats.indexOf(String(p)) : p;
        result.points.push({ x: x, y: v, kind: kindOf(t) });
      });
      var spec = parameter(param);
      if (!result.xCats && typeof spec.low === "number") result.xSpan = [spec.low, spec.high];
    } else if (kind === "parameter-history") {
      var sampled = DATA.trials.map(function (t) { return t.params[param]; }).filter(function (v) {
        return v !== undefined;
      });
      result.yCats = categories(param, sampled);
      DATA.trials.forEach(function (t) {
        var p = t.params[param];
        if (p === undefined) return;
        var y = result.yCats ? result.yCats.indexOf(String(p)) : p;
        result.points.push({ x: t.number, y: y, kind: kindOf(t) });
      });
      var range = parameter(param);
      if (!result.yCats && typeof range.low === "number") result.ySpan = [range.low, range.high];
      result.xInteger = true;
    } else if (kind === "pareto-front") {
      var first = DATA.objectives[0].name;
      var second = DATA.objectives[1].name;
      var front = [];
      completed.forEach(function (t) {
        var a = t.objectives[first];
        var b = t.objectives[second];
        if (typeof a !== "number" || typeof b !== "number") return;
        var k = kindOf(t);
        result.points.push({ x: a, y: b, kind: k });
        if (k !== "trial") front.push([a, b]);
      });
      front.sort(function (p, q) { return p[0] - q[0]; });
      result.line = front.length > 1 ? front : null;
    } else if (kind === "trial-durations") {
      DATA.trials.forEach(function (t) {
        if (typeof t.duration !== "number") return;
        result.points.push({ x: t.number, y: t.duration, kind: kindOf(t) });
      });
      result.xInteger = true;
    }
    return result;
  }

  function bounds(values, span) {
    var all = values.slice();
    if (span) { all.push(span[0]); all.push(span[1]); }
    var low = Math.min.apply(null, all);
    var high = Math.max.apply(null, all);
    if (low === high) { low -= 1; high += 1; }
    var pad = (high - low) * 0.08;
    return [low - pad, high + pad];
  }

  function integerTicks(low, high) {
    var first = Math.ceil(low);
    var last = Math.floor(high);
    var step = Math.max(1, Math.ceil((last - first) / 6));
    var ticks = [];
    for (var t = first; t <= last; t += step) ticks.push(t);
    return ticks;
  }

  function draw() {
    var data = figure();
    output.textContent = "";
    if (!data.points.length) {
      var empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "The study recorded nothing to plot for this choice.";
      output.appendChild(empty);
      return;
    }

    var style = look();
    var width = 640;
    var height = 420;
    var margin = { left: 86, right: 28, top: 58, bottom: 74 };
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;

    var xs = data.points.map(function (p) { return p.x; });
    var ys = data.points.map(function (p) { return p.y; });
    if (data.line) data.line.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); });
    var showStop = el.stop.checked && DATA.stopped_at && data.xInteger &&
      el.kind.value === "objective-history";
    if (showStop) xs.push(DATA.stopped_at - 1);

    var xRange = data.xCats ? [-0.5, data.xCats.length - 0.5]
      : data.xInteger ? [Math.min.apply(null, xs) - 0.5, Math.max.apply(null, xs) + 0.5]
      : bounds(xs, data.xSpan);
    var yRange = data.yCats ? [-0.5, data.yCats.length - 0.5] : bounds(ys, data.ySpan);

    function toX(v) { return margin.left + (v - xRange[0]) / (xRange[1] - xRange[0]) * plotW; }
    function toY(v) { return margin.top + plotH - (v - yRange[0]) / (yRange[1] - yRange[0]) * plotH; }

    var svg = node("svg", {
      xmlns: kit.SVG_NS,
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      "font-family": "Helvetica Neue, Helvetica, Arial, sans-serif"
    });
    svg.appendChild(node("rect", { x: 0, y: 0, width: width, height: height, fill: style.background }));

    // Value grid and labels on the vertical axis.
    var yTicks = data.yCats
      ? data.yCats.map(function (_, i) { return i; })
      : kit.niceTicks(yRange[0], yRange[1], 6);
    yTicks.forEach(function (tick, i) {
      var y = toY(tick);
      svg.appendChild(node("line", {
        x1: margin.left, y1: y, x2: margin.left + plotW, y2: y,
        stroke: style.grid, "stroke-width": 1, "stroke-dasharray": "3 4"
      }));
      svg.appendChild(node("text", {
        x: margin.left - 12, y: y + 4, "text-anchor": "end", "font-size": 12.5, fill: style.muted
      }, data.yCats ? data.yCats[i] : format(tick)));
    });
    var xTicks = data.xCats
      ? data.xCats.map(function (_, i) { return i; })
      : data.xInteger ? integerTicks(xRange[0], xRange[1])
      : kit.niceTicks(xRange[0], xRange[1], 6);
    xTicks.forEach(function (tick, i) {
      svg.appendChild(node("text", {
        x: toX(tick), y: margin.top + plotH + 22, "text-anchor": "middle",
        "font-size": 12.5, fill: style.muted
      }, data.xCats ? data.xCats[i] : format(tick)));
    });

    if (showStop) {
      var sx = toX(DATA.stopped_at - 1);
      svg.appendChild(node("line", {
        x1: sx, y1: margin.top, x2: sx, y2: margin.top + plotH,
        stroke: style.muted, "stroke-width": 1.2, "stroke-dasharray": "5 4"
      }));
      svg.appendChild(node("text", {
        x: sx - 6, y: margin.top + 14, "text-anchor": "end", "font-size": 12, fill: style.muted
      }, "early stop"));
    }

    var lineOn = data.line && (el.kind.value !== "objective-history" || el.bestLine.checked);
    if (lineOn) {
      var coords = [];
      var previous = null;
      data.line.forEach(function (p) {
        var x = toX(p[0]);
        var y = toY(p[1]);
        // The best-so-far curve is a staircase: it only moves when a trial beats it.
        if (el.kind.value === "objective-history" && previous !== null) {
          coords.push(x + "," + previous);
        }
        coords.push(x + "," + y);
        previous = y;
      });
      svg.appendChild(node("polyline", {
        points: coords.join(" "), fill: "none", stroke: style.bars[1],
        "stroke-width": 2, "stroke-dasharray": el.kind.value === "pareto-front" ? "6 4" : "none"
      }));
    }

    // Marks: ordinary trials first, so the chosen and Pareto ones sit on top.
    var colours = { trial: style.bars[0], pareto: style.bars[2], best: style.bars[1] };
    var highlight = el.highlight.checked;
    var order = { trial: 0, pareto: 1, best: 2 };
    data.points.slice().sort(function (p, q) {
      return (highlight ? order[p.kind] : 0) - (highlight ? order[q.kind] : 0);
    }).forEach(function (p) {
      var kind = highlight ? p.kind : "trial";
      svg.appendChild(node("circle", {
        cx: toX(p.x), cy: toY(p.y), r: kind === "trial" ? 4.5 : 6,
        fill: colours[kind], "fill-opacity": kind === "trial" ? 0.75 : 1,
        stroke: style.background, "stroke-width": 1
      }));
    });

    // Axis lines last, so they sit over the grid.
    svg.appendChild(node("line", {
      x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH,
      stroke: style.ink, "stroke-width": 1, opacity: 0.55
    }));
    svg.appendChild(node("line", {
      x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH,
      stroke: style.ink, "stroke-width": 1, opacity: 0.55
    }));

    if (el.title.value) {
      svg.appendChild(node("text", {
        x: width / 2, y: 30, "text-anchor": "middle", "font-size": 18, "font-weight": "600",
        "letter-spacing": "0.2", fill: style.ink
      }, el.title.value));
    }
    if (el.y.value) {
      svg.appendChild(node("text", {
        x: 18, y: margin.top + plotH / 2, "text-anchor": "middle", "font-size": 14,
        fill: style.muted, transform: "rotate(-90 18 " + (margin.top + plotH / 2) + ")"
      }, el.y.value));
    }
    if (el.x.value) {
      svg.appendChild(node("text", {
        x: margin.left + plotW / 2, y: height - 16, "text-anchor": "middle",
        "font-size": 14, fill: style.muted
      }, el.x.value));
    }

    if (el.legend.checked) {
      var entries = [["trial", "trial"]];
      if (highlight) {
        if (data.points.some(function (p) { return p.kind === "pareto"; })) entries.push(["pareto", "Pareto"]);
        if (data.points.some(function (p) { return p.kind === "best"; })) entries.push(["best", "chosen"]);
      }
      var lx = margin.left + plotW;
      var ly = margin.top - 14;
      var labels = entries.map(function (e) { return e[1]; });
      if (lineOn) labels.push(el.kind.value === "pareto-front" ? "front" : "best so far");
      // Laid out right to left from the plot's corner, so it never meets the title.
      var x = lx;
      for (var i = labels.length - 1; i >= 0; i--) {
        var label = labels[i];
        var textWidth = label.length * 6.8;
        x -= textWidth;
        svg.appendChild(node("text", { x: x, y: ly + 4, "font-size": 12, fill: style.ink }, label));
        if (i < entries.length) {
          svg.appendChild(node("circle", { cx: x - 9, cy: ly, r: 4.5, fill: colours[entries[i][0]] }));
          x -= 22;
        } else {
          svg.appendChild(node("line", {
            x1: x - 20, y1: ly, x2: x - 4, y2: ly, stroke: style.bars[1], "stroke-width": 2
          }));
          x -= 30;
        }
      }
    }

    output.style.background = style.background;
    output.appendChild(svg);
  }

  /* Labels follow the choice of chart until the reader edits them. */
  function syncLabels() {
    var kind = el.kind.value;
    var objective = el.objective.value;
    var param = el.parameter.value;
    var humanize = kit.humanize;
    var texts = {
      "objective-history": [humanize(objective) + " over the trials", "Trial", humanize(objective)],
      "parameter-slice": [humanize(objective) + " by " + param, humanize(param), humanize(objective)],
      "parameter-history": [humanize(param) + " over the trials", "Trial", humanize(param)],
      "pareto-front": [
        "Pareto front",
        DATA.objectives[0] ? humanize(DATA.objectives[0].name) : "",
        DATA.objectives[1] ? humanize(DATA.objectives[1].name) : ""
      ],
      "trial-durations": ["Trial durations", "Trial", "Seconds"]
    }[kind];
    el.title.value = DATA.title + ": " + texts[0];
    el.x.value = texts[1];
    el.y.value = texts[2];
  }

  /* Only the controls the chosen chart uses are shown. */
  function showControls() {
    var kind = el.kind.value;
    var uses = {
      objective: kind === "objective-history" || kind === "parameter-slice",
      parameter: kind === "parameter-slice" || kind === "parameter-history",
      "best-line": kind === "objective-history",
      stop: kind === "objective-history" && !!DATA.stopped_at
    };
    Object.keys(uses).forEach(function (key) {
      var label = el.form.querySelector('[data-for="' + key + '"]');
      if (label) label.hidden = !uses[key];
    });
  }

  DATA.objectives.forEach(function (o) {
    var option = document.createElement("option");
    option.value = o.name;
    option.textContent = o.name;
    el.objective.appendChild(option);
  });
  DATA.parameters.forEach(function (p) {
    var option = document.createElement("option");
    option.value = p.name;
    option.textContent = p.name;
    el.parameter.appendChild(option);
  });

  function fileName(extension) {
    return kit.fileName(el.title.value || el.kind.value, extension);
  }
  document.getElementById("tc-svg").addEventListener("click", function () {
    kit.downloadSvg(output, fileName("svg"));
  });
  document.getElementById("tc-png").addEventListener("click", function () {
    kit.downloadPng(output, look().background, fileName("png"));
  });

  [el.kind, el.objective, el.parameter].forEach(function (control) {
    control.addEventListener("change", function () {
      showControls();
      syncLabels();
      draw();
    });
  });
  el.form.addEventListener("input", draw);
  el.form.addEventListener("change", draw);

  showControls();
  syncLabels();
  draw();
})();
