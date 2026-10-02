/* SPDX-License-Identifier: MIT
 *
 * Narrow a comparison table: its metric rows by name, by group and by an
 * explicit pick, and its planner columns by a pick.
 *
 * A run logs a few dozen metrics; the reader usually wants three. Typing
 * keeps the rows whose name contains every word typed, and also narrows the
 * metric picker's list. The group boxes hide whole families -- timings, the
 * planners' internal counters -- at once. A row shows only if it passes all
 * three. The planner picker hides columns, and "best" is then re-marked
 * among the planners left showing: a best chosen from a hidden column would
 * point at nothing. Everything happens in the page; the rows are all there.
 *
 * Both picks are remembered per page and table. What is stored is what was
 * taken out, so a metric or planner that appears later is shown by default.
 * Storage can be unavailable, so every access is guarded.
 */
(function () {
  "use strict";

  function load(key) {
    try {
      var raw = window.localStorage.getItem(key);
      var parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      return [];
    }
  }

  function save(key, removed) {
    try {
      window.localStorage.setItem(key, JSON.stringify(removed));
    } catch (e) {
      /* A remembered pick is a convenience, never a requirement. */
    }
  }

  /* One dropdown of checkboxes: restores its pick, keeps its summary count,
     and handles Select all / Clear, Escape and a click elsewhere. */
  function picker(element, key, onChange) {
    var choices = Array.prototype.slice.call(element.querySelectorAll("[data-choice-name]"));
    var summary = element.querySelector("[data-picker-summary]");
    var label = summary.getAttribute("data-picker-label");
    var removed = load(key);
    choices.forEach(function (choice) {
      choice.checked = removed.indexOf(choice.getAttribute("data-choice-name")) === -1;
    });

    function changed() {
      save(
        key,
        choices.filter(function (c) { return !c.checked; }).map(function (c) {
          return c.getAttribute("data-choice-name");
        })
      );
      onChange();
    }

    function refresh() {
      var ticked = choices.filter(function (c) { return c.checked; }).length;
      summary.textContent = label + " (" + ticked + " of " + choices.length + ")";
    }

    function setListed(checked) {
      // Only the choices still listed, so "Clear" after typing "time" clears
      // the timings and nothing else.
      choices.forEach(function (choice) {
        if (!choice.closest("label").hidden) choice.checked = checked;
      });
      changed();
    }

    choices.forEach(function (choice) { choice.addEventListener("change", changed); });
    element.querySelector("[data-picker-all]").addEventListener("click", function () {
      setListed(true);
    });
    element.querySelector("[data-picker-none]").addEventListener("click", function () {
      setListed(false);
    });
    element.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && element.open) {
        element.open = false;
        summary.focus();
      }
    });
    document.addEventListener("click", function (event) {
      if (element.open && !element.contains(event.target)) element.open = false;
    });

    return { choices: choices, refresh: refresh };
  }

  function remark(rows, hiddenColumns) {
    rows.forEach(function (row) {
      var direction = row.getAttribute("data-direction");
      var cells = Array.prototype.slice.call(row.querySelectorAll("td[data-value]"));
      cells.forEach(function (cell) { cell.classList.remove("is-best"); });
      if (!direction) return;
      var visible = cells.filter(function (cell) {
        return !hiddenColumns[cell.getAttribute("data-col")];
      });
      var values = visible.map(function (cell) {
        return parseFloat(cell.getAttribute("data-value"));
      });
      if (values.length < 2) return;
      var high = Math.max.apply(null, values);
      var low = Math.min.apply(null, values);
      if (high === low) return;
      var best = direction === "minimize" ? low : high;
      visible.forEach(function (cell, index) {
        if (values[index] === best) cell.classList.add("is-best");
      });
    });
  }

  document.querySelectorAll(".metric-filter[data-filter-for]").forEach(function (control) {
    var tableId = control.getAttribute("data-filter-for");
    var table = document.getElementById(tableId);
    if (!table) return;
    var page = window.location.pathname + "#" + tableId;
    var box = control.querySelector("input[type=search]");
    var toggles = Array.prototype.slice.call(control.querySelectorAll("[data-group-toggle]"));
    var count = control.querySelector("[data-filter-count]");
    var rows = Array.prototype.slice.call(table.querySelectorAll("tbody tr[data-metric]"));
    var columnCells = Array.prototype.slice.call(table.querySelectorAll("[data-col]"));
    var builder = control.querySelector("[data-build-chart]");
    var builderBase = builder ? builder.getAttribute("href") : null;

    var metrics = picker(
      control.querySelector('[data-picker="metric"]'),
      "pomdp-compare-metrics:" + page,
      apply
    );
    var planners = picker(
      control.querySelector('[data-picker="planner"]'),
      "pomdp-compare-planners:" + page,
      apply
    );

    function apply() {
      var words = box.value.toLowerCase().split(/\s+/).filter(Boolean);
      var matches = function (name) {
        return words.every(function (word) { return name.indexOf(word) !== -1; });
      };
      var groups = {};
      toggles.forEach(function (toggle) {
        groups[toggle.getAttribute("data-group-toggle")] = toggle.checked;
      });
      var chosen = {};
      metrics.choices.forEach(function (choice) {
        chosen[choice.getAttribute("data-choice-name")] = choice.checked;
      });

      var shown = 0;
      rows.forEach(function (row) {
        var name = row.getAttribute("data-metric");
        var visible =
          groups[row.getAttribute("data-group")] !== false &&
          chosen[name] !== false &&
          matches(name.toLowerCase());
        row.hidden = !visible;
        if (visible) shown += 1;
      });
      // The text box narrows the metric list too; a hidden choice keeps its tick.
      metrics.choices.forEach(function (choice) {
        choice.closest("label").hidden = !matches(
          choice.getAttribute("data-choice-name").toLowerCase()
        );
      });

      var hiddenColumns = {};
      planners.choices.forEach(function (choice) {
        if (!choice.checked) hiddenColumns[choice.getAttribute("data-planner-choice")] = true;
      });
      columnCells.forEach(function (cell) {
        cell.hidden = !!hiddenColumns[cell.getAttribute("data-col")];
      });
      remark(rows, hiddenColumns);

      // The chart builder opens with what the table is showing: its planners,
      // and its first metric row that is a logged metric.
      if (builder) {
        var names = planners.choices.filter(function (c) { return c.checked; }).map(function (c) {
          return c.getAttribute("data-choice-name");
        });
        var first = rows.filter(function (row) {
          var name = row.getAttribute("data-metric");
          return !row.hidden && name !== "episodes" && name !== "average_return best trial";
        })[0];
        var query = new URLSearchParams();
        if (names.length) query.set("planners", names.join(","));
        if (first) query.set("metric", first.getAttribute("data-metric"));
        builder.setAttribute("href", builderBase + (query.toString() ? "?" + query : ""));
      }

      metrics.refresh();
      planners.refresh();
      count.textContent = shown === rows.length ? "" : shown + " of " + rows.length + " metrics";
    }

    box.addEventListener("input", apply);
    toggles.forEach(function (toggle) { toggle.addEventListener("change", apply); });
    apply();
  });
})();
