/* SPDX-License-Identifier: MIT
 *
 * Narrow a comparison table's metric rows by name and by group.
 *
 * A run logs a few dozen metrics; the reader usually wants three. Typing
 * keeps the rows whose name contains every word typed, and the group boxes
 * hide whole families -- timings, the planners' internal counters -- at
 * once. Everything happens in the page: the rows are all there already.
 */
(function () {
  "use strict";

  document.querySelectorAll(".metric-filter[data-filter-for]").forEach(function (control) {
    var table = document.getElementById(control.getAttribute("data-filter-for"));
    if (!table) return;
    var box = control.querySelector("input[type=search]");
    var toggles = Array.prototype.slice.call(control.querySelectorAll("[data-group-toggle]"));
    var count = control.querySelector("[data-filter-count]");
    var rows = Array.prototype.slice.call(table.querySelectorAll("tbody tr[data-metric]"));

    function apply() {
      var words = box.value.toLowerCase().split(/\s+/).filter(Boolean);
      var groups = {};
      toggles.forEach(function (toggle) {
        groups[toggle.getAttribute("data-group-toggle")] = toggle.checked;
      });
      var shown = 0;
      rows.forEach(function (row) {
        var name = row.getAttribute("data-metric").toLowerCase();
        var visible =
          groups[row.getAttribute("data-group")] !== false &&
          words.every(function (word) { return name.indexOf(word) !== -1; });
        row.hidden = !visible;
        if (visible) shown += 1;
      });
      count.textContent = shown === rows.length ? "" : shown + " of " + rows.length + " metrics";
    }

    box.addEventListener("input", apply);
    toggles.forEach(function (toggle) { toggle.addEventListener("change", apply); });
    apply();
  });
})();
