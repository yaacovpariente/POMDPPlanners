/* SPDX-License-Identifier: MIT
 *
 * Export a metric table as CSV: the numbers, not a picture of them.
 *
 * One handler for every metric table on the site. A table names its planners
 * in its header cells (data-planner, data-col, data-episodes) and carries each
 * value at full precision on its cell (data-value, data-low, data-high,
 * data-best-trial), so the export does not depend on how a cell is drawn.
 *
 * The CSV is long -- one row per environment, planner and metric -- because a
 * long table loads into pandas or Excel the same way whatever the metrics and
 * planners were. "As shown" follows the table's filters: hidden rows and
 * hidden planner columns are left out. "All data" ignores them.
 */
(function () {
  "use strict";

  if (window.POMDPTableExport) return;
  window.POMDPTableExport = true;

  function field(value) {
    if (value === null || value === undefined) return "";
    var text = String(value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function attr(cell, name) {
    return cell && cell.hasAttribute(name) ? cell.getAttribute(name) : "";
  }

  function csv(table, shownOnly) {
    var env = table.getAttribute("data-env") || "";
    var planners = Array.prototype.slice.call(table.querySelectorAll("thead [data-planner]"));
    var rows = Array.prototype.slice.call(
      table.querySelectorAll("tbody tr[data-metric]:not([data-export-skip])")
    );
    var lines = [[
      "environment", "planner", "metric", "value", "ci_lower", "ci_upper",
      "episodes", "best_trial_value"
    ].join(",")];
    rows.forEach(function (row) {
      if (shownOnly && row.hidden) return;
      var metric = row.getAttribute("data-metric");
      planners.forEach(function (head) {
        if (shownOnly && head.hidden) return;
        var cell = row.querySelector('td[data-col="' + head.getAttribute("data-col") + '"]');
        lines.push([
          env,
          head.getAttribute("data-planner"),
          metric,
          attr(cell, "data-value"),
          attr(cell, "data-low"),
          attr(cell, "data-high"),
          head.getAttribute("data-episodes") || "",
          attr(cell, "data-best-trial")
        ].map(field).join(","));
      });
    });
    return lines.join("\n") + "\n";
  }

  function save(text, name) {
    var url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    var link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // Delegated, so a table that arrives later -- inside a collapsed section --
  // is exported by the same handler.
  document.addEventListener("click", function (event) {
    var button = event.target.closest && event.target.closest("[data-export-table]");
    if (!button) return;
    var table = document.getElementById(button.getAttribute("data-export-table"));
    if (!table) return;
    var shownOnly = button.getAttribute("data-export-mode") !== "all";
    var base = (table.getAttribute("data-export-name") || "metrics")
      .replace(/[^A-Za-z0-9_-]+/g, "_");
    save(csv(table, shownOnly), base + (shownOnly ? "" : "_all") + ".csv");
  });

  window.POMDPTableExport = { csv: csv };
})();
