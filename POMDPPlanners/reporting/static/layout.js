/* SPDX-License-Identifier: MIT
 *
 * Cards or list, for the pages that show a grid of them.
 *
 * The choice is remembered per browser rather than per page, so setting it
 * once holds while the reader walks down experiment, run and planner. Storage
 * can be unavailable (a private window, blocked site data), so every access is
 * guarded and the page renders as cards when it fails.
 */
(function () {
  "use strict";

  var KEY = "pomdp-results-layout";
  var buttons = Array.prototype.slice.call(document.querySelectorAll("[data-layout]"));
  if (!buttons.length) return;

  var grids = Array.prototype.slice.call(document.querySelectorAll(".cards"));

  function read() {
    try {
      return window.localStorage.getItem(KEY);
    } catch (e) {
      return null;
    }
  }

  function write(mode) {
    try {
      window.localStorage.setItem(KEY, mode);
    } catch (e) {
      /* A remembered preference is a convenience, never a requirement. */
    }
  }

  function apply(mode) {
    grids.forEach(function (grid) {
      grid.classList.toggle("as-list", mode === "list");
    });
    buttons.forEach(function (button) {
      button.setAttribute("aria-pressed", button.dataset.layout === mode ? "true" : "false");
    });
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      write(button.dataset.layout);
      apply(button.dataset.layout);
    });
  });

  apply(read() === "list" ? "list" : "cards");
})();
