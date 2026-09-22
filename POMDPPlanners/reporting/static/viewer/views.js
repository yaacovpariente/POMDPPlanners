/* SPDX-License-Identifier: MIT
 *
 * The episode page's view tabs: one episode, several recordings of it.
 *
 * Panels are hidden rather than rebuilt, so switching away from the 3D replay
 * and back does not reload the trace. The canvas is sized from its container,
 * which measures zero while hidden, so every switch re-announces a resize —
 * without it the replay comes back stretched or blank.
 */
(function () {
  "use strict";

  var root = document.querySelector(".views");
  if (!root) return;

  var tabs = Array.prototype.slice.call(root.querySelectorAll(".tab"));
  var panels = Array.prototype.slice.call(root.querySelectorAll(".view"));

  function show(name) {
    tabs.forEach(function (tab) {
      tab.setAttribute("aria-pressed", tab.dataset.view === name ? "true" : "false");
    });
    panels.forEach(function (panel) {
      panel.hidden = panel.dataset.view !== name;
    });
    window.dispatchEvent(new Event("resize"));
  }

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () { show(tab.dataset.view); });
  });
})();
