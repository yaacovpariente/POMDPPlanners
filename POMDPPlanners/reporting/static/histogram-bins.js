/* SPDX-License-Identifier: MIT
 *
 * Let the reader choose a returns histogram's number of bins.
 *
 * The page draws each histogram with the automatic count. A new count asks
 * the server to redraw it (data-histogram is the redraw URL), so the binning
 * and the drawing live in one place, the Python that drew the page. Bins
 * stay shared across planners because the server bins them together.
 *
 * The choice is remembered per page and histogram, like the table filters.
 * Storage can be unavailable, so every access is guarded.
 */
(function () {
  "use strict";

  if (window.POMDPHistogramBins) return;
  window.POMDPHistogramBins = true;

  function storageKey(figure) {
    return "pomdp-histogram-bins:" + window.location.pathname + "#" +
      figure.getAttribute("data-histogram-key");
  }

  function load(figure) {
    try {
      var value = parseInt(window.localStorage.getItem(storageKey(figure)), 10);
      return isNaN(value) ? null : value;
    } catch (e) {
      return null;
    }
  }

  function save(figure, bins) {
    try {
      if (bins === null) window.localStorage.removeItem(storageKey(figure));
      else window.localStorage.setItem(storageKey(figure), String(bins));
    } catch (e) {
      /* A remembered count is a convenience, never a requirement. */
    }
  }

  function clamp(figure, value) {
    var input = figure.querySelector(".histogram-bins input");
    var low = parseInt(input.min, 10) || 1;
    var high = parseInt(input.max, 10) || 100;
    return Math.min(high, Math.max(low, value));
  }

  /* Redraw one histogram. A request id drops a slow answer that arrives
     after a newer one, so typing 1 then 12 never ends on 1 bin. */
  function draw(figure, bins) {
    var auto = parseInt(figure.getAttribute("data-auto-bins"), 10);
    var input = figure.querySelector(".histogram-bins input");
    var reset = figure.querySelector("[data-bins-auto]");
    var plot = figure.querySelector(".histogram-plot");
    input.value = String(bins);
    reset.hidden = bins === auto;
    var request = (figure.pomdpBinsRequest || 0) + 1;
    figure.pomdpBinsRequest = request;
    var url = figure.getAttribute("data-histogram") + "&bins=" + bins;
    fetch(url, { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.text();
      })
      .then(function (svg) {
        if (figure.pomdpBinsRequest !== request) return;
        plot.innerHTML = svg;
        figure.removeAttribute("data-bins-error");
      })
      .catch(function () {
        /* The chart already drawn stays; it is still a true histogram. */
        if (figure.pomdpBinsRequest === request) figure.setAttribute("data-bins-error", "");
      });
  }

  function choose(figure, raw) {
    var value = parseInt(raw, 10);
    if (isNaN(value)) return;
    var bins = clamp(figure, value);
    var auto = parseInt(figure.getAttribute("data-auto-bins"), 10);
    save(figure, bins === auto ? null : bins);
    draw(figure, bins);
  }

  function restore(root) {
    var figures = root.querySelectorAll("figure[data-histogram]");
    Array.prototype.forEach.call(figures, function (figure) {
      if (figure.pomdpBinsRestored) return;
      figure.pomdpBinsRestored = true;
      var stored = load(figure);
      if (stored !== null) draw(figure, clamp(figure, stored));
    });
  }

  // Delegated, so a histogram that arrives later -- inside a collapsed
  // section -- is handled by the same listeners. "change" fires on Enter,
  // on blur and on each stepper click, not on every keystroke.
  document.addEventListener("change", function (event) {
    var input = event.target;
    if (!input.matches || !input.matches(".histogram-bins input")) return;
    choose(input.closest("figure[data-histogram]"), input.value);
  });
  document.addEventListener("click", function (event) {
    var button = event.target.closest && event.target.closest("[data-bins-auto]");
    if (!button) return;
    var figure = button.closest("figure[data-histogram]");
    choose(figure, figure.getAttribute("data-auto-bins"));
  });
  document.addEventListener("pomdp:inserted", function (event) {
    restore(event.detail || document);
  });

  // The script sits right after the first histogram, so later ones are not
  // parsed yet; restoring once the whole document is in picks them all up.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { restore(document); });
  } else {
    restore(document);
  }
})();
