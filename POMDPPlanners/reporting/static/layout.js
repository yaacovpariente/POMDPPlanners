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

  themeSwitch();

  var KEY = "pomdp-results-layout";
  var MODES = { cards: 1, list: 1, table: 1 };
  var buttons = Array.prototype.slice.call(document.querySelectorAll("[data-layout]"));
  if (!buttons.length) return;

  var listings = Array.prototype.slice.call(document.querySelectorAll(".listing"));

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
    listings.forEach(function (listing) {
      var cards = listing.querySelector(".cards");
      var table = listing.querySelector(".table-view");
      if (cards) {
        cards.hidden = mode === "table";
        cards.classList.toggle("as-list", mode === "list");
      }
      if (table) table.hidden = mode !== "table";
    });
    buttons.forEach(function (button) {
      button.setAttribute("aria-pressed", button.dataset.layout === mode ? "true" : "false");
    });
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      var mode = MODES[button.dataset.layout] ? button.dataset.layout : "cards";
      write(mode);
      apply(mode);
    });
  });

  var stored = read();
  apply(MODES[stored] ? stored : "cards");

  /* Auto / Light / Dark. "Auto" removes the stamp rather than writing a
     value, so the page goes back to following `prefers-color-scheme` and
     keeps following it when the system flips. */
  function themeSwitch() {
    var THEME_KEY = "pomdp-results-theme";
    var choices = Array.prototype.slice.call(document.querySelectorAll("[data-theme-choice]"));
    if (!choices.length) return;

    function current() {
      try {
        var value = window.localStorage.getItem(THEME_KEY);
        return value === "dark" || value === "light" ? value : "system";
      } catch (e) {
        return "system";
      }
    }

    function applyTheme(theme) {
      if (theme === "system") {
        delete document.documentElement.dataset.theme;
      } else {
        document.documentElement.dataset.theme = theme;
      }
      choices.forEach(function (button) {
        button.setAttribute(
          "aria-pressed",
          button.dataset.themeChoice === theme ? "true" : "false"
        );
      });
    }

    choices.forEach(function (button) {
      button.addEventListener("click", function () {
        var theme = button.dataset.themeChoice;
        try {
          if (theme === "system") window.localStorage.removeItem(THEME_KEY);
          else window.localStorage.setItem(THEME_KEY, theme);
        } catch (e) {
          /* The switch still works for this page; it just will not be kept. */
        }
        applyTheme(theme);
      });
    });

    applyTheme(current());
  }
})();
