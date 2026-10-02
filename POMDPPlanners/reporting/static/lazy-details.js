/* SPDX-License-Identifier: MIT
 *
 * Collapsed sections whose contents are built only when first opened.
 *
 * A study can run thousands of trials and an evaluation hundreds of episodes.
 * The server writes such a section's contents into a <template>, which the
 * browser parses but does not lay out, load images for, or run scripts from.
 * Opening the section moves the contents into the page. Scripts inside it are
 * then run in order, because a script moved out of a template never runs on
 * its own; one the page has already loaded is not loaded twice.
 */
(function () {
  "use strict";

  // A page with two collapsed sections includes this file twice; once is enough.
  if (window.POMDPLazyDetails) return;
  window.POMDPLazyDetails = true;

  function runScripts(sources, done) {
    var next = sources.shift();
    if (!next) return done();
    var loaded = Array.prototype.some.call(document.scripts, function (script) {
      return script.getAttribute("src") === next;
    });
    if (loaded) return runScripts(sources, done);
    var script = document.createElement("script");
    script.src = next;
    script.onload = script.onerror = function () {
      runScripts(sources, done);
    };
    document.body.appendChild(script);
  }

  function expand(section) {
    var template = section.querySelector(":scope > template");
    if (!template) return;
    var content = template.content.cloneNode(true);
    var sources = [];
    Array.prototype.slice.call(content.querySelectorAll("script[src]")).forEach(function (s) {
      sources.push(s.getAttribute("src"));
      s.remove();
    });
    template.replaceWith(content);
    runScripts(sources, function () {
      // The view picker in layout.js styles listings present at load; this
      // tells it a new one has arrived.
      document.dispatchEvent(new CustomEvent("pomdp:inserted", { detail: section }));
    });
  }

  // Delegated, not bound per section: this file is included after the first
  // collapsed section, before any later one exists. "toggle" does not bubble,
  // so it is caught on the way down.
  document.addEventListener(
    "toggle",
    function (event) {
      var section = event.target;
      if (section.matches && section.matches("details[data-lazy]") && section.open) {
        expand(section);
      }
    },
    true
  );
})();
