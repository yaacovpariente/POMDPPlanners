// SPDX-License-Identifier: MIT
//
// Hover tips for the tuning charts. Every mark carries an SVG <title>, which
// the browser already shows on hover, so the charts work without this file.
// The native tip is slow to appear and cannot be styled, so this moves each
// title into a data attribute and shows it in a themed box beside the pointer.
(function () {
  "use strict";

  var tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.hidden = true;
  document.body.appendChild(tip);

  var marks = document.querySelectorAll(".tuning-chart .pt");
  marks.forEach(function (mark) {
    var title = mark.querySelector("title");
    if (!title) return;
    mark.setAttribute("data-tip", title.textContent);
    // Removed, not hidden: a <title> left in place still raises the browser's
    // own tip, and two tips at once read as a glitch.
    title.remove();
    mark.setAttribute("tabindex", "0");
  });

  function show(mark, x, y) {
    tip.textContent = mark.getAttribute("data-tip");
    tip.hidden = false;
    var box = tip.getBoundingClientRect();
    var left = Math.min(x + 14, window.innerWidth - box.width - 8);
    var top = y - box.height - 12 < 8 ? y + 16 : y - box.height - 12;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = top + "px";
  }

  function hide() {
    tip.hidden = true;
  }

  document.addEventListener("mousemove", function (event) {
    var mark = event.target.closest && event.target.closest(".tuning-chart .pt[data-tip]");
    if (mark) show(mark, event.clientX, event.clientY);
    else hide();
  });
  document.addEventListener("focusin", function (event) {
    var mark = event.target.closest && event.target.closest(".tuning-chart .pt[data-tip]");
    if (!mark) return hide();
    var rect = mark.getBoundingClientRect();
    show(mark, rect.right, rect.top);
  });
  document.addEventListener("scroll", hide, true);
})();
