# SPDX-License-Identifier: MIT
"""``.. episode-viewer::`` — replay a recorded episode in 3D on a docs page.

Usage, in any page::

    .. episode-viewer:: traces/tiger.json

       Optional caption, parsed as reStructuredText.

The argument is a trace JSON written by ``Environment.cache_trace``, relative
to the page (or to the docs root when it starts with ``/``). The page gets an
``<iframe>`` onto a small generated page that runs the same viewer the results
site runs: three.js, ``renderer-core.js``, the trace's scene module and
``trace-player.js``.

Why an iframe per viewer. ``trace-player.js`` finds its canvas, HUD and
controls by id and expects one viewer per document. The maze, push and laser
tag pages each show several worlds. An iframe gives every viewer its own
document, so the player runs unchanged, two scene modules never share a page's
globals, and a viewer below the fold is not started until it is scrolled to
(``loading="lazy"``), which keeps a page from opening several WebGL contexts
at once.

Why the viewer is copied at build time. The JavaScript lives in the package,
under ``POMDPPlanners/reporting/static``, and is served from there by the
results site. The build copies it into ``_static/pomdp-viewer/`` rather than
keeping a second copy under ``docs/``, so a fix to a scene reaches the docs on
the next build and there is nothing to keep in step by hand. The viewer markup
comes from :func:`POMDPPlanners.reporting.pages.trace_viewer_html` for the same
reason.

The build fails, rather than rendering an empty frame, when a trace is missing,
is not a trace, or names a payload kind that has no scene module.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri

# Everything the build writes lives under this directory of the HTML output.
OUTPUT_DIR = "_static/pomdp-viewer"


def _package_static() -> Path:
    """The package's viewer assets, the one copy the results site also serves."""
    from POMDPPlanners.reporting import pages  # pylint: disable=import-outside-toplevel

    return Path(pages.__file__).resolve().parent / "static"


def _scene_file(payload_kind: str) -> Path:
    """The scene module that draws ``payload_kind``, by the results site's rule."""
    from POMDPPlanners.reporting.pages import (  # pylint: disable=import-outside-toplevel
        scene_script_path,
    )

    relative = scene_script_path(payload_kind, static_root="").lstrip("/")
    return _package_static() / relative


class episode_viewer(nodes.General, nodes.Element):  # pylint: disable=invalid-name
    """One embedded viewer; its children are the caption."""


def _visit_html(self: Any, node: episode_viewer) -> None:
    page = self.builder.get_target_uri(self.builder.current_docname)
    src = relative_uri(page, f"{OUTPUT_DIR}/embed/{node['stem']}.html")
    title = self.encode(node["title"])
    self.body.append(
        '<figure class="episode-viewer">'
        f'<iframe src="{self.encode(src)}" title="{title}" loading="lazy" '
        'allow="fullscreen" '
        'style="display:block;width:100%;aspect-ratio:16/10;border:0;border-radius:8px;">'
        "</iframe>"
    )
    if node.children:
        self.body.append("<figcaption>")


def _depart_html(self: Any, node: episode_viewer) -> None:
    if node.children:
        self.body.append("</figcaption>")
    self.body.append("</figure>\n")


def _skip(self: Any, node: episode_viewer) -> None:
    del self, node
    raise nodes.SkipNode


class EpisodeViewerDirective(SphinxDirective):
    """``.. episode-viewer:: <trace.json>`` with an optional caption body."""

    required_arguments = 1
    has_content = True
    option_spec = {"title": directives.unchanged}

    def run(self) -> List[nodes.Node]:
        relative, absolute = self.env.relfn2path(self.arguments[0], self.env.docname)
        source = Path(absolute)
        where = f"{self.env.doc2path(self.env.docname)}:{self.lineno}"
        if not source.is_file():
            raise ExtensionError(f"{where}: episode-viewer trace {relative} does not exist.")
        try:
            trace = json.loads(source.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ExtensionError(f"{where}: {relative} is not JSON ({error}).") from error
        kind = trace.get("payload_kind") if isinstance(trace, dict) else None
        if not isinstance(kind, str) or not kind:
            raise ExtensionError(f"{where}: {relative} has no payload_kind; not a trace.")
        if not _scene_file(kind).is_file():
            raise ExtensionError(
                f"{where}: {relative} has payload kind {kind!r}, and no scene module "
                f"draws it ({_scene_file(kind).name} is missing)."
            )

        stem = source.stem
        known: Dict[str, str] = self.env.episode_viewer_traces  # type: ignore[attr-defined]
        other = known.get(stem)
        if other is not None and other["source"] != str(source):
            raise ExtensionError(
                f"{where}: two traces share the file name {stem}.json "
                f"({other['source']} and {source}); rename one."
            )
        known[stem] = {"source": str(source), "kind": kind, "docname": self.env.docname}
        self.env.note_dependency(str(source))

        node = episode_viewer()
        node["stem"] = stem
        node["title"] = self.options.get(
            "title", f"3D replay of a recorded {kind.split('.', 1)[0].replace('_', ' ')} episode"
        )
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def _init_env(app: Sphinx, env: Any, docnames: List[str]) -> None:
    del app, docnames
    if not hasattr(env, "episode_viewer_traces"):
        env.episode_viewer_traces = {}


def _purge(app: Sphinx, env: Any, docname: str) -> None:
    del app
    traces = getattr(env, "episode_viewer_traces", {})
    for stem in [s for s, info in traces.items() if info["docname"] == docname]:
        del traces[stem]


def _merge(app: Sphinx, env: Any, docnames: List[str], other: Any) -> None:
    del app, docnames
    _init_env(None, env, [])  # type: ignore[arg-type]
    env.episode_viewer_traces.update(getattr(other, "episode_viewer_traces", {}))


EMBED_PAGE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="../site.css">
<style>
  /* The viewer fills the frame's viewport, less one line for the status.
     Its height is fixed rather than flexed: a scene may add a panel of its
     own under the viewport (firefighting draws its wind belief there), and
     a flexed viewer is what that panel squeezes to zero height. The panel
     scrolls below instead. */
  html, body {{ margin: 0; background: transparent; }}
  /* The docs theme is light. site.css allows dark too, and a frame whose
     colour scheme differs from its page is painted with an opaque canvas,
     which showed as a black band behind the status line. */
  :root {{ color-scheme: light; }}
  .viewer {{ aspect-ratio: auto; height: calc(100vh - 30px); }}
  .viewer-status {{ margin: 6px 2px 0; font-size: 12px; line-height: 18px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
</style>
</head>
<body>
{viewer}
<script>
  /* A browser will not let a file:// page fetch its trace. Say how to fix
     that instead of leaving the bare "Failed to fetch" the player reports. */
  if (location.protocol === "file:") {{
    var status = document.getElementById("viewer-status");
    status.textContent = "The 3D replay needs the docs served over HTTP: "
      + "python -m http.server -d docs/_build/html";
    status.classList.add("failed");
  }}
</script>
</body>
</html>
"""


def _write_assets(app: Sphinx, exception: Exception | None) -> None:
    """Copy the package's viewer and write one embed page per trace."""
    if exception is not None or app.builder.format != "html":
        return
    from POMDPPlanners.reporting.pages import (  # pylint: disable=import-outside-toplevel
        trace_viewer_html,
    )

    static = _package_static()
    out = Path(app.outdir) / OUTPUT_DIR
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(static / "viewer", out / "viewer")
    (out / "vendor").mkdir(parents=True)
    shutil.copy2(static / "vendor" / "three.min.js", out / "vendor" / "three.min.js")
    shutil.copy2(static / "site.css", out / "site.css")

    (out / "traces").mkdir()
    (out / "embed").mkdir()
    traces: Dict[str, Dict[str, str]] = getattr(app.env, "episode_viewer_traces", {})
    for stem, info in sorted(traces.items()):
        shutil.copy2(info["source"], out / "traces" / f"{stem}.json")
        title = f"{stem.replace('_', ' ')} episode replay"
        page = EMBED_PAGE.format(
            title=title,
            viewer=trace_viewer_html(f"../traces/{stem}.json", info["kind"], static_root=".."),
        )
        (out / "embed" / f"{stem}.html").write_text(page, encoding="utf-8")


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_node(
        episode_viewer,
        html=(_visit_html, _depart_html),
        latex=(_skip, None),
        text=(_skip, None),
        man=(_skip, None),
        texinfo=(_skip, None),
    )
    app.add_directive("episode-viewer", EpisodeViewerDirective)
    app.connect("env-before-read-docs", _init_env)
    app.connect("env-purge-doc", _purge)
    app.connect("env-merge-info", _merge)
    app.connect("build-finished", _write_assets)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
