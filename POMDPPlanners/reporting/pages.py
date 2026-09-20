# SPDX-License-Identifier: MIT

"""The site's HTML.

Hand-written strings rather than a template engine, because the site must run
from a plain install with no extra dependency, and because there are six
pages. Every value that comes from a run — an environment name, a policy name,
a metric key — is written by :func:`html` and is therefore escaped: those
names come from user code and reach the page unchanged.
"""

from datetime import datetime, timezone
from html import escape
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

from POMDPPlanners.core.simulation.traces import ArtifactKind
from POMDPPlanners.reporting import charts
from POMDPPlanners.reporting.artifacts import EpisodeArtifact, preferred
from POMDPPlanners.reporting.store import (
    EnvironmentView,
    ExperimentView,
    PolicyView,
    RunView,
)


def html(value: object) -> str:
    """Escape a value for insertion into HTML text or an attribute.

    Args:
        value: Anything; it is stringified first.

    Returns:
        The escaped string.
    """
    return escape(str(value), quote=True)


def _url(*parts: object) -> str:
    return "/" + "/".join(quote(str(p), safe="") for p in parts)


def run_url(run: RunView) -> str:
    """URL of a run's page."""
    return _url("run", run.store_index, run.experiment_id, run.run_id)


def env_url(run: RunView, env: str) -> str:
    """URL of one environment's page within a run."""
    return _url("run", run.store_index, run.experiment_id, run.run_id, "env", env)


def policy_url(run: RunView, env: str, policy: str) -> str:
    """URL of one planner's page within a run."""
    return _url("run", run.store_index, run.experiment_id, run.run_id, "env", env, "policy", policy)


def episode_url(run: RunView, env: str, policy: str, index: int) -> str:
    """URL of one episode's page."""
    return policy_url(run, env, policy) + f"/episode/{index}"


def artifact_url(run: RunView, relative_path: str) -> str:
    """URL that serves one artifact's bytes."""
    prefix = _url("artifact", run.store_index, run.experiment_id, run.run_id)
    return prefix + "/" + "/".join(quote(p, safe="") for p in relative_path.split("/"))


def _timestamp(value: Optional[int]) -> str:
    if not value:
        return "—"
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def layout(title: str, breadcrumbs: Sequence[Tuple[str, Optional[str]]], body: str) -> str:
    """Wrap a page body in the shared shell.

    Args:
        title: Page title, also the browser tab's text.
        breadcrumbs: ``(label, href)`` pairs; a ``None`` href is the current
            page and is not linked.
        body: The page's inner HTML.

    Returns:
        A complete HTML document.
    """
    crumbs = " / ".join(
        f'<a href="{html(href)}">{html(label)}</a>' if href else f"<span>{html(label)}</span>"
        for label, href in breadcrumbs
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        f"<title>{html(title)}</title>"
        '<link rel="stylesheet" href="/static/site.css">'
        # An empty data icon, so the browser does not request /favicon.ico and
        # log a 404 against a server that has nothing to serve there.
        '<link rel="icon" href="data:,">'
        "</head><body>"
        f'<header class="topbar"><a class="brand" href="/">POMDPPlanners results</a>'
        f'<nav class="crumbs">{crumbs}</nav></header>'
        f'<main class="page">{body}</main>'
        "</body></html>"
    )


def _chip(status: str) -> str:
    """A run's status as a coloured chip.

    Args:
        status: MLflow's status string.

    Returns:
        HTML for the chip.
    """
    tone = {"FINISHED": "ok", "FAILED": "bad", "RUNNING": "busy"}.get(status.upper(), "flat")
    return f'<span class="chip chip-{tone}">{html(status.title())}</span>'


def _stats(items: Sequence[Tuple[str, str]]) -> str:
    """A row of labelled figures.

    Args:
        items: ``(label, value)`` pairs; a pair whose value is empty is skipped
            rather than shown as a blank figure.

    Returns:
        HTML for the row, or an empty string when nothing is worth showing.
    """
    cells = "".join(
        f'<div class="stat"><span class="stat-label">{html(label)}</span>'
        f'<span class="stat-value">{value}</span></div>'
        for label, value in items
        if value
    )
    return f'<div class="stats">{cells}</div>' if cells else ""


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    head = "".join(f"<th>{html(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    if not body:
        return '<p class="empty">Nothing here.</p>'
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def index_page(experiments: Sequence[ExperimentView], roots: Sequence[object]) -> str:
    """The landing page: every experiment found under the served roots."""
    cards = []
    for experiment in experiments:
        latest = experiment.runs[0] if experiment.runs else None
        envs = sorted({e.name for run in experiment.runs for e in run.environments})
        pills = " ".join(f'<span class="pill">{html(name)}</span>' for name in envs[:6])
        href = _url("experiment", experiment.store_index, experiment.experiment_id)
        cards.append(
            f'<a class="card" href="{html(href)}">'
            f'<h2 class="card-title">{html(experiment.name)}</h2>'
            + _stats(
                [
                    ("Runs", str(len(experiment.runs))),
                    ("Environments", str(len(envs)) if envs else ""),
                    ("Latest run", html(_timestamp(latest.start_time if latest else None))),
                ]
            )
            + (f'<p class="pills">{pills}</p>' if pills else "")
            + f'<p class="path"><code>{html(experiment.store_path)}</code></p>'
            "</a>"
        )
    served = ", ".join(f"<code>{html(root)}</code>" for root in roots)
    body = (
        "<h1>Experiments</h1>"
        f'<p class="note">{len(experiments)} experiment(s), read from the MLflow stores '
        f"under {served}.</p>"
    )
    if not cards:
        body += '<p class="empty">No MLflow store here holds a finished run.</p>'
    else:
        body += f'<div class="cards">{"".join(cards)}</div>'
    return layout("Experiments", [("Experiments", None)], body)


def experiment_page(experiment: ExperimentView) -> str:
    """One experiment's runs."""
    cards = []
    for run in experiment.runs:
        pills = " ".join(f'<span class="pill">{html(e.name)}</span>' for e in run.environments)
        episodes = sum(len(p.episodes) for e in run.environments for p in e.policies)
        cards.append(
            f'<a class="card" href="{html(run_url(run))}">'
            f'<h2 class="card-title">{html(run.run_name)} {_chip(run.status)}</h2>'
            + _stats(
                [
                    ("Started", html(_timestamp(run.start_time))),
                    ("Environments", str(len(run.environments))),
                    ("Episodes", str(episodes) if episodes else ""),
                ]
            )
            + (f'<p class="pills">{pills}</p>' if pills else "")
            + f'<p class="path"><code>{html(run.run_id)}</code></p>'
            "</a>"
        )
    return layout(
        experiment.name,
        [("Experiments", "/"), (experiment.name, None)],
        f"<h1>{html(experiment.name)}</h1>"
        f'<p class="note">{len(experiment.runs)} run(s), newest first.</p>'
        + (f'<div class="cards">{"".join(cards)}</div>' if cards else _table([], [])),
    )


def _metric_table(run: RunView, env: EnvironmentView) -> str:
    per_policy = {p.name: run.metrics_for(env.name, p.name) for p in env.policies}
    names: List[str] = sorted(
        {name for metrics in per_policy.values() for name in charts.base_metric_names(metrics)}
    )
    if not names:
        return '<p class="empty">This run logged no metrics for this environment.</p>'

    rows = []
    for name in names:
        row = [html(name)]
        for policy in env.policies:
            metrics = per_policy[policy.name]
            value = metrics.get(name)
            if value is None:
                row.append('<span class="dim">—</span>')
                continue
            low = metrics.get(name + charts.CI_LOWER_SUFFIX)
            high = metrics.get(name + charts.CI_UPPER_SUFFIX)
            cell = f"{value:.4g}"
            if low is not None and high is not None and high > low:
                cell += f' <span class="dim">[{low:.3g}, {high:.3g}]</span>'
            row.append(cell)
        rows.append(row)
    return _table(["Metric"] + [p.name for p in env.policies], rows)


def _comparison_charts(run: RunView, env: EnvironmentView, limit: int = 6) -> str:
    # A comparison of one planner is a row of single bars: it says nothing the
    # planner's own figures do not, and it pushes everything else off screen.
    if len(env.policies) < 2:
        return ""
    per_policy = {p.name: run.metrics_for(env.name, p.name) for p in env.policies}
    names: List[str] = sorted(
        {name for metrics in per_policy.values() for name in charts.base_metric_names(metrics)}
    )
    # Headline metrics first: these are the ones a comparison is usually about.
    preferred_order = [
        "average_return",
        "goal_reaching_rate",
        "return_cvar",
        "average_actual_num_steps",
    ]
    ordered = [n for n in preferred_order if n in names] + [
        n for n in names if n not in preferred_order
    ]

    blocks = []
    for name in ordered[:limit]:
        series = []
        for policy in env.policies:
            metrics = per_policy[policy.name]
            if name not in metrics:
                continue
            value = metrics[name]
            series.append(
                (
                    policy.name,
                    value,
                    metrics.get(name + charts.CI_LOWER_SUFFIX, value),
                    metrics.get(name + charts.CI_UPPER_SUFFIX, value),
                )
            )
        svg = charts.comparison_svg(name, series)
        if svg:
            blocks.append(f'<figure class="chart-card">{svg}</figure>')
    if not blocks:
        return ""
    return f'<div class="charts">{"".join(blocks)}</div>'


def _policy_cards(run: RunView, env: EnvironmentView) -> str:
    """One card per planner, carrying the figures a comparison is usually about.

    Args:
        run: The run being shown.
        env: The environment whose planners to card.

    Returns:
        HTML for the card grid.
    """
    cards = []
    for policy in env.policies:
        metrics = run.metrics_for(env.name, policy.name)

        def figure(key: str, fmt: str = "{:.3g}") -> str:
            value = metrics.get(key)
            return fmt.format(value) if value is not None else ""

        cards.append(
            f'<a class="card" href="{html(policy_url(run, env.name, policy.name))}">'
            f'<h3 class="card-title">{html(policy.name)}</h3>'
            f'<p class="note">{html(policy.policy_type or "planner")}</p>'
            + _stats(
                [
                    ("Avg. return", figure("average_return")),
                    ("Goal rate", figure("goal_reaching_rate", "{:.0%}")),
                    ("Avg. steps", figure("average_actual_num_steps")),
                    ("Episodes", str(len(policy.episodes)) if policy.episodes else ""),
                ]
            )
            + "</a>"
        )
    return f'<div class="cards">{"".join(cards)}</div>' if cards else ""


def run_page(run: RunView) -> str:
    """One run: its environments, and the comparison across planners in each."""
    sections = []
    for env in run.environments:
        sections.append(
            f'<section class="env"><h2><a href="{html(env_url(run, env.name))}">'
            f"{html(env.name)}</a></h2>"
            + _policy_cards(run, env)
            + _comparison_charts(run, env)
            + f"<details><summary>All metrics</summary>{_metric_table(run, env)}</details>"
            + "</section>"
        )

    params = _table(
        ["Parameter", "Value"],
        [[html(k), html(v)] for k, v in sorted(run.params.items())],
    )
    return layout(
        run.run_name,
        [
            ("Experiments", "/"),
            (run.experiment_name, _url("experiment", run.store_index, run.experiment_id)),
            (run.run_name, None),
        ],
        f"<h1>{html(run.run_name)} {_chip(run.status)}</h1>"
        f'<p class="note">Started {html(_timestamp(run.start_time))} · '
        f"run <code>{html(run.run_id)}</code></p>"
        + "".join(sections)
        + f"<details><summary>Run parameters</summary>{params}</details>",
    )


def environment_page(run: RunView, env: EnvironmentView) -> str:
    """One environment within one run: its planners and their episode counts."""
    env_plots = "".join(
        f'<figure><img src="{html(artifact_url(run, env.name + "/" + a.relative_path))}" '
        f'alt="{html(a.relative_path)}"><figcaption>{html(a.relative_path)}</figcaption></figure>'
        for a in env.artifacts
        if a.kind is ArtifactKind.PLOT
    )
    return layout(
        env.name,
        [
            ("Experiments", "/"),
            (run.experiment_name, _url("experiment", run.store_index, run.experiment_id)),
            (run.run_name, run_url(run)),
            (env.name, None),
        ],
        f"<h1>{html(env.name)}</h1>"
        f'<p class="note">{len(env.policies)} planner(s) in run '
        f'<a href="{html(run_url(run))}">{html(run.run_name)}</a>.</p>'
        + _policy_cards(run, env)
        + _comparison_charts(run, env)
        + (f'<h2>Plots</h2><section class="gallery">{env_plots}</section>' if env_plots else "")
        + f"<details><summary>All metrics</summary>{_metric_table(run, env)}</details>",
    )


def policy_page(run: RunView, env: EnvironmentView, policy: PolicyView) -> str:
    """One planner on one environment: every episode it ran."""
    cards = []
    for index, artifacts in policy.episodes.items():
        summary = next((a.summary for a in artifacts if a.summary), None)
        # The recorded path is the thumbnail when the episode drew one: a
        # grid of paths says more about a planner at a glance than a grid of
        # identical first frames would.
        thumb_artifact = next(
            (a for a in artifacts if a.kind in (ArtifactKind.GIF, ArtifactKind.PLOT)), None
        )
        thumb = (
            f'<img loading="lazy" class="thumb" alt="Recorded path of episode {index}" '
            f'src="{html(artifact_url(run, f"{env.name}/{policy.name}/{thumb_artifact.relative_path}"))}">'
            if thumb_artifact
            else '<div class="thumb thumb-empty"><span>No recorded path</span></div>'
        )
        outcome = ""
        if summary and summary.reach_terminal_state is not None:
            outcome = "terminal" if summary.reach_terminal_state else "out of steps"
        cards.append(
            f'<a class="card episode-card" '
            f'href="{html(episode_url(run, env.name, policy.name, index))}">'
            f"{thumb}"
            f'<h3 class="card-title">Episode {index}</h3>'
            + _stats(
                [
                    (
                        "Return",
                        f"{summary.discounted_return:.2f}"
                        if summary and summary.discounted_return is not None
                        else "",
                    ),
                    ("Steps", str(summary.num_steps) if summary and summary.num_steps else ""),
                    ("Ended", html(outcome)),
                ]
            )
            + "</a>"
        )
    plots = "".join(
        f'<figure><img src="{html(artifact_url(run, f"{env.name}/{policy.name}/{a.relative_path}"))}" '
        f'alt="{html(a.relative_path)}"><figcaption>{html(a.relative_path)}</figcaption></figure>'
        for a in policy.plots
        if a.kind is ArtifactKind.PLOT
    )
    return layout(
        f"{policy.name} on {env.name}",
        [
            ("Experiments", "/"),
            (run.experiment_name, _url("experiment", run.store_index, run.experiment_id)),
            (run.run_name, run_url(run)),
            (env.name, env_url(run, env.name)),
            (policy.name, None),
        ],
        f"<h1>{html(policy.name)}</h1>"
        f'<p class="note">{len(policy.episodes)} episode(s) on '
        f'<a href="{html(env_url(run, env.name))}">{html(env.name)}</a> · '
        "each card shows that episode's recorded path.</p>"
        + (
            f'<div class="cards episodes">{"".join(cards)}</div>'
            if cards
            else '<p class="empty">This planner produced no episode artifacts.</p>'
        )
        + (f'<h2>Plots</h2><section class="gallery">{plots}</section>' if plots else ""),
    )


def view_label(artifact: EpisodeArtifact) -> str:
    """Name one artifact as a way of watching the episode.

    The label says what the viewer will see, not which file it came from:
    "3D replay" and "Recorded path" are the two ways the same episode is shown,
    and a GIF that is not a path — a board, a grid — is not called one.

    Args:
        artifact: The artifact to label.

    Returns:
        A short label for its tab.
    """
    if artifact.kind is ArtifactKind.TRACE:
        return "3D replay"
    if artifact.kind is ArtifactKind.VIDEO:
        return "Recorded video"
    if artifact.kind is ArtifactKind.GIF:
        name = artifact.relative_path.rsplit("/", 1)[-1]
        return "Recorded path" if "path" in name else "Recorded animation"
    return "Plot"


def _player_html(run: RunView, env: str, policy: str, artifact: EpisodeArtifact) -> str:
    """Build the player for one artifact, chosen from its kind alone."""
    src = artifact_url(run, f"{env}/{policy}/{artifact.relative_path}")
    if artifact.player == "video":
        return (
            f'<video class="player" controls preload="metadata" playsinline '
            f'src="{html(src)}"></video>'
        )
    if artifact.player == "image":
        return f'<img class="player" src="{html(src)}" alt="{html(view_label(artifact))}">'
    # trace-viewer
    return (
        f'<div class="viewer" id="viewer" data-trace="{html(src)}" '
        f'data-payload-kind="{html(artifact.payload_kind or "")}">'
        '<canvas id="viewer-canvas"></canvas>'
        '<div class="viewer-hud">'
        '<span id="hud-step">step —</span>'
        '<span id="hud-action">action —</span>'
        '<span id="hud-pos">x — y —</span>'
        '<span id="hud-reward">reward —</span>'
        '<span id="hud-return">return —</span>'
        '<span id="hud-belief">belief —</span>'
        "</div>"
        '<div class="viewer-bar">'
        '<button id="play" type="button">Pause</button>'
        '<input id="scrub" type="range" min="0" max="1" step="0.01" value="0">'
        '<select id="speed"><option value="0.5">0.5x</option>'
        '<option value="1" selected>1x</option><option value="2">2x</option></select>'
        '<span class="cams">'
        '<button type="button" data-cam="board" aria-pressed="true">Board</button>'
        '<button type="button" data-cam="overhead" aria-pressed="false">Top</button>'
        '<button type="button" data-cam="chase" aria-pressed="false">Chase</button>'
        '<button type="button" data-cam="orbit" aria-pressed="false">Orbit</button>'
        "</span></div>"
        "</div>"
        '<p class="viewer-status" id="viewer-status">Loading trace…</p>'
        '<script src="/static/vendor/three.min.js"></script>'
        '<script src="/static/viewer/renderer-core.js"></script>'
        f'<script src="{scene_script_path(artifact.payload_kind or "")}"></script>'
        '<script src="/static/viewer/trace-player.js"></script>'
    )


def scene_script_path(payload_kind: str) -> str:
    """Path of the scene module that draws ``payload_kind``.

    Derived by convention rather than looked up in a table, so adding an
    environment means adding one file and editing nothing: a kind of
    ``light_dark.v1`` loads ``scenes/light-dark.js``, which registers itself
    under its own kind. A table here would be a merge conflict every time a new
    environment is migrated, and a second place to forget to update.
    """
    name = payload_kind.split(".", 1)[0].replace("_", "-")
    return f"/static/viewer/scenes/{name}.js"


def episode_page(
    run: RunView,
    env: EnvironmentView,
    policy: PolicyView,
    index: int,
    artifacts: Sequence[EpisodeArtifact],
) -> str:
    """One episode, with every way its artifacts allow of watching it.

    An episode usually leaves two records of the same run: the trace, which the
    3D viewer replays, and the recorded path the environment drew while it ran.
    Both are offered, as tabs over one another, because they answer different
    questions — the replay shows what the planner believed, the path shows
    where it went.
    """
    chosen = preferred(artifacts)
    # Every artifact that can be shown, preferred one first, so the tab that is
    # open on arrival is the richest view the episode produced.
    playable = [chosen] if chosen else []
    playable += [a for a in artifacts if a is not chosen and a.kind is not ArtifactKind.TRACE]

    tabs = "".join(
        f'<button type="button" class="tab" data-view="{i}" '
        f'aria-pressed="{"true" if i == 0 else "false"}">{html(view_label(a))}</button>'
        for i, a in enumerate(playable)
    )
    panels = "".join(
        f'<div class="view" data-view="{i}"{"" if i == 0 else " hidden"}>'
        f"{_player_html(run, env.name, policy.name, a)}</div>"
        for i, a in enumerate(playable)
    )
    views = (
        f'<div class="views"><div class="tabs">{tabs}</div>{panels}</div>'
        + '<script src="/static/viewer/views.js"></script>'
        if playable
        else '<p class="empty">This episode produced no artifact the site can play.</p>'
    )

    summary = next((a.summary for a in artifacts if a.summary), None)
    facts = _stats(
        [
            (
                "Discounted return",
                f"{summary.discounted_return:.2f}"
                if summary and summary.discounted_return is not None
                else "",
            ),
            ("Steps", str(summary.num_steps) if summary and summary.num_steps else ""),
            (
                "Ended",
                ("terminal state" if summary.reach_terminal_state else "out of steps")
                if summary and summary.reach_terminal_state is not None
                else "",
            ),
            ("Views", str(len(playable)) if len(playable) > 1 else ""),
        ]
    )

    others = "".join(
        f'<li><a href="{html(artifact_url(run, f"{env.name}/{policy.name}/{a.relative_path}"))}">'
        f'{html(a.relative_path)}</a> <span class="dim">({html(a.kind.value)})</span></li>'
        for a in artifacts
    )
    return layout(
        f"Episode {index} · {policy.name}",
        [
            ("Experiments", "/"),
            (run.experiment_name, _url("experiment", run.store_index, run.experiment_id)),
            (run.run_name, run_url(run)),
            (env.name, env_url(run, env.name)),
            (policy.name, policy_url(run, env.name, policy.name)),
            (f"Episode {index}", None),
        ],
        f"<h1>Episode {index}</h1>"
        f'<p class="note"><a href="{html(policy_url(run, env.name, policy.name))}">'
        f"{html(policy.name)}</a> on "
        f'<a href="{html(env_url(run, env.name))}">{html(env.name)}</a></p>'
        f"{facts}{views}"
        f'<details><summary>Files</summary><ul class="files">{others}</ul></details>',
    )


def not_found(message: str) -> str:
    """A 404 body."""
    return layout(
        "Not found",
        [("Experiments", "/"), ("Not found", None)],
        f'<h1>Not found</h1><p class="empty">{html(message)}</p>',
    )
