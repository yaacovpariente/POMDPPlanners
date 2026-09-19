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


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    head = "".join(f"<th>{html(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    if not body:
        return '<p class="empty">Nothing here.</p>'
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def index_page(experiments: Sequence[ExperimentView], roots: Sequence[object]) -> str:
    """The landing page: every experiment found under the served roots."""
    rows = []
    for experiment in experiments:
        latest = experiment.runs[0] if experiment.runs else None
        rows.append(
            [
                f'<a href="{html(_url("experiment", experiment.store_index, experiment.experiment_id))}">'
                f"{html(experiment.name)}</a>",
                str(len(experiment.runs)),
                html(_timestamp(latest.start_time if latest else None)),
                f'<code class="dim">{html(experiment.store_path)}</code>',
            ]
        )
    served = "".join(f"<li><code>{html(root)}</code></li>" for root in roots)
    return layout(
        "Experiments",
        [("Experiments", None)],
        "<h1>Experiments</h1>"
        f'<p class="note">Read from {len(experiments)} experiment(s) across the MLflow stores '
        f'found under:</p><ul class="dim">{served}</ul>'
        + _table(["Experiment", "Runs", "Latest run", "Store"], rows),
    )


def experiment_page(experiment: ExperimentView) -> str:
    """One experiment's runs."""
    rows = []
    for run in experiment.runs:
        env_names = ", ".join(html(e.name) for e in run.environments) or "—"
        rows.append(
            [
                f'<a href="{html(run_url(run))}">{html(run.run_name)}</a>',
                html(run.status),
                html(_timestamp(run.start_time)),
                env_names,
                f'<code class="dim">{html(run.run_id)}</code>',
            ]
        )
    return layout(
        experiment.name,
        [("Experiments", "/"), (experiment.name, None)],
        f"<h1>{html(experiment.name)}</h1>"
        + _table(["Run", "Status", "Started", "Environments", "Run id"], rows),
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


def run_page(run: RunView) -> str:
    """One run: its environments, and the comparison across planners in each."""
    sections = []
    for env in run.environments:
        policy_links = " ".join(
            f'<a class="pill" href="{html(policy_url(run, env.name, p.name))}">{html(p.name)}</a>'
            for p in env.policies
        )
        sections.append(
            f'<section class="env"><h2><a href="{html(env_url(run, env.name))}">'
            f"{html(env.name)}</a></h2>"
            f'<p class="pills">{policy_links}</p>'
            + _comparison_charts(run, env)
            + _metric_table(run, env)
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
        f"<h1>{html(run.run_name)}</h1>"
        f'<p class="note">{html(run.status)} · started {html(_timestamp(run.start_time))} · '
        f"run <code>{html(run.run_id)}</code></p>"
        + "".join(sections)
        + f"<details><summary>Run parameters</summary>{params}</details>",
    )


def environment_page(run: RunView, env: EnvironmentView) -> str:
    """One environment within one run: its planners and their episode counts."""
    rows = []
    for policy in env.policies:
        episodes = policy.episodes
        kinds = sorted({a.kind.value for a in policy.artifacts})
        rows.append(
            [
                f'<a href="{html(policy_url(run, env.name, policy.name))}">{html(policy.name)}</a>',
                html(policy.policy_type or "—"),
                str(len(episodes)),
                ", ".join(html(k) for k in kinds) or "—",
            ]
        )
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
        + _comparison_charts(run, env)
        + _table(["Planner", "Type", "Episodes", "Artifact kinds"], rows)
        + _metric_table(run, env)
        + (f'<section class="gallery">{env_plots}</section>' if env_plots else ""),
    )


def policy_page(run: RunView, env: EnvironmentView, policy: PolicyView) -> str:
    """One planner on one environment: every episode it ran."""
    rows = []
    for index, artifacts in policy.episodes.items():
        chosen = preferred(artifacts)
        kinds = ", ".join(sorted({a.kind.value for a in artifacts}))
        rows.append(
            [
                f'<a href="{html(episode_url(run, env.name, policy.name, index))}">'
                f"Episode {index}</a>",
                html(kinds),
                html(chosen.player if chosen else "—"),
            ]
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
        f'<p class="note">on <a href="{html(env_url(run, env.name))}">{html(env.name)}</a></p>'
        + _table(["Episode", "Artifacts", "Player"], rows)
        + (f'<section class="gallery">{plots}</section>' if plots else ""),
    )


def _player_html(run: RunView, env: str, policy: str, artifact: EpisodeArtifact) -> str:
    """Build the player for one artifact, chosen from its kind alone."""
    src = artifact_url(run, f"{env}/{policy}/{artifact.relative_path}")
    if artifact.player == "video":
        return (
            f'<video class="player" controls preload="metadata" playsinline '
            f'src="{html(src)}"></video>'
        )
    if artifact.player == "image":
        return f'<img class="player" src="{html(src)}" alt="Episode replay">'
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
        '<script src="/static/viewer/scenes/light-dark.js"></script>'
        '<script src="/static/viewer/trace-player.js"></script>'
    )


def episode_page(
    run: RunView,
    env: EnvironmentView,
    policy: PolicyView,
    index: int,
    artifacts: Sequence[EpisodeArtifact],
) -> str:
    """One episode, with the player its artifacts earn."""
    chosen = preferred(artifacts)
    player = (
        _player_html(run, env.name, policy.name, chosen)
        if chosen
        else '<p class="empty">This episode produced no artifact the site can play.</p>'
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
        f'<p class="note">{html(policy.name)} on {html(env.name)} · '
        f"played as <strong>{html(chosen.kind.value if chosen else 'nothing')}</strong></p>"
        f"{player}"
        f"<h2>Files</h2><ul>{others}</ul>",
    )


def not_found(message: str) -> str:
    """A 404 body."""
    return layout(
        "Not found",
        [("Experiments", "/"), ("Not found", None)],
        f'<h1>Not found</h1><p class="empty">{html(message)}</p>',
    )
