# SPDX-License-Identifier: MIT

"""The site's HTML.

Hand-written strings rather than a template engine, because the site must run
from a plain install with no extra dependency, and because there are six
pages. Every value that comes from a run — an environment name, a policy name,
a metric key — is written by :func:`html` and is therefore escaped: those
names come from user code and reach the page unchanged.
"""

import json

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

from POMDPPlanners.core.simulation import tuning_run_layout as run_layout
from POMDPPlanners.core.simulation.metrics import CommonMetricName
from POMDPPlanners.core.simulation.traces import ArtifactKind
from POMDPPlanners.reporting import charts, tuning, tuning_charts
from POMDPPlanners.reporting.artifacts import EpisodeArtifact, preferred
from POMDPPlanners.reporting.scenes import scene_script_url
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
    return _url(
        "run",
        run.store_index,
        run.experiment_id,
        run.run_id,
        "env",
        env,
        "policy",
        policy,
    )


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
        (f'<a href="{html(href)}">{html(label)}</a>' if href else f"<span>{html(label)}</span>")
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
        # Stamped before the first paint: applying the reader's theme from a
        # script at the end of the body shows them the other one first.
        "<script>(function(){try{var t=localStorage.getItem('pomdp-results-theme');"
        "if(t==='dark'||t==='light')document.documentElement.dataset.theme=t;}catch(e){}})();"
        "</script>"
        "</head><body>"
        f'<header class="topbar"><a class="brand" href="/">POMDPPlanners results</a>'
        f'<nav class="crumbs">{crumbs}</nav>'
        f"{_theme_toggle()}</header>"
        f'<main class="page">{body}</main>'
        '<script src="/static/layout.js"></script>'
        "</body></html>"
    )


def _theme_toggle() -> str:
    """The light/dark switch in the top bar.

    Three states rather than two: a reader who has chosen neither should keep
    following their system, which is what most of them want and what the site
    did before there was a switch at all.

    Returns:
        HTML for the switch.
    """
    return (
        '<div class="theme-switch" role="group" aria-label="Theme">'
        + "".join(
            f'<button type="button" class="tab" data-theme-choice="{mode}" '
            f'aria-pressed="false">{label}</button>'
            for mode, label in (
                ("system", "Auto"),
                ("light", "Light"),
                ("dark", "Dark"),
            )
        )
        + "</div>"
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


def _live_toggle() -> str:
    """The switch that plays every episode on the page at once.

    Off by default: a page of playing scenes is work a reader has not asked
    for, and the still frame is enough to choose an episode from.

    Returns:
        HTML for the switch.
    """
    return (
        '<button type="button" class="tab" data-live aria-pressed="false" '
        'title="Play every episode on this page, on repeat">Live</button>'
    )


def _layout_toggle() -> str:
    """The view picker that sits beside a page's heading.

    The choice is the reader's, not the page's: cards to compare a handful of
    runs, a list to scan forty, a table to read the numbers down a column. It
    is remembered per browser, so the site does not argue with whoever set it
    every time they open another page.

    Returns:
        HTML for the picker.
    """
    return (
        '<div class="layout-switch" role="group" aria-label="View">'
        + "".join(
            f'<button type="button" class="tab" data-layout="{mode}" '
            f'aria-pressed="{"true" if mode == "cards" else "false"}">{label}</button>'
            for mode, label in (
                ("cards", "Cards"),
                ("list", "List"),
                ("table", "Table"),
            )
        )
        + "</div>"
    )


@dataclass(frozen=True)
class ListingItem:
    """One thing in a list of things — an experiment, a run, a planner, an episode.

    Every listing on the site is built from these rather than from page-specific
    HTML, so that all three views come from one description of the item and a
    new listing gets them all for free.

    Attributes:
        title: The item's name, as plain text.
        href: Where it links.
        chip: Optional status chip, already HTML.
        subtitle: A line under the title, as plain text.
        fields: ``(label, value)`` figures; the table's columns come from these
            labels, in the order the first item lists them.
        tags: Short labels, shown as pills on a card and joined in the table.
        footnote: A path or id, set in monospace and last.
        thumb: Optional thumbnail HTML.
        env: The environment the item belongs to, as ``data-env`` on its card
            and row, so a page's environment filter can hide it.
    """

    title: str
    href: str
    chip: str = ""
    subtitle: str = ""
    fields: Sequence[Tuple[str, str]] = ()
    tags: Sequence[str] = ()
    footnote: str = ""
    thumb: str = ""
    env: str = ""


def _listing(
    items: Sequence[ListingItem],
    empty: str = "Nothing here.",
    tags_label: str = "Tags",
) -> str:
    """Render one listing in all three views, with only the chosen one shown.

    All three are written into the page rather than fetched or rebuilt on
    demand: a listing is small, and switching view must not cost a round trip
    or lose the reader's place.

    Args:
        items: What to list.
        empty: What to say when there is nothing.
        tags_label: Heading for the tag column, when the items carry tags.

    Returns:
        HTML for the listing.
    """
    if not items:
        return f'<p class="empty">{html(empty)}</p>'

    cards = []
    for item in items:
        cards.append(
            f'<a class="card{" has-thumb" if item.thumb else ""}" href="{html(item.href)}"'
            + (f' data-env="{html(item.env)}"' if item.env else "")
            + ">"
            + item.thumb
            + f'<h3 class="card-title">{html(item.title)}{" " + item.chip if item.chip else ""}</h3>'
            + (f'<p class="note">{html(item.subtitle)}</p>' if item.subtitle else "")
            + _stats(item.fields)
            + (
                '<p class="pills">'
                + " ".join(f'<span class="pill">{html(t)}</span>' for t in item.tags)
                + "</p>"
                if item.tags
                else ""
            )
            + (f'<p class="path"><code>{html(item.footnote)}</code></p>' if item.footnote else "")
            + "</a>"
        )

    # Columns follow the first item that names them, so a listing whose items
    # carry different figures still reads as one table.
    labels: List[str] = []
    for item in items:
        for label, _ in item.fields:
            if label not in labels:
                labels.append(label)
    has_tags = any(item.tags for item in items)
    has_footnote = any(item.footnote for item in items)

    head = (
        "<th>Name</th>"
        + "".join(f"<th>{html(label)}</th>" for label in labels)
        + (f"<th>{html(tags_label)}</th>" if has_tags else "")
        + ("<th>Id</th>" if has_footnote else "")
    )
    body = []
    for item in items:
        values = dict(item.fields)
        cells = [
            f'<td><a href="{html(item.href)}">{html(item.title)}</a>'
            f'{" " + item.chip if item.chip else ""}</td>'
        ]
        cells += [f'<td class="num">{values.get(label) or "—"}</td>' for label in labels]
        if has_tags:
            cells.append(f'<td>{", ".join(html(t) for t in item.tags) or "—"}</td>')
        if has_footnote:
            cells.append(f"<td><code>{html(item.footnote)}</code></td>")
        env_attr = f' data-env="{html(item.env)}"' if item.env else ""
        body.append(f"<tr{env_attr}>" + "".join(cells) + "</tr>")

    return (
        '<div class="listing">'
        f'<div class="cards">{"".join(cards)}</div>'
        '<div class="scroll table-view" hidden>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        "</div></div>"
    )


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


#: How a run's params are grouped for reading, by name prefix, in order. A
#: param matching none of them goes under "Run setup". The prefixes are the
#: ones the simulator and the optimizer write.
PARAMETER_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("Environment", "env_"),
    ("Policy", "policy_"),
    ("Fixed planner settings", "constant_"),
)
RUN_SETUP_GROUP = "Run setup"


def run_parameters(params: Dict[str, str], omit_prefixes: Sequence[str] = ()) -> str:
    """A run's params as one collapsed table per group, prefix stripped.

    One flat table of a config run is sixty rows where the environment's
    settings, the planner's and the run's own are interleaved alphabetically;
    grouped, a reader opens the one they came for.

    Args:
        params: The run's params.
        omit_prefixes: Params to leave out because the page shows them
            elsewhere.

    Returns:
        HTML for the groups that have a param; nothing for an empty one.
    """
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for key, value in sorted(params.items()):
        if any(key.startswith(prefix) for prefix in omit_prefixes):
            continue
        label, name = RUN_SETUP_GROUP, key
        for group, prefix in PARAMETER_GROUPS:
            if key.startswith(prefix) and len(key) > len(prefix):
                # "env__x" is a private attribute logged as-is; its name
                # reads as "x" once the prefix is gone.
                label, name = group, key[len(prefix) :].lstrip("_") or key
                break
        grouped.setdefault(label, []).append((name, value))
    order = [group for group, _ in PARAMETER_GROUPS] + [RUN_SETUP_GROUP]
    return "".join(
        f'<details class="param-group"><summary>{html(group)} parameters '
        f"({len(grouped[group])})</summary>"
        + _table(["Parameter", "Value"], [[html(k), html(v)] for k, v in grouped[group]])
        + "</details>"
        for group in order
        if grouped.get(group)
    )


def index_page(experiments: Sequence[ExperimentView], roots: Sequence[object]) -> str:
    """The landing page: every experiment found under the served roots."""
    items = []
    for experiment in experiments:
        latest = experiment.runs[0] if experiment.runs else None
        envs = sorted({e.name for run in experiment.runs for e in run.environments})
        groups = tuning.group_studies(experiment.runs)
        if groups:
            # A study is several MLflow runs; counting them would count one
            # piece of work three times.
            members = set().union(*(g.member_ids for g in groups))
            others = sum(1 for r in experiment.runs if r.run_id not in members)
            counts = [("Tuning studies", str(len(groups)))] + (
                [("Other runs", str(others))] if others else []
            )
        else:
            counts = [("Runs", str(len(experiment.runs)))]
        items.append(
            ListingItem(
                title=experiment.name,
                href=_url("experiment", experiment.store_index, experiment.experiment_id),
                fields=counts
                + [
                    ("Latest run", _timestamp(latest.start_time if latest else None)),
                ],
                tags=envs,
                footnote=str(experiment.store_path),
            )
        )
    served = ", ".join(f"<code>{html(root)}</code>" for root in roots)
    return layout(
        "Experiments",
        [("Experiments", None)],
        f'<div class="page-head"><h1>Experiments</h1>{_layout_toggle()}</div>'
        f'<p class="note">{len(experiments)} experiment(s), read from the MLflow stores '
        f"under {served}.</p>"
        + _listing(items, "No MLflow store here holds a finished run.", "Environments"),
    )


def experiment_page(experiment: ExperimentView) -> str:
    """One experiment's runs, with each tuning study as one entry.

    A tuning study is stored as a parent run, a run per tuned planner and an
    evaluation run. Listed one card per run, one study reads as three pieces
    of work, two of them repeating the third; so its runs are folded into a
    single card, and the parts are reached from inside it. An experiment with
    no tuning study lists exactly as it always has.
    """
    groups = tuning.group_studies(experiment.runs)
    anchors = {g.anchor.run_id: g for g in groups}
    hidden = set().union(*(g.member_ids for g in groups)) if groups else set()
    items = []
    for run in experiment.runs:
        if run.run_id in anchors:
            items.append(_study_item(anchors[run.run_id]))
            continue
        if run.run_id in hidden:
            continue
        episodes = sum(len(p.episodes) for e in run.environments for p in e.policies)
        items.append(
            ListingItem(
                title=run.run_name,
                href=run_url(run),
                chip=_chip(run.status),
                subtitle=run_kind_label(run),
                fields=[
                    ("Started", _timestamp(run.start_time)),
                    ("Episodes", str(episodes)),
                ],
                tags=[e.name for e in run.environments],
                footnote=run.run_id,
            )
        )
    return layout(
        experiment.name,
        [("Experiments", "/"), (experiment.name, None)],
        f'<div class="page-head"><h1>{html(experiment.name)}</h1>{_layout_toggle()}</div>'
        + (
            f'<p class="note">{len(items)} entr{"y" if len(items) == 1 else "ies"}, newest '
            "first. A tuning study is one entry: its tuned planners and its evaluation are "
            "inside it.</p>"
            if groups
            else f'<p class="note">{len(experiment.runs)} run(s), newest first.</p>'
        )
        + _listing(items, "This experiment has no runs the site can read.", "Environments"),
    )


def _export_attrs(value, low=None, high=None, best_trial=None) -> str:
    """The data attributes a value cell carries for the CSV export.

    Written with ``repr`` so the export has the run's full precision, not the
    four digits the cell shows.
    """
    attrs = f' data-value="{value!r}"'
    if low is not None and high is not None:
        attrs += f' data-low="{low!r}" data-high="{high!r}"'
    if best_trial is not None:
        attrs += f' data-best-trial="{best_trial!r}"'
    return attrs


def _export_buttons(table_id: str, filtered: bool = False) -> str:
    """The CSV export beside a metric table, for every metric table on the site.

    The CSV is long: one row per (environment, planner, metric) with value,
    confidence bounds, episodes and, for a tuned planner, the best trial's
    value. A long table loads into pandas or Excel the same way whatever the
    metrics and planners were, where a wide one would change shape with them.

    Args:
        table_id: The table's id.
        filtered: The table has filters; then the default export follows them
            and a second button exports everything.

    Returns:
        HTML for the buttons and the shared script that handles them.
    """
    buttons = (
        f'<button type="button" class="tab" data-export-table="{html(table_id)}" '
        f'data-export-mode="shown">Export CSV{" (as shown)" if filtered else ""}</button>'
    )
    if filtered:
        buttons += (
            f'<button type="button" class="tab" data-export-table="{html(table_id)}" '
            'data-export-mode="all">Export CSV (all data)</button>'
        )
    return (
        f'<div class="table-export">{buttons}</div>'
        '<script src="/static/table-export.js"></script>'
    )


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text)


def _metric_table(run: RunView, env: EnvironmentView) -> str:
    per_policy = {p.name: run.metrics_for(env.name, p.name) for p in env.policies}
    names: List[str] = sorted(
        {name for metrics in per_policy.values() for name in charts.base_metric_names(metrics)}
    )
    if not names:
        return '<p class="empty">This run logged no metrics for this environment.</p>'

    # Two columns per planner — the estimate and its interval — rather than an
    # interval tucked in beside the number, where it reads as part of it.
    head = (
        "<tr><th rowspan='2'>Metric</th>"
        + "".join(
            f"<th colspan='2' class='group' data-col='{i}' data-planner=\"{html(p.name)}\" "
            f"data-episodes='{len(p.episodes)}'>{html(p.name)}</th>"
            for i, p in enumerate(env.policies)
        )
        + "</tr><tr>"
        + "".join("<th>Value</th><th>Confidence interval</th>" for _ in env.policies)
        + "</tr>"
    )

    rows = []
    for name in names:
        cells = [f"<td>{html(name)}</td>"]
        for column, policy in enumerate(env.policies):
            metrics = per_policy[policy.name]
            value = metrics.get(name)
            if value is None:
                cells.append(f'<td class="dim" data-col="{column}">—</td><td class="dim">—</td>')
                continue
            low = metrics.get(name + charts.CI_LOWER_SUFFIX)
            high = metrics.get(name + charts.CI_UPPER_SUFFIX)
            interval = (
                f"{low:.3g} – {high:.3g}"
                if low is not None and high is not None and high > low
                else '<span class="dim">—</span>'
            )
            cells.append(
                f'<td class="num" data-col="{column}"{_export_attrs(value, low, high)}>'
                f'{value:.4g}</td><td class="num ci">{interval}</td>'
            )
        rows.append(f'<tr data-metric="{html(name)}">' + "".join(cells) + "</tr>")

    table_id = f"metrics-{_slug(run.run_id[:12])}-{_slug(env.name)}"
    return (
        _export_buttons(table_id)
        + f'<div class="scroll"><table class="metrics" id="{html(table_id)}" '
        f'data-env="{html(env.name)}" data-export-name="{html(env.name)}_metrics">'
        f"<thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="note">Intervals are the ones the run computed, at whatever '
        "confidence level it was configured with; a dash means it reported none.</p>"
    )


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
        CommonMetricName.TASK_COMPLETION_RATE.value,
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
    # The figures worth a card, when the run logged them. A run that logs none
    # of these — a new environment with metrics of its own — falls back to the
    # first few it did log, so the card is never empty for want of a name this
    # module happens to know.
    headline = [
        ("average_return", "Avg. return", "{:.3g}"),
        (CommonMetricName.TASK_COMPLETION_RATE.value, "Task completion", "{:.0%}"),
        ("average_actual_num_steps", "Avg. steps", "{:.3g}"),
    ]

    items = []
    for policy in env.policies:
        metrics = run.metrics_for(env.name, policy.name)
        shown = [
            (label, fmt.format(metrics[key])) for key, label, fmt in headline if key in metrics
        ]
        if not shown:
            shown = [
                (name.replace("_", " ").capitalize(), f"{metrics[name]:.3g}")
                for name in charts.base_metric_names(metrics)[:3]
            ]
        items.append(
            ListingItem(
                title=policy.name,
                href=policy_url(run, env.name, policy.name),
                subtitle=policy.policy_type or "planner",
                fields=shown + [("Episodes", str(len(policy.episodes)))],
            )
        )
    return _listing(items, "This run logged no planners for this environment.")


def run_page(
    run: RunView, children: Sequence[RunView] = (), parent: Optional[RunView] = None
) -> str:
    """One run: its environments, and the comparison across planners in each.

    Args:
        run: The run.
        children: Runs nested under it -- a tuning study's tuned planners and
            its evaluation -- listed after its own results.
        parent: The run it is nested under, linked from the page.

    Returns:
        A complete HTML document.
    """
    sections = []
    for env in run.environments:
        sections.append(
            f'<section class="env"><h2><a href="{html(env_url(run, env.name))}">'
            f"{html(env.name)}</a></h2>"
            f'<p class="note"><a href="{html(chart_url(run, env.name))}">'
            "Build a chart from these metrics</a></p>"
            + _policy_cards(run, env)
            + _comparison_charts(run, env)
            + f"<details><summary>All metrics</summary>{_metric_table(run, env)}</details>"
            + "</section>"
        )

    params = run_parameters(run.params)
    return layout(
        run.run_name,
        [
            ("Experiments", "/"),
            (
                run.experiment_name,
                _url("experiment", run.store_index, run.experiment_id),
            ),
            (run.run_name, None),
        ],
        f'<div class="page-head"><h1>{html(run.run_name)} {_chip(run.status)}</h1>'
        f"{_layout_toggle()}</div>"
        f'<p class="note">{html(run_kind_label(run) + " · ") if run_kind_label(run) else ""}'
        f"Started {html(_timestamp(run.start_time))} · "
        f"run <code>{html(run.run_id)}</code>"
        + (
            f' · part of <a href="{html(run_url(parent))}">{html(parent.run_name)}</a>'
            if parent is not None
            else ""
        )
        + "</p>"
        + "".join(sections)
        + _child_runs(children)
        + params,
    )


def _episode_returns(policy: PolicyView) -> List[float]:
    """The discounted return of each of one planner's episodes, in order.

    Read from the traces rather than from a metric, because a metric is the
    summary and this plot is about the episodes behind it.

    Args:
        policy: The planner whose episodes to read.

    Returns:
        One return per episode that recorded one.
    """
    returns = []
    for artifacts in policy.episodes.values():
        summary = next((a.summary for a in artifacts if a.summary), None)
        if summary and summary.discounted_return is not None:
            returns.append(summary.discounted_return)
    return returns


def _returns_chart(env: EnvironmentView, heading: str = "Discounted return per episode") -> str:
    """The histogram of episode returns for every planner in one environment.

    Args:
        env: The environment being shown.
        heading: Heading to put above the plot.

    Returns:
        HTML for the section, or an empty string when no episode recorded a
        return — a run of videos alone, for instance.
    """
    series = [(policy.name, _episode_returns(policy)) for policy in env.policies]
    svg = charts.histogram_svg(series, value_label="Discounted return", count_label="Episodes")
    if not svg:
        return ""
    episodes = sum(len(values) for _, values in series)
    return (
        f"<h2>{html(heading)}</h2>"
        f'<figure class="chart-card wide">{svg}</figure>'
        f'<p class="note">How many of the {episodes} episode(s) ended in each range of '
        "discounted return. The bins are shared across planners so the bars can be "
        "compared; the returns are read from the episodes themselves, not from a "
        "logged average.</p>"
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
            (
                run.experiment_name,
                _url("experiment", run.store_index, run.experiment_id),
            ),
            (run.run_name, run_url(run)),
            (env.name, None),
        ],
        f'<div class="page-head"><h1>{html(env.name)}</h1>{_layout_toggle()}</div>'
        f'<p class="note">{len(env.policies)} planner(s) in run '
        f'<a href="{html(run_url(run))}">{html(run.run_name)}</a> · '
        f'<a href="{html(chart_url(run, env.name))}">build a chart</a>.</p>'
        + _policy_cards(run, env)
        + _comparison_charts(run, env)
        + _returns_chart(env)
        + (
            "<details><summary>Plots the run drew</summary>"
            f'<section class="gallery">{env_plots}</section></details>'
            if env_plots
            else ""
        )
        + f"<details><summary>All metrics</summary>{_metric_table(run, env)}</details>",
    )


def chart_url(run: RunView, env: str) -> str:
    """URL of one environment's chart builder."""
    return env_url(run, env) + "/chart"


def chart_builder_page(run: RunView, env: EnvironmentView) -> str:
    """A chart of this environment's metrics, built by the reader.

    The comparison charts on the other pages are the site's choice: one metric
    each, every planner, the site's labels. A figure for a paper is the
    author's choice, so this page hands over the planners, the metric, the
    three pieces of text and the download, and draws in the browser from the
    numbers the run logged.

    Args:
        run: The run being shown.
        env: The environment whose metrics to plot.

    Returns:
        A complete HTML document.
    """
    return layout(
        f"Chart · {env.name}",
        [
            ("Experiments", "/"),
            (
                run.experiment_name,
                _url("experiment", run.store_index, run.experiment_id),
            ),
            (run.run_name, run_url(run)),
            (env.name, env_url(run, env.name)),
            ("Chart", None),
        ],
        f"<h1>Build a chart</h1>"
        f'<p class="note">{html(env.name)} in run '
        f'<a href="{html(run_url(run))}">{html(run.run_name)}</a>. '
        "Everything plotted is a metric this run logged; the error bars are its "
        "confidence intervals.</p>"
        + chart_builder_html(run, env, read_query=True)
        + '<script src="/static/figure-kit.js"></script>'
        '<script src="/static/chart-builder.js"></script>',
    )


def chart_builder_html(run: RunView, env: EnvironmentView, read_query: bool = False) -> str:
    """One planner-comparison chart builder: its controls, canvas and data.

    Controls are found by role within the builder's own container, not by
    page-wide id, so a page can carry one builder per environment. The page
    must load ``figure-kit.js`` and ``chart-builder.js`` after it.

    Args:
        run: The run whose metrics to plot.
        env: The environment whose planners to plot.
        read_query: Start from ``?planners=`` and ``?metric=`` in the address,
            for the standalone builder page.

    Returns:
        HTML for the builder.
    """
    data = {
        "environment": env.name,
        "policies": [
            {"name": policy.name, "metrics": run.metrics_for(env.name, policy.name)}
            for policy in env.policies
        ],
        "ci": {
            "lower": charts.CI_LOWER_SUFFIX,
            "upper": charts.CI_UPPER_SUFFIX,
        },
    }
    # In a <script type="application/json"> the only sequence that can end the
    # element early is "</", so that is what is escaped.
    payload = json.dumps(data).replace("</", "<\\/")
    return (
        f'<div class="builder" data-chart-builder{" data-read-query" if read_query else ""}>'
        '<form class="builder-controls" data-role="form">'
        "<fieldset><legend>Planners</legend>"
        '<p class="note">Tick the ones to plot; the box under each is the name it carries in the figure.</p>'
        '<div data-role="policies"></div></fieldset>'
        '<label>Metric <select data-role="metric"></select></label>'
        '<label>Title <input data-role="title" type="text" autocomplete="off"></label>'
        '<label>Value axis <input data-role="y" type="text" autocomplete="off"></label>'
        '<label>Planner axis <input data-role="x" type="text" autocomplete="off"></label>'
        '<label>Style <select data-role="style">'
        '<option value="mono">Paper, mono</option>'
        '<option value="colour" selected>Paper, colour</option>'
        '<option value="slide">Slide, dark</option>'
        "</select></label>"
        '<label>Orientation <select data-role="orient">'
        '<option value="vertical">Vertical bars</option>'
        '<option value="horizontal">Horizontal bars</option>'
        "</select></label>"
        '<label class="check"><input data-role="errors" type="checkbox" checked> '
        "Show confidence intervals</label>"
        '<label class="check"><input data-role="values" type="checkbox" checked> '
        "Print the value on each bar</label>"
        '<div class="builder-actions">'
        '<button type="button" data-role="svg" class="tab">Download SVG</button>'
        '<button type="button" data-role="png" class="tab">Download PNG</button>'
        '<button type="button" data-role="csv" class="tab">Download data (CSV)</button>'
        "</div></form>"
        '<figure class="builder-canvas" data-role="output"></figure>'
        f'<script type="application/json" data-role="data">{payload}</script>'
        "</div>"
    )


def _thumbnail(
    run: RunView,
    env: str,
    policy: str,
    index: int,
    artifacts: Sequence[EpisodeArtifact],
) -> str:
    """The picture that stands for one episode in a list of them.

    An episode with a trace gets a frame of the 3D scene, drawn in the browser
    from that trace — the same view the episode page opens with, so a card
    looks like what clicking it gives. The recorded image is served underneath
    and stays put until a frame has actually been drawn over it, which is what
    a page with no WebGL, or a payload kind with no scene module yet, keeps.

    Anything else falls back by artifact kind, like the player does, so an
    environment the site has never heard of still gets a thumbnail: a drawn
    recording if it made one, else the first frame of its video, else a plain
    placeholder.

    Args:
        run: The run the episode belongs to.
        env: Environment name.
        policy: Planner name.
        index: Episode index, for the alternative text.
        artifacts: That episode's artifacts.

    Returns:
        HTML for the thumbnail.
    """
    recorded = '<div class="thumb thumb-empty"><span>No recording</span></div>'
    for kind in (ArtifactKind.PLOT, ArtifactKind.VIDEO):
        artifact = next((a for a in artifacts if a.kind is kind), None)
        if artifact is None:
            continue
        src = artifact_url(run, f"{env}/{policy}/{artifact.relative_path}")
        if kind is ArtifactKind.VIDEO:
            # metadata alone is enough for the first frame, and muted+playsinline
            # keeps a grid of them from ever making noise or going fullscreen.
            recorded = (
                f'<video class="thumb thumb-recorded" src="{html(src)}" '
                'preload="metadata" muted playsinline></video>'
            )
        else:
            recorded = (
                f'<img loading="lazy" class="thumb thumb-recorded" '
                f'alt="{html(view_label(artifact))} of episode {index}" src="{html(src)}">'
            )
        break

    trace = next((a for a in artifacts if a.kind is ArtifactKind.TRACE), None)
    if trace is None:
        return recorded
    trace_src = artifact_url(run, f"{env}/{policy}/{trace.relative_path}")
    return (
        recorded + f'<canvas class="thumb thumb-scene" data-thumb-trace="{html(trace_src)}" '
        f'aria-label="Scene from episode {index}" hidden></canvas>'
    )


def _thumbnail_scripts(policies: Sequence[PolicyView]) -> str:
    """The scripts that draw scene thumbnails for a page's episode cards.

    Loaded only when some episode on the page has a trace, and one scene module
    per payload kind present, so a page of videos pays for none of it.

    Args:
        policies: The planners whose episodes the page lists.

    Returns:
        HTML script tags, or an empty string.
    """
    kinds = sorted(
        {
            artifact.payload_kind
            for policy in policies
            for artifacts in policy.episodes.values()
            for artifact in artifacts
            if artifact.kind is ArtifactKind.TRACE and artifact.payload_kind
        }
    )
    if not kinds:
        return ""
    scenes = "".join(f'<script src="{html(scene_script_url(k))}"></script>' for k in kinds)
    return (
        '<script src="/static/vendor/three.min.js"></script>'
        '<script src="/static/viewer/renderer-core.js"></script>'
        + scenes
        + '<script src="/static/viewer/scene-cards.js"></script>'
    )


def _episode_items(run: RunView, env: EnvironmentView, policy: PolicyView) -> List[ListingItem]:
    """One listing item per episode of one planner, each with its thumbnail.

    Shared by the planner page and the tuning view's evaluation section, so an
    episode card looks the same wherever it is listed.

    Args:
        run: The run the episodes belong to.
        env: Their environment.
        policy: Their planner.

    Returns:
        The items, in episode order.
    """
    items = []
    for index, artifacts in policy.episodes.items():
        summary = next((a.summary for a in artifacts if a.summary), None)
        outcome = ""
        if summary and summary.reach_terminal_state is not None:
            outcome = "terminal" if summary.reach_terminal_state else "out of steps"
        items.append(
            ListingItem(
                title=f"Episode {index}",
                href=episode_url(run, env.name, policy.name, index),
                thumb=_thumbnail(run, env.name, policy.name, index, artifacts),
                fields=[
                    (
                        "Return",
                        (
                            f"{summary.discounted_return:.2f}"
                            if summary and summary.discounted_return is not None
                            else ""
                        ),
                    ),
                    (
                        "Steps",
                        str(summary.num_steps) if summary and summary.num_steps else "",
                    ),
                    ("Ended", outcome),
                ],
                tags=sorted({view_label(a) for a in artifacts}),
            )
        )
    return items


def policy_page(run: RunView, env: EnvironmentView, policy: PolicyView) -> str:
    """One planner on one environment: every episode it ran."""
    items = _episode_items(run, env, policy)
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
            (
                run.experiment_name,
                _url("experiment", run.store_index, run.experiment_id),
            ),
            (run.run_name, run_url(run)),
            (env.name, env_url(run, env.name)),
            (policy.name, None),
        ],
        f'<div class="page-head"><h1>{html(policy.name)}</h1>'
        f'<div class="layout-switch">{_live_toggle()}{_layout_toggle()}</div></div>'
        f'<p class="note">{len(policy.episodes)} episode(s) on '
        f'<a href="{html(env_url(run, env.name))}">{html(env.name)}</a> · '
        "each card shows that episode's own recording.</p>"
        + _listing(items, "This planner produced no episode artifacts.", "Recordings")
        + _returns_chart(EnvironmentView(name=env.name, policies=[policy]))
        + (
            "<details><summary>Plots the run drew</summary>"
            f'<section class="gallery">{plots}</section></details>'
            if plots
            else ""
        )
        + _thumbnail_scripts([policy]),
    )


def view_label(artifact: EpisodeArtifact) -> str:
    """Name one artifact as a way of watching the episode.

    The label says what the viewer will see, not which file it came from.

    Args:
        artifact: The artifact to label.

    Returns:
        A short label for its tab.
    """
    if artifact.kind is ArtifactKind.TRACE:
        return "3D replay"
    if artifact.kind is ArtifactKind.VIDEO:
        return "Recorded video"
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
    return trace_viewer_html(src, artifact.payload_kind or "")


def trace_viewer_html(trace_src: str, payload_kind: str, static_root: str = "/static") -> str:
    """The 3D trace viewer: its canvas, HUD, controls and scripts.

    One definition for every page that replays a trace. The results site
    serves the viewer at ``/static``; the Sphinx docs embed the same markup in
    a page of their own under ``_static/pomdp-viewer`` and pass a relative
    root. ``trace-player.js`` finds its elements by these ids, so a second copy
    of this markup would be a second contract to keep in step.

    Args:
        trace_src: URL of the trace JSON, relative to the page or absolute.
        payload_kind: The trace's payload kind, which picks the scene module.
        static_root: URL prefix under which ``vendor/`` and ``viewer/`` live.

    Returns:
        HTML for the viewer and the scripts that drive it.
    """
    return (
        f'<div class="viewer" id="viewer" data-trace="{html(trace_src)}" '
        f'data-payload-kind="{html(payload_kind)}">'
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
        f'<script src="{html(static_root)}/vendor/three.min.js"></script>'
        f'<script src="{html(static_root)}/viewer/renderer-core.js"></script>'
        f'<script src="{html(scene_script_url(payload_kind, static_root))}"></script>'
        f'<script src="{html(static_root)}/viewer/trace-player.js"></script>'
    )


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
                (
                    f"{summary.discounted_return:.2f}"
                    if summary and summary.discounted_return is not None
                    else ""
                ),
            ),
            ("Steps", str(summary.num_steps) if summary and summary.num_steps else ""),
            (
                "Ended",
                (
                    ("terminal state" if summary.reach_terminal_state else "out of steps")
                    if summary and summary.reach_terminal_state is not None
                    else ""
                ),
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
            (
                run.experiment_name,
                _url("experiment", run.store_index, run.experiment_id),
            ),
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


# -- tuning studies ----------------------------------------------------------

_RUN_KIND_LABELS = {
    run_layout.RUN_KIND_STUDY: "Tuning study",
    run_layout.RUN_KIND_CONFIG: "Tuned planner",
    run_layout.RUN_KIND_EVALUATION: "Evaluation of the tuned planners",
}


def run_kind_label(run: RunView) -> str:
    """Say what part of a tuning study a run is, or nothing for a plain run.

    Args:
        run: Any run.

    Returns:
        A short label, or an empty string.
    """
    if run.run_kind in _RUN_KIND_LABELS:
        return _RUN_KIND_LABELS[run.run_kind]
    if tuning.is_tuning_config_run(run):
        return _RUN_KIND_LABELS[run_layout.RUN_KIND_CONFIG]
    return ""


def _number(value: object, fmt: str = "{:.4g}") -> str:
    """Format a value from a run: numbers compactly, anything else as text."""
    if isinstance(value, bool):
        return html(value)
    if isinstance(value, int):
        return html(value)
    if isinstance(value, float):
        return html(fmt.format(value))
    if value is None:
        return '<span class="dim">—</span>'
    return html(value)


def _interval(low: Optional[float], high: Optional[float]) -> str:
    if low is None or high is None or not high > low:
        return '<span class="dim">—</span>'
    return f"{low:.3g} – {high:.3g}"


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _early_stopping_text(study: tuning.TuningStudy) -> str:
    """One line on whether early stopping ended the study."""
    if study.early_stopping_fired:
        return f"stopped after {study.stopped_at_trial} trials"
    if study.early_stopping is None:
        return "off" if study.has_summary else ""
    return "did not fire"


def _trials_text(study: tuning.TuningStudy) -> str:
    completed = study.n_trials_completed
    budget = study.n_trials_budget
    if completed is not None and budget is not None:
        return f"{completed} of {budget}"
    if budget is not None:
        return f"budget {budget}"
    return ""


def _study_stats(study: tuning.TuningStudy) -> str:
    return _stats(
        [
            # Not repeated here: the planner and environment (the page title),
            # the best and Pareto trials (marked in the trial table) and the
            # episodes per trial (the comparison's column heading).
            ("Trials run", html(_trials_text(study))),
            ("Early stopping", html(_early_stopping_text(study))),
            ("Tuning time", html(_duration(study.optimization_time_seconds))),
        ]
    )


def _objective_comparison(
    study: tuning.TuningStudy,
    evaluation: Optional[Tuple[RunView, EnvironmentView, PolicyView]],
) -> str:
    """The best trial's score beside the fresh evaluation's, per objective.

    The two are shown side by side because they differ for a reason: the best
    trial was chosen as the best of several noisy scores, so its own score is
    biased upward, and the evaluation run is what the planner does on its own
    episode count.
    """
    eval_metrics: Dict[str, float] = {}
    eval_episodes = ""
    if evaluation is not None:
        eval_run, eval_env, eval_policy = evaluation
        eval_metrics = eval_run.metrics_for(eval_env.name, eval_policy.name)
        eval_episodes = f" ({len(eval_policy.episodes)} episodes)" if eval_policy.episodes else ""
    trial_episodes = f" ({study.episodes_per_trial} episodes)" if study.episodes_per_trial else ""

    names = [name for name, _ in study.objectives] or sorted(study.best_trial_scores)
    rows = []
    for name in names:
        direction = next((d for n, d in study.objectives if n == name), "")
        value, low, high = study.best_trial_scores.get(name, (None, None, None))
        evaluated = eval_metrics.get(name)
        rows.append(
            [
                html(name),
                html(direction or ""),
                f'<span class="num">{_number(value)}</span>',
                f'<span class="num ci">{_interval(low, high)}</span>',
                f'<span class="num">{_number(evaluated)}</span>',
                '<span class="num ci">'
                + _interval(
                    eval_metrics.get(name + charts.CI_LOWER_SUFFIX),
                    eval_metrics.get(name + charts.CI_UPPER_SUFFIX),
                )
                + "</span>",
            ]
        )
    return _table(
        [
            "Objective",
            "Direction",
            f"Best trial{trial_episodes}",
            "Interval",
            f"Evaluation{eval_episodes}",
            "Interval",
        ],
        rows,
    )


def _parameter_table(study: tuning.TuningStudy) -> str:
    rows = []
    for parameter in study.parameters:
        if parameter.choices is not None:
            space = "one of " + ", ".join(html(c) for c in parameter.choices)
        elif parameter.low is not None and parameter.high is not None:
            space = f"{_number(parameter.low)} – {_number(parameter.high)}"
        else:
            space = html(parameter.range_text) or '<span class="dim">—</span>'
        position = parameter.position
        range_bar = (
            f'<span class="range-bar" title="{position:.0%} of the way through the range">'
            f'<span class="range-mark" style="left:{position * 100:.1f}%"></span></span>'
            if position is not None
            else ""
        )
        rows.append(
            [
                html(parameter.name),
                f'<span class="num">{_number(parameter.best)}</span>',
                space,
                range_bar,
            ]
        )
    if not rows:
        return '<p class="empty">This run recorded no search space.</p>'
    return _table(["Parameter", "Chosen", "Search range", "Where in the range"], rows) + (
        '<p class="note">A value pressed against either end of its range suggests the '
        "range was too narrow.</p>"
    )


def _trial_table(study: tuning.TuningStudy) -> str:
    if not study.trials:
        return (
            '<p class="empty">This run carries no trial records. Runs logged before they '
            "were copied into MLflow keep them under <code>tuning_diagnostics/</code> in the "
            "study's cache directory.</p>"
        )
    objective_names = [name for name, _ in study.objectives] or sorted(
        {k for t in study.trials for k in t.objective_values}
    )
    param_names = [p.name for p in study.parameters] or sorted(
        {k for t in study.trials for k in t.params}
    )
    pareto = set(study.pareto_trial_numbers) or {t.number for t in study.trials if t.is_pareto}

    head = (
        "<th>Trial</th><th>State</th>"
        + "".join(f'<th class="group">{html(n)}</th>' for n in objective_names)
        + "".join(f"<th>{html(n)}</th>" for n in param_names)
        + "<th>Duration</th>"
    )
    body = []
    for trial in study.trials:
        marks = []
        is_best = trial.number == study.best_trial_number
        if is_best:
            marks.append('<span class="chip chip-ok">Best</span>')
        if trial.number in pareto or trial.is_pareto:
            marks.append('<span class="chip">Pareto</span>')
        cells = [f'<td class="num">#{trial.number} {" ".join(marks)}</td>']
        cells.append(f"<td>{html(trial.state.title())}</td>")
        cells += [
            f'<td class="num">{_number(trial.objective_values.get(n))}</td>'
            for n in objective_names
        ]
        cells += [f'<td class="num">{_number(trial.params.get(n))}</td>' for n in param_names]
        cells.append(f'<td class="num">{html(_duration(trial.duration_seconds)) or "—"}</td>')
        body.append(f'<tr{" class=best" if is_best else ""}>' + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table class="metrics trials">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        '<p class="note">Each objective value is the mean over that trial\'s own '
        "episodes. Pareto trials are not beaten "
        "on every objective by any other trial; the best one is the Pareto trial with the "
        "highest normalized score.</p>"
    )


#: Above this many rows the trial table is built only when its section opens.
#: The rows are written into the page either way; a <template> keeps the
#: browser from laying out thousands of them that nobody has asked to see.
TRIALS_INLINE_LIMIT = 200

#: Above this many episodes their cards are built only when the section opens:
#: each card fetches its trace and builds a 3D scene, which for a hundred
#: episodes is work the reader may never want.
EPISODES_INLINE_LIMIT = 24


def _collapsible(summary: str, body: str, lazy: bool) -> str:
    """A section closed by default, in the site's <details> pattern.

    Args:
        summary: The line the reader clicks, as plain text.
        body: The section's HTML.
        lazy: Hold the body in a <template> until the section is first
            opened, for a body too large to lay out up front.

    Returns:
        HTML for the section.
    """
    inner = f"<template>{body}</template>" if lazy else body
    return (
        f'<details{" data-lazy" if lazy else ""}><summary>{html(summary)}</summary>'
        f"{inner}</details>" + ('<script src="/static/lazy-details.js"></script>' if lazy else "")
    )


def _trials_section(study: tuning.TuningStudy) -> str:
    """The trial table, collapsed: a study can run thousands of trials."""
    if not study.trials:
        return _trial_table(study)
    return _collapsible(
        f"All trials ({len(study.trials)})",
        _trial_table(study),
        lazy=len(study.trials) > TRIALS_INLINE_LIMIT,
    )


def _chart_grid(svgs: Sequence[str]) -> str:
    cards = "".join(f'<figure class="chart-card">{svg}</figure>' for svg in svgs if svg)
    return f'<div class="charts">{cards}</div>' if cards else ""


def _diagnostic_charts(run: RunView, study: tuning.TuningStudy) -> str:
    """The study's diagnostic charts, drawn here from its trial records.

    The PNGs the optimizer drew stay in the run for reading offline; they are
    linked, not shown, because a picture cannot say which trial a point is or
    follow the page's theme.
    """
    if not study.trials:
        return (
            '<p class="empty">This run carries no trial records, so there is nothing to '
            "chart.</p>"
        )
    sections = [
        (
            "Objectives over the trials",
            "Each dot is a trial; the line is the best value found so far. A line still "
            "climbing at the last trial means the budget ran out before the search settled.",
            tuning_charts.objective_history(study)
            + [tuning_charts.pareto_front(study), tuning_charts.front_quality(study)],
            True,
        ),
        (
            "Objectives ranked, with their intervals",
            "Trials ordered best first. Where the intervals overlap, the ranking is "
            "within the noise of the episodes behind each score.",
            tuning_charts.objective_confidence_intervals(study),
            True,
        ),
        (
            "Parameters over the trials",
            "What the sampler drew, trial by trial.",
            tuning_charts.parameter_history(study),
            True,
        ),
        (
            "Objective against each parameter",
            "A slope running into the edge of a range says the range stops short.",
            tuning_charts.parameter_slices(study),
            False,
        ),
        (
            "Metrics the study did not optimize",
            "What the chosen parameters cost on everything else the episodes measured.",
            tuning_charts.secondary_metrics(study),
            False,
        ),
        ("Trial durations", "", [tuning_charts.trial_durations(study)], False),
    ]
    blocks = []
    for heading, note, svgs, open_ in sections:
        grid = _chart_grid(svgs)
        if not grid:
            continue
        count = sum(1 for svg in svgs if svg)
        blocks.append(
            f'<details class="chart-group"{" open" if open_ else ""}>'
            f"<summary>{html(heading)} ({count})</summary>"
            + (f'<p class="note">{html(note)}</p>' if note else "")
            + grid
            + "</details>"
        )
    return (
        f'<p><a class="tab" href="{html(tuning_chart_url(run))}">Build a chart</a> '
        '<span class="dim">— an editable figure of these diagnostics, as SVG or PNG.</span></p>'
        '<p class="note">Hover a point to see its trial and parameters. Green marks the '
        "Pareto trials, red the chosen one. Parameter importances are not drawn: Optuna "
        "computes them from the live study, which the run does not keep.</p>"
        + "".join(blocks)
        + '<script src="/static/chart-tips.js"></script>'
    )


def _evaluation_section(
    study: tuning.TuningStudy,
    evaluation: Optional[Tuple[RunView, EnvironmentView, PolicyView]],
    compare_url: str = "",
) -> str:
    if evaluation is None:
        if study.evaluation_run_id:
            return (
                '<p class="empty">The evaluation run this study names, '
                f"<code>{html(study.evaluation_run_id)}</code>, is not in the served stores, "
                "or does not hold this planner.</p>"
            )
        return (
            '<p class="empty">No evaluation run is linked to this study. '
            "<code>run_optimize_and_evaluate</code> runs one and links it here.</p>"
        )
    eval_run, eval_env, eval_policy = evaluation
    single = EnvironmentView(name=eval_env.name, policies=[eval_policy])
    return (
        f'<p class="note">Raw evaluation run: '
        f'<a href="{html(run_url(eval_run))}">{html(eval_run.run_name)}</a> · '
        f'<a href="{html(policy_url(eval_run, eval_env.name, eval_policy.name))}">'
        f"{html(eval_policy.name)} on {html(eval_env.name)}</a>. Its headline figures are "
        "in the comparison at the top of the page.</p>"
        + (
            f'<p><a class="compare-link" href="{html(compare_url)}">Compare with the other '
            f"planners on {html(eval_env.name)}</a></p>"
            if compare_url
            else ""
        )
        + f'<p><a class="tab" href="{html(chart_url(eval_run, eval_env.name))}">'
        "Build a chart from this evaluation</a></p>"
        + _collapsible(
            f"Episodes ({len(eval_policy.episodes)})",
            _listing(
                _episode_items(eval_run, eval_env, eval_policy),
                "This evaluation produced no episode artifacts.",
                "Recordings",
            )
            # The scene scripts go inside the section, so a lazy one loads
            # them only when it opens.
            + _thumbnail_scripts([eval_policy]),
            lazy=len(eval_policy.episodes) > EPISODES_INLINE_LIMIT,
        )
        + _returns_chart(single, "Discounted return per evaluation episode")
    )


def _evaluation_metrics(
    evaluation: Optional[Tuple[RunView, EnvironmentView, PolicyView]],
) -> str:
    """Every metric the evaluation logged, collapsed under the comparison.

    It sits under the comparison rather than with the episodes because it is
    the full version of that table's evaluation column.
    """
    if evaluation is None:
        return ""
    eval_run, eval_env, eval_policy = evaluation
    single = EnvironmentView(name=eval_env.name, policies=[eval_policy])
    return (
        f"<details><summary>All evaluation metrics</summary>{_metric_table(eval_run, single)}"
        "</details>"
    )


def _compare_url(
    evaluation: Optional[Tuple[RunView, EnvironmentView, PolicyView]],
    parent: Optional[RunView],
) -> str:
    """The study page's comparison block for this planner's environment.

    Empty unless another planner was evaluated on the same environment and
    the study page exists to hold the block.
    """
    if evaluation is None or parent is None:
        return ""
    _, eval_env, _ = evaluation
    if len(eval_env.policies) < 2:
        return ""
    return run_url(parent) + "#" + compare_anchor(eval_env.name)


def find_evaluation(
    study: tuning.TuningStudy, evaluation_run: Optional[RunView]
) -> Optional[Tuple[RunView, EnvironmentView, PolicyView]]:
    """Find the tuned planner inside its evaluation run.

    The evaluation run holds every planner the study tuned, filed by
    environment name and planner name, so those two names pick this one out.
    A run with a single environment or a single planner needs no name.

    Args:
        study: The study.
        evaluation_run: The run its tag names, or ``None``.

    Returns:
        ``(run, environment, planner)``, or ``None`` when it cannot be found.
    """
    if evaluation_run is None:
        return None
    env = evaluation_run.environment(study.environment_name or "")
    if env is None and len(evaluation_run.environments) == 1:
        env = evaluation_run.environments[0]
    if env is None:
        return None
    policy = env.policy(study.policy_name or "")
    if policy is None and len(env.policies) == 1:
        policy = env.policies[0]
    if policy is None:
        return None
    return evaluation_run, env, policy


def tuning_page(
    run: RunView,
    study: tuning.TuningStudy,
    evaluation: Optional[Tuple[RunView, EnvironmentView, PolicyView]] = None,
    parent: Optional[RunView] = None,
) -> str:
    """One tuned planner: the study that chose its parameters, and its fresh evaluation.

    Args:
        run: The config run.
        study: The study read from it.
        evaluation: The planner inside its evaluation run, when linked.
        parent: The study run this config run is nested under.

    Returns:
        A complete HTML document.
    """
    # The chosen values and their ranges are the "Chosen parameters" table.
    params = run_parameters(
        run.params,
        omit_prefixes=(run_layout.BEST_PARAM_PREFIX, run_layout.PARAM_RANGE_PREFIX),
    )
    title = study_label(study)
    crumbs: List[Tuple[str, Optional[str]]] = [
        ("Experiments", "/"),
        (run.experiment_name, _url("experiment", run.store_index, run.experiment_id)),
    ]
    if parent is not None:
        crumbs.append(("Tuning study", run_url(parent)))
    crumbs.append((title, None))
    study_note = (
        f' · part of <a href="{html(run_url(parent))}">tuning study ' f"{html(parent.run_name)}</a>"
        if parent is not None
        else ""
    )
    thin = (
        ""
        if study.has_summary
        else '<p class="note">This run predates the study summary, so this view is built from '
        "its parameters alone.</p>"
    )
    return layout(
        title,
        crumbs,
        f'<div class="page-head"><h1>{html(title)} {_chip(run.status)}</h1>'
        f"{_layout_toggle()}</div>"
        f'<p class="note">Hyperparameter tuning{html(_class_note(study))} · '
        f"started {html(_timestamp(run.start_time))} · "
        f"run <code>{html(run.run_name)}</code>{study_note}</p>"
        + thin
        + _study_stats(study)
        + "<h2>Best trial against the evaluation</h2>"
        + _objective_comparison(study, evaluation)
        + _evaluation_metrics(evaluation)
        + "<h2>Chosen parameters</h2>"
        + _parameter_table(study)
        + "<h2>Trials</h2>"
        + _trials_section(study)
        + "<h2>Diagnostic charts</h2>"
        + _diagnostic_charts(run, study)
        + "<h2>Evaluation</h2>"
        + _evaluation_section(study, evaluation, _compare_url(evaluation, parent))
        + params,
    )


def _class_note(study: tuning.TuningStudy) -> str:
    """The planner and environment classes, where the title's names do not say them."""
    notes = []
    if study.planner and study.planner != study.policy_name:
        notes.append(f"planner class {study.planner}")
    if study.environment and study.environment != study.environment_name:
        notes.append(f"environment class {study.environment}")
    return (" · " + ", ".join(notes)) if notes else ""


def study_label(study: tuning.TuningStudy) -> str:
    """Name a tuned planner the way a reader would: planner on environment."""
    planner = study.policy_name or study.planner or "Planner"
    environment = study.environment_name or study.environment or "an environment"
    return f"{planner} on {environment}"


def group_title(group: tuning.StudyGroup) -> str:
    """Name a tuning study for its card and its page."""
    if len(group.configs) == 1:
        return f"{study_label(group.configs[0].study)} — tuning study"
    if group.configs:
        planners = _distinct(c.study.planner or c.study.policy_name or "" for c in group.configs)
        environments = _study_environments(group)
        count = f"{len(planners)} planner{'s' if len(planners) != 1 else ''}"
        where = environments[0] if len(environments) == 1 else f"{len(environments)} environments"
        return f"Tuning study of {count} on {where}"
    return group.anchor.run_name


def _distinct(values: Iterable[str]) -> List[str]:
    """Values in first-seen order, without repeats or blanks."""
    seen: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _study_environments(group: tuning.StudyGroup) -> List[str]:
    return _distinct(c.study.environment_name or c.study.environment or "" for c in group.configs)


def _headline(name: str, value: Optional[float]) -> str:
    return f"{name} {value:.4g}" if value is not None else ""


def _config_scores(config: tuning.TunedConfig) -> List[Tuple[str, str]]:
    """The best trial's and the evaluation's score on the first objective."""
    study = config.study
    names = [str(n) for n, _ in study.objectives] or sorted(study.best_trial_scores)
    if not names:
        return []
    name = names[0]
    best = study.best_trial_scores.get(name, (None, None, None))[0]
    evaluation = find_evaluation(study, config.evaluation_run)
    evaluated = ""
    if evaluation is not None:
        eval_run, eval_env, eval_policy = evaluation
        value = eval_run.metrics_for(eval_env.name, eval_policy.name).get(name)
        if value is not None:
            evaluated = f"{_headline(name, value)} ({len(eval_policy.episodes)} episodes)"
    return [("Best trial", html(_headline(name, best))), ("Evaluation", html(evaluated))]


def _study_item(group: tuning.StudyGroup) -> ListingItem:
    """The one card that stands for a whole tuning study."""
    anchor = group.anchor
    if len(group.configs) == 1:
        config = group.configs[0]
        fields = [
            ("Trials run", html(_trials_text(config.study))),
            ("Early stopping", html(_early_stopping_text(config.study))),
            *_config_scores(config),
        ]
        href = run_url(config.run)
    else:
        # Which planners, by name: a count alone sends the reader through to
        # find out what the study was about.
        planners = _distinct(c.study.planner or c.study.policy_name or "" for c in group.configs)
        fields = [("Planners", html(", ".join(planners)))]
        href = run_url(anchor)
    environments = _study_environments(group)
    return ListingItem(
        title=group_title(group),
        href=href,
        chip=_chip(anchor.status),
        fields=fields + [("Started", _timestamp(anchor.start_time))],
        tags=environments,
        footnote=anchor.run_id,
    )


def _best_parameters_text(study: tuning.TuningStudy, limit: int = 4) -> str:
    chosen = [(p.name, p.best) for p in study.parameters if p.best is not None]
    shown = ", ".join(
        f"{name}={value:.3g}" if isinstance(value, float) else f"{name}={value}"
        for name, value in chosen[:limit]
    )
    return shown + (f" and {len(chosen) - limit} more" if len(chosen) > limit else "")


def compare_anchor(environment: str) -> str:
    """The id of an environment's comparison block on the study page."""
    return "compare-" + "".join(c if c.isalnum() else "-" for c in environment)


def _comparison_rows(group: tuning.StudyGroup) -> Dict[str, List[tuple]]:
    """Each environment's evaluated planners: ``env -> [(config, eval_run, env, policy)]``."""
    rows: Dict[str, List[tuple]] = {}
    for config in group.configs:
        evaluation = find_evaluation(config.study, config.evaluation_run)
        if evaluation is None:
            continue
        eval_run, eval_env, eval_policy = evaluation
        rows.setdefault(eval_env.name, []).append((config, eval_run, eval_env, eval_policy))
    return rows


#: Metrics whose better direction is known by name. Anything else -- the
#: planners' own policy_info_* counters, most environment-specific counts --
#: gets no "best" mark: calling a larger tree better would be a guess.
_HIGHER_IS_BETTER = {
    "average_return",
    "return_cvar",
    "return_value_at_risk",
    CommonMetricName.TASK_COMPLETION_RATE.value,
    CommonMetricName.ENDED_BY_GOAL_RATE.value,
}
_LOWER_IS_BETTER = {
    CommonMetricName.ENDED_BY_FAILURE_RATE.value,
    CommonMetricName.ENDED_BY_TIMEOUT_RATE.value,
    CommonMetricName.AVERAGE_EPISODE_LENGTH.value,
    CommonMetricName.COLLISION_RATE.value,
    CommonMetricName.AVERAGE_COLLISIONS.value,
    CommonMetricName.AVERAGE_DANGEROUS_AREA_STEPS.value,
    CommonMetricName.AVERAGE_DANGEROUS_ENCOUNTERS.value,
    CommonMetricName.AVERAGE_NEAR_MISSES.value,
    "average_actual_num_steps",
}


def metric_direction(name: str, objectives: Sequence[Tuple[str, str]] = ()) -> Optional[str]:
    """Whether a larger or a smaller value of a metric is better, when that is known.

    Args:
        name: Metric name.
        objectives: The study's ``(metric, direction)`` pairs, which win.

    Returns:
        ``"maximize"``, ``"minimize"``, or ``None`` when the name does not say.
    """
    for objective, direction in objectives:
        if objective == name:
            return str(direction)
    if name in _HIGHER_IS_BETTER:
        return "maximize"
    if (
        name in _LOWER_IS_BETTER
        or name.endswith("_time")
        or "dangerous" in name
        or "collision" in name
    ):
        return "minimize"
    return None


def metric_group(name: str) -> str:
    """The filter group a metric belongs to: timing, planner internals, or outcome."""
    if name.endswith("_time"):
        return "timing"
    if name.startswith("policy_info_"):
        return "policy"
    return "outcome"


_GROUP_LABELS = (
    ("outcome", "Outcomes"),
    ("timing", "Timings"),
    ("policy", "Planner internals"),
)


def _picker(kind: str, label: str, groups: Sequence[Tuple[str, str]], count: int) -> str:
    """A dropdown of checkboxes, the shared shell of the metric and planner pickers.

    A <details> rather than a custom widget: it opens and closes from the
    keyboard as it is, and every checkbox inside is reachable by Tab.

    Args:
        kind: ``"metric"`` or ``"planner"``; names the data attributes.
        label: The summary's word, e.g. "Metrics".
        groups: ``(legend, checkboxes HTML)``; an empty legend draws none.
        count: How many choices there are.

    Returns:
        HTML for the picker.
    """
    fieldsets = "".join(
        "<fieldset>"
        + (f"<legend>{html(legend)}</legend>" if legend else "")
        + boxes
        + "</fieldset>"
        for legend, boxes in groups
        if boxes
    )
    return (
        f'<details class="metric-picker" data-picker="{kind}">'
        f'<summary data-picker-summary data-picker-label="{html(label)}">'
        f"{html(label)} ({count} of {count})</summary>"
        '<div class="metric-picker-panel">'
        '<div class="metric-picker-actions">'
        '<button type="button" class="tab" data-picker-all>Select all</button>'
        '<button type="button" class="tab" data-picker-none>Clear</button></div>'
        + fieldsets
        + "</div></details>"
    )


def _planner_picker(planners: Sequence[str]) -> str:
    """A dropdown choosing which planner columns show."""
    boxes = "".join(
        f'<label class="check"><input type="checkbox" data-planner-choice="{index}" '
        f'data-choice-name="{html(name)}" checked> {html(name)}</label>'
        for index, name in enumerate(planners)
    )
    return _picker("planner", "Planners", [("", boxes)], len(planners))


def _metric_picker(choices: Sequence[Tuple[str, str, str]]) -> str:
    """A dropdown choosing exactly which metric rows show, grouped like the toggles.

    Args:
        choices: ``(row key, label, group)`` for every row of the table.

    Returns:
        HTML for the picker.
    """
    groups = [
        (
            label,
            "".join(
                f'<label class="check"><input type="checkbox" data-metric-choice="{html(row)}" '
                f'data-choice-name="{html(row)}" checked> {html(text)}</label>'
                for row, text, group in choices
                if group == key
            ),
        )
        for key, label in _GROUP_LABELS
    ]
    return _picker("metric", "Metrics", groups, len(choices))


def _comparison_block(environment: str, rows: Sequence[tuple]) -> str:
    """One environment's planners side by side: a table, then the run's charts.

    The table has a row per metric and a column per planner. A run logs a
    few dozen metrics and a study compares a handful of planners, so the
    long side runs down the page and every planner's figure for one metric
    sits on one line, where it can be compared. Every metric the evaluation
    logged is there with its interval, the best trial's return is a first
    row for reference, and the best value in a row is marked when the
    metric's direction is known. A filter box and group toggles narrow the
    rows. The charts are the evaluation run page's own, for these planners.
    """
    study = rows[0][0].study
    objectives = [(str(n), str(d)) for n, d in study.objectives]
    per_planner = [r[1].metrics_for(r[2].name, r[3].name) for r in rows]
    names = sorted({n for metrics in per_planner for n in charts.base_metric_names(metrics)})
    leading = [
        n for n in ("average_return", CommonMetricName.TASK_COMPLETION_RATE.value) if n in names
    ]
    names = leading + [n for n in names if n not in leading]

    def best_of(values: List[Optional[float]], direction: Optional[str]) -> Optional[float]:
        present = [v for v in values if v is not None]
        if direction is None or len(present) < 2 or len(set(present)) == 1:
            return None
        return min(present) if direction == "minimize" else max(present)

    def cell(
        column: int,
        value: Optional[float],
        mark: Optional[float],
        low=None,
        high=None,
        best_trial=None,
    ) -> str:
        # The value and column ride on the cell, so the planner picker can
        # re-mark the best among the planners left showing.
        if value is None:
            return f'<td class="dim" data-col="{column}">—</td>'
        text = f'<span class="cell-value">{value:.4g}</span>'
        if low is not None and high is not None and high > low:
            text += f' <span class="ci">({low:.3g} – {high:.3g})</span>'
        best = mark is not None and value == mark
        return (
            f'<td class="num{" is-best" if best else ""}" data-col="{column}"'
            f"{_export_attrs(value, low, high, best_trial)}>"
            f'{text} <span class="chip chip-ok best-chip">best</span></td>'
        )

    def direction_note(direction: Optional[str]) -> str:
        if direction is None:
            return ""
        arrow = "higher is better" if direction == "maximize" else "lower is better"
        return f' <span class="dim">({arrow})</span>'

    body = []
    best_trial = [r[0].study.best_trial_scores.get("average_return", (None,))[0] for r in rows]
    reference_direction = metric_direction("average_return", objectives)
    reference_mark = best_of(best_trial, reference_direction)
    body.append(
        '<tr class="reference" data-metric="average_return best trial" data-group="outcome"'
        f' data-direction="{reference_direction or ""}" data-export-skip>'
        "<td>average_return, best trial"
        '<span class="dim"> — the tuning estimate, for reference</span></td>'
        + "".join(cell(i, v, reference_mark) for i, v in enumerate(best_trial))
        + "</tr>"
    )
    body.append(
        '<tr data-metric="episodes" data-group="outcome" data-export-skip>'
        "<td>episodes evaluated</td>"
        + "".join(
            f'<td class="num" data-col="{i}">{len(r[3].episodes)}</td>' for i, r in enumerate(rows)
        )
        + "</tr>"
    )
    for name in names:
        direction = metric_direction(name, objectives)
        mark = best_of([m.get(name) for m in per_planner], direction)
        cells = "".join(
            cell(
                i,
                m.get(name),
                mark,
                m.get(name + charts.CI_LOWER_SUFFIX),
                m.get(name + charts.CI_UPPER_SUFFIX),
                rows[i][0].study.best_trial_scores.get(name, (None,))[0],
            )
            for i, m in enumerate(per_planner)
        )
        body.append(
            f'<tr data-metric="{html(name)}" data-group="{metric_group(name)}" '
            f'data-direction="{direction or ""}">'
            f"<td>{html(name)}{direction_note(direction)}</td>{cells}</tr>"
        )
    head = "<th>Metric</th>" + "".join(
        f'<th class="group" data-col="{i}" data-planner="{html(r[3].name)}" '
        f'data-episodes="{len(r[3].episodes)}">'
        f'<a href="{html(run_url(r[0].run))}">{html(r[3].name)}</a></th>'
        for i, r in enumerate(rows)
    )
    controls = (
        '<div class="metric-filter" data-filter-for="compare-table-'
        f'{html(compare_anchor(environment))}">'
        '<input type="search" placeholder="Filter metrics by name" aria-label="Filter metrics">'
        + _planner_picker([r[3].name for r in rows])
        + _metric_picker(
            [
                ("average_return best trial", "average_return, best trial", "outcome"),
                ("episodes", "episodes evaluated", "outcome"),
            ]
            + [(name, name, metric_group(name)) for name in names]
        )
        + "".join(
            f'<label class="check"><input type="checkbox" data-group-toggle="{key}" checked> '
            f"{label}</label>"
            for key, label in _GROUP_LABELS
        )
        + '<span class="dim" data-filter-count></span></div>'
        + _export_buttons(f"compare-table-{compare_anchor(environment)}", filtered=True)
    )
    eval_run, eval_env = rows[0][1], rows[0][2]
    shown = EnvironmentView(
        name=eval_env.name,
        policies=[r[3] for r in rows],
        artifacts=eval_env.artifacts,
    )
    return (
        f'<section class="env" id="{html(compare_anchor(environment))}" '
        f'data-env="{html(environment)}">'
        f"<h2>Compare Tuned planners on {html(environment)}</h2>"
        + controls
        + '<div class="scroll"><table class="metrics compare" '
        f'id="compare-table-{html(compare_anchor(environment))}" '
        f'data-env="{html(environment)}" data-export-name="{html(environment)}_compare">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        '<p class="note">Every metric the evaluation logged, with its confidence interval. '
        '"Best" is marked only where the metric says which way is better. Each planner\'s '
        f'name opens its tuning view. <a href="{html(env_url(eval_run, eval_env.name))}">All '
        "of this environment's evaluation</a>.</p>"
        # The builder opens from what the table shows: the page script ticks
        # the planners and picks the metric the table has left visible.
        + '<details class="inline-builder" data-inline-builder><summary>Build a chart</summary>'
        + chart_builder_html(eval_run, shown)
        + "</details>"
        + _comparison_charts(eval_run, shown)
        + _returns_chart(shown, "Discounted return per evaluation episode")
        + "</section>"
    )


def _comparison_section(group: tuning.StudyGroup) -> str:
    """A block per environment where more than one planner was evaluated."""
    rows = _comparison_rows(group)
    blocks = "".join(
        _comparison_block(environment, env_rows)
        for environment, env_rows in rows.items()
        if len(env_rows) > 1
    )
    return blocks


def study_page(group: tuning.StudyGroup) -> str:
    """A tuning study: one row per tuned planner, each opening its tuning view.

    Args:
        group: The study.

    Returns:
        A complete HTML document.
    """
    run = group.anchor
    title = group_title(group)
    items = [
        ListingItem(
            title=study_label(config.study),
            href=run_url(config.run),
            chip=_chip(config.run.status),
            subtitle=_best_parameters_text(config.study),
            fields=[
                ("Trials run", html(_trials_text(config.study))),
                ("Early stopping", html(_early_stopping_text(config.study))),
                *_config_scores(config),
            ],
            env=config.study.environment_name or config.study.environment or "",
        )
        for config in group.configs
    ]
    evaluation = group.evaluation_run
    params = run_parameters(run.params)
    comparison = _comparison_section(group)
    environments = _study_environments(group)
    several_environments = len(environments) > 1
    tuning_builder = (
        '<details class="inline-builder"><summary>Build a tuning chart</summary>'
        '<p class="note">A figure of one tuned planner\'s trials. Its data is fetched when '
        "you pick the planner.</p>"
        + tuning_chart_builder_html(
            configs=[
                (
                    study_label(config.study),
                    tuning_chart_json_url(config.run),
                    config.study.environment_name or config.study.environment or "",
                )
                for config in group.configs
            ]
        )
        + "</details>"
        if group.configs
        else ""
    )
    environment_filter = (
        '<div class="metric-filter env-filter" data-env-filter>'
        + _picker(
            "environment",
            "Environments",
            [
                (
                    "",
                    "".join(
                        f'<label class="check"><input type="checkbox" '
                        f'data-choice-name="{html(name)}" checked> {html(name)}</label>'
                        for name in environments
                    ),
                )
            ],
            len(environments),
        )
        + '<span class="dim">Show only the environments you pick; the choice is kept '
        "for this page.</span></div>"
        if several_environments
        else ""
    )
    scripts = (
        '<script src="/static/figure-kit.js"></script>'
        '<script src="/static/chart-builder.js"></script>'
        '<script src="/static/tuning-chart-builder.js"></script>'
        '<script src="/static/metric-filter.js"></script>'
    )
    return layout(
        title,
        [
            ("Experiments", "/"),
            (run.experiment_name, _url("experiment", run.store_index, run.experiment_id)),
            (title, None),
        ],
        f'<div class="page-head"><h1>{html(title)} {_chip(run.status)}</h1>'
        f"{_layout_toggle()}</div>"
        f'<p class="note">Started {html(_timestamp(run.start_time))} · '
        f"run <code>{html(run.run_name)}</code>"
        + (
            f' · raw evaluation run <a href="{html(run_url(evaluation))}">'
            f"{html(evaluation.run_name)}</a>"
            if evaluation is not None
            else ""
        )
        + "</p>"
        + environment_filter
        # With several environments the comparison is what the page is for,
        # so it comes first; with one, the planners lead and it follows.
        + (comparison if several_environments else "")
        + "<h2>Tuned planners</h2>"
        + _listing(items, "No planner in this study finished tuning.")
        + ("" if several_environments else comparison)
        + tuning_builder
        + params
        + scripts,
    )


def _child_runs(children: Sequence[RunView]) -> str:
    """The runs nested under a study run, as a listing."""
    if not children:
        return ""
    items = [
        ListingItem(
            title=child.run_name,
            href=run_url(child),
            chip=_chip(child.status),
            subtitle=run_kind_label(child),
            fields=[("Started", _timestamp(child.start_time))],
            footnote=child.run_id,
        )
        for child in children
    ]
    return "<h2>Runs in this study</h2>" + _listing(items)


def tuning_chart_url(run: RunView) -> str:
    """URL of a tuned planner's diagnostics chart builder."""
    return run_url(run) + "/tuning-chart"


def tuning_chart_data(study: tuning.TuningStudy) -> Dict[str, object]:
    """What the diagnostics builder draws from: the trials and the study's frame.

    Args:
        study: The study.

    Returns:
        A JSON-serializable mapping.
    """
    return {
        "title": study_label(study),
        "objectives": [
            {"name": str(name), "direction": str(direction)} for name, direction in study.objectives
        ],
        "parameters": [
            {
                "name": p.name,
                "low": p.low,
                "high": p.high,
                "choices": list(p.choices) if p.choices is not None else None,
            }
            for p in study.parameters
        ],
        "best": study.best_trial_number,
        "pareto": sorted(
            set(study.pareto_trial_numbers) | {t.number for t in study.trials if t.is_pareto}
        ),
        "stopped_at": study.stopped_at_trial,
        "trials": [
            {
                "number": t.number,
                "state": t.state,
                "params": t.params,
                "objectives": t.objective_values,
                "duration": t.duration_seconds,
            }
            for t in study.trials
        ],
    }


def tuning_chart_page(
    run: RunView, study: tuning.TuningStudy, parent: Optional[RunView] = None
) -> str:
    """A figure of the study's diagnostics, built by the reader.

    The counterpart of :func:`chart_builder_page` for a tuning study: the
    tuning view's charts are the site's, this one is the author's, with the
    same text fields, looks and downloads.

    Args:
        run: The config run.
        study: The study read from it.
        parent: The study run, for the breadcrumbs.

    Returns:
        A complete HTML document.
    """
    title = study_label(study)
    crumbs: List[Tuple[str, Optional[str]]] = [
        ("Experiments", "/"),
        (run.experiment_name, _url("experiment", run.store_index, run.experiment_id)),
    ]
    if parent is not None:
        crumbs.append(("Tuning study", run_url(parent)))
    crumbs += [(title, run_url(run)), ("Chart", None)]
    return layout(
        f"Chart · {title}",
        crumbs,
        "<h1>Build a chart</h1>"
        f'<p class="note">Tuning of <a href="{html(run_url(run))}">{html(title)}</a>. '
        "Everything plotted is a trial the study recorded.</p>"
        + tuning_chart_builder_html(data=tuning_chart_data(study))
        + '<script src="/static/figure-kit.js"></script>'
        '<script src="/static/tuning-chart-builder.js"></script>',
    )


def tuning_chart_json_url(run: RunView) -> str:
    """URL of one tuned planner's builder data, fetched by a study page's builder."""
    return run_url(run) + "/tuning-chart.json"


def tuning_chart_builder_html(
    data: Optional[Dict[str, object]] = None,
    configs: Sequence[Tuple[str, str, str]] = (),
) -> str:
    """One tuning-diagnostics builder: its controls and canvas, and its data.

    Either the data is embedded, for a single tuned planner, or the builder
    offers ``configs`` to choose from and fetches the chosen one's data. The
    page must load ``figure-kit.js`` and ``tuning-chart-builder.js`` after it.

    Args:
        data: One study's :func:`tuning_chart_data`, to embed.
        configs: ``(label, data URL, environment)`` per tuned planner, for a
            builder that fetches.

    Returns:
        HTML for the builder.
    """
    kinds = [
        ("objective-history", "Objective over the trials"),
        ("parameter-slice", "Objective against a parameter"),
        ("parameter-history", "Parameter over the trials"),
        ("pareto-front", "Pareto front"),
        ("trial-durations", "Trial durations"),
    ]
    if data is not None and len(data.get("objectives") or []) != 2:  # type: ignore[arg-type]
        kinds = [k for k in kinds if k[0] != "pareto-front"]
    chooser = (
        '<label>Tuned planner <select data-role="config">'
        + "".join(
            f'<option value="{html(url)}" data-env="{html(env)}">{html(label)}</option>'
            for label, url, env in configs
        )
        + "</select></label>"
        if configs
        else ""
    )
    embedded = (
        '<script type="application/json" data-role="data">'
        + json.dumps(data).replace("</", "<\\/")
        + "</script>"
        if data is not None
        else ""
    )
    return (
        '<div class="builder" data-tuning-chart-builder>'
        '<form class="builder-controls" data-role="form">'
        + chooser
        + '<label>Chart <select data-role="kind">'
        + "".join(f'<option value="{k}">{html(label)}</option>' for k, label in kinds)
        + "</select></label>"
        '<label data-for="objective">Objective <select data-role="objective"></select></label>'
        '<label data-for="parameter">Parameter <select data-role="parameter"></select></label>'
        '<label>Title <input data-role="title" type="text" autocomplete="off"></label>'
        '<label>Vertical axis <input data-role="y" type="text" autocomplete="off"></label>'
        '<label>Horizontal axis <input data-role="x" type="text" autocomplete="off"></label>'
        '<label>Style <select data-role="style">'
        '<option value="mono">Paper, mono</option>'
        '<option value="colour" selected>Paper, colour</option>'
        '<option value="slide">Slide, dark</option>'
        "</select></label>"
        '<label class="check" data-for="best-line"><input data-role="best-line" '
        'type="checkbox" checked> Best-so-far line</label>'
        '<label class="check" data-for="stop"><input data-role="stop" type="checkbox" checked> '
        "Early-stop marker</label>"
        '<label class="check"><input data-role="highlight" type="checkbox" checked> '
        "Mark the chosen and Pareto trials</label>"
        '<label class="check"><input data-role="legend" type="checkbox" checked> Legend</label>'
        '<div class="builder-actions">'
        '<button type="button" data-role="svg" class="tab">Download SVG</button>'
        '<button type="button" data-role="png" class="tab">Download PNG</button>'
        "</div></form>"
        '<figure class="builder-canvas" data-role="output"></figure>' + embedded + "</div>"
    )


def not_found(message: str) -> str:
    """A 404 body."""
    return layout(
        "Not found",
        [("Experiments", "/"), ("Not found", None)],
        f'<h1>Not found</h1><p class="empty">{html(message)}</p>',
    )
