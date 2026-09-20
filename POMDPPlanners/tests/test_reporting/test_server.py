# SPDX-License-Identifier: MIT

"""Tests for run discovery and the server's routes.

The routes are driven through :class:`Router` rather than over a socket: the
HTTP layer is a dozen lines of glue, and a socket test would only make these
slower and flakier. One test does stand a real server up, to prove the
artifact bytes actually reach a client.
"""

import shutil
import threading
from http import HTTPStatus
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.reporting.server import Router, build_server, parse_range
from POMDPPlanners.reporting.store import RunIndex, find_stores, tracking_uri_for
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_light_dark_pinned_kwargs,
)

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "agent_path_0.mp4"

TRACE_ENV = "ContinuousLightDarkPOMDP"
VIDEO_ENV = "CarlaPOMDP"


def _episode(length: int = 3):
    history = []
    for step in range(length):
        is_last = step == length - 1
        particles = [np.array([float(step), 5.0]), np.array([float(step) + 0.3, 4.8])]
        history.append(
            StepData(
                state=np.array([float(step), 5.0]),
                action=None if is_last else "right",
                next_state=None if is_last else np.array([float(step + 1), 5.0]),
                observation=None if is_last else np.array([float(step) + 0.1, 5.0]),
                reward=None if is_last else -2.0,
                belief=WeightedParticleBelief(
                    particles=particles, log_weights=np.log(np.ones(2) / 2)
                ),
            )
        )
    return history


@pytest.fixture(name="run_dir")
def run_dir_fixture(tmp_path: Path) -> Path:
    """A run directory shaped exactly as a simulation leaves one.

    Two environments on purpose: one that writes a trace and a GIF, and one
    that writes only an MP4. The second is the CARLA / Isaac Lab / nuPlan case
    the artifact-kind dispatch exists for.
    """
    # pylint: disable-next=import-outside-toplevel
    import mlflow
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = tmp_path / "run-a" / "mlruns"
    store.mkdir(parents=True)
    mlflow.set_tracking_uri(f"file://{store}")
    mlflow.set_experiment("results_site_test")

    env = ContinuousLightDarkPOMDP(discount_factor=0.95, **continuous_light_dark_pinned_kwargs())

    staging = tmp_path / "staging"
    trace_viz = staging / TRACE_ENV / "PFT_DPW" / "visualizations"
    trace_viz.mkdir(parents=True)
    for index in range(2):
        env.cache_trace(
            history=_episode(3),
            output_dir=trace_viz,
            episode_index=index,
            policy_name="PFT_DPW",
        )
        (trace_viz / f"agent_path_{index}.gif").write_bytes(b"GIF89a-stand-in")
    trace_plots = staging / TRACE_ENV / "PFT_DPW" / "plots"
    trace_plots.mkdir(parents=True)
    (trace_plots / "discounted_returns_histogram.png").write_bytes(b"\x89PNG-stand-in")
    (staging / TRACE_ENV / "policy_comparison_histogram.png").write_bytes(b"\x89PNG-stand-in")

    video_viz = staging / VIDEO_ENV / "POMCPOW" / "visualizations"
    video_viz.mkdir(parents=True)
    shutil.copy(FIXTURE_VIDEO, video_viz / "agent_path_0.mp4")

    with mlflow.start_run(run_name="environment_policy_comparison"):
        mlflow.log_param("num_environments", 2)
        mlflow.log_param("env_0_name", TRACE_ENV)
        mlflow.log_param("env_0_policy_0_name", "PFT_DPW")
        mlflow.log_param("env_0_policy_0_type", "PFT_DPW")
        # A second planner with metrics but no artifacts: a comparison run has
        # more than one planner per environment, and the run page only draws
        # its comparison charts when there is something to compare.
        mlflow.log_param("env_0_policy_1_name", "POMCP")
        mlflow.log_param("env_0_policy_1_type", "POMCP")
        mlflow.log_param("env_1_name", VIDEO_ENV)
        mlflow.log_param("env_1_policy_0_name", "POMCPOW")
        mlflow.log_param("env_1_policy_0_type", "POMCPOW")
        mlflow.log_metric(f"{TRACE_ENV}_PFT_DPW_average_return", -12.5)
        mlflow.log_metric(f"{TRACE_ENV}_PFT_DPW_average_return_ci_lower", -18.0)
        mlflow.log_metric(f"{TRACE_ENV}_PFT_DPW_average_return_ci_upper", -7.0)
        mlflow.log_metric(f"{TRACE_ENV}_POMCP_average_return", -20.0)
        mlflow.log_metric(f"{VIDEO_ENV}_POMCPOW_average_return", -3.25)
        # One call per child, which is what ``BaseSimulator`` does: logging the
        # environment directory itself would nest it one level deeper and the
        # fixture would stop matching a real run's layout.
        for env_name in (TRACE_ENV, VIDEO_ENV):
            for item in sorted((staging / env_name).iterdir()):
                mlflow.log_artifact(str(item), env_name)

    return tmp_path


@pytest.fixture(name="router")
def router_fixture(run_dir: Path) -> Router:
    """A router over the fixture run directory."""
    return Router(RunIndex([run_dir]))


def _run(router: Router):
    experiment = router.index.experiments[0]
    return experiment, experiment.runs[0]


def _get(router: Router, path: str):
    status, media_type, body = router.resolve(path)
    return status, media_type, body.decode("utf-8", errors="replace")


def test_find_stores_locates_a_nested_mlruns(run_dir: Path):
    """Discovery walks for stores, because every run directory holds its own.

    Purpose: This project points MLflow at ``<run_dir>/mlruns``, so there is no
    single tracking store to configure — only a tree to search.

    Given: A results tree with one run directory in it.
    When: The roots are searched.
    Then: The nested store is found.
    """
    stores = find_stores([run_dir])
    assert stores == [run_dir / "run-a" / "mlruns"]


def test_index_builds_the_environment_and_planner_hierarchy(router: Router):
    """A run is indexed as environments, each with its planners and episodes.

    Purpose: One MLflow run can cover several environments and planners, so the
    row the site shows is (run, environment, planner), not run.

    Given: A run over two environments with one planner each.
    When: The index is built.
    Then: Both environments appear with their planner and episode artifacts.
    """
    _, run = _run(router)
    assert sorted(env.name for env in run.environments) == sorted([TRACE_ENV, VIDEO_ENV])

    trace_env = run.environment(TRACE_ENV)
    policy = trace_env.policy("PFT_DPW")
    assert list(policy.episodes) == [0, 1]
    assert {a.kind.value for a in policy.episodes[0]} == {"trace", "gif"}

    video_policy = run.environment(VIDEO_ENV).policy("POMCPOW")
    assert list(video_policy.episodes) == [0]
    assert [a.kind.value for a in video_policy.episodes[0]] == ["video"]


def test_metrics_are_split_by_environment_and_planner(router: Router):
    """Metric keys glue names with an underscore, so the prefix is stripped by length.

    Purpose: Splitting on ``_`` would break on every real environment name.

    Given: A run with prefixed metric keys.
    When: The metrics for one environment and planner are read.
    Then: Only that pair's metrics come back, with bare names.
    """
    _, run = _run(router)
    metrics = run.metrics_for(TRACE_ENV, "PFT_DPW")
    assert metrics["average_return"] == pytest.approx(-12.5)
    assert metrics["average_return_ci_lower"] == pytest.approx(-18.0)
    assert "average_return" in run.metrics_for(VIDEO_ENV, "POMCPOW")
    assert run.metrics_for(VIDEO_ENV, "POMCPOW")["average_return"] == pytest.approx(-3.25)


def test_index_route_lists_the_experiment(router: Router):
    """The landing page lists every experiment found."""
    status, media_type, body = _get(router, "/")
    assert status == HTTPStatus.OK
    assert media_type.startswith("text/html")
    assert "results_site_test" in body


def test_the_route_chain_reaches_an_episode(router: Router):
    """Experiment, run, environment, planner and episode each render.

    Purpose: The hierarchy is the site's whole navigation, and a broken link
    in the middle of it is invisible from either end.

    Given: The fixture run.
    When: Each page in the chain is requested.
    Then: Each returns HTML naming the next level down.
    """
    experiment, run = _run(router)

    status, _, body = _get(
        router, f"/experiment/{experiment.store_index}/{experiment.experiment_id}"
    )
    assert status == HTTPStatus.OK
    assert run.run_name in body

    base = f"/run/{run.store_index}/{run.experiment_id}/{run.run_id}"
    status, _, body = _get(router, base)
    assert status == HTTPStatus.OK
    assert TRACE_ENV in body and "PFT_DPW" in body
    # The comparison plot the MLflow UI gives, drawn inline.
    assert "<svg" in body and "average_return" in body

    status, _, body = _get(router, f"{base}/env/{TRACE_ENV}")
    assert status == HTTPStatus.OK
    assert "PFT_DPW" in body

    status, _, body = _get(router, f"{base}/env/{TRACE_ENV}/policy/PFT_DPW")
    assert status == HTTPStatus.OK
    assert "Episode 0" in body and "Episode 1" in body

    status, _, body = _get(router, f"{base}/env/{TRACE_ENV}/policy/PFT_DPW/episode/0")
    assert status == HTTPStatus.OK
    assert 'id="viewer"' in body


def test_a_trace_episode_gets_the_viewer_and_a_video_episode_gets_a_video(router: Router):
    """The player on an episode page follows the artifact kind.

    Purpose: This is the dispatch the site exists to prove. The Light-Dark
    episode has a trace, so it gets the 3D viewer; the CARLA-shaped episode has
    only an MP4, so it gets a <video> — with no environment-specific code.

    Given: One episode of each shape in the same run.
    When: Both episode pages render.
    Then: One carries the trace viewer and its scripts, the other a <video>.
    """
    _, run = _run(router)
    base = f"/run/{run.store_index}/{run.experiment_id}/{run.run_id}"

    _, _, trace_page = _get(router, f"{base}/env/{TRACE_ENV}/policy/PFT_DPW/episode/0")
    assert 'id="viewer"' in trace_page
    assert "/static/viewer/renderer-core.js" in trace_page
    assert "/static/viewer/scenes/light-dark.js" in trace_page
    assert "<video" not in trace_page

    _, _, video_page = _get(router, f"{base}/env/{VIDEO_ENV}/policy/POMCPOW/episode/0")
    assert "<video" in video_page
    assert 'id="viewer"' not in video_page
    assert "agent_path_0.mp4" in video_page


def test_a_store_with_no_file_metadata_is_read_from_the_database_beside_it(tmp_path: Path):
    """An ``mlruns`` holding only artifacts is read from its ``mlflow.db``.

    Purpose: MLflow 3.6 and later refuse the local file backend on some
    installations and write metadata to SQLite instead, leaving ``mlruns``
    with artifacts and no ``meta.yaml``. Assuming the file store made every
    such run invisible — the site listed nothing and said nothing was wrong.

    Given: A run directory of each shape.
    When: Each is asked for its tracking URI.
    Then: The one with no ``meta.yaml`` resolves to the database.
    """
    file_store = tmp_path / "file-run" / "mlruns"
    (file_store / "0" / "abc").mkdir(parents=True)
    (file_store / "0" / "meta.yaml").write_text("experiment_id: '0'\n", encoding="utf-8")

    sql_store = tmp_path / "sql-run" / "mlruns"
    (sql_store / "0" / "abc" / "artifacts").mkdir(parents=True)
    database = sql_store.parent / "mlflow.db"
    database.write_bytes(b"SQLite format 3\x00")

    assert tracking_uri_for(file_store) == f"file://{file_store}"
    assert tracking_uri_for(sql_store) == f"sqlite:///{database}"


def test_an_episode_offers_every_recording_of_itself(router: Router):
    """Both records of one episode are reachable, not just the richest.

    Purpose: An episode leaves a trace and the path the environment drew while
    it ran. Playing only the trace hides the recorded path behind a download
    link, and the path is what most readers want to see first.

    Given: The fixture episode, which wrote a trace and a path GIF.
    When: Its page renders.
    Then: It carries a tab per recording, the 3D replay open, and the path
        image is on the page rather than only linked.
    """
    _, run = _run(router)
    base = f"/run/{run.store_index}/{run.experiment_id}/{run.run_id}"

    _, _, page = _get(router, f"{base}/env/{TRACE_ENV}/policy/PFT_DPW/episode/0")

    assert ">3D replay<" in page and ">Recorded path<" in page
    assert 'class="tab" data-view="0" aria-pressed="true"' in page
    assert 'id="viewer"' in page
    assert '<img class="player"' in page and "agent_path_0.gif" in page
    assert "/static/viewer/views.js" in page


def test_a_planner_page_shows_each_episode_and_its_recorded_path(router: Router):
    """The episode list is a grid of paths, with each episode's outcome on it.

    Purpose: Which episode is worth opening is a question about the episodes,
    not about their file names: the path shows where the planner went and the
    trace's own numbers say how it did.

    Given: The fixture planner, with two episodes.
    When: Its page renders.
    Then: Each episode links out, shows its recorded path as the thumbnail, and
        carries the return the trace recorded.
    """
    _, run = _run(router)
    base = f"/run/{run.store_index}/{run.experiment_id}/{run.run_id}"

    _, _, page = _get(router, f"{base}/env/{TRACE_ENV}/policy/PFT_DPW")

    assert "Episode 0" in page and "Episode 1" in page
    assert 'class="thumb"' in page and "agent_path_0.gif" in page
    assert ">Return<" in page


def test_artifact_route_serves_bytes_with_the_right_media_type(router: Router):
    """Artifacts are served from the run directory with a usable media type."""
    _, run = _run(router)
    base = f"/artifact/{run.store_index}/{run.experiment_id}/{run.run_id}"

    status, media_type, body = router.resolve(
        f"{base}/{VIDEO_ENV}/POMCPOW/visualizations/agent_path_0.mp4"
    )
    assert status == HTTPStatus.OK
    assert media_type == "video/mp4"
    assert body == FIXTURE_VIDEO.read_bytes()

    status, media_type, body = router.resolve(
        f"{base}/{TRACE_ENV}/PFT_DPW/visualizations/trace_0.json"
    )
    assert status == HTTPStatus.OK
    assert media_type == "application/json"
    assert b"light_dark.v1" in body


def test_artifact_route_refuses_a_path_that_escapes_the_run(router: Router):
    """A traversal attempt is refused rather than normalised.

    Purpose: The artifact path comes straight from the URL, so it is the one
    place a request can name a file outside the run directory.
    """
    _, run = _run(router)
    status, _, _ = router.resolve(
        f"/artifact/{run.store_index}/{run.experiment_id}/{run.run_id}/../../meta.yaml"
    )
    assert status == HTTPStatus.NOT_FOUND


def test_static_route_serves_the_vendored_viewer(router: Router):
    """The viewer's JavaScript ships with the package and is served locally.

    Purpose: The site must work with no network, which means three.js is
    vendored rather than pulled from a CDN.
    """
    status, media_type, body = router.resolve("/static/vendor/three.min.js")
    assert status == HTTPStatus.OK
    assert media_type.startswith("text/javascript")
    assert b"three.js" in body.lower() or b"THREE" in body

    status, _, _ = router.resolve("/static/../server.py")
    assert status == HTTPStatus.NOT_FOUND


@pytest.mark.parametrize(
    "path",
    [
        "/experiment/0/does-not-exist",
        "/run/0/nope/nope",
        "/nonsense",
    ],
)
def test_unknown_routes_return_not_found(router: Router, path):
    """An unknown URL is a 404 page, not a traceback."""
    status, media_type, body = _get(router, path)
    assert status == HTTPStatus.NOT_FOUND
    assert media_type.startswith("text/html")
    assert "Not found" in body


def test_a_real_server_serves_a_page_and_an_artifact(run_dir: Path):
    """End to end over a socket, on an OS-chosen port.

    Purpose: The router tests skip the HTTP glue; this one proves a browser
    would actually get the page and the artifact bytes.
    """
    server, index = build_server([run_dir], host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        with urlopen(f"http://{host}:{port}/") as response:
            assert response.status == HTTPStatus.OK
            assert b"results_site_test" in response.read()

        run = index.experiments[0].runs[0]
        url = (
            f"http://{host}:{port}/artifact/{run.store_index}/{run.experiment_id}/"
            f"{run.run_id}/{VIDEO_ENV}/POMCPOW/visualizations/agent_path_0.mp4"
        )
        with urlopen(url) as response:
            assert response.headers["Content-Type"] == "video/mp4"
            assert response.read() == FIXTURE_VIDEO.read_bytes()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "header, expected",
    [
        ("bytes=0-99", (0, 99)),
        ("bytes=100-", (100, 999)),
        ("bytes=-50", (950, 999)),
        ("bytes=900-5000", (900, 999)),
        (None, None),
        ("", None),
        ("items=0-9", None),
        ("bytes=0-9,20-29", None),
        ("bytes=abc-def", None),
        ("bytes=1000-1100", None),
        ("bytes=50-10", None),
    ],
)
def test_parse_range_reads_one_byte_range(header, expected):
    """A single byte range is honoured; anything else falls back to the whole file.

    Purpose: Video seeking is byte ranges. A server that mishandles them
    either replays an episode from the start or refuses to scrub, and both look
    like a broken viewer rather than a broken server.

    Given: The range headers a media element and a confused client send.
    When: They are parsed against a 1000-byte file.
    Then: Valid single ranges are clamped to the file; the rest yield None, so
        the caller answers with the whole file, which is always legal.
    """
    assert parse_range(header, 1000) == expected


def test_server_serves_a_video_byte_range(run_dir: Path):
    """A range request on an episode video returns 206 and just those bytes.

    Purpose: This is the path CARLA, Isaac Lab and nuPlan episodes use, and a
    range request is how a <video> element seeks within one.

    Given: A served run whose episode artifact is an MP4.
    When: The client asks for bytes 10-99, and then for the whole file.
    Then: The partial response carries the right status, Content-Range and
        bytes, and the full response matches the file on disk.
    """
    server, index = build_server([run_dir], host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        run = index.experiments[0].runs[0]
        url = (
            f"http://{host}:{port}/artifact/{run.store_index}/{run.experiment_id}/"
            f"{run.run_id}/{VIDEO_ENV}/POMCPOW/visualizations/agent_path_0.mp4"
        )
        expected = FIXTURE_VIDEO.read_bytes()

        request = Request(url, headers={"Range": "bytes=10-99"})
        with urlopen(request) as response:
            assert response.status == HTTPStatus.PARTIAL_CONTENT
            assert response.headers["Content-Range"] == f"bytes 10-99/{len(expected)}"
            assert response.read() == expected[10:100]

        with urlopen(url) as response:
            assert response.status == HTTPStatus.OK
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.read() == expected
    finally:
        server.shutdown()
        server.server_close()
