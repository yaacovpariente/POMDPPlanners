# pylint: disable=protected-access  # Tests need to inspect internal state
"""Tests for :func:`install_signal_handlers`.

The handler is installed via :func:`signal.signal` which has process-wide
state. To keep tests hermetic, an autouse fixture saves and restores the
SIGTERM / SIGINT handlers around every test in this file.
"""

import os
import signal
import threading
from typing import Callable
from unittest.mock import MagicMock

import pytest

from POMDPPlanners.simulations.simulations_deployment.run_progress.notify import (
    NullNotifier,
    install_signal_handlers,
)


@pytest.fixture(autouse=True)
def _save_restore_signal_handlers():
    """Save SIGTERM/SIGINT before every test, restore after."""
    saved = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)


def test_install_handlers_returns_uninstall_callable():
    """Verify install_signal_handlers returns a callable for cleanup.

    Purpose: Validates the lifecycle contract: install returns uninstall.

    Given: A NullNotifier on the main thread.
    When: install_signal_handlers is called.
    Then: The return value is callable, and after calling it, the original
        SIGTERM/SIGINT handlers are restored.

    Test type: unit
    """
    original_term = signal.getsignal(signal.SIGTERM)
    original_int = signal.getsignal(signal.SIGINT)

    uninstall = install_signal_handlers(NullNotifier(), run_id="rid-1")

    assert callable(uninstall)
    assert signal.getsignal(signal.SIGTERM) is not original_term
    assert signal.getsignal(signal.SIGINT) is not original_int

    uninstall()

    assert signal.getsignal(signal.SIGTERM) is original_term
    assert signal.getsignal(signal.SIGINT) is original_int


def test_uninstall_is_idempotent():
    """Verify calling the returned uninstall callable twice does not raise.

    Purpose: Validates that double-cleanup (e.g. atexit + __exit__) is safe.

    Given: An installed pair of signal handlers.
    When: The uninstall callable is invoked twice.
    Then: Neither invocation raises.

    Test type: unit
    """
    uninstall = install_signal_handlers(NullNotifier(), run_id="rid-2")

    uninstall()
    uninstall()


def test_handler_invokes_notifier_run_failed(monkeypatch):
    """Verify the installed handler emits run_failed with the run_id.

    Purpose: Validates the core "notify on signal" contract.

    Given: A MagicMock Notifier with handlers installed for run_id 'rid-x'.
    When: The SIGTERM handler is fetched and invoked manually with a dummy
        frame, os.kill mocked out so the test process is not killed.
    Then: notifier.run_failed was called once, with 'rid-x' as first arg
        and an error string starting with 'signal:SIGTERM'.

    Test type: unit
    """
    notifier = MagicMock()
    install_signal_handlers(notifier, run_id="rid-x")

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    monkeypatch.setattr(os, "kill", lambda *_args, **_kwargs: None)

    handler(signal.SIGTERM, None)

    notifier.run_failed.assert_called_once()
    args, _ = notifier.run_failed.call_args
    assert args[0] == "rid-x"
    assert args[1].startswith("signal:SIGTERM")


def test_handler_chains_to_previous_callable():
    """Verify the installed handler invokes the prior handler when callable.

    Purpose: Validates chaining so existing handlers (e.g. QueueLoggerManager
    cleanup at logger.py:256-260) still run on signal.

    Given: A prior SIGTERM handler that records invocation, then
        install_signal_handlers wrapping a NullNotifier on top of it.
    When: The new SIGTERM handler is invoked with a dummy frame.
    Then: The prior handler is invoked exactly once.

    Test type: unit
    """
    prior_calls: list[tuple[int, object]] = []

    def prior_handler(signum: int, frame: object) -> None:
        prior_calls.append((signum, frame))

    signal.signal(signal.SIGTERM, prior_handler)
    install_signal_handlers(NullNotifier(), run_id="rid-3")

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    handler(signal.SIGTERM, None)

    assert prior_calls == [(signal.SIGTERM, None)]


def test_handler_with_sig_dfl_redelivers_signal(monkeypatch):
    """Verify SIG_DFL fallback re-delivers the signal via os.kill.

    Purpose: Validates the non-callable-prev path: when there is no prior
    handler chain, default OS handling is restored and the signal is
    re-delivered so the process exits as it normally would.

    Given: Default handlers (SIG_DFL) on SIGTERM, then install_signal_handlers.
    When: The new SIGTERM handler runs.
    Then: os.kill(getpid(), SIGTERM) is invoked exactly once.

    Test type: unit
    """
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    install_signal_handlers(NullNotifier(), run_id="rid-4")

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    handler(signal.SIGTERM, None)

    assert kill_calls == [(os.getpid(), signal.SIGTERM)]


def test_handler_with_sig_ign_redelivers_signal(monkeypatch):
    """Verify SIG_IGN prev-handler also goes through the re-deliver path.

    Purpose: Validates that ``signal.SIG_IGN`` (int, not callable) is treated
    like SIG_DFL — re-delivered. The OS then ignores the re-delivered signal.

    Given: SIG_IGN as the prior SIGINT handler.
    When: The new SIGINT handler runs.
    Then: os.kill is called with the current pid and SIGINT.

    Test type: unit
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    install_signal_handlers(NullNotifier(), run_id="rid-5")

    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler)

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    handler(signal.SIGINT, None)

    assert kill_calls == [(os.getpid(), signal.SIGINT)]


def test_handler_swallows_notifier_exception(monkeypatch):
    """Verify a raising notifier does not break the signal-handler path.

    Purpose: Validates the signal handler's own try/except around notifier.

    Given: A Notifier whose run_failed raises.
    When: The signal handler is invoked.
    Then: The exception is swallowed; the re-deliver path still runs.

    Test type: unit
    """
    notifier = MagicMock()
    notifier.run_failed.side_effect = RuntimeError("notifier broken")
    install_signal_handlers(notifier, run_id="rid-6")

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    handler(signal.SIGTERM, None)  # must not raise

    notifier.run_failed.assert_called_once()
    assert kill_calls == [(os.getpid(), signal.SIGTERM)]


def test_install_on_non_main_thread_does_not_raise():
    """Verify install_signal_handlers is a no-op when off the main thread.

    Purpose: Validates the ValueError-swallow path so Dask/PBS workers
    instantiating BaseSimulator off-thread do not crash.

    Given: A thread that is not the main thread.
    When: install_signal_handlers is called inside that thread.
    Then: It returns a callable without raising, and the returned uninstall
        is also safe to call.

    Test type: unit
    """
    uninstall_holder: list[Callable[[], None]] = []

    def worker() -> None:
        uninstall_holder.append(install_signal_handlers(NullNotifier(), run_id="rid-7"))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)

    assert len(uninstall_holder) == 1
    uninstall_holder[0]()  # idempotent + safe
