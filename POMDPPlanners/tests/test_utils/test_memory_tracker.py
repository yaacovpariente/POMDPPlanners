"""Tests for memory tracker utility.

This module tests the memory tracking functionality, including:
- Basic memory tracking operations
- Different tracking modes (lightweight, detailed, sampling)
- Memory leak detection
- Context manager functionality
- Integration with logging
- Error handling and edge cases
"""

import gc
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from POMDPPlanners.utils.memory_tracker import (
    MemoryTracker,
    memory_monitor,
    profile_memory,
    TRACEMALLOC_AVAILABLE,
    PSUTIL_AVAILABLE,
    MEMORY_PROFILER_AVAILABLE,
)


class TestMemoryTracker:
    """Test cases for MemoryTracker class."""

    def test_initialization_default(self):
        """Test default initialization."""
        tracker = MemoryTracker()
        assert tracker.enable_tracking is False  # Should be False by default
        assert tracker.tracking_mode == "lightweight"
        assert tracker.sample_rate == 1.0
        assert tracker.interval_seconds == 5.0
        assert tracker.leak_threshold_mb == 100.0
        assert len(tracker.checkpoints) == 0
        assert len(tracker.snapshots) == 0

    def test_initialization_with_logger(self):
        """Test initialization with custom logger."""
        logger = Mock()
        tracker = MemoryTracker(logger=logger, enable_tracking=True)
        assert tracker.logger == logger
        assert tracker.enable_tracking is True

    def test_initialization_tracking_modes(self):
        """Test initialization with different tracking modes."""
        # Lightweight mode
        tracker = MemoryTracker(enable_tracking=True, tracking_mode="lightweight")
        assert tracker.tracking_mode == "lightweight"
        assert not tracker.tracemalloc_enabled

        # Detailed mode (if tracemalloc available)
        tracker = MemoryTracker(enable_tracking=True, tracking_mode="detailed")
        assert tracker.tracking_mode == "detailed"
        if TRACEMALLOC_AVAILABLE:
            assert tracker.tracemalloc_enabled
        else:
            assert not tracker.tracemalloc_enabled

        # Sampling mode
        tracker = MemoryTracker(enable_tracking=True, tracking_mode="sampling", sample_rate=0.5)
        assert tracker.tracking_mode == "sampling"
        assert tracker.sample_rate == 0.5

    def test_environment_detection(self):
        """Test automatic environment-based tracking enablement."""
        with patch.dict(os.environ, {"ENABLE_MEMORY_PROFILING": "true"}):
            tracker = MemoryTracker()
            assert tracker.enable_tracking is True

        with patch.dict(os.environ, {"DEBUG": "true"}):
            tracker = MemoryTracker()
            assert tracker.enable_tracking is True

        with patch.dict(os.environ, {"CI": "true"}):
            tracker = MemoryTracker()
            assert tracker.enable_tracking is True

        # Test with no environment variables
        with patch.dict(os.environ, {}, clear=True):
            tracker = MemoryTracker()
            assert tracker.enable_tracking is False

    def test_checkpoint_disabled_tracking(self):
        """Test checkpoint when tracking is disabled."""
        tracker = MemoryTracker(enable_tracking=False)
        result = tracker.checkpoint("test")
        assert result == {}
        assert len(tracker.checkpoints) == 0

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_checkpoint_enabled_tracking(self):
        """Test checkpoint when tracking is enabled."""
        tracker = MemoryTracker(enable_tracking=True, tracking_mode="lightweight")
        result = tracker.checkpoint("test")

        assert "label" in result
        assert "timestamp" in result
        assert "rss_mb" in result
        assert "vms_mb" in result
        assert "percent" in result
        assert result["label"] == "test"
        assert len(tracker.checkpoints) == 1

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_checkpoint_with_gc(self):
        """Test checkpoint with garbage collection."""
        tracker = MemoryTracker(
            enable_tracking=True, interval_seconds=0.0
        )  # Disable interval filtering

        # Create some objects
        data = [i for i in range(10000)]
        tracker.checkpoint("before_cleanup")

        # Delete objects and checkpoint with GC
        del data
        result = tracker.checkpoint("after_cleanup", force_gc=True)

        assert len(tracker.checkpoints) == 2
        assert result["label"] == "after_cleanup"

    def test_sampling_mode(self):
        """Test sampling-based tracking."""
        tracker = MemoryTracker(
            enable_tracking=True, tracking_mode="sampling", sample_rate=0.0  # Never sample
        )

        # Should not create checkpoints with 0% sample rate
        tracker.checkpoint("test1")
        tracker.checkpoint("test2")
        assert len(tracker.checkpoints) == 0

    def test_interval_mode(self):
        """Test time-based interval tracking."""
        tracker = MemoryTracker(
            enable_tracking=True, tracking_mode="lightweight", interval_seconds=1.0
        )

        # First checkpoint should be recorded
        tracker.checkpoint("first")
        assert len(tracker.checkpoints) == 1

        # Immediate second checkpoint should be skipped
        tracker.checkpoint("second")
        assert len(tracker.checkpoints) == 1

        # Wait and checkpoint should be recorded
        time.sleep(1.1)
        tracker.checkpoint("third")
        assert len(tracker.checkpoints) == 2

    def test_get_peak_usage(self):
        """Test peak usage calculation."""
        tracker = MemoryTracker(enable_tracking=False)
        assert tracker.get_peak_usage() == 0.0

        # Mock some checkpoints
        tracker.checkpoints = [
            {"rss_mb": 100.0},
            {"rss_mb": 150.0},
            {"rss_mb": 120.0},
        ]
        assert tracker.get_peak_usage() == 150.0

    def test_get_memory_growth(self):
        """Test memory growth calculation."""
        tracker = MemoryTracker(enable_tracking=False)
        assert tracker.get_memory_growth() == 0.0

        # Mock some checkpoints
        tracker.checkpoints = [
            {"rss_mb": 100.0},
            {"rss_mb": 150.0},
            {"rss_mb": 120.0},
        ]
        assert tracker.get_memory_growth() == 20.0  # 120 - 100

    def test_detect_memory_leak(self):
        """Test memory leak detection."""
        tracker = MemoryTracker(enable_tracking=False, leak_threshold_mb=50.0)

        # No checkpoints
        assert not tracker.detect_memory_leak()

        # Mock checkpoints with leak
        tracker.checkpoints = [
            {"rss_mb": 100.0},
            {"rss_mb": 200.0},  # 100MB growth > 50MB threshold
        ]
        assert tracker.detect_memory_leak()

        # Mock checkpoints without leak
        tracker.checkpoints = [
            {"rss_mb": 100.0},
            {"rss_mb": 120.0},  # 20MB growth < 50MB threshold
        ]
        assert not tracker.detect_memory_leak()

    def test_detect_memory_leak_custom_threshold(self):
        """Test memory leak detection with custom threshold."""
        tracker = MemoryTracker(enable_tracking=False)

        # Mock checkpoints
        tracker.checkpoints = [
            {"rss_mb": 100.0},
            {"rss_mb": 150.0},  # 50MB growth
        ]

        # Should detect leak with 30MB threshold
        assert tracker.detect_memory_leak(threshold_mb=30.0)

        # Should not detect leak with 60MB threshold
        assert not tracker.detect_memory_leak(threshold_mb=60.0)

    @pytest.mark.skipif(not TRACEMALLOC_AVAILABLE, reason="tracemalloc not available")
    def test_compare_snapshots(self):
        """Test snapshot comparison."""
        tracker = MemoryTracker(enable_tracking=True, tracking_mode="detailed")

        # Take two snapshots
        tracker.checkpoint("first")
        tracker.checkpoint("second")

        # Should have snapshots
        assert len(tracker.snapshots) == 2

        # Compare snapshots (should not raise exception)
        tracker.compare_snapshots("first", "second")

    def test_compare_snapshots_tracemalloc_disabled(self):
        """Test snapshot comparison when tracemalloc is disabled."""
        tracker = MemoryTracker(enable_tracking=True, tracking_mode="lightweight")

        # Should not have snapshots
        assert len(tracker.snapshots) == 0

        # Should handle gracefully
        tracker.compare_snapshots("first", "second")

    def test_save_report(self):
        """Test saving memory report to file."""
        tracker = MemoryTracker(enable_tracking=True)  # Enable tracking for this test

        # Mock some checkpoints
        tracker.checkpoints = [
            {"label": "test1", "rss_mb": 100.0, "percent": 5.0, "timestamp": 1234567890.0},
            {"label": "test2", "rss_mb": 150.0, "percent": 7.5, "timestamp": 1234567891.0},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = Path(f.name)

        try:
            tracker.save_report(filepath)
            assert filepath.exists()

            # Check file contents
            content = filepath.read_text()
            assert "Memory Tracking Report" in content
            assert "test1" in content
            assert "test2" in content
            assert "100.0 MB" in content
            assert "150.0 MB" in content
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_save_report_disabled_tracking(self):
        """Test saving report when tracking is disabled."""
        tracker = MemoryTracker(enable_tracking=False)

        # Create a file path that doesn't exist yet
        filepath = Path(tempfile.mktemp(suffix=".txt"))

        try:
            tracker.save_report(filepath)
            # Should not create file when tracking is disabled
            assert not filepath.exists()
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_cleanup(self):
        """Test cleanup functionality."""
        tracker = MemoryTracker(enable_tracking=True, tracking_mode="detailed")

        # Add some data
        tracker.checkpoint("test")
        assert len(tracker.checkpoints) > 0

        # Cleanup
        tracker.cleanup()
        assert len(tracker.checkpoints) == 0
        assert len(tracker.snapshots) == 0
        assert not tracker.tracemalloc_enabled

    def test_log_summary(self):
        """Test logging summary."""
        logger = Mock()
        tracker = MemoryTracker(logger=logger, enable_tracking=True)  # Enable tracking

        # Mock checkpoints
        tracker.checkpoints = [
            {"rss_mb": 100.0},
            {"rss_mb": 150.0},
        ]

        tracker.log_summary()

        # Should call logger.info (may be called multiple times due to initialization)
        assert logger.info.call_count >= 1
        # Find the call with Memory Summary
        summary_call = None
        for call in logger.info.call_args_list:
            if "Memory Summary" in call[0][0]:
                summary_call = call[0][0]
                break
        assert summary_call is not None
        assert "100.0 MB" in summary_call
        assert "150.0 MB" in summary_call

    def test_log_summary_no_checkpoints(self):
        """Test logging summary with no checkpoints."""
        logger = Mock()
        tracker = MemoryTracker(logger=logger, enable_tracking=False)

        tracker.log_summary()

        # Should not call logger for summary when no checkpoints (but may call for initialization)
        summary_calls = [
            call for call in logger.info.call_args_list if "Memory Summary" in call[0][0]
        ]
        assert len(summary_calls) == 0


class TestMemoryMonitor:
    """Test cases for memory_monitor context manager."""

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_monitor_context_manager(self):
        """Test memory monitor context manager."""
        logger = Mock()

        with memory_monitor(threshold_mb=1000.0, logger=logger, enable_tracking=True) as monitor:
            assert monitor is not None
            assert isinstance(monitor, MemoryTracker)
            assert monitor.enable_tracking is True
            assert monitor.tracking_mode == "lightweight"

            # Should have start checkpoint
            assert len(monitor.checkpoints) >= 1

        # Note: After cleanup, checkpoints are cleared, so we can't check for end checkpoint

    def test_memory_monitor_disabled(self):
        """Test memory monitor when disabled."""
        with memory_monitor(enable_tracking=False) as monitor:
            assert monitor is None

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_monitor_high_usage(self):
        """Test memory monitor with high usage threshold."""
        logger = Mock()

        with memory_monitor(threshold_mb=0.1, logger=logger, enable_tracking=True) as monitor:
            # Create some memory usage
            data = [i for i in range(100000)]
            monitor.checkpoint("high_usage")
            del data

        # Should log warning about high memory usage
        logger.warning.assert_called()

    def test_memory_monitor_exception_handling(self):
        """Test memory monitor exception handling."""
        logger = Mock()

        with pytest.raises(ValueError):
            with memory_monitor(logger=logger, enable_tracking=True) as monitor:
                if monitor:
                    monitor.checkpoint("before_exception")
                raise ValueError("Test exception")

        # Should still clean up properly
        assert True  # If we get here, cleanup worked


class TestProfileMemory:
    """Test cases for profile_memory decorator."""

    def test_profile_memory_decorator_disabled(self):
        """Test profile_memory decorator when profiling is disabled."""

        @profile_memory
        def test_function():
            return "test"

        # Should return original function when profiling disabled
        result = test_function()
        assert result == "test"

    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    def test_profile_memory_decorator_enabled(self):
        """Test profile_memory decorator when profiling is enabled."""
        with patch.dict(os.environ, {"ENABLE_MEMORY_PROFILING": "true"}):

            @profile_memory
            def test_function():
                return "test"

            # Should return profiled function
            result = test_function()
            assert result == "test"

    def test_profile_memory_decorator_debug_mode(self):
        """Test profile_memory decorator in debug mode."""
        with patch.dict(os.environ, {"DEBUG": "true"}):

            @profile_memory
            def test_function():
                return "test"

            result = test_function()
            assert result == "test"


class TestIntegration:
    """Integration tests for memory tracker."""

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_full_workflow(self):
        """Test complete memory tracking workflow."""
        logger = Mock()
        tracker = MemoryTracker(
            logger=logger,
            enable_tracking=True,
            tracking_mode="lightweight",
            leak_threshold_mb=50.0,
            interval_seconds=0.0,  # Disable interval filtering
        )

        # Initial checkpoint
        tracker.checkpoint("initial")
        assert len(tracker.checkpoints) == 1

        # Create some memory usage
        data1 = [i for i in range(50000)]
        tracker.checkpoint("after_data1")

        data2 = [i for i in range(100000)]
        tracker.checkpoint("after_data2")

        # Clean up
        del data1, data2
        gc.collect()
        tracker.checkpoint("after_cleanup")

        # Check results
        assert len(tracker.checkpoints) == 4
        assert tracker.get_peak_usage() > 0
        # Memory growth can be negative due to garbage collection
        assert tracker.get_memory_growth() >= -10.0  # Allow for some negative growth

        # Log summary
        tracker.log_summary()
        logger.info.assert_called()

        # Save report
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = Path(f.name)

        try:
            tracker.save_report(filepath)
            assert filepath.exists()
        finally:
            if filepath.exists():
                filepath.unlink()

        # Cleanup
        tracker.cleanup()
        assert len(tracker.checkpoints) == 0

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with missing psutil
        with patch("POMDPPlanners.utils.memory_tracker.PSUTIL_AVAILABLE", False):
            tracker = MemoryTracker(enable_tracking=True)
            result = tracker.checkpoint("test")
            # Should handle gracefully
            assert "label" in result
            assert result["rss_mb"] == 0.0

        # Test with missing tracemalloc
        with patch("POMDPPlanners.utils.memory_tracker.TRACEMALLOC_AVAILABLE", False):
            tracker = MemoryTracker(enable_tracking=True, tracking_mode="detailed")
            assert not tracker.tracemalloc_enabled

    def test_memory_tracker_with_simulator_usage(self):
        """Test memory tracker in a simulator-like scenario."""
        tracker = MemoryTracker(
            enable_tracking=True, tracking_mode="lightweight", interval_seconds=0.0
        )

        # Simulate simulator workflow
        tracker.checkpoint("simulator_init")

        # Simulate environment creation
        tracker.checkpoint("after_env_creation")

        # Simulate policy creation
        tracker.checkpoint("after_policy_creation")

        # Simulate simulation execution
        tracker.checkpoint("after_simulation")

        # Simulate cleanup
        tracker.checkpoint("after_cleanup")

        # Check workflow
        assert len(tracker.checkpoints) == 5
        assert tracker.checkpoints[0]["label"] == "simulator_init"
        assert tracker.checkpoints[-1]["label"] == "after_cleanup"

        # Test leak detection
        leak_detected = tracker.detect_memory_leak(threshold_mb=1.0)
        # Should not detect leak for this simple test
        assert not leak_detected

        # Cleanup
        tracker.cleanup()


# Fixtures for testing
@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()


@pytest.fixture
def temp_file():
    """Temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        filepath = Path(f.name)
    yield filepath
    if filepath.exists():
        filepath.unlink()


# Performance tests (marked as slow)
@pytest.mark.slow
class TestPerformance:
    """Performance tests for memory tracker."""

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_checkpoint_performance(self):
        """Test checkpoint performance."""
        tracker = MemoryTracker(
            enable_tracking=True, tracking_mode="lightweight", interval_seconds=0.0
        )

        start_time = time.time()
        for i in range(100):
            tracker.checkpoint(f"test_{i}")
        end_time = time.time()

        # Should complete 100 checkpoints in reasonable time
        duration = end_time - start_time
        assert duration < 50.0  # Accommodate slower CI environments
        assert len(tracker.checkpoints) == 100

    def test_disabled_tracking_performance(self):
        """Test performance when tracking is disabled."""
        tracker = MemoryTracker(enable_tracking=False)

        start_time = time.time()
        for i in range(1000):
            tracker.checkpoint(f"test_{i}")
        end_time = time.time()

        # Should be very fast when disabled
        duration = end_time - start_time
        assert duration < 0.1  # Should be very fast
        assert len(tracker.checkpoints) == 0
