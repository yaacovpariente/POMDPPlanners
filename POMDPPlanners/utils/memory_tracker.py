"""Memory tracking utilities for POMDP simulations.

This module provides memory tracking capabilities for monitoring memory usage
during simulation execution. It includes both lightweight monitoring for production
use and detailed profiling for development and debugging.

Key Features:
- Conditional profiling (only when enabled)
- Multiple tracking modes (lightweight, detailed, sampling-based)
- Memory leak detection with configurable thresholds
- Integration with existing logging infrastructure
- Support for tracemalloc, psutil, and memory_profiler
"""

import gc
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    import tracemalloc

    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from memory_profiler import profile

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


class MemoryTracker:
    """Memory tracking utility for simulator operations.

    Provides configurable memory monitoring with minimal overhead when disabled.
    Supports multiple tracking modes and integrates with existing logging.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        enable_tracking: Optional[bool] = None,
        tracking_mode: str = "lightweight",
        sample_rate: float = 1.0,
        interval_seconds: float = 5.0,
        leak_threshold_mb: float = 100.0,
    ):
        """Initialize memory tracker.

        Args:
            logger: Logger instance for output. If None, uses print().
            enable_tracking: Whether to enable tracking. If None, auto-detects
                           based on environment variables.
            tracking_mode: Tracking mode - "lightweight", "detailed", or "sampling".
            sample_rate: For sampling mode, fraction of calls to track (0.0-1.0).
            interval_seconds: For time-based tracking, minimum interval between checks.
            leak_threshold_mb: Threshold for memory leak detection in MB.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.tracking_mode = tracking_mode
        self.sample_rate = sample_rate
        self.interval_seconds = interval_seconds
        self.leak_threshold_mb = leak_threshold_mb

        # Auto-detect if tracking should be enabled
        if enable_tracking is None:
            self.enable_tracking = self._should_enable_tracking()
        else:
            self.enable_tracking = enable_tracking

        # Initialize tracking components
        self.checkpoints: List[Dict[str, Any]] = []
        self.snapshots: List[Tuple[str, Any, float]] = []
        self.last_checkpoint_time = 0.0

        # Initialize system monitoring
        if PSUTIL_AVAILABLE and self.enable_tracking:
            self.process = psutil.Process()
        else:
            self.process = None

        # Initialize tracemalloc if available and enabled
        self.tracemalloc_enabled = False
        if (
            TRACEMALLOC_AVAILABLE
            and self.enable_tracking
            and tracking_mode in ["detailed", "sampling"]
        ):
            try:
                tracemalloc.start()
                self.tracemalloc_enabled = True
                self._log_info("Memory tracking ENABLED with tracemalloc (expect 10-30% slowdown)")
            except Exception as e:
                self._log_warning(f"Failed to start tracemalloc: {e}")
        elif self.enable_tracking:
            self._log_info("Memory tracking ENABLED (lightweight mode, minimal overhead)")
        else:
            self._log_info("Memory tracking DISABLED (no performance impact)")

    def _should_enable_tracking(self) -> bool:
        """Determine if tracking should be enabled based on environment."""
        return (
            os.getenv("ENABLE_MEMORY_PROFILING", "false").lower() == "true"
            or os.getenv("DEBUG", "false").lower() == "true"
            or os.getenv("CI", "false").lower() == "true"
        )

    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[INFO] {message}")

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"[WARNING] {message}")

    def _log_debug(self, message: str) -> None:
        """Log debug message."""
        if self.logger:
            self.logger.debug(message)
        else:
            print(f"[DEBUG] {message}")

    def checkpoint(self, label: str, force_gc: bool = True) -> Dict[str, Any]:
        """Record memory checkpoint.

        Args:
            label: Label for this checkpoint.
            force_gc: Whether to force garbage collection before measurement.

        Returns:
            Dictionary with memory information.
        """
        if not self.enable_tracking:
            return {}

        # Force garbage collection if requested
        if force_gc:
            gc.collect()

        current_time = time.time()
        checkpoint = {
            "label": label,
            "timestamp": current_time,
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "percent": 0.0,
        }

        # Get memory information if psutil is available
        if self.process:
            try:
                memory_info = self.process.memory_info()
                checkpoint.update(
                    {
                        "rss_mb": memory_info.rss / 1024 / 1024,
                        "vms_mb": memory_info.vms / 1024 / 1024,
                        "percent": self.process.memory_percent(),
                    }
                )
            except Exception as e:
                self._log_warning(f"Failed to get memory info: {e}")

        # Handle different tracking modes
        should_record = True
        if self.tracking_mode == "sampling":
            if random.random() > self.sample_rate:
                should_record = False
        elif self.tracking_mode == "lightweight":
            if current_time - self.last_checkpoint_time < self.interval_seconds:
                should_record = False

        # Take tracemalloc snapshot if enabled
        if self.tracemalloc_enabled:
            try:
                snapshot = tracemalloc.take_snapshot()
                self.snapshots.append((label, snapshot, checkpoint["rss_mb"]))
            except Exception as e:
                self._log_warning(f"Failed to take tracemalloc snapshot: {e}")

        # Only record checkpoint if we should record
        if should_record:
            self.checkpoints.append(checkpoint)
            self.last_checkpoint_time = current_time

        self._log_debug(f"Memory checkpoint '{label}': {checkpoint['rss_mb']:.1f} MB RSS")
        return checkpoint

    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB."""
        if not self.checkpoints:
            return 0.0
        return max(cp["rss_mb"] for cp in self.checkpoints)

    def get_memory_growth(self) -> float:
        """Get total memory growth from first to last checkpoint."""
        if len(self.checkpoints) < 2:
            return 0.0
        return self.checkpoints[-1]["rss_mb"] - self.checkpoints[0]["rss_mb"]

    def log_summary(self) -> None:
        """Log memory usage summary."""
        if not self.enable_tracking:
            return
        if not self.checkpoints:
            return

        initial = self.checkpoints[0]["rss_mb"]
        final = self.checkpoints[-1]["rss_mb"]
        peak = self.get_peak_usage()
        growth = self.get_memory_growth()

        self._log_info(
            f"Memory Summary - Initial: {initial:.1f} MB, "
            f"Final: {final:.1f} MB, Peak: {peak:.1f} MB, "
            f"Growth: {growth:.1f} MB"
        )

    def detect_memory_leak(self, threshold_mb: Optional[float] = None) -> bool:
        """Detect potential memory leaks.

        Args:
            threshold_mb: Threshold for leak detection. Uses instance default if None.

        Returns:
            True if potential leak detected.
        """
        if len(self.checkpoints) < 2:
            return False

        threshold = threshold_mb or self.leak_threshold_mb
        growth = self.get_memory_growth()

        if growth > threshold:
            self._log_warning(
                f"Potential memory leak detected: {growth:.1f} MB growth "
                f"(threshold: {threshold} MB)"
            )
            return True
        return False

    def compare_snapshots(self, label1: str, label2: str, limit: int = 10) -> None:
        """Compare tracemalloc snapshots between two checkpoints.

        Args:
            label1: Label of first checkpoint.
            label2: Label of second checkpoint.
            limit: Maximum number of differences to show.
        """
        if not self.tracemalloc_enabled:
            self._log_warning("tracemalloc not enabled - cannot compare snapshots")
            return

        snap1 = next((s[1] for s in self.snapshots if s[0] == label1), None)
        snap2 = next((s[1] for s in self.snapshots if s[0] == label2), None)

        if not snap1 or not snap2:
            self._log_warning(f"Could not find snapshots for {label1} or {label2}")
            return

        try:
            top_stats = snap2.compare_to(snap1, "lineno")
            self._log_info(f"Memory differences between {label1} and {label2}:")
            for i, stat in enumerate(top_stats[:limit], 1):
                self._log_info(f"  #{i}: {stat}")
        except Exception as e:
            self._log_warning(f"Failed to compare snapshots: {e}")

    def save_report(self, filepath: Path) -> None:
        """Save memory tracking report to file.

        Args:
            filepath: Path to save the report.
        """
        if not self.enable_tracking:
            return
        if not self.checkpoints:
            return

        try:
            with open(filepath, "w") as f:
                f.write("Memory Tracking Report\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Tracking Mode: {self.tracking_mode}\n")
                f.write(f"Sample Rate: {self.sample_rate}\n")
                f.write(f"Interval: {self.interval_seconds}s\n")
                f.write(f"Leak Threshold: {self.leak_threshold_mb} MB\n\n")

                f.write("Checkpoints:\n")
                f.write("-" * 20 + "\n")
                for cp in self.checkpoints:
                    f.write(
                        f"{cp['label']}: {cp['rss_mb']:.1f} MB RSS, "
                        f"{cp['percent']:.1f}% at {cp['timestamp']:.2f}s\n"
                    )

                f.write("\nSummary:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Peak Usage: {self.get_peak_usage():.1f} MB\n")
                f.write(f"Total Growth: {self.get_memory_growth():.1f} MB\n")
                f.write(f"Leak Detected: {self.detect_memory_leak()}\n")

            self._log_info(f"Memory report saved to: {filepath}")
        except Exception as e:
            self._log_warning(f"Failed to save memory report: {e}")

    def cleanup(self) -> None:
        """Clean up tracking resources."""
        if self.tracemalloc_enabled:
            try:
                tracemalloc.stop()
                self.tracemalloc_enabled = False
            except Exception as e:
                self._log_warning(f"Failed to stop tracemalloc: {e}")

        # Clear stored data
        self.checkpoints.clear()
        self.snapshots.clear()

        self._log_debug("Memory tracker cleaned up")


@contextmanager
def memory_monitor(
    threshold_mb: float = 1000.0,
    logger: Optional[logging.Logger] = None,
    enable_tracking: bool = True,
):
    """Context manager for memory monitoring.

    Args:
        threshold_mb: Alert threshold for memory usage.
        logger: Logger instance for output.
        enable_tracking: Whether to enable tracking.

    Yields:
        MemoryTracker instance.
    """
    if not enable_tracking:
        yield None
        return
    if not PSUTIL_AVAILABLE:
        yield None
        return

    tracker = MemoryTracker(logger=logger, enable_tracking=True, tracking_mode="lightweight")

    try:
        tracker.checkpoint("start")
        yield tracker
    finally:
        tracker.checkpoint("end")

        # Check for high memory usage
        if tracker.process:
            try:
                current_mb = tracker.process.memory_info().rss / 1024 / 1024
                if current_mb > threshold_mb:
                    tracker._log_warning(f"High memory usage: {current_mb:.1f} MB")
            except Exception:
                pass

        tracker.log_summary()
        tracker.cleanup()


def profile_memory(func):
    """Decorator for memory profiling functions.

    Only profiles if memory_profiler is available and tracking is enabled.
    """
    if not MEMORY_PROFILER_AVAILABLE:
        return func

    # Check if profiling should be enabled
    if not (
        os.getenv("ENABLE_MEMORY_PROFILING", "false").lower() == "true"
        or os.getenv("DEBUG", "false").lower() == "true"
    ):
        return func

    return profile(func)


# Example usage and testing
if __name__ == "__main__":
    import tempfile

    # Test the memory tracker
    tracker = MemoryTracker(enable_tracking=True, tracking_mode="lightweight")

    # Simulate some memory usage
    tracker.checkpoint("initial")

    # Create some objects
    data = [i for i in range(100000)]
    tracker.checkpoint("after_data_creation")

    # Clear data
    del data
    gc.collect()
    tracker.checkpoint("after_cleanup")

    # Log summary
    tracker.log_summary()

    # Test leak detection
    tracker.detect_memory_leak()

    # Save report
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tracker.save_report(Path(f.name))
        print(f"Report saved to: {f.name}")

    # Cleanup
    tracker.cleanup()

    # Test context manager
    with memory_monitor(threshold_mb=100) as monitor:
        if monitor:
            monitor.checkpoint("context_test")
            test_data = [i for i in range(50000)]
            monitor.checkpoint("context_test_end")
