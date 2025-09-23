import atexit
import logging
import queue
import signal
import threading
import time
from datetime import datetime
from logging.handlers import QueueHandler
from pathlib import Path
from typing import Any, Dict, Optional, Deque


class QueueLoggerManager:
    """Centralized queue-based logger manager with individual task log files.

    This manager solves the "too many open files" problem in heavy multiprocessing
    scenarios by using a single writer thread that handles all file I/O operations.
    Worker processes only interact with a memory queue, eliminating file descriptor
    leaks while maintaining individual log files per task.

    Key benefits:
    - Workers never open file handles directly (prevents FD leaks)
    - Single writer thread manages individual file handlers per task
    - Smart handler pooling and cleanup
    - Maintains individual log files as before
    - Scales to thousands of concurrent processes
    """

    def __init__(
        self,
        max_handlers: int = 100,
        cleanup_interval: int = 60,
        handler_timeout: int = 300,
    ):
        self._log_queue: "queue.Queue[Optional[logging.LogRecord]]" = queue.Queue(
            maxsize=10000
        )  # Bounded queue
        self._writer_thread: Optional[threading.Thread] = None
        self._task_handlers: Dict[str, logging.Handler] = {}  # task_id -> file handler
        self._handler_ref_count: Dict[logging.Handler, int] = {}  # handler -> reference count
        self._handler_last_used: Dict[logging.Handler, float] = {}  # handler -> timestamp
        self._loggers: Dict[str, Dict[str, Any]] = {}  # logger_name -> config
        self._cleanup_timer: Optional[threading.Timer] = None
        self._shutdown_event = threading.Event()
        self._max_handlers = max_handlers
        self._cleanup_interval = cleanup_interval
        self._handler_timeout = handler_timeout
        self._setup_cleanup_handlers()

    def start(self):
        """Start the background logging thread."""
        if self._writer_thread is None or not self._writer_thread.is_alive():
            self._writer_thread = threading.Thread(
                target=self._log_writer_worker, name="QueueLogWriter", daemon=True
            )
            self._writer_thread.start()

            # Start periodic cleanup
            self._start_periodic_cleanup()

    def stop(self):
        """Stop the background logging thread and cleanup resources."""
        if self._shutdown_event.is_set():
            return  # Already stopping

        self._shutdown_event.set()

        # Stop cleanup timer
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        if self._writer_thread and self._writer_thread.is_alive():
            # Send shutdown signal through queue
            try:
                self._log_queue.put_nowait(None)
            except queue.Full:
                pass  # Queue full, thread will timeout and check shutdown event
            self._writer_thread.join(timeout=5.0)

    def get_queue_logger(
        self,
        task_id: str,
        cache_dir: Optional[Path] = None,
        debug: bool = False,
        console_output: bool = True,
    ) -> logging.Logger:
        """Get a logger that writes to the centralized queue with individual task file.

        Args:
            task_id: Unique task identifier for individual log file
            cache_dir: Directory for log files
            debug: Enable debug mode with verbose logging
            console_output: Enable/disable console output

        Returns:
            Configured logger instance that writes to the queue
        """

        logger_name = f"queue.{task_id}"

        if logger_name not in self._loggers:
            # Create logger with queue handler
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG if debug else logging.INFO)

            # Remove any existing handlers to prevent conflicts
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Add queue handler - this is the ONLY handler workers use
            queue_handler = QueueHandler(self._log_queue)
            logger.addHandler(queue_handler)

            # Store configuration for the writer thread
            self._loggers[logger_name] = {
                "task_id": task_id,
                "cache_dir": cache_dir,
                "debug": debug,
                "console_output": console_output,
            }

            # Ensure writer thread is started
            self.start()

        return logging.getLogger(logger_name)

    def _log_writer_worker(self):
        """Background thread that processes log queue and writes to individual task files."""

        console_handler = None

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get log record from queue (blocks until available)
                    record = self._log_queue.get(timeout=1.0)

                    # Shutdown signal
                    if record is None:
                        break

                    config = self._loggers.get(record.name, {})
                    task_id = config.get("task_id")

                    if not task_id:
                        continue

                    # Get or create task-specific file handler
                    file_handler = self._get_or_create_task_handler(task_id, config)

                    # Write to task-specific file
                    if file_handler:
                        try:
                            file_handler.emit(record)
                            file_handler.flush()  # Ensure immediate write
                        except Exception:
                            pass  # Don't let handler errors crash writer thread

                    # Write to shared console handler if enabled
                    if config.get("console_output", True):
                        if console_handler is None:
                            console_handler = self._create_console_handler(
                                config.get("debug", False)
                            )

                        try:
                            console_handler.emit(record)
                        except Exception:
                            pass

                    self._log_queue.task_done()

                except queue.Empty:
                    continue  # Timeout, check shutdown event

                except Exception:
                    # Don't let logging errors crash the writer thread
                    pass

        finally:
            # Cleanup all task handlers
            for handler in self._task_handlers.values():
                try:
                    handler.close()
                except:
                    pass

            if console_handler:
                try:
                    console_handler.close()
                except:
                    pass

    def _get_or_create_task_handler(self, task_id: str, config: dict) -> Optional[logging.Handler]:
        """Get or create file handler for specific task (maintains individual files)."""

        if task_id not in self._task_handlers:
            cache_dir = config.get("cache_dir")
            if not cache_dir:
                return None

            # Create task-specific log file (same pattern as original)
            logs_dir = cache_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use the full task ID for filename (maintains existing behavior)
            task_name = task_id.replace(".", "_")
            log_file = logs_dir / f"{task_name}_{timestamp}.log"

            # Create handler in writer thread (not worker processes)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.DEBUG if config.get("debug") else logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
            )
            handler.setFormatter(formatter)

            # Track handler for cleanup
            self._task_handlers[task_id] = handler
            self._handler_ref_count[handler] = 1
            self._handler_last_used[handler] = time.time()

            # Check if we need emergency cleanup
            if len(self._task_handlers) > self._max_handlers * 0.8:
                self._emergency_cleanup()

        else:
            # Update reference count and last used time
            handler = self._task_handlers[task_id]
            self._handler_ref_count[handler] = self._handler_ref_count.get(handler, 0) + 1
            self._handler_last_used[handler] = time.time()

        return self._task_handlers.get(task_id)

    def _create_console_handler(self, debug: bool) -> logging.Handler:
        """Create console handler."""

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG if debug else logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s: %(pathname)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)

        return handler

    def _start_periodic_cleanup(self):
        """Start periodic cleanup of unused handlers."""
        if not self._shutdown_event.is_set():
            self._cleanup_unused_handlers()
            # Schedule next cleanup
            self._cleanup_timer = threading.Timer(
                self._cleanup_interval, self._start_periodic_cleanup
            )
            self._cleanup_timer.start()

    def _cleanup_unused_handlers(self):
        """Clean up handlers that haven't been used recently."""

        current_time = time.time()
        handlers_to_cleanup = []

        for task_id, handler in self._task_handlers.items():
            last_used = self._handler_last_used.get(handler, current_time)
            if current_time - last_used > self._handler_timeout:
                handlers_to_cleanup.append((task_id, handler))

        # Clean up old handlers
        for task_id, handler in handlers_to_cleanup:
            try:
                handler.close()
                del self._task_handlers[task_id]
                if handler in self._handler_ref_count:
                    del self._handler_ref_count[handler]
                if handler in self._handler_last_used:
                    del self._handler_last_used[handler]
            except Exception:
                pass  # Don't let cleanup errors affect logging

    def _emergency_cleanup(self):
        """Force cleanup when approaching resource limits."""
        # Close oldest 50% of handlers
        sorted_handlers = sorted(
            [
                (task_id, self._handler_last_used.get(handler, 0))
                for task_id, handler in self._task_handlers.items()
            ],
            key=lambda x: x[1],
        )

        cleanup_count = len(sorted_handlers) // 2
        for task_id, _ in sorted_handlers[:cleanup_count]:
            if task_id in self._task_handlers:
                try:
                    handler = self._task_handlers[task_id]
                    handler.close()
                    del self._task_handlers[task_id]
                    if handler in self._handler_ref_count:
                        del self._handler_ref_count[handler]
                    if handler in self._handler_last_used:
                        del self._handler_last_used[handler]
                except Exception:
                    pass

    def _setup_cleanup_handlers(self):
        """Setup cleanup for process exit and signals."""
        atexit.register(self.stop)
        try:
            signal.signal(signal.SIGTERM, lambda sig, frame: self.stop())
            signal.signal(signal.SIGINT, lambda sig, frame: self.stop())
        except ValueError:
            # Signal setup might fail in some environments (e.g., threads)
            pass


# Global singleton instance
_queue_logger_manager = None


def get_queue_logger_manager() -> QueueLoggerManager:
    """Get the global queue logger manager instance."""
    global _queue_logger_manager
    if _queue_logger_manager is None:
        _queue_logger_manager = QueueLoggerManager()
    return _queue_logger_manager


def get_logger(
    name: str,
    level: int = logging.INFO,
    output_dir: Optional[Path] = None,
    debug: bool = False,
    console_output: bool = True,
    use_queue: bool = False,
) -> logging.Logger:
    """Get a configured logger for POMDP experiments and algorithm execution.

    This utility creates standardized loggers for tracking experimental progress,
    algorithm execution, and debugging information. It supports both console
    and file logging with configurable levels and formatting.

    The logger automatically handles timestamped log files, directory creation,
    and proper formatter configuration for both development and production use.

    Args:
        name: Logger identifier (typically module name or algorithm name)
        level: Base logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        output_dir: Directory for log files (creates logs/ subdirectory if provided)
        debug: Enable debug mode with verbose logging and detailed formatting
        console_output: Enable/disable console output (useful for batch experiments)
        use_queue: Enable queue-based logging for heavy multiprocessing scenarios

    Returns:
        Configured logger instance ready for use

    Note:
        Queue-based logging is recommended for heavy multiprocessing workloads
        to prevent "too many open files" errors. It uses a single writer thread
        to handle all file I/O operations while workers only interact with a
        memory queue, maintaining individual log files per task.

    Example:
        Basic logging setup for algorithm execution::

            from POMDPPlanners.utils.logger import get_logger
            import logging

            # Create logger for POMCP algorithm
            logger = get_logger("POMCP_Tiger", level=logging.INFO)

            # Log algorithm progress
            logger.info("Starting POMCP planning on Tiger POMDP")
            logger.info("Tree construction complete: 1000 simulations")
            logger.warning("Low particle count detected: 50 particles")
            logger.error("Planning failed: invalid belief state")

    Example:
        File logging for experimental studies::

            from pathlib import Path

            # Setup file logging for experiment tracking
            experiment_dir = Path("experiments/tiger_study_2024")
            logger = get_logger(
                name="TigerExperiment",
                level=logging.INFO,
                output_dir=experiment_dir,
                console_output=True
            )

            # Log files will be created at:
            # experiments/tiger_study_2024/logs/TigerExperiment_20240315_143022.log

            logger.info("Starting Tiger POMDP comparative study")
            logger.info("Environments: TigerPOMDP")
            logger.info("Planners: POMCP, PFT-DPW, SparsePFT")
            logger.info("Episodes per run: 100")

    Example:
        Debug mode for algorithm development::

            # Enable detailed debug logging
            debug_logger = get_logger(
                name="POMCP_Debug",
                debug=True,                # Enables DEBUG level + detailed formatting
                output_dir=Path("debug_logs"),
                console_output=True
            )

            debug_logger.debug("UCB values: [0.45, 0.67, 0.23]")
            debug_logger.debug("Selected action: listen (index=0)")
            debug_logger.debug("Tree depth reached: 15/20")
            debug_logger.info("Simulation complete: reward=8.5")

    Example:
        Batch experiment logging (console disabled)::

            # For batch experiments where console output clutters results
            batch_logger = get_logger(
                name="BatchExperiment",
                level=logging.INFO,
                output_dir=Path("batch_results"),
                console_output=False      # Only file logging
            )

            # All output goes to file only - clean console for batch processing
            for run in range(100):
                batch_logger.info(f"Run {run+1}: reward={reward:.3f}, steps={steps}")

    Example:
        Integration with simulation framework::

            from POMDPPlanners.simulations.episodes import run_episode
            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
            from POMDPPlanners.core.belief import get_initial_belief

            # Setup logging for episode execution
            episode_logger = get_logger("EpisodeRunner", level=logging.INFO)

            # Create environment and planner
            env = TigerPOMDP(discount_factor=0.95)
            planner = POMCP(
                environment=env,
                discount_factor=0.95,
                depth=20,
                exploration_constant=1.0,
                name="POMCP_Tiger",
                n_simulations=1000
            )

            initial_belief = get_initial_belief(env, n_particles=200)

            # Run episode with logging
            history = run_episode(
                environment=env,
                policy=planner,
                initial_belief=initial_belief,
                num_steps=50,
                logger=episode_logger     # Pass logger to episode runner
            )

            episode_logger.info(f"Episode completed: {len(history.history)} steps")
            episode_logger.info(f"Total reward: {history.total_return:.3f}")

    Logging Best Practices:
        **Log Levels**:
        - DEBUG: Detailed algorithm internals, variable values
        - INFO: Algorithm progress, major milestones, results
        - WARNING: Suboptimal conditions, parameter issues
        - ERROR: Algorithm failures, invalid inputs
        - CRITICAL: System failures, experiment termination

        **Message Format**:
        - Use descriptive messages with context
        - Include relevant numerical values
        - Timestamp and location automatically included

        **Performance Considerations**:
        - Disable debug logging in production experiments
        - Use file-only logging for batch processing
        - Consider log rotation for long-running experiments

    File Organization:
        Log files are automatically organized with timestamps:
        ```
        output_dir/
        └── logs/
            ├── POMCP_Tiger_20240315_143022.log
            ├── PFT_DPW_CartPole_20240315_143155.log
            └── Experiment_Batch_20240315_144301.log
        ```

    Thread Safety:
        The logger is thread-safe and suitable for use with distributed
        computing frameworks like Dask or Joblib for parallel experiments.
    """

    if use_queue:
        # Use queue-based logging for better resource management in multiprocessing
        manager = get_queue_logger_manager()
        return manager.get_queue_logger(
            task_id=name,
            cache_dir=output_dir,
            debug=debug,
            console_output=console_output,
        )
    else:
        # Original individual logger implementation
        return _create_individual_logger(name, level, output_dir, debug, console_output)


def _create_individual_logger(
    name: str, level: int, output_dir: Optional[Path], debug: bool, console_output: bool
) -> logging.Logger:
    """Create individual logger (original implementation).

    This is the original logger creation logic, preserved for backward compatibility
    and scenarios where queue-based logging is not needed.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler only if console_output is True
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if debug else level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s: %(pathname)s:%(lineno)d - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Add file handler if output_dir is provided
    if output_dir is not None:
        # Convert string to Path if needed
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        # Create logs directory
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use logger name (which includes policy name) in the filename
        name_part = name.replace(".", "_")  # Replace dots with underscores for filename
        log_file = logs_dir / f"{name_part}_{timestamp}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG if debug else level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Only log to console if console_output is enabled
        if console_output:
            logger.info(f"Logging to file: {log_file}")

    return logger


def get_queue_logger_diagnostics() -> Dict[str, Any]:
    """Get diagnostic information about the queue-based logging system.

    Returns:
        Dictionary containing diagnostic information including queue size,
        handler count, writer thread status, and resource usage.
    """

    try:
        manager = get_queue_logger_manager()

        return {
            "queue_size": manager._log_queue.qsize(),
            "writer_thread_alive": manager._writer_thread and manager._writer_thread.is_alive(),
            "registered_loggers": len(manager._loggers),
            "active_handlers": len(manager._task_handlers),
            "max_handlers": manager._max_handlers,
            "shutdown_event_set": manager._shutdown_event.is_set(),
        }
    except Exception as e:
        return {"error": str(e)}


def cleanup_all_loggers():
    """Emergency cleanup of all logging resources.

    This function should be called when shutting down the application
    to ensure all logging resources are properly cleaned up.
    """

    global _queue_logger_manager
    if _queue_logger_manager is not None:
        _queue_logger_manager.stop()
        _queue_logger_manager = None


def reset_logger_state():
    """Reset the global logger state for testing.

    This function ensures clean state between test runs by:
    - Stopping any running queue manager
    - Clearing the global singleton
    - Removing all Python loggers created by this module
    """
    global _queue_logger_manager

    # Stop and clear the queue manager
    if _queue_logger_manager is not None:
        _queue_logger_manager.stop()
        _queue_logger_manager = None

    # Clear all loggers that start with "queue." to reset state
    import logging

    loggers_to_clear = []
    for name in logging.Logger.manager.loggerDict.keys():
        if name.startswith("queue."):
            loggers_to_clear.append(name)

    for name in loggers_to_clear:
        logger = logging.getLogger(name)
        # Remove all handlers
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except:
                pass
        # Remove from manager
        if name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[name]
