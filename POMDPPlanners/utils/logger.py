# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

import atexit
import logging
import queue
import signal
import threading
import time
from datetime import datetime
from logging.handlers import MemoryHandler, QueueHandler
from pathlib import Path
from typing import Any, Dict, Optional


class ConditionalMemoryHandler(MemoryHandler):
    """Memory handler that buffers logs and only flushes on failure or explicit request.

    This handler is designed for failure-only logging scenarios where you want to
    capture all logs during execution but only write them to disk/console when
    a failure occurs. This dramatically reduces I/O overhead for successful operations.

    Attributes:
        should_flush: Flag to manually trigger flushing of buffered logs
    """

    def __init__(self, capacity: int, target: logging.Handler):
        """Initialize the conditional memory handler.

        Args:
            capacity: Maximum number of log records to buffer before auto-flush
            target: The target handler to flush logs to (FileHandler, StreamHandler, etc.)
        """
        super().__init__(capacity, target=target)
        self.should_flush = False

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """Determine if buffered logs should be flushed.

        Flushes occur when:
        - Manual trigger via trigger_flush()
        - ERROR or CRITICAL level message
        - Buffer capacity reached

        Args:
            record: The log record being processed

        Returns:
            True if logs should be flushed, False otherwise
        """
        return self.should_flush or record.levelno >= logging.ERROR or super().shouldFlush(record)

    def trigger_flush(self):
        """Manually trigger flush of all buffered logs to target handler."""
        self.should_flush = True
        self.flush()


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

    def _log_writer_worker(self):  # pylint: disable=too-many-branches
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
                        except Exception:  # pylint: disable=broad-exception-caught
                            pass  # Don't let handler errors crash writer thread

                    # Write to shared console handler if enabled
                    if config.get("console_output", True):
                        if console_handler is None:
                            console_handler = self._create_console_handler(
                                config.get("debug", False)
                            )

                        try:
                            console_handler.emit(record)
                        except Exception:  # pylint: disable=broad-exception-caught
                            pass

                    self._log_queue.task_done()

                except queue.Empty:
                    continue  # Timeout, check shutdown event

                except Exception:  # pylint: disable=broad-exception-caught
                    # Don't let logging errors crash the writer thread
                    pass

        finally:
            # Cleanup all task handlers
            for handler in self._task_handlers.values():
                try:
                    handler.close()
                except:  # pylint: disable=bare-except
                    pass

            if console_handler:
                try:
                    console_handler.close()
                except:  # pylint: disable=bare-except
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
            handler: logging.Handler = logging.FileHandler(log_file)
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
            except Exception:  # pylint: disable=broad-exception-caught
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
                except Exception:  # pylint: disable=broad-exception-caught
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
    global _queue_logger_manager  # pylint: disable=global-statement
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
        Basic console logging::

            >>> import logging
            >>> logger = get_logger("POMCP_Tiger", level=logging.INFO)
            >>> logger.info("Starting POMCP planning")  # doctest: +SKIP

    Example:
        File logging with output directory::

            >>> from pathlib import Path
            >>> logger = get_logger(
            ...     name="Experiment",
            ...     output_dir=Path("/tmp/test_logs"),
            ...     console_output=False
            ... )
            >>> logger.info("Test message")  # doctest: +SKIP

    Example:
        Debug mode with detailed logging::

            >>> debug_logger = get_logger(
            ...     name="Debug_Test",
            ...     debug=True,
            ...     console_output=False
            ... )
            >>> debug_logger.debug("Debug info")  # doctest: +SKIP
            >>> debug_logger.info("Regular info")  # doctest: +SKIP

    Logging Best Practices:
        - DEBUG: Detailed algorithm internals, variable values
        - INFO: Algorithm progress, major milestones, results
        - WARNING: Suboptimal conditions, parameter issues
        - ERROR: Algorithm failures, invalid inputs
        - CRITICAL: System failures, experiment termination

    Note:
        Log files are automatically organized with timestamps in output_dir/logs/.
        The logger is thread-safe for use with distributed computing frameworks.
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

    # If logger already has handlers, return it as-is to avoid creating multiple log files
    if logger.handlers:
        return logger

    # Remove any existing handlers (should be empty at this point, but just in case)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # If no output_dir and no console_output, disable propagation to prevent any logging
    # This is the "no_logs" case where we want to completely disable logging
    # Add a NullHandler to prevent Python's "lastResort" handler from outputting to stderr
    if output_dir is None and not console_output:
        logger.propagate = False
        # Add NullHandler to prevent lastResort handler from being used
        # This ensures no output is produced when logging is disabled
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)

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
            logger.info("Logging to file: %s", log_file)

    return logger


def get_queue_logger_diagnostics() -> Dict[str, Any]:
    """Get diagnostic information about the queue-based logging system.

    Returns:
        Dictionary containing diagnostic information including queue size,
        handler count, writer thread status, and resource usage.
    """

    try:
        manager = get_queue_logger_manager()

        return {  # pylint: disable=protected-access
            "queue_size": manager._log_queue.qsize(),
            "writer_thread_alive": manager._writer_thread and manager._writer_thread.is_alive(),
            "registered_loggers": len(manager._loggers),
            "active_handlers": len(manager._task_handlers),
            "max_handlers": manager._max_handlers,
            "shutdown_event_set": manager._shutdown_event.is_set(),
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {"error": str(e)}


def cleanup_all_loggers():
    """Emergency cleanup of all logging resources.

    This function should be called when shutting down the application
    to ensure all logging resources are properly cleaned up.
    """

    global _queue_logger_manager  # pylint: disable=global-statement
    if _queue_logger_manager is not None:
        _queue_logger_manager.stop()
        _queue_logger_manager = None


def reset_logger_state():
    """Reset the global logger state for testing.

    This function ensures clean state between test runs by:
    - Stopping any running queue manager
    - Clearing the global singleton
    - Resetting the task logger manager
    - Removing all Python loggers created by this module
    """
    global _queue_logger_manager, _task_logger_manager  # pylint: disable=global-statement

    # Stop and clear the queue manager
    if _queue_logger_manager is not None:
        _queue_logger_manager.stop()
        _queue_logger_manager = None

    # Reset the task logger manager
    if _task_logger_manager is not None:
        _task_logger_manager = None

    # Clear all loggers that start with "queue." to reset state

    loggers_to_clear = []
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("queue."):
            loggers_to_clear.append(name)

    for name in loggers_to_clear:
        logger = logging.getLogger(name)
        # Remove all handlers
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except:  # pylint: disable=bare-except
                pass
        # Remove from manager
        if name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[name]


class TaskLoggerManager:
    """Manages task logger configuration and buffered handlers without polluting logger objects.

    This manager maintains a registry of configured loggers and their associated
    memory handlers, eliminating the need to set dynamic attributes on logger objects.

    Attributes:
        _configured_loggers: Dictionary mapping logger names to their configuration state
        _memory_handlers: Dictionary mapping logger names to their ConditionalMemoryHandler list
        _lock: Thread lock for safe concurrent access
    """

    def __init__(self):
        self._configured_loggers: Dict[str, bool] = {}
        self._memory_handlers: Dict[str, list] = {}
        self._lock = threading.Lock()

    def get_or_create_logger(
        self,
        logger_name: str,
        output_dir: Optional[Path],
        debug: bool,
        console_output: bool,
        use_queue: bool,
        log_only_on_failure: bool,
    ) -> logging.Logger:
        """Get or create a task logger with optional buffering.

        Args:
            logger_name: Unique name for the logger
            output_dir: Directory for log files (creates logs/ subdirectory if provided)
            debug: Enable debug mode with verbose logging
            console_output: Enable/disable console output
            use_queue: Enable queue-based logging for multiprocessing
            log_only_on_failure: Buffer logs in memory and only flush on failure

        Returns:
            Configured logger instance ready for task execution
        """
        with self._lock:
            logger = logging.getLogger(logger_name)

            # If logger already has handlers and is in our registry, return it
            if logger.handlers and logger_name in self._configured_loggers:
                return logger

            # Get base logger (which will create handlers)
            logger = get_logger(
                name=logger_name,
                debug=debug,
                output_dir=output_dir,
                console_output=console_output if not log_only_on_failure else False,
                use_queue=use_queue,
            )

            # Add buffered handlers if log_only_on_failure is enabled
            if log_only_on_failure and logger_name not in self._memory_handlers:
                self._setup_buffered_handlers(logger_name, logger)

            # Mark logger as configured in our registry
            self._configured_loggers[logger_name] = True

            return logger

    def _setup_buffered_handlers(self, logger_name: str, logger: logging.Logger) -> None:
        """Wrap logger handlers with ConditionalMemoryHandler for buffering.

        Args:
            logger_name: Name of the logger (for tracking in registry)
            logger: The logger instance to configure with buffered handlers
        """
        memory_handlers = []

        # Replace each existing handler with a buffered version
        for handler in logger.handlers[:]:
            # Create memory buffer that holds logs until failure
            memory_handler = ConditionalMemoryHandler(
                capacity=10000,  # Hold up to 10k log records
                target=handler,
            )
            memory_handler.setLevel(handler.level)
            if handler.formatter:
                memory_handler.setFormatter(handler.formatter)

            # Replace original handler with buffered version
            logger.removeHandler(handler)
            logger.addHandler(memory_handler)
            memory_handlers.append(memory_handler)

        # Store references in our registry (not on the logger object)
        self._memory_handlers[logger_name] = memory_handlers

    def flush_buffered_logs(self, logger_name: str) -> None:
        """Flush buffered logs to file/console when failure occurs.

        Args:
            logger_name: Name of the logger to flush
        """
        try:
            with self._lock:
                if logger_name in self._memory_handlers:
                    for handler in self._memory_handlers[logger_name]:
                        if isinstance(handler, ConditionalMemoryHandler):
                            handler.trigger_flush()
        except Exception:  # pylint: disable=broad-exception-caught
            # Don't let flush errors affect the main task
            pass

    def cleanup_logger(
        self,
        logger_name: str,
        episode_failed: bool = False,
        log_only_on_failure: bool = False,
    ) -> None:
        """Clean up task logger resources with buffering awareness.

        Handles cleanup for both buffered and non-buffered loggers:
        - For buffered loggers: Flushes on failure, discards on success
        - For non-buffered loggers: Always flushes

        Args:
            logger_name: Name of the logger to clean up
            episode_failed: Whether the episode failed
            log_only_on_failure: Whether buffering is enabled for this logger
        """
        try:
            logger = logging.getLogger(logger_name)
            # Only flush if episode failed OR log_only_on_failure is disabled
            should_flush = episode_failed or not log_only_on_failure

            with self._lock:
                # Check if we have buffered handlers for this logger
                _ = logger_name in self._memory_handlers

            for handler in logger.handlers:
                if isinstance(handler, ConditionalMemoryHandler):
                    if should_flush:
                        # Flush buffered logs for failed episodes or normal logging
                        if hasattr(handler, "flush"):
                            handler.flush()
                    else:
                        # Discard buffered logs for successful episodes with log_only_on_failure
                        handler.buffer.clear()
                elif hasattr(handler, "flush"):
                    # Always flush non-buffered handlers
                    handler.flush()
        except Exception:  # pylint: disable=broad-exception-caught
            # Don't let cleanup errors affect the main task
            pass


# Global singleton instance
_task_logger_manager: Optional[TaskLoggerManager] = None


def get_task_logger_manager() -> TaskLoggerManager:
    """Get the global task logger manager instance."""
    global _task_logger_manager  # pylint: disable=global-statement
    if _task_logger_manager is None:
        _task_logger_manager = TaskLoggerManager()
    return _task_logger_manager


def setup_task_logger_with_buffering(
    logger_name: str,
    output_dir: Optional[Path],
    debug: bool,
    console_output: bool,
    use_queue: bool,
    log_only_on_failure: bool,
) -> logging.Logger:
    """Set up a task logger with optional buffering for failure-only logging.

    This is a convenience function that delegates to TaskLoggerManager to avoid
    setting dynamic attributes on logger objects.

    Args:
        logger_name: Unique name for the logger
        output_dir: Directory for log files (creates logs/ subdirectory if provided)
        debug: Enable debug mode with verbose logging
        console_output: Enable/disable console output
        use_queue: Enable queue-based logging for multiprocessing
        log_only_on_failure: Buffer logs in memory and only flush on failure

    Returns:
        Configured logger instance ready for task execution
    """
    manager = get_task_logger_manager()
    return manager.get_or_create_logger(
        logger_name=logger_name,
        output_dir=output_dir,
        debug=debug,
        console_output=console_output,
        use_queue=use_queue,
        log_only_on_failure=log_only_on_failure,
    )


def flush_buffered_task_logs(logger_name: str) -> None:
    """Flush buffered logs to file/console when failure occurs.

    This function triggers the flush of all buffered log records for loggers
    configured with ConditionalMemoryHandler (when log_only_on_failure is enabled).

    Args:
        logger_name: Name of the logger to flush
    """
    manager = get_task_logger_manager()
    manager.flush_buffered_logs(logger_name)


def cleanup_task_logger(
    logger_name: str,
    episode_failed: bool = False,
    log_only_on_failure: bool = False,
) -> None:
    """Clean up task logger resources with buffering awareness.

    Handles cleanup for both buffered and non-buffered loggers:
    - For buffered loggers: Flushes on failure, discards on success
    - For non-buffered loggers: Always flushes

    Args:
        logger_name: Name of the logger to clean up
        episode_failed: Whether the episode failed
        log_only_on_failure: Whether buffering is enabled for this logger
    """
    manager = get_task_logger_manager()
    manager.cleanup_logger(
        logger_name=logger_name,
        episode_failed=episode_failed,
        log_only_on_failure=log_only_on_failure,
    )
