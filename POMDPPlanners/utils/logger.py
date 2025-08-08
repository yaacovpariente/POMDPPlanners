import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

def get_logger(name: str, level: int = logging.INFO, output_dir: Optional[Path] = None, debug: bool = False, console_output: bool = True) -> logging.Logger:
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
        
    Returns:
        Configured logger instance ready for use
        
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
    """
    Get a configured logger instance with optional file and console output.
    
    Args:
        name (str): The name for the logger
        level (int): The logging level (default: logging.INFO)
        output_dir (Optional[Path]): Directory to store log files. If None, only console logging is enabled.
        debug (bool): Whether to enable debug logging (default: False)
        console_output (bool): Whether to enable console output (default: True). 
                              Set to False to disable console output while keeping file logging.
        
    Returns:
        logging.Logger: Configured logger instance
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
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(pathname)s:%(lineno)d - %(message)s')
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
        name_part = name.replace('.', '_')  # Replace dots with underscores for filename
        log_file = logs_dir / f"{name_part}_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG if debug else level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Only log to console if console_output is enabled
        if console_output:
            logger.info(f"Logging to file: {log_file}")
    
    return logger
