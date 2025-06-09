"""
Logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import colorlog


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    use_colors: bool = True
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        format_string: Custom format string
        use_colors: Whether to use colored output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Default format
    if format_string is None:
        if use_colors:
            format_string = (
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            format_string = (
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if use_colors:
        formatter = colorlog.ColoredFormatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        # File formatter (no colors)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Set specific loggers to appropriate levels
    logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


class TrainingLogger:
    """Custom logger for training metrics and progress."""
    
    def __init__(self, name: str = 'training'):
        self.logger = get_logger(name)
        
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log the start of an epoch."""
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
        
    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: float, 
                     epoch_time: float):
        """Log the end of an epoch."""
        self.logger.info(
            f"Epoch {epoch} completed - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
    def log_best_model(self, epoch: int, metric: float, metric_name: str):
        """Log when a new best model is found."""
        self.logger.info(
            f"New best model at epoch {epoch} - "
            f"{metric_name}: {metric:.4f}"
        )
        
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping."""
        self.logger.warning(
            f"Early stopping triggered at epoch {epoch} "
            f"(patience: {patience})"
        )
        
    def log_training_complete(self, total_time: float, best_metric: float):
        """Log training completion."""
        self.logger.info(
            f"Training completed in {total_time:.2f}s - "
            f"Best metric: {best_metric:.4f}"
        )
        
    def log_financial_metrics(self, metrics: dict):
        """Log financial performance metrics."""
        self.logger.info("Financial Metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {name}: {value:.4f}")
            else:
                self.logger.info(f"  {name}: {value}")


def log_system_info():
    """Log system information for debugging."""
    import torch
    import platform
    import psutil
    
    logger = get_logger('system')
    
    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("CUDA: Not available")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"RAM: {memory.total / 1e9:.1f} GB total, "
               f"{memory.available / 1e9:.1f} GB available")
    
    # CPU info
    logger.info(f"CPU: {psutil.cpu_count()} cores")


def log_config(config: dict, logger_name: str = 'config'):
    """Log configuration parameters."""
    logger = get_logger(logger_name)
    
    logger.info("Configuration:")
    for section, params in config.items():
        logger.info(f"  {section.upper()}:")
        if isinstance(params, dict):
            for key, value in params.items():
                logger.info(f"    {key}: {value}")
        else:
            logger.info(f"    {params}")


def log_model_summary(model, logger_name: str = 'model'):
    """Log model architecture summary."""
    logger = get_logger(logger_name)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Summary:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1e6:.2f} MB")
    
    # Log architecture
    logger.info("  Architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                logger.info(f"    {name}: {module.__class__.__name__} ({params:,} params)")


def log_data_info(datamodule, logger_name: str = 'data'):
    """Log data module information."""
    logger = get_logger(logger_name)
    
    logger.info("Data Information:")
    logger.info(f"  Symbols: {datamodule.symbols}")
    logger.info(f"  Date range: {datamodule.start_date} to {datamodule.end_date}")
    logger.info(f"  Sequence length: {datamodule.sequence_length}")
    logger.info(f"  Prediction horizon: {datamodule.prediction_horizon}")
    logger.info(f"  Batch size: {datamodule.batch_size}")
    logger.info(f"  Target type: {datamodule.target_type}")
    logger.info(f"  Features: {datamodule.get_num_features()}")
    
    if hasattr(datamodule, 'train_dataset') and datamodule.train_dataset:
        logger.info(f"  Train samples: {len(datamodule.train_dataset)}")
    if hasattr(datamodule, 'val_dataset') and datamodule.val_dataset:
        logger.info(f"  Validation samples: {len(datamodule.val_dataset)}")
    if hasattr(datamodule, 'test_dataset') and datamodule.test_dataset:
        logger.info(f"  Test samples: {len(datamodule.test_dataset)}")