"""
logger.py
---------
This module provides a unified and customized logging setup for the project.
It implements colored console outputs for better readability using `colorama`, 
along with a file handler that safely records all logs to `reports/results/project.log`.
This ensures consistency in error tracking and information logging across all modules.
"""

import logging
import sys
from colorama import Fore, Style, init
from src.config import RESULTS_DIR

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Ensure the console can render non-ASCII characters (e.g. arrows, sigma,
# emojis) instead of crashing on Windows' default cp1252 code page.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

class ColoredFormatter(logging.Formatter):
    """Custom formatting for console output with colors based on level."""
    
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelno, '')
        formatted_msg = super().format(record)
        return color + formatted_msg + Style.RESET_ALL

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with the unified format.
    Outputs to both console (colored) and a log file.
    """
    logger = logging.getLogger(name)
    
    # Only set up handlers if it doesn't already have them
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Log Formatting
        log_format = '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
        
        # Console Handler (Colored)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(log_format)
        console_handler.setFormatter(console_formatter)
        
        # File Handler (Plain text)
        log_file = RESULTS_DIR / 'project.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger
