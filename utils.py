import os
import logging
import json
from typing import Dict, Tuple, Optional

class SearchCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "search_cache.json")
        self.cache: Dict[str, Tuple[str, str, str]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save cache to disk"""
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    def get(self, question: str) -> Optional[Tuple[str, str, str]]:
        """Get cached results for a question"""
        return tuple(self.cache[question]) if question in self.cache else None

    def set(self, question: str, results: Tuple[str, str, str]):
        """Cache results for a question"""
        self.cache[question] = results
        self._save_cache()

def setup_logging(save_dir, filename='agent.log', console_output=False):
    """Configure logging to both file and console
    
    Args:
        save_dir (str): Directory to save log files
        filename (str): Name of the log file (default: 'agent.log')
        
    Returns:
        logger: Configured logging instance
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a new logger with a unique name based on the filename
    logger_name = f"{__name__}.{filename}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler - only if explicitly requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)  # Less verbose for console
        logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(save_dir, filename))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # More verbose for file
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (which outputs to console)
    logger.propagate = False
    
    return logger