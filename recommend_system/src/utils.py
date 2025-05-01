from pathlib import Path
from loguru import logger
import traceback # Keep for potential unexpected errors

def save_file(filepath: Path, content: str, file_type: str = "text", overwrite_check: bool = False) -> bool:
    """Saves content to a file with error handling, using Loguru for messages."""
    try:
        if overwrite_check and filepath.exists():
            logger.warning(f"Skipping save for {file_type} to {filepath} (already exists).")
            return True # Consider existing as success in this context
        # Ensure directory exists before writing
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding='utf-8')
        logger.info(f"Successfully saved {file_type} to {filepath}")
        return True
    except IOError as e:
        logger.error(f"IOError saving {file_type} file {filepath}: {e}")
        return False
    except Exception as e:
        # Use logger.exception to automatically include traceback
        logger.exception(f"An unexpected error occurred while saving {file_type} file {filepath}")
        return False

# Add other utility functions here if needed in the future 