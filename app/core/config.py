import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Validation ---
def validate_config():
    """Ensures all required environment variables are present."""
    missing = []
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    
    if missing:
        # Use a simple print here because the logger might not be configured yet
        error_msg = f"❌ CONFIGURATION ERROR: Missing required environment variables: {', '.join(missing)}"
        print(error_msg)
        raise RuntimeError(error_msg)

# --- Logging Setup ---
def setup_logging():
    """Configures a professional logging format for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Silence noisy third-party logs
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("codebase-assistant")
