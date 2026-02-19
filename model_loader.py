import logging
import time
from pathlib import Path

import gdown
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "yolov10x_best.pt"
GDRIVE_FILE_ID = "15YJAUuHYJQlMm0_rjlC-e_VJPmAvjeiE"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def download_model(max_retries: int = MAX_RETRIES) -> Path | None:
    """
    Download the YOLOv10 model from Google Drive if not already present.

    Retries up to `max_retries` times on failure.

    Returns:
        Path to the model file, or None if all attempts fail.
    """
    if MODEL_PATH.exists():
        logger.info("Model already exists at: %s — skipping download.", MODEL_PATH)
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Model not found locally. Downloading from Google Drive...")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Download attempt %d / %d ...", attempt, max_retries)
            gdown.download(GDRIVE_URL, str(MODEL_PATH), quiet=False)

            if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
                logger.info("Model downloaded successfully → %s", MODEL_PATH)
                return MODEL_PATH
            else:
                raise RuntimeError("Downloaded file is empty or missing.")

        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                logger.info("Retrying in %ds...", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    logger.error("All %d download attempts failed. Model unavailable.", max_retries)
    # Clean up any partial download
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    return None


def load_model() -> YOLO | None:
    """
    Download (if needed) and load the YOLOv10 model.

    Returns:
        A loaded YOLO instance, or None on failure.
    """
    model_path = download_model()
    if model_path is None:
        logger.error("Cannot load model: download failed.")
        return None

    try:
        logger.info("Loading YOLO model from: %s", model_path)
        model = YOLO(str(model_path))
        logger.info("Model loaded successfully. Classes: %s", list(model.names.values()))
        return model
    except Exception as exc:
        logger.exception("Failed to load YOLO model: %s", exc)
        return None