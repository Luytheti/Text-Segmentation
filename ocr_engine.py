pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


import logging
from typing import TypedDict

import cv2
import numpy as np
import pytesseract
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.40
PADDING = 10

PSM_BY_LABEL: dict[str, int] = {
    "Title":          7,
    "Section-header": 7,
    "Page-header":    7,
    "Page-footer":    7,
    "Caption":        7,
    "Text":           6,
    "List-item":      6,
    "Footnote":       11,
    "Table":          6,
    "Formula":        6,
}
DEFAULT_PSM = 6

NON_TEXT_LABELS = {"Picture"}

SPECIAL_LABELS = {
    "Table":          "[TABLE]",
    "Formula":        "[FORMULA]",
    "Caption":        "[CAPTION]",
    "Footnote":       "[FOOTNOTE]",
    "Page-footer":    "[FOOTER]",
    "Page-header":    "[HEADER]",
    "Section-header": "[SECTION]",
    "Title":          "[TITLE]",
    "List-item":      "[LIST]",
}

COLUMN_GAP_RATIO = 0.08


class RegionResult(TypedDict):
    bbox: list[int]
    text: str
    confidence: float
    label: str
    skipped: bool


class AnalysisResult(TypedDict):
    annotated_image: np.ndarray
    extracted_text: str
    regions: list[RegionResult]
    layout: str

def load_ocr_reader(use_gpu: bool = False) -> dict:

    try:
        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR ready. Version: %s", version)
        return {"available": True, "version": str(version)}
    except pytesseract.TesseractNotFoundError:
        raise EnvironmentError(
            "Tesseract binary not found. "
            "Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and ensure it is on your system PATH."
        )


def _pad_bbox(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = PADDING
) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(w, x2 + padding),
        min(h, y2 + padding),
    )


def _preprocess_full_image(image_array: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)


def _deskew_region(region: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region
    coords = np.column_stack(np.where(gray < 200))

    if coords.shape[0] < 50:
        return region

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return region

    h, w = region.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(region, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _preprocess_region(region: np.ndarray, label: str = "Text") -> np.ndarray:

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    # Upscale small regions — Tesseract accuracy drops sharply below 30px height
    h, w = gray.shape
    if h < 60:
        scale = 60 / h
        gray = cv2.resize(gray, (int(w * scale), 60), interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=8,
    )

    # Dilate slightly to help Tesseract connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    return thresh

def _detect_layout(regions: list[dict], page_width: int) -> str:
    """Detect single vs multi-column layout via horizontal coverage gap."""
    if not regions or page_width == 0:
        return "single-column"

    coverage = np.zeros(page_width, dtype=np.int32)
    for r in regions:
        x1, _, x2, _ = r["bbox"]
        coverage[x1:x2] += 1

    third = page_width // 3
    gap_width = int(np.sum(coverage[third: 2 * third] == 0))

    if gap_width > page_width * COLUMN_GAP_RATIO:
        logger.info("Multi-column layout detected (gap=%d px).", gap_width)
        return "multi-column"
    return "single-column"


def _sort_reading_order(regions: list[dict], layout: str, page_width: int) -> list[dict]:
    """Sort regions into natural reading order."""
    if layout == "single-column" or not regions:
        return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    mid   = page_width // 2
    left  = sorted([r for r in regions if r["bbox"][0] < mid],  key=lambda r: r["bbox"][1])
    right = sorted([r for r in regions if r["bbox"][0] >= mid], key=lambda r: r["bbox"][1])
    return left + right


def _run_tesseract(region: np.ndarray, label: str = "Text") -> str:

    psm = PSM_BY_LABEL.get(label, DEFAULT_PSM)
    config = f"--oem 3 --psm {psm}"

    try:
        pil_img = Image.fromarray(region)
        text = pytesseract.image_to_string(pil_img, config=config)
        return text.strip()
    except Exception as exc:
        logger.warning("Tesseract failed on region (label=%s): %s", label, exc)
        return ""


def analyze_image(
    image: "Image.Image | np.ndarray",
    model: YOLO,
    reader: dict,
) -> AnalysisResult:

    image_array = np.array(image)
    page_h, page_w = image_array.shape[:2]

    preprocessed    = _preprocess_full_image(image_array)
    results         = model(preprocessed)
    annotated_bgr   = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes.data
    logger.info("YOLO detected %d raw boxes.", len(boxes))

    detected_regions: list[RegionResult] = []

    for box in boxes:
        if len(box) < 6:
            continue

        x1, y1, x2, y2, conf, cls_id = map(float, box.tolist())
        conf     = float(conf)

        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_name        = model.names.get(int(cls_id), "unknown")

        x1p, y1p, x2p, y2p = _pad_bbox(image_array, x1, y1, x2, y2)
        cropped             = image_array[y1p:y2p, x1p:x2p]

        if cropped.size == 0:
            continue

        # Skip OCR for pictures
        if cls_name in NON_TEXT_LABELS:
            detected_regions.append(RegionResult(
                bbox=[x1p, y1p, x2p, y2p],
                text="",
                confidence=conf,
                label=cls_name,
                skipped=True,
            ))
            continue

        # Preprocess and OCR with label-aware PSM
        deskewed     = _deskew_region(cropped)
        region_clean = _preprocess_region(deskewed, label=cls_name)
        ocr_text     = _run_tesseract(region_clean, label=cls_name)

        # Prepend semantic tag for structured region types
        prefix = SPECIAL_LABELS.get(cls_name, "")
        if prefix and ocr_text:
            ocr_text = f"{prefix} {ocr_text}"

        detected_regions.append(RegionResult(
            bbox=[x1p, y1p, x2p, y2p],
            text=ocr_text,
            confidence=conf,
            label=cls_name,
            skipped=False,
        ))

    layout           = _detect_layout(detected_regions, page_w)
    detected_regions = _sort_reading_order(detected_regions, layout, page_w)
    combined_text    = "\n\n".join(r["text"] for r in detected_regions if r["text"])

    logger.info(
        "OCR complete: %d regions kept (%s layout), %d words extracted.",
        len(detected_regions), layout, len(combined_text.split()),
    )

    return AnalysisResult(
        annotated_image=annotated_image,
        extracted_text=combined_text,
        regions=detected_regions,
        layout=layout,
    )