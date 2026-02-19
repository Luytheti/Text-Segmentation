"""
report_exporter.py
------------------
Generates a structured Markdown report from document analysis results.

The report includes:
  - Document metadata (filename, pages, stats)
  - Per-page section breakdown grouped by region type
    (Titles, Headers, Body Text, Tables, Formulas, Lists, Captions)
  - A Q&A log of the full chat session
  - Formatted for clean rendering on GitHub, Notion, or any Markdown viewer

This is a standalone utility — no Streamlit dependency — so it can also be
used in batch / headless pipelines.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Maps YOLO label → Markdown section heading used in the report
LABEL_TO_SECTION: dict[str, str] = {
    "Title":          "## Title",
    "Section-header": "### Section",
    "Page-header":    "<!-- header -->",   # Suppressed from body, shown in metadata
    "Page-footer":    "<!-- footer -->",
    "Text":           "",                  # Body text — no heading prefix needed
    "List-item":      "",                  # Rendered as bullet list
    "Caption":        "*Caption:*",
    "Footnote":       "*Footnote:*",
    "Table":          "#### Table",
    "Formula":        "#### Formula",
    "Picture":        "",                  # No text to render
}

# Labels that render as bullet-list items
LIST_LABELS = {"List-item"}

# Labels to skip in the body (metadata-only)
SKIP_IN_BODY = {"Page-header", "Page-footer", "Picture"}


def _clean_text(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal blank lines."""
    lines = [l.rstrip() for l in text.strip().splitlines()]
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned)


def _regions_to_markdown(regions: list[dict], layout: str) -> str:
    """
    Convert a list of RegionResult dicts into a Markdown body string.

    Groups consecutive list-items into a single bullet list block.
    Applies appropriate Markdown headings per region type.
    """
    lines: list[str] = []
    list_buffer: list[str] = []

    def _flush_list():
        if list_buffer:
            lines.extend(f"- {item}" for item in list_buffer)
            list_buffer.clear()
            lines.append("")

    for region in regions:
        label   = region.get("label", "Text")
        text    = region.get("text", "").strip()
        skipped = region.get("skipped", False)

        if skipped or label in SKIP_IN_BODY or not text:
            _flush_list()
            continue

        # Strip semantic prefix tags added by ocr_engine
        for tag in ("[TABLE]", "[FORMULA]", "[CAPTION]", "[FOOTNOTE]",
                    "[FOOTER]", "[HEADER]", "[SECTION]", "[TITLE]", "[LIST]"):
            text = text.replace(tag, "").strip()

        if not text:
            continue

        if label in LIST_LABELS:
            list_buffer.append(text)
        else:
            _flush_list()
            md_prefix = LABEL_TO_SECTION.get(label, "")
            if md_prefix and not md_prefix.startswith("<!--"):
                lines.append(f"{md_prefix}")
            lines.append(text)
            lines.append("")

    _flush_list()
    layout_note = f"> *Layout detected: **{layout}***\n" if layout else ""
    return layout_note + "\n".join(lines)


def generate_markdown_report(
    filename: str,
    page_results: list[dict],
    chat_messages: list[dict] | None = None,
) -> str:
    """
    Generate a full structured Markdown report from document analysis results.

    Args:
        filename:      Original uploaded file name.
        page_results:  List of per-page result dicts from analyze_image.
                       Each dict must have keys: page, regions, extracted_text,
                       and optionally layout.
        chat_messages: Optional list of {"role": "user"|"assistant", "content": str}
                       dicts from the chat session Q&A log.

    Returns:
        A formatted Markdown string ready to be saved as .md or downloaded.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Aggregate stats ---
    total_pages   = len(page_results)
    all_regions   = [r for pr in page_results for r in pr.get("regions", [])]
    total_regions = len(all_regions)
    total_words   = sum(
        len(pr.get("extracted_text", "").split()) for pr in page_results
    )
    avg_conf = (
        sum(r.get("confidence", 0) for r in all_regions) / total_regions
        if total_regions else 0.0
    )

    # --- Label distribution ---
    label_counts: dict[str, int] = {}
    for r in all_regions:
        lbl = r.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    label_table_rows = "\n".join(
        f"| {lbl} | {cnt} |" for lbl, cnt in sorted(label_counts.items())
    )

    # --- Build document ---
    sections: list[str] = []

    # Header
    sections.append(f"# Document Analysis Report\n")
    sections.append(
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **File** | `{filename}` |\n"
        f"| **Generated** | {now} |\n"
        f"| **Pages** | {total_pages} |\n"
        f"| **Regions detected** | {total_regions} |\n"
        f"| **Words extracted** | {total_words} |\n"
        f"| **Avg. detection confidence** | {avg_conf:.1%} |\n"
    )

    # Region type breakdown
    if label_counts:
        sections.append(
            f"\n## Region Type Breakdown\n\n"
            f"| Label | Count |\n"
            f"|---|---|\n"
            f"{label_table_rows}\n"
        )

    sections.append("\n---\n")

    # Per-page content
    sections.append("# Extracted Content\n")
    for pr in page_results:
        page_num = pr.get("page", "?")
        layout   = pr.get("layout", "single-column")
        regions  = pr.get("regions", [])
        n_words  = len(pr.get("extracted_text", "").split())

        sections.append(f"## Page {page_num}\n")
        sections.append(f"*{len(regions)} regions · {n_words} words · {layout} layout*\n")

        body = _regions_to_markdown(regions, layout)
        if body.strip():
            sections.append(body)
        else:
            sections.append("*No text extracted from this page.*\n")

        sections.append("\n---\n")

    # Q&A log
    if chat_messages:
        sections.append("# Chat Q&A Log\n")
        for msg in chat_messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "user":
                sections.append(f"**Q:** {content}\n")
            else:
                sections.append(f"**A:** {content}\n")
        sections.append("\n---\n")

    sections.append(
        f"*Report generated by Document Segmentation & Text Retrieval · "
        f"Rajas Daryapurkar · {now}*\n"
    )

    report = "\n".join(sections)
    logger.info(
        "Markdown report generated: %d pages, %d words, %d Q&A turns.",
        total_pages, total_words, len(chat_messages) if chat_messages else 0,
    )
    return report