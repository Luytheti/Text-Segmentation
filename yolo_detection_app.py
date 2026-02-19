import json
import os
import tempfile

import cv2
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from PIL import Image

from model_loader import load_model
from ocr_engine import analyze_image, load_ocr_reader
from rag_agent import RAGAgent
from report_exporter import generate_markdown_report

def _safe_remove(path: str, retries: int = 5, delay: float = 0.3) -> None:

    import time
    for attempt in range(retries):
        try:
            os.remove(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
    # Last attempt — if still locked, log and move on rather than crashing
    import logging
    logging.getLogger(__name__).warning("Could not delete temp file: %s", path)


st.set_page_config(
    page_title="DocSeg · Text Retrieval",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <div style="text-align:center; padding-bottom:8px;">
        <h2 style="color:#dc7ef3; margin-bottom:0;">Rajas Daryapurkar</h2>
        <h1 style="color:#8cf3ea; margin-top:4px;">Document Segmentation & Text Retrieval</h1>
        <p style="color:#aaa;">
            Upload an image or PDF · Detect regions · Extract text · Chat with your document
        </p>
    </div>
    <hr style="border-color:#333;">
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLO model…")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Loading OCR engine…")
def get_ocr_reader():
    return load_ocr_reader()


model = get_model()
if model is None:
    st.error("❌ Failed to load the YOLO model. Check your connection and restart.")
    st.stop()


with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Required to enable the document chat feature.",
    )
    st.divider()
    confidence_threshold = st.slider(
        "YOLO Confidence Threshold",
        min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        help="Detections below this score are discarded.",
    )
    stream_mode = st.toggle(
        "Stream responses",
        value=True,
        help="Show LLM answer token-by-token as it's generated.",
    )
    show_sources = st.toggle(
        "Show source attribution",
        value=True,
        help="Display document chunks used to generate each answer.",
    )
    st.divider()
    st.markdown(
        "**Pipeline**\n\n"
        "1. YOLOv10 detects document regions\n"
        "2. Pytesseract extracts text per region\n"
        "3. Layout analysis (single/multi-column)\n"
        "4. FAISS + MMR retrieval\n"
        "5. Query rewriting → Gemini answer\n"
        "6. Source attribution surfaced to UI"
    )


uploaded_file = st.file_uploader(
    "📂 Upload Image or PDF",
    type=["jpg", "jpeg", "png", "pdf"],
    help="Supported formats: JPG, PNG, PDF",
)


def _init_state():
    defaults = {
        "all_text": "",
        "all_regions": [],
        "page_results": [],
        "rag_agent": None,
        "messages": [],
        "last_file_name": None,
        "processing_done": False,
        "last_sources": [],          # Source chunks from the most recent answer
        "last_cited_bboxes": [],     # Bboxes highlighted in the heatmap
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

if uploaded_file and uploaded_file.name != st.session_state["last_file_name"]:
    st.session_state.update({
        "all_text": "",
        "all_regions": [],
        "page_results": [],
        "messages": [],
        "processing_done": False,
        "last_sources": [],
        "last_cited_bboxes": [],
        "last_file_name": uploaded_file.name,
    })
    if st.session_state["rag_agent"] is not None:
        st.session_state["rag_agent"].reset()
        st.session_state["rag_agent"] = None


def _build_heatmap(image_array: np.ndarray, cited_bboxes: list[list[int]]) -> np.ndarray:
    """
    Draw a semi-transparent yellow highlight over regions that were cited
    in the most recent RAG answer.

    Args:
        image_array:  Original page image as RGB numpy array.
        cited_bboxes: List of [x1, y1, x2, y2] bounding boxes to highlight.

    Returns:
        Annotated RGB image with heatmap overlay.
    """
    overlay = image_array.copy()
    highlight_color = (255, 230, 50)   # Yellow

    for bbox in cited_bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), highlight_color, thickness=-1)

    # Blend overlay with original
    result = cv2.addWeighted(overlay, 0.35, image_array, 0.65, 0)

    # Draw border on highlighted regions for clarity
    for bbox in cited_bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 180, 0), thickness=2)

    return result


def _find_cited_regions(sources: list[dict], all_regions: list[dict]) -> list[list[int]]:

    if not sources or not all_regions:
        return []

    # Combine all source snippet text for fuzzy matching
    source_text = " ".join(s.get("snippet", "") for s in sources).lower()

    cited_bboxes = []
    for region in all_regions:
        region_text = region.get("text", "").strip().lower()
        if not region_text or len(region_text) < 10:
            continue
        # Check if a meaningful chunk of this region's text appears in the sources
        words = region_text.split()
        sample = " ".join(words[:6])    # Match on first 6 words
        if sample and sample in source_text:
            cited_bboxes.append(region["bbox"])

    return cited_bboxes


def process_page(image: Image.Image, page_num: int) -> dict:
    reader = get_ocr_reader()
    result = analyze_image(image, model, reader)
    return {
        "page":            page_num,
        "original_image":  np.array(image),
        "annotated_image": result["annotated_image"],
        "extracted_text":  result["extracted_text"],
        "regions":         result["regions"],
        "layout":          result.get("layout", "single-column"),
    }


def display_page_result(page_result: dict):
    """Render one page result inside an expander."""
    page_num  = page_result["page"]
    regions   = page_result["regions"]
    text      = page_result["extracted_text"]
    layout    = page_result.get("layout", "single-column")
    word_count = len(text.split())

    cited_bboxes = st.session_state.get("last_cited_bboxes", [])

    layout_icon = "⬜" if layout == "single-column" else "⬛"
    with st.expander(
        f"📄 Page {page_num}  ·  {len(regions)} regions  ·  "
        f"{word_count} words  ·  {layout_icon} {layout}",
        expanded=(page_num == 1),
    ):
        col1, col2 = st.columns([1, 1])

        with col1:
            # Show heatmap if there are cited regions on this page
            if cited_bboxes:
                original = page_result.get("original_image")
                if original is not None:
                    heatmap = _build_heatmap(original, cited_bboxes)
                    st.image(heatmap, caption="🟡 Cited regions highlighted", use_column_width=True)
                else:
                    st.image(page_result["annotated_image"],
                             caption="Detected Regions", use_column_width=True)
            else:
                st.image(page_result["annotated_image"],
                         caption="Detected Regions", use_column_width=True)

            # Per-region confidence breakdown
            if regions:
                st.markdown("**Region Confidence Scores**")
                for i, r in enumerate(regions, 1):
                    label = r.get("label", "text")
                    conf  = r.get("confidence", 0.0)
                    skipped = r.get("skipped", False)
                    bar_color = (
                        "#aaa" if skipped else
                        "#8cf3ea" if conf >= 0.7 else
                        "#f3c08c" if conf >= 0.5 else
                        "#f38c8c"
                    )
                    suffix = " *(skipped)*" if skipped else ""
                    st.markdown(
                        f"<small>Region {i} · <b>{label}</b>{suffix} · "
                        f"<span style='color:{bar_color}'>{conf:.0%}</span></small>",
                        unsafe_allow_html=True,
                    )
                    st.progress(conf)

        with col2:
            st.text_area(
                "Extracted Text",
                text if text else "(No text detected)",
                height=380,
                key=f"text_area_page_{page_num}",
            )

        st.download_button(
            f"⬇️ Download Page {page_num} JSON",
            data=json.dumps(regions, indent=2).encode(),
            file_name=f"page_{page_num}_regions.json",
            mime="application/json",
            key=f"dl_json_{page_num}",
        )


if uploaded_file and not st.session_state["processing_done"]:
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if file_ext in {"jpg", "jpeg", "png"}:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analysing image…"):
            result = process_page(image, page_num=1)

        st.session_state["page_results"] = [result]
        st.session_state["all_text"]     = result["extracted_text"]
        st.session_state["all_regions"]  = result["regions"]
        st.session_state["processing_done"] = True

    elif file_ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        doc = None
        try:
            doc          = fitz.open(tmp_path)
            total_pages  = doc.page_count
            progress_bar = st.progress(0, text="Starting…")
            page_results = []
            all_text_parts = []

            for i in range(total_pages):
                progress_bar.progress(
                    i / total_pages,
                    text=f"Processing page {i + 1} of {total_pages}…",
                )
                page = doc.load_page(i)
                pix  = page.get_pixmap(dpi=150)
                img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                result = process_page(img, page_num=i + 1)
                page_results.append(result)
                all_text_parts.append(result["extracted_text"])

            progress_bar.progress(1.0, text="✅ All pages processed.")

            st.session_state["page_results"]    = page_results
            st.session_state["all_text"]        = "\n\n".join(all_text_parts)
            st.session_state["all_regions"]     = [
                r for pr in page_results for r in pr["regions"]
            ]
            st.session_state["processing_done"] = True

        finally:
            # Close fitz BEFORE deleting — on Windows, PyMuPDF holds a file
            # handle until explicitly released, causing WinError 32 otherwise.
            if doc is not None:
                doc.close()
                doc = None
            _safe_remove(tmp_path)


if st.session_state["processing_done"] and st.session_state["page_results"]:
    page_results = st.session_state["page_results"]
    all_regions  = st.session_state["all_regions"]
    all_text     = st.session_state["all_text"]

    total_pages   = len(page_results)
    total_regions = len(all_regions)
    total_words   = len(all_text.split())
    avg_conf = (
        sum(r.get("confidence", 0) for r in all_regions) / total_regions
        if total_regions else 0
    )

    # --- Summary metrics ---
    st.markdown("### 📊 Document Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages", total_pages)
    c2.metric("Regions Detected", total_regions)
    c3.metric("Words Extracted", total_words)
    c4.metric("Avg. Confidence", f"{avg_conf:.0%}")

    # --- Region type breakdown chart ---
    label_counts: dict[str, int] = {}
    for r in all_regions:
        lbl = r.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    if label_counts:
        with st.expander("📈 Region Type Breakdown", expanded=False):
            import pandas as pd
            df = pd.DataFrame(
                {"Label": list(label_counts.keys()),
                 "Count": list(label_counts.values())}
            ).sort_values("Count", ascending=False)
            st.bar_chart(df.set_index("Label"))

    st.divider()

    # --- Per-page results ---
    st.markdown("### 📄 Page Results")
    for pr in page_results:
        display_page_result(pr)

    st.divider()

    # --- Export buttons ---
    col_json, col_md = st.columns(2)

    with col_json:
        full_report = {
            "summary": {
                "total_pages": total_pages,
                "total_regions": total_regions,
                "total_words": total_words,
                "avg_confidence": round(avg_conf, 4),
            },
            "pages": [
                {
                    "page":           pr["page"],
                    "layout":         pr.get("layout", "single-column"),
                    "extracted_text": pr["extracted_text"],
                    "regions":        pr["regions"],
                }
                for pr in page_results
            ],
        }
        st.download_button(
            "📦 Export Full Report (JSON)",
            data=json.dumps(full_report, indent=2).encode(),
            file_name="full_report.json",
            mime="application/json",
        )

    with col_md:
        md_report = generate_markdown_report(
            filename=st.session_state.get("last_file_name", "document"),
            page_results=page_results,
            chat_messages=st.session_state.get("messages", []),
        )
        st.download_button(
            "📝 Export Markdown Report",
            data=md_report.encode(),
            file_name="report.md",
            mime="text/markdown",
        )


if st.session_state["processing_done"] and api_key and st.session_state["all_text"]:

    if st.session_state["rag_agent"] is None:
        agent = RAGAgent(api_key=api_key)
        with st.spinner("🔍 Indexing document for chat…"):
            success = agent.process_document(st.session_state["all_text"])
        if success:
            st.session_state["rag_agent"] = agent
            st.success("✅ Document indexed! You can now ask questions below.")
        else:
            st.error("❌ Failed to index document for chat.")

    if st.session_state["rag_agent"] is not None:
        st.divider()
        st.markdown("### 💬 Chat with Document")

        # Render chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask a question about the document…")

        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                agent = st.session_state["rag_agent"]

                if stream_mode:
                    # --- Streaming path ---
                    # st.write_stream was added in Streamlit 1.31.
                    # This manual loop works on all versions.
                    placeholder = st.empty()
                    response_text = ""
                    for token in agent.get_response_stream(prompt):
                        response_text += token
                        placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                    sources = []   # Sources not available in streaming mode
                else:
                    # --- Standard path with source attribution ---
                    with st.spinner("Thinking…"):
                        response_text, sources = agent.get_response(prompt)
                    st.markdown(response_text)

                # Source attribution panel
                if show_sources and sources:
                    st.session_state["last_sources"] = sources

                    # Find which regions were cited and update heatmap state
                    cited = _find_cited_regions(
                        sources, st.session_state["all_regions"]
                    )
                    st.session_state["last_cited_bboxes"] = cited

                    with st.expander(
                        f"📎 Source Attribution ({len(sources)} chunks)", expanded=False
                    ):
                        for i, src in enumerate(sources, 1):
                            snippet = src.get("snippet", "")
                            score   = src.get("relevance_score")
                            score_str = f" · relevance: `{score:.3f}`" if score else ""
                            st.markdown(
                                f"**Chunk {i}**{score_str}\n\n"
                                f"> {snippet}"
                            )

                    if cited:
                        st.info(
                            f"🟡 {len(cited)} region(s) highlighted in the page view above.",
                            icon="ℹ️",
                        )

            st.session_state["messages"].append(
                {"role": "assistant", "content": response_text}
            )

            # Re-render Markdown export button with updated Q&A log
            if st.session_state["page_results"]:
                md_report = generate_markdown_report(
                    filename=st.session_state.get("last_file_name", "document"),
                    page_results=st.session_state["page_results"],
                    chat_messages=st.session_state["messages"],
                )
                st.download_button(
                    "📝 Export Markdown Report (with Q&A)",
                    data=md_report.encode(),
                    file_name="report_with_qa.md",
                    mime="text/markdown",
                    key=f"md_export_chat_{len(st.session_state['messages'])}",
                )

elif st.session_state["processing_done"] and not api_key:
    st.info("💡 Enter your Google Gemini API Key in the sidebar to enable document chat.")