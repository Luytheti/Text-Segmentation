# 🔍 Document Segmentation & Text Retrieval

> **AI-powered document analysis pipeline** — upload any image or PDF, detect structured regions with YOLOv10, extract text with Tesseract OCR, and chat with your document using a RAG pipeline backed by Google Gemini.


---

##  What It Does

This project combines **document layout analysis**, **OCR**, and **Retrieval-Augmented Generation (RAG)** into a single end-to-end pipeline:

1. **Upload** an image or multi-page PDF
2. **YOLO detects** 11 document region types — Title, Text, Table, Figure, Formula, List, Header, Footer, Caption, Footnote, Section-header
3. **Tesseract extracts** text from each region with PSM modes tuned per region type
4. **Layout analysis** detects single vs. multi-column pages and re-orders regions into correct reading order
5. **RAG pipeline** indexes the extracted text into a FAISS vector store with MMR retrieval
6. **Chat** with your document — answers are grounded in the actual content with source attribution and streaming responses

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │     Document Ingestion     │
          │  (Image / PDF via PyMuPDF) │
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   YOLOv10 Layout Analysis  │
          │  11 region classes detected│
          │  Adaptive preprocessing   │
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │    Region-Type-Aware OCR   │
          │  Tesseract + PSM per label │
          │  Deskew · Threshold · Scale│
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │    Multi-Column Layout     │
          │  Gap detection · Re-order  │
          │  Semantic prefix tagging   │
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │        RAG Pipeline        │
          │  MiniLM Embeddings · FAISS │
          │  Query Rewriting · MMR     │
          │  Gemini 1.5 Flash · Stream │
          │  Source Attribution        │
          └───────────────────────────┘
```

---

##  Key Features

### Document Processing
| Feature | Detail |
|---|---|
| **Region Detection** | YOLOv10x fine-tuned on 11 document layout classes |
| **OCR Engine** | Tesseract with per-region PSM tuning (PSM 6/7/11) |
| **Layout Detection** | Horizontal gap analysis for multi-column detection |
| **Reading Order** | Column-aware sorting (left→right, top→bottom) |
| **Preprocessing** | Adaptive thresholding, deskewing, upscaling |
| **Region Tagging** | Semantic prefixes: `[TABLE]`, `[TITLE]`, `[FORMULA]` etc. |
| **Picture Handling** | Picture regions skipped — no wasted OCR cycles |

### RAG Chat Pipeline
| Feature | Detail |
|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` — local, no API cost |
| **Vector Store** | FAISS with AVX2 acceleration |
| **Retrieval** | MMR (Maximal Marginal Relevance) — diverse, non-redundant chunks |
| **Query Rewriting** | LLM rewrites user questions into better retrieval queries |
| **LLM** | Google Gemini 1.5 Flash |
| **Streaming** | Token-by-token response rendering |
| **Source Attribution** | Shows which document chunks grounded each answer |
| **Heatmap Overlay** | Cited regions highlighted on the original page image |

### Export
- **JSON export** — per-page or full document with all region metadata
- **Markdown report** — structured report with section breakdown + Q&A log

---

##  Project Structure

```
├── app.py                  # Streamlit UI — main entry point
├── ocr_engine.py           # YOLO inference + Tesseract OCR pipeline
├── rag_agent.py            # RAG pipeline — embeddings, retrieval, LLM
├── model_loader.py         # YOLOv10 model download + loading
├── report_exporter.py      # Markdown report generation
└── models/
    └── yolov10x_best.pt    # Fine-tuned YOLOv10 weights (auto-downloaded)
```

---

## ⚙️ Setup

### 1. Install Tesseract (Windows)
Download and install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).  
Ensure **"Add to PATH"** is checked during installation.

### 2. Install Python dependencies

```bash
pip install streamlit ultralytics pytesseract opencv-python pillow pymupdf
pip install langchain langchain-community langchain-google-genai
pip install sentence-transformers faiss-cpu gdown
```

### 3. Run

```bash
streamlit run app.py
```

The YOLOv10 model (~140MB) will auto-download on first run.

---

##  Usage

1. Open the app in your browser (`http://localhost:8501`)
2. Enter your **Google Gemini API key** in the sidebar
3. Upload an **image** (JPG/PNG) or **PDF**
4. The pipeline runs automatically — view detected regions, confidence scores, and extracted text per page
5. Once indexed, ask questions in the **Chat with Document** section
6. Export results as **JSON** or a formatted **Markdown report**

---

##  YOLO Model Classes

| Class | Description |
|---|---|
| `Title` | Document or section title |
| `Text` | Body paragraph text |
| `Section-header` | Sub-section heading |
| `List-item` | Bullet or numbered list entry |
| `Table` | Tabular data region |
| `Figure` / `Picture` | Images, diagrams, charts |
| `Formula` | Mathematical equations |
| `Caption` | Figure or table caption |
| `Footnote` | Footer annotation |
| `Page-header` | Repeating page header |
| `Page-footer` | Repeating page footer |

---

##  Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Layout Detection | YOLOv10x (custom fine-tuned) |
| OCR | Tesseract 5 via pytesseract |
| Image Processing | OpenCV, PIL |
| PDF Parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (MiniLM-L6-v2) |
| Vector Store | FAISS |
| LLM | Google Gemini 1.5 Flash |
| RAG Framework | LangChain |

---

## 👤 Author

**Rajas Daryapurkar**

---

*Document Segmentation & Text Retrieval — End-to-end document AI pipeline*