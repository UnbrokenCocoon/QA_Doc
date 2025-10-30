Certainly! Below is a **comprehensive, British English** markdown summary of your application’s **import sequence, initialisation logic, and core pipeline architecture**, mirroring the style and structure of your previous document.

---

# Application Architecture and Initialisation Overview

The provided script defines a **production-ready RAG (Retrieval-Augmented Generation) evaluation pipeline** for policy and curriculum documents. It is designed for deployment in containerised environments (e.g., Hugging Face Spaces) and features **modular imports**, **resource-aware model loading**, **robust error handling**, and a **Gradio-based user interface**.

The initialisation phase carefully sequences imports to **minimise startup latency**, **avoid runtime downloads**, and **ensure compatibility** with restricted environments.

---

## Summary of Initialisation and Core Components

### 1. Import Strategy and Dependency Management

| Phase | Action | Rationale |
| :--- | :--- | :--- |
| **Timestamped Logging** | Logs each import phase with ISO timestamp (`YYYY-MM-DD HH:MM:SS`) and emoji indicators. | Enables **debugging and performance monitoring** in CI/CD or cloud logs. |
| **Standard Library First** | Imports `os`, `json`, `re`, `sys`, `shutil`, `tempfile`, `subprocess`, `random`. | Ensures foundational utilities are available before third-party dependencies. |
| **Numerical Stack** | Loads `numpy` (as `np`) and `pandas` (as `pd`). | Supports data manipulation, chunking, and result aggregation. |
| **Path Handling** | Uses `pathlib.Path` for cross-platform file operations. | Improves readability and avoids string-based path errors. |
| **Document Processing** | Imports `pypdf` (for PDF text extraction) and `openpyxl` (for Excel formatting). | Enables end-to-end conversion from DOCX → PDF → evaluation → formatted Excel output. |
| **Vector Search** | Imports `faiss` for efficient similarity search over embeddings. | Provides **low-latency retrieval** from large document corpora. |
| **NLP (NLTK)** | Sets `NLTK_DATA` to `/app/nltk_data` and imports `nltk` **without runtime downloads**. | Assumes **pre-cached tokenisers** (e.g., via Docker), avoiding network calls at startup. |
| **AI Embeddings (Delayed)** | Defers `SentenceTransformer` import until after logging, then loads `all-MiniLM-L6-v2` from `/app/hf_cache`. | Prevents **startup hangs**; ensures model is loaded from **pre-downloaded cache** (critical for HF Spaces). |
| **Google GenAI** | Attempts to import `google.genai`; gracefully degrades if unavailable. | Allows **optional LLM integration**; fails safely in environments without API access. |
| **UI Layer (Last)** | Imports `gradio` only after all backend dependencies are ready. | Reduces memory pressure and ensures backend is fully initialised before UI launch. |

---

### 2. Resource Management and Environment Safeguards

| Feature | Description |
| :--- | :--- |
| **Hugging Face Cache Cleanup** | Automatically checks Hugging Face cache size (`~/.cache/huggingface` or `HF_HOME`). If >10 GB, **deletes the entire cache** to prevent disk exhaustion in ephemeral environments (e.g., HF Spaces). Includes safeguards against scanning excessively large directories. |
| **Pre-Cached Models** | Both **NLTK data** and the **SentenceTransformer model** are expected to be pre-installed in `/app/nltk_data` and `/app/hf_cache`, respectively—enabling **zero-download startup** in containers. |
| **LibreOffice-Based DOCX→PDF** | Uses `libreoffice --headless` for reliable DOCX conversion, with **60-second timeout** and **temporary user profile** (`/tmp`) to avoid permission issues in sandboxed runtimes. |

---

### 3. Core Evaluation Pipeline

| Component | Function | Purpose |
| :--- | :--- | :--- |
| **PDF Chunking** | `chunk_pdf_by_words()` | Splits PDFs into **semantic fragments** using punctuation and discourse connectives (e.g., *however*, *therefore*). Aggregates fragments into **minimum-word chunks** (default: 250 words) with **cross-page spillover handling**. |
| **RAG Artifact Loading** | `load_rag_artifacts()` | Loads a **pre-built FAISS index**, corresponding **text chunks** (from Parquet/CSV), and optional **embeddings** (NumPy) for a given jurisdiction (e.g., `Wales`). |
| **LLM-Powered Evaluation** | `evaluate_chunk()` | Uses **Google Gemini** to score each report chunk against retrieved source excerpts. Enforces **strict JSON output**, includes **retry logic** (up to 10 attempts with fallback API key), and generates **debug reports** on failure. |
| **Batch Evaluation** | `run_rag_evaluation()` | Orchestrates evaluation across all chunks, captures errors per chunk, and returns a structured DataFrame. |
| **Excel Export** | `save_evaluation_to_excel()` | Produces a **professionally formatted Excel file** with:<br>• **Conditional colour scale** (red → green) on accuracy scores,<br>• **Wrapped text** in comment/reference columns,<br>• **Auto-adjusted column widths** (references column twice as wide),<br>• **Average score row** appended at the bottom. |

---

### 4. Deployment Modes

| Mode | Description |
| :--- | :--- |
| **Self-Evaluation** | `evaluate_report_against_rag_index()` evaluates a report **against itself** (useful for QA or coherence checks). Builds a **temporary FAISS index** from the input PDF. |
| **Cross-Reference Evaluation** | `evaluate_custom_report()` evaluates a user-uploaded DOCX against a **pre-built jurisdictional knowledge base** (e.g., *England*, *Finland*). Requires a matching FAISS index in the `Data/` directory. |
| **Gradio Web Interface** | Provides a **user-friendly UI** for non-technical users to upload reports and select reference indexes. Integrates with **Hugging Face Secrets** for secure API key management. |

---

### 5. Design Principles

- **Idempotent & Safe**: No side effects on input files; all outputs written to temporary directories.
- **Environment-Aware**: Detects missing dependencies (e.g., LibreOffice, GenAI) and fails gracefully.
- **UK-Conforming**: Uses British English spelling and policy-relevant evaluation criteria (e.g., accuracy of dates, paraphrasing tolerance).
- **Audit-Ready**: Every evaluation includes **evidence excerpts with page numbers**, enabling traceability.
- **Scalable**: Designed for **ephemeral cloud runtimes** (e.g., HF Spaces) with cache cleanup and timeout protections.

---

This architecture transforms **unstructured policy documents** into **quantitatively assessed, evidence-backed evaluations**, supporting quality assurance in academic, governmental, or NGO settings—particularly where **accountability, accuracy, and reproducibility** are paramount.
