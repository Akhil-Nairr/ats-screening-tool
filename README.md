# 🧭 ATS Screening Tool v2 — with Rule‑Based Auto‑Tailor

This version adds **Auto‑Tailor**:
- When a resume's **match** or **ATS** score is below your threshold,
- The tool creates a **tailored** version by injecting JD keywords and missing skills,
- Re-scores to show **before vs after**,
- And lets you **download** the tailored resume (DOCX/TXT).

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## How tailoring works (non-AI)
- Extracts salient JD terms via TF‑IDF.
- Finds skills present vs missing using a configurable dictionary.
- Builds a short **Tailored Summary**, **Core Skills**, and **JD‑Fit Highlights** section with action verbs.
- Appends your original resume content (nothing is lost).

> For semantic rewriting, plug in sentence-transformers or an LLM and replace `build_tailored_resume`.
