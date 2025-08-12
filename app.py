
import io
import re
import json
import time
from typing import List, Tuple, Dict

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional export
try:
    import docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

############################
# Parsing helpers
############################

def read_txt(file) -> str:
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        file.seek(0)
        return file.read().decode("latin-1", errors="ignore")

def read_pdf(file) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception:
        return ""

def read_docx(file) -> str:
    try:
        import docx
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def load_file(file) -> str:
    name = file.name.lower()
    if name.endswith(".txt"):
        return read_txt(file)
    if name.endswith(".pdf"):
        return read_pdf(file)
    if name.endswith(".docx"):
        return read_docx(file)
    return ""

############################
# Core scoring
############################

def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        max_df=0.95,
        min_df=1,
    )

def score_similarity(jd: str, resume: str) -> float:
    vect = build_vectorizer()
    try:
        X = vect.fit_transform([jd, resume])
        sim = cosine_similarity(X[0:1], X[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0

def ats_readability_score(text: str) -> int:
    # Heuristic ATS-friendliness score [0..100]
    score = 100
    if re.search(r"table|columns|graphic|image", text, re.I):
        score -= 10
    specials = len(re.findall(r"[^a-zA-Z0-9\s.,;:()&/\-+']", text))
    if specials > 50:
        score -= 10
    wc = len(text.split())
    if wc < 150:
        score -= 15
    if wc < 30:
        score -= 30
    return max(0, min(100, score))

############################
# Skills & Sections
############################

def load_default_skills() -> Dict[str, List[str]]:
    return {
        "Programming": ["python", "java", "c++", "sql", "r", "scala", "javascript"],
        "Data": ["power bi", "tableau", "excel", "pandas", "numpy", "spark", "bigquery", "snowflake", "looker"],
        "Cloud/DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "airflow", "git", "linux", "databricks"],
        "ML/Stats": ["machine learning", "deep learning", "regression", "classification", "nlp", "time series"],
        "PM/Process": ["agile", "scrum", "jira", "kanban", "stakeholder management", "roadmap"],
        "Customer Support": ["zendesk", "sla", "csat", "kpi", "incident management"],
        "Generic": ["communication", "leadership", "problem solving", "presentation", "documentation", "mentoring"]
    }

def find_skills(text: str, skills: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    text_l = " " + text.lower() + " "
    present = []
    all_skills = []
    for _, items in skills.items():
        for s in items:
            all_skills.append(s)
            if f" {s.lower()} " in text_l or s.lower() in text_l:
                present.append(s)
    missing = [s for s in all_skills if s not in present]
    return sorted(list(set(present))), sorted(list(set(missing)))

def heuristic_sections(text: str) -> Dict[str, str]:
    sections = {}
    lowered = text.lower()
    anchors = [
        "summary", "objective", "experience", "work experience",
        "projects", "education", "skills", "certifications", "achievements"
    ]
    indices = []
    for a in anchors:
        i = lowered.find(a)
        if i != -1:
            indices.append((i, a))
    indices.sort()
    for idx, (pos, name) in enumerate(indices):
        end = indices[idx + 1][0] if idx + 1 < len(indices) else len(text)
        sections[name.title()] = text[pos:end].strip()
    return sections

############################
# Rule-based Auto-Tailoring
############################

STOPWORDS = set("""a an and the to of in for with on by at from as is are be this that it its using use used via into across per within within-based""".split())

def top_jd_terms(jd: str, k: int = 20) -> List[str]:
    # basic tf-idf on the JD itself to pick salient ngrams
    vect = TfidfVectorizer(lowercase=True, ngram_range=(1,2), stop_words="english", max_features=2000)
    try:
        X = vect.fit_transform([jd])
        terms = vect.get_feature_names_out()
        # approximate importance by idf; with single doc, idf won't vary much, but we'll keep order
        # fallback to term frequencies
        return [t for t in terms if not t.isnumeric()][:k]
    except Exception:
        # simple keyword fallback
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9+\-_/]*", jd.lower())
        words = [w for w in words if w not in STOPWORDS]
        uniq = []
        for w in words:
            if w not in uniq:
                uniq.append(w)
        return uniq[:k]

BULLET_VERBS = ["Delivered", "Automated", "Owned", "Optimized", "Analyzed", "Built", "Improved", "Implemented", "Led", "Reduced"]

def make_jd_fit_highlights(missing_terms: List[str], present_terms: List[str], limit: int = 6) -> List[str]:
    picks = (missing_terms[: (limit//2)] + present_terms[: (limit//2)])[:limit]
    bullets = []
    for i, kw in enumerate(picks):
        verb = BULLET_VERBS[i % len(BULLET_VERBS)]
        bullets.append(f"{verb} {kw} initiatives aligned to role requirements; quantified impact where possible (e.g., +X% efficiency, -Y hrs/week).")
    return bullets

def build_tailored_resume(original_text: str, jd_text: str, skills_dict: Dict[str, List[str]], max_skills: int = 18) -> str:
    # Normalize
    resume_text = normalize_text(original_text)
    jd_text_n = normalize_text(jd_text)

    # Extract skills present/missing
    present, missing = find_skills(resume_text, skills_dict)

    # Select keywords from JD
    jd_terms = [t for t in top_jd_terms(jd_text_n, k=30) if len(t) > 2]

    # Choose skills to inject: prioritize missing that also appear in JD terms
    jd_term_set = set([t.lower() for t in jd_terms])
    missing_priority = [s for s in missing if s.lower() in jd_term_set]
    fill_more = [s for s in missing if s not in missing_priority]
    injected_skills = (missing_priority + fill_more)[: max(0, max_skills - len(present))]

    # Compose a tailored header + skills + highlights
    header = [
        "SUMMARY (Tailored for JD)",
        f"Results-driven professional aligning to requirements such as: {', '.join(jd_terms[:10])}.",
    ]

    skills_list = sorted(list(set(present + injected_skills)))[:max_skills]
    skills_block = ["", "CORE SKILLS (JD-Aligned)", ", ".join(skills_list)]

    highlights = make_jd_fit_highlights(injected_skills, present_terms=present, limit=6)
    highlights_block = ["", "JD-FIT HIGHLIGHTS", *[f"- {b}" for b in highlights]]

    # Merge with original content, keeping original first headings
    stitched = "\n".join(header + skills_block + highlights_block + ["", "—" * 20, "", resume_text])
    return stitched

def export_docx(text: str) -> bytes:
    if not DOCX_AVAILABLE:
        return b""
    from docx import Document
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

############################
# UI
############################

st.set_page_config(page_title="ATS Screening Tool v2 (with Auto‑Tailor)", page_icon="🧭", layout="wide")
st.title("🧭 ATS Screening Tool v2")
st.write("Upload resumes, paste a JD, get similarity & ATS scores — and optionally auto‑tailor resumes to the JD and see **before/after** results.")

with st.sidebar:
    st.header("Settings")
    default_skills = load_default_skills()
    custom_skills = st.text_area("Add custom skills (comma-separated)", placeholder="dbt, redshift, presto, looker, databricks")
    add_bucket = st.text_input("Optional: custom bucket name (e.g., 'Analytics Engg')", "")
    if st.button("Add Skills"):
        if custom_skills.strip():
            items = [s.strip() for s in custom_skills.split(",") if s.strip()]
            bucket = add_bucket.strip() or "Custom"
            default_skills.setdefault(bucket, [])
            default_skills[bucket].extend(items)
            st.success(f"Added {len(items)} skills to '{bucket}'.")

    st.markdown("---")
    st.subheader("Auto‑Tailor Options")
    enable_tailor = st.checkbox("Enable rule‑based tailoring when match < threshold", value=True)
    sim_threshold = st.slider("Similarity threshold to trigger tailoring", 0.0, 1.0, 0.35, 0.01)
    ats_threshold = st.slider("ATS threshold to trigger tailoring", 0, 100, 75, 1)
    max_skills = st.slider("Max skills in tailored resume", 8, 30, 18, 1)
    export_format = st.selectbox("Export tailored files as", ["DOCX", "TXT"])
    st.caption("Note: Tailoring is rule‑based (no AI). It injects JD keywords & missing skills to lift match scores ethically.")

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste the JD here", height=260, placeholder="Paste full JD...")
    st.caption("A detailed JD improves scoring accuracy.")

with col2:
    st.subheader("Resumes")
    files = st.file_uploader("Upload resumes (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

run = st.button("Run Screening", type="primary")

if run:
    if not jd_text or not files:
        st.error("Please paste a Job Description and upload at least one resume.")
    else:
        jd_text_n = normalize_text(jd_text)
        results = []
        tailored_packages = []  # (filename, bytes, mime)

        with st.spinner("Scoring resumes..."):
            for f in files:
                raw = load_file(f)
                text_n = normalize_text(raw)

                # BEFORE
                sim_before = score_similarity(jd_text_n, text_n)
                ats_before = ats_readability_score(text_n)
                present, missing = find_skills(text_n, default_skills)

                tailored_text = None
                sim_after = sim_before
                ats_after = ats_before

                should_tailor = enable_tailor and ((sim_before < sim_threshold) or (ats_before < ats_threshold))

                if should_tailor:
                    tailored_text = build_tailored_resume(text_n, jd_text_n, default_skills, max_skills=max_skills)
                    sim_after = score_similarity(jd_text_n, tailored_text)
                    ats_after = ats_readability_score(tailored_text)

                    # Prepare export
                    if export_format == "DOCX" and DOCX_AVAILABLE:
                        data = export_docx(tailored_text)
                        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        outname = f.name.rsplit(".", 1)[0] + "_TAILORED.docx"
                    else:
                        data = tailored_text.encode("utf-8")
                        mime = "text/plain"
                        outname = f.name.rsplit(".", 1)[0] + "_TAILORED.txt"
                    tailored_packages.append((outname, data, mime))

                results.append({
                    "file_name": f.name,
                    "similarity_before": sim_before,
                    "ats_before": ats_before,
                    "similarity_after": sim_after,
                    "ats_after": ats_after,
                    "present_skills": present,
                    "missing_skills": missing,
                    "tailored": tailored_text is not None,
                    "tailored_preview": (tailored_text[:1200] + "…") if (tailored_text and len(tailored_text) > 1200) else tailored_text
                })
                time.sleep(0.05)

        st.success("Done!")
        st.markdown("### Results (Before → After)")

        results.sort(key=lambda x: x["similarity_after"], reverse=True)

        for i, r in enumerate(results, start=1):
            title = f"{i}. {r['file_name']} — Match: {r['similarity_before']*100:.1f}% → {r['similarity_after']*100:.1f}% | ATS: {r['ats_before']}/100 → {r['ats_after']}/100"
            with st.expander(title):
                st.write("**Matched Skills:** ", ", ".join(r["present_skills"]) or "—")
                st.write("**Missing Skills (pre-tailor):** ", ", ".join(r["missing_skills"][:30]) or "—")
                if r["tailored"]:
                    st.markdown("**Tailored Preview (first ~1,200 chars)**")
                    st.code(r["tailored_preview"] or "—")
                else:
                    st.caption("No tailoring applied (scores above thresholds).")

        # Bulk download tailored files
        if any(r["tailored"] for r in results) and len(tailored_packages) > 0:
            import io, zipfile
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
                for (name, data, _) in tailored_packages:
                    z.writestr(name, data)
            bio.seek(0)
            st.download_button("Download all tailored resumes (ZIP)", data=bio.read(), file_name="tailored_resumes.zip", mime="application/zip")

        # Summary table
        import pandas as pd
        df = pd.DataFrame([{
            "file_name": r["file_name"],
            "match_before_%": round(r["similarity_before"]*100, 1),
            "match_after_%": round(r["similarity_after"]*100, 1),
            "ats_before": r["ats_before"],
            "ats_after": r["ats_after"],
            "tailored_applied": r["tailored"]
        } for r in results])
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download summary CSV", data=csv_bytes, file_name="ats_screening_results_v2.csv", mime="text/csv")

st.markdown("---")
st.caption("Auto‑Tailor is deterministic and rule‑based (no LLM). It injects missing skills & JD terms to improve alignment, then re‑scores. For a smarter rewrite, plug in embeddings or LLMs.")
