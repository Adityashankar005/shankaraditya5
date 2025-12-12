# app.py
import streamlit as st
import io
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from PyPDF2 import PdfReader

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="PDF Keyword Paragraph Extractor + WordCloud", layout="wide")
st.title("📄 PDF Keyword Paragraph Extractor & Word Cloud")

# Ensure NLTK resources (cached)
@st.cache_data(show_spinner=False)
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")
    return True

ensure_nltk()

# -----------------------
# Helpers
# -----------------------
def extract_paragraphs_from_pdf_bytes_py(pdf_bytes):
    """
    Extract text from PDF bytes using PyPDF2 and split into paragraphs.
    """
    paragraphs = []
    if not pdf_bytes:
        return paragraphs

    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        full_text.append(text)

    full_text_str = "\n".join(full_text)
    raw_pars = re.split(r'\n{2,}', full_text_str)
    for p in raw_pars:
        p = p.strip()
        if p:
            paragraphs.append(p)
    return paragraphs

def normalize_text(t):
    return re.sub(r'\s+', ' ', t).strip()

def filter_paragraphs(paragraphs, keywords, min_len=50):
    kws = [k.strip().lower() for k in keywords if k.strip()]
    matched = []
    for p in paragraphs:
        if len(p) < min_len:
            continue
        low = p.lower()
        if any(k in low for k in kws):
            matched.append(p)
    return matched

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload PDF (optional)", type=["pdf"])
use_repo_pdf = st.sidebar.checkbox("Use repo PDF if present (Annual-Report-2024-25.pdf)", value=True)
keywords_input = st.sidebar.text_input("Keywords (comma-separated)", value="semiconductor, aerospace, xev, orderbook")
min_len = st.sidebar.number_input("Minimum paragraph length (chars)", min_value=10, max_value=2000, value=50)
show_raw = st.sidebar.checkbox("Show matched paragraphs", value=True)
max_paragraphs_display = st.sidebar.number_input("Max paragraphs to display", min_value=10, max_value=1000, value=200)

# -----------------------
# Load PDF bytes: uploaded -> repo file (optional)
# -----------------------
pdf_bytes = None
if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    st.sidebar.success(f"Loaded uploaded file: {uploaded_file.name}")
else:
    if use_repo_pdf and os.path.exists("Annual-Report-2024-25.pdf"):
        try:
            with open("Annual-Report-2024-25.pdf", "rb") as f:
                pdf_bytes = f.read()
            st.sidebar.success("Loaded repo PDF: Annual-Report-2024-25.pdf")
        except Exception as e:
            st.sidebar.error(f"Failed to load repo PDF: {e}")

if pdf_bytes is None:
    st.info("Upload a PDF or add 'Annual-Report-2024-25.pdf' to the repo and enable 'Use repo PDF'.")
    st.stop()

# -----------------------
# Extract & filter paragraphs
# -----------------------
with st.spinner("Extracting paragraphs from PDF..."):
    paragraphs = extract_paragraphs_from_pdf_bytes_py(pdf_bytes)

st.write(f"Total paragraphs extracted: {len(paragraphs)}")

keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
if not keywords:
    st.sidebar.error("Please enter at least one keyword.")
    st.stop()

matched = filter_paragraphs(paragraphs, keywords, min_len=min_len)
st.write(f"Paragraphs matching keywords ({len(keywords)} keywords): {len(matched)}")

# Display matched paragraphs
if show_raw and matched:
    st.subheader("Matched Paragraphs")
    for i, p in enumerate(matched[:max_paragraphs_display], 1):
        st.markdown(f"**Paragraph {i}:** {normalize_text(p)}")

if not matched:
    st.info("No paragraphs matched your keywords. Try broader keywords or reduce Minimum paragraph length.")
    st.stop()

# -----------------------
# Tokenize -> WordCloud
# -----------------------
combined = " ".join(matched).lower()
combined = re.sub(r'[^a-z0-9\s]', ' ', combined)
combined = re.sub(r'\s+', ' ', combined)

stop_words = set(stopwords.words("english"))
custom_stop = {
    'sansera','engineering','limited','report','annual','company','million','mn','fy','page',
    'figure','table','note','notes','financial','statement','statutory','operations','operational','group','india'
}
stop_words |= custom_stop

tokens = [w for w in word_tokenize(combined) if w.isalnum() and len(w) > 2 and w not in stop_words]

# Frequency table
freq = pd.Series(tokens).value_counts().reset_index()
freq.columns = ["token", "count"]
st.subheader("Top tokens (filtered)")
st.dataframe(freq.head(50))

# WordCloud
st.subheader("Word Cloud")
if tokens:
    wc = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(" ".join(tokens))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No tokens available to build word cloud after filtering stopwords.")

# Download matched paragraphs
df_out = pd.DataFrame({"paragraph": matched})
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button("Download matched paragraphs (CSV)", csv_bytes, file_name="matched_paragraphs.csv", mime="text/csv")

# Keyword counts
st.subheader("Keyword match counts")
keyword_counts = {k: sum(1 for p in matched if k.lower() in p.lower()) for k in keywords}
st.write(keyword_counts)

st.markdown("---")
st.write("Tip: to auto-load the Annual Report from the repo, add `Annual-Report-2024-25.pdf` to the repository root and enable 'Use repo PDF' in the sidebar.")
