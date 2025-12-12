import streamlit as st
import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Only download punkt once (quiet to avoid noisy logs)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Sansera Report - Keyword Extractor", layout="wide")
st.title("Sansera Annual Report — Keyword Paragraph Extractor")
st.write("Upload a PDF, enter multiple keywords (comma-separated). Extracts paragraphs containing all keywords and shows top words.")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
keywords_input = st.text_input("Enter keywords (comma separated):")

if uploaded_pdf and keywords_input:
    keywords = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]
    if not keywords:
        st.warning("Please enter at least one keyword.")
    else:
        # Extract text from PDF
        text_data = ""
        with pdfplumber.open(uploaded_pdf) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    # basic cleanup
                    extracted = extracted.replace("-\n", "").replace("\n", " ")
                    text_data += extracted + "\n\n"

        # Split into paragraphs (double newline) and filter
        raw_paragraphs = [p.strip() for p in text_data.split("\n\n") if p.strip()]
        matching_paragraphs = []
        for para in raw_paragraphs:
            p_lower = para.lower()
            if all(k in p_lower for k in keywords):
                matching_paragraphs.append(para)

        st.subheader("Extracted Paragraphs")
        if not matching_paragraphs:
            st.info("No paragraphs found containing all specified keywords.")
        else:
            # show paragraphs in an expandable area
            for i, p in enumerate(matching_paragraphs, 1):
                with st.expander(f"Paragraph {i} (click to expand)"):
                    st.write(p)

            # Build token list for frequency analysis
            combined = " ".join(matching_paragraphs).lower()
            tokens = [t for t in word_tokenize(combined) if t.isalpha() and len(t) > 1]

            # remove stopwords
            stopwords = set(nltk.corpus.stopwords.words("english"))
            tokens = [t for t in tokens if t not in stopwords]

            freq = Counter(tokens)
            top = freq.most_common(20)

            if top:
                words, counts = zip(*top)
                # horizontal bar chart (most readable)
                fig, ax = plt.subplots(figsize=(10,6))
                ax.barh(words[::-1], counts[::-1])
                ax.set_xlabel("Frequency")
                ax.set_title("Top words from extracted paragraphs")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No words available to show frequency plot.")
