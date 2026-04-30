import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Base Reset ---- */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ---- Background ---- */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* ---- Hide default streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; }

/* ---- Top badge ---- */
.top-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #818cf8;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}

/* ---- Hero title ---- */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    color: #f0f1f5;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(90deg, #818cf8 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---- Subtitle ---- */
.hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 2.5rem;
    font-weight: 300;
}

/* ---- Card ---- */
.card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
}

/* ---- Label ---- */
.input-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.6rem;
}

/* ---- Textarea ---- */
.stTextArea textarea {
    background: #0d0f14 !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    resize: none !important;
    transition: border-color 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.08) !important;
}
.stTextArea textarea::placeholder { color: #3d4456 !important; }

/* ---- Button ---- */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ---- Result boxes ---- */
.result-positive {
    background: rgba(16, 185, 129, 0.07);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-negative {
    background: rgba(239, 68, 68, 0.07);
    border: 1px solid rgba(239, 68, 68, 0.22);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-icon { font-size: 2rem; line-height: 1; }
.result-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.result-positive .result-label { color: #34d399; }
.result-negative .result-label { color: #f87171; }
.result-text {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f0f1f5;
    margin-bottom: 2px;
}
.result-conf {
    font-size: 0.78rem;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- Confidence bar ---- */
.conf-bar-wrap {
    background: #1f2330;
    border-radius: 100px;
    height: 4px;
    margin-top: 10px;
    overflow: hidden;
}
.conf-bar-fill-pos {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.6s ease;
}
.conf-bar-fill-neg {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ef4444, #f87171);
}

/* ---- Warning ---- */
.warn-box {
    background: rgba(245, 158, 11, 0.07);
    border: 1px solid rgba(245, 158, 11, 0.22);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #fbbf24;
    font-size: 0.85rem;
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

/* ---- Footer ---- */
.footer {
    margin-top: 3rem;
    text-align: center;
    font-size: 0.75rem;
    color: #3d4456;
    line-height: 1.8;
}
.footer a { color: #6366f1; text-decoration: none; }
.footer a:hover { color: #818cf8; }

/* ---- Divider ---- */
.divider {
    height: 1px;
    background: #1f2330;
    margin: 1.5rem 0;
}

/* ---- How it works ---- */
.steps-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
}
.step {
    flex: 1;
    background: #0d0f14;
    border: 1px solid #1f2330;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.step-num {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #6366f1;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.step-text {
    font-size: 0.78rem;
    color: #9ca3af;
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load model + vectorizer
# =========================
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
stemmer = PorterStemmer()


# =========================
# Preprocessing
# =========================
def clean_text(text):
    text = re.sub(r'@[\w]*', '', text)
    text = re.sub(r'[^a-zA-Z#]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if len(w) > 3]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


# =========================
# Hero Section
# =========================
st.markdown('<div class="top-badge">🧠 ML · NLP · Text Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Detect <span>Hate Speech</span><br>in Any Text</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Paste a tweet, comment, or any sentence — the model tells you what\'s behind it.</div>', unsafe_allow_html=True)

# =========================
# Input Card
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="input-label">📝 Input Text</div>', unsafe_allow_html=True)
text = st.text_area(
    label="",
    placeholder="Type or paste text here... e.g. 'I love how kind people can be.'",
    height=130,
    label_visibility="collapsed"
)
analyze_btn = st.button("Analyze Sentiment →")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Result
# =========================
if analyze_btn:
    if text.strip() == "":
        st.markdown('<div class="warn-box">⚠️ Please enter some text to analyze.</div>', unsafe_allow_html=True)
    else:
        cleaned = clean_text(text)
        if cleaned.strip() == "":
            st.markdown('<div class="warn-box">⚠️ Text is too short or has no meaningful words after processing.</div>', unsafe_allow_html=True)
        else:
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]

            if prediction == 0:
                confidence = round(probability[0] * 100, 2)
                st.markdown(f"""
                <div class="result-positive">
                    <div class="result-icon">😊</div>
                    <div style="flex:1">
                        <div class="result-label">Result</div>
                        <div class="result-text">Normal / Non-Hate Speech</div>
                        <div class="result-conf">Confidence: {confidence}%</div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill-pos" style="width:{confidence}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = round(probability[1] * 100, 2)
                st.markdown(f"""
                <div class="result-negative">
                    <div class="result-icon">😠</div>
                    <div style="flex:1">
                        <div class="result-label">Result</div>
                        <div class="result-text">Hate / Offensive Speech</div>
                        <div class="result-conf">Confidence: {confidence}%</div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill-neg" style="width:{confidence}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =========================
# How it works
# =========================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="input-label">⚙️ How it works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="steps-row">
    <div class="step">
        <div class="step-num">Step 01</div>
        <div class="step-text">Text is cleaned &amp; stemmed using NLP preprocessing</div>
    </div>
    <div class="step">
        <div class="step-num">Step 02</div>
        <div class="step-text">TF-IDF vectorizer converts words to numeric features</div>
    </div>
    <div class="step">
        <div class="step-num">Step 03</div>
        <div class="step-text">ML model classifies and returns confidence score</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("""
<div class="footer">
    Built with Python · Scikit-learn · Streamlit<br>
    <span style="color:#4b5563">Special thanks to my teacher for the guidance &amp; support 🙏</span>
</div>
""", unsafe_allow_html=True)
