import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="YouTube Comments Sentiment Analyzer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Import a modern font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Main container styling with a space-inspired gradient */
    .main {
        background: linear-gradient(135deg, #0d1b2a, #1b263b, #415a77);
        color: #e0e1dd;
        font-family: 'Poppins', sans-serif;
        min-height: 100vh;
        padding: 20px;
    }

    /* Header styling with glowing effect */
    .header {
        color: #00f5d4;
        font-size: 56px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
        text-shadow: 0 0 15px rgba(0, 245, 212, 0.7), 0 0 30px rgba(0, 245, 212, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }

    /* Subheader styling */
    .subheader {
        color: #a9bcd0;
        font-size: 22px;
        font-weight: 300;
        text-align: center;
        margin-bottom: 40px;
        letter-spacing: 1px;
    }

    /* Sidebar styling with a sleek, dark theme */
    .stSidebar {
        background: linear-gradient(to bottom, #0d1b2a, #1b263b);
        color: #e0e1dd;
        border-right: 1px solid rgba(0, 245, 212, 0.2);
    }
    .stSidebar h2 {
        color: #00f5d4;
        font-size: 28px;
        font-weight: 600;
        text-shadow: 0 0 5px rgba(0, 245, 212, 0.5);
    }
    .stSidebar .stSelectbox {
        background-color: #1b263b;
        color: #e0e1dd;
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(0, 245, 212, 0.3);
        transition: all 0.3s ease;
    }
    .stSidebar .stSelectbox:hover {
        border-color: #00f5d4;
        box-shadow: 0 0 10px rgba(0, 245, 212, 0.4);
    }

    /* Input box with a glowing border */
    .stTextInput > div > div > input {
        background-color: #1b263b;
        color: #e0e1dd;
        border: 2px solid #00f5d4;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 0 8px rgba(0, 245, 212, 0.3);
    }
    .stTextInput > div > div > input:focus {
        outline: none;
        border-color: #00f5d4;
        box-shadow: 0 0 15px rgba(0, 245, 212, 0.6);
    }

    /* Button with a gradient and hover animation */
    .stButton > button {
        background: linear-gradient(90deg, #00f5d4, #00c4b4);
        color: #0d1b2a;
        border: none;
        border-radius: 12px;
        padding: 14px 40px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.4s ease;
        box-shadow: 0 0 10px rgba(0, 245, 212, 0.5);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00c4b4, #00f5d4);
        transform: translateY(-3px);
        box-shadow: 0 5px 20px rgba(0, 245, 212, 0.8);
    }

    /* Result box with a futuristic card design */
    .result-box {
        background: rgba(27, 38, 59, 0.9);
        color: #e0e1dd;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 245, 212, 0.2);
        margin-top: 30px;
        border: 1px solid rgba(0, 245, 212, 0.3);
        backdrop-filter: blur(5px);
        transition: transform 0.3s ease;
    }
    .result-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 245, 212, 0.4);
    }
    .result-box h3 {
        color: #00f5d4;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
        text-shadow: 0 0 5px rgba(0, 245, 212, 0.5);
    }
    .result-box p {
        font-size: 16px;
        margin: 8px 0;
        line-height: 1.6;
    }

    /* Bar chart styling with a futuristic look */
    .stBarChart {
        background: rgba(27, 38, 59, 0.8);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(0, 245, 212, 0.3);
        box-shadow: 0 5px 15px rgba(0, 245, 212, 0.2);
    }

    /* Warning message styling */
    .stAlert {
        background: rgba(255, 87, 87, 0.1);
        color: #ff5757;
        border: 1px solid #ff5757;
        border-radius: 10px;
        padding: 10px;
        font-weight: 400;
    }

    /* Footer styling with a subtle glow */
    .footer {
        text-align: center;
        color: #a9bcd0;
        margin-top: 50px;
        font-size: 14px;
        font-weight: 300;
        text-shadow: 0 0 5px rgba(0, 245, 212, 0.3);
    }

    /* Glowing animation for the header */
    @keyframes glow {
        0% {
            text-shadow: 0 0 15px rgba(0, 245, 212, 0.7), 0 0 30px rgba(0, 245, 212, 0.5);
        }
        100% {
            text-shadow: 0 0 25px rgba(0, 245, 212, 0.9), 0 0 40px rgba(0, 245, 212, 0.7);
        }
    }

    /* Divider styling */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(0, 245, 212, 0.5), transparent);
        margin: 40px 0;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2>🚀 Model Selection</h2>", unsafe_allow_html=True)
    model_options = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "SVM": "svm_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "XGBoost": "xgboost_model.pkl"
    }
    selected_model_name = st.selectbox("Choose a Model", list(model_options.keys()), key="model_select")
    model_file = model_options[selected_model_name]

try:
    model = joblib.load(model_file)
    tfidf = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure all .pkl files are in the folder.")
    st.stop()

try:
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.warning("label_encoder.pkl not found. Assuming 0=negative, 1=neutral, 2=positive.")
    class FakeLabelEncoder:
        def inverse_transform(self, y):
            mapping = {0: "negative", 1: "neutral", 2: "positive"}
            return [mapping[i] for i in y]
        @property
        def classes_(self):
            return ["negative", "neutral", "positive"]
    le = FakeLabelEncoder()

st.markdown("<h1 class='header'>🌟 YouTube Comments Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Enter a YouTube-style comment below to predict its sentiment (Positive, Negative, Neutral) with a touch of magic!</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    comment = st.text_input("", placeholder="Type your comment here...", key="comment_input")
    if st.button("Analyze Sentiment"):
        if comment:
            stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"])
            cleaned_comment = ' '.join([word for word in re.sub(r'[^\w\s]', '', str(comment).lower()).split() if word not in stop_words])
            comment_tfidf = tfidf.transform([cleaned_comment])

            prediction = model.predict(comment_tfidf)
            sentiment = le.inverse_transform(prediction)[0]

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(comment_tfidf)[0]
                confidence = max(probs) * 100
                prob_df = pd.DataFrame({'Sentiment': le.classes_, 'Probability': probs})
            else:
                confidence = "N/A (Model does not support probabilities)"
                prob_df = None

            st.markdown(f"""
                <div class='result-box'>
                    <h3>Prediction Result</h3>
                    <p><b>Sentiment:</b> {sentiment}</p>
                    <p><b>Confidence:</b> {confidence if isinstance(confidence, str) else f'{confidence:.2f}%'}</p>
                    <p><b>Model Used:</b> {selected_model_name}</p>
                </div>
            """, unsafe_allow_html=True)

            if prob_df is not None:
                st.markdown("<h3 style='color: #00f5d4; margin-top: 40px; text-shadow: 0 0 5px rgba(0, 245, 212, 0.5);'>📊 Probability Breakdown</h3>", unsafe_allow_html=True)
                st.bar_chart(prob_df.set_index('Sentiment'), height=350)

        else:
            st.warning("Please enter a comment to analyze.")

st.markdown("<hr><p class='footer'>Powered by advanced machine learning</p>", unsafe_allow_html=True)
