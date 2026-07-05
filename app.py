import streamlit as st
import librosa
import numpy as np
import pickle

st.set_page_config(page_title="AI Audio Emotion Detector", layout="wide")
st.title("🧠 AI Audio Emotion Recognition Engine")

@st.cache_resource
def load_ai_model():
    with open("emotion_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["classes"]

try:
    model, classes = load_ai_model()
    st.sidebar.success("✅ AI Model Loaded!")
except Exception as e:
    st.sidebar.error(f"❌ Model Error: {e}")

st.sidebar.markdown("### 📊 Project Overview")
st.sidebar.info("Dataset: RAVDESS Speech\n\nFeatures: 40 MFCCs")

tab1, tab2 = st.tabs(["🔮 Predict Emotion", "📊 Model Insights"])

with tab1:
    st.header("🎵 Upload an Audio File")
    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("Analyze Emotion 🚀"):
            X, sample_rate = librosa.load(uploaded_file, sr=22050)
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)

            prediction = model.predict(mfccs_processed)[0]
            st.success(f"### Detected Emotion: **{prediction.upper()}**")

with tab2:
    st.header("📈 Confusion Matrix Insight")
    st.info("Aap ki notebook ka analysis batata hai ke 'Calm' sab se accurate hai.")
