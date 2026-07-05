# 🔮 VibeCheck AI: Voice Emotion Detector

A highly optimized Speech Emotion Recognition (SER) application that processes live microphone recordings or uploaded audio streams to classify sentiment into four categories: Happy, Sad, Angry, or Calm.

## 🚀 Live Demo
Try the application live on Hugging Face here: https://huggingface.co/spaces/ByJaveria/vibecheck-ai
## 🛠️ How It Works
- **Feature Extraction:** Audio inputs are converted into 40 Mel-Frequency Cepstral Coefficients (MFCCs) using `librosa`.
- **Lightweight Inference:** Instead of loading heavy deep learning frameworks on the server, the app evaluates prediction layers instantly using native **NumPy matrix multiplication** ($Z = W \cdot X + B$).
- **Native Streaming:** Uses browser-native HTML5 audio capture via Streamlit to allow instant microphone recording.

## 📦 Tech Stack
- Streamlit (UI & Custom Cyberpunk Styling)
- Librosa & Soundfile (Audio Processing)
- NumPy (Mathematical Inference Matrix)
