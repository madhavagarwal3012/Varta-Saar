# 🤖 Varta-Saar (वार्ता-सार)

> **Transform raw, multi-speaker meeting recordings and media streams into clear, multi-model AI executive intelligence and structured PDF reports.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**Varta-Saar** (Sanskrit for *"Essence of Conversation"*) is an enterprise-grade AI meeting assistant designed to bridge the gap between unstructured conversation and actionable decision-making. 

Unlike single-LLM solutions, Varta-Saar processes speech through an ensemble pipeline—combining **AssemblyAI**, **BERTopic**, **OpenAI (GPT-4o)**, **Google Gemini**, **Perplexity Sonar**, and **Groq (Llama-3)**—to deliver cross-validated meeting summaries, precise speaker diarization, sentiment tracking, and publication-ready PDF exports.

---

## ✨ Key Features & Technical Highlights

### 🎙️ 1. Multi-Lingual Speech Processing
* **Automatic Language Detection & Translation:** Accepts audio/video in non-English languages and automatically translates transcripts into English using Google Gemini models.
* **Speaker Diarization:** Uses AssemblyAI models to tag distinct speakers with precise millisecond timestamps ([HH:MM:SS] Speaker A).
* **Media Stream Ingestion:** Native support for standard audio/video files (.mp3, .m4a, .mp4, .mov) and direct stream extraction via yt-dlp and FFmpeg.

### 🧠 2. Multi-AI Consensus Architecture
To eliminate single-model bias and hallucinations, Varta-Saar queries an ensemble of frontier LLM models:
* **Perplexity AI (Sonar):** Real-time web-aware context synthesis.
* **OpenAI (GPT-4o-mini):** Precise bulleted action items and structural summaries.
* **Google Gemini (1.5 Flash):** Deep contextual translation and long-form narrative structure.
* **Groq (Llama-3.3-70b):** Low-latency summary formatting and validation.

### 📊 3. Topic Modeling & Sentiment Metrics
* **BERTopic Integration:** Leverages vector representations (CountVectorizer) to extract underlying themes and cluster topic keywords across discussion threads.
* **Granular Sentiment Analysis:** Evaluates overall meeting tone (Positive, Neutral, Negative) across speaker turns.

### 📄 4. Production-Grade PDF Report Generation
* **ReportLab Integration:** Generates clean, standalone PDF reports using standard typography, formatted tables, custom line-wrapping, and dynamic page layouts.
* **Defensive Text Sanitization:** Built-in regex sanitizers strip non-printable unicode characters, malformed markdown, and HTML break tags to guarantee zero PDF compilation errors.

---

## 🏗️ System Architecture Pipeline

┌─────────────────────────┐
│ Audio / Video / YouTube │
└────────────┬────────────┘
             │
             ▼
   [ FFmpeg Processing ] (Formats to 192kbps MP3 / Verifies File Integrity)
             │
             ▼
   [ AssemblyAI Engine ] (Transcribes, Detects Language, Tags Speakers, Evaluates Sentiment)
             │
             ▼
   [ BERTopic Modeling ] (Clustering & Key-Phrase Extraction)
             │
             ▼
 ┌────────────────────────────────────────────────────────┐
 │            Multi-AI Synthesis Engine                   │
 │ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
 │ │  Perplexity  │ │ OpenAI GPT-4 │ │  Google Gemini   │ │
 │ └──────────────┘ └──────────────┘ └──────────────────┘ │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │               Cleaned Executive Report                 │
 │      (Consolidated Insights + Speaker Logs)            │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
               [ ReportLab PDF Generator ]

---

## 🔒 Security & Privacy Commitments

* **Zero Permanent Data Retention:** Uploaded media files and converted audio chunks are stored in temporary system folders (tempfile) and purged immediately upon pipeline completion via Python finally execution blocks.
* **API Key Safeguards:** API credentials are loaded dynamically via st.secrets or local environment variables. Keys are never logged, exposed to the front end, or hardcoded.
* **Client-Side File Processing:** PDF generation runs entirely in-memory using io.BytesIO streams, serving binary base64 downloads directly to the client browser.

---

## 🛠️ Quickstart & Local Setup

### Prerequisites
* **Python 3.9+**
* **FFmpeg:** Required for audio/video stream conversion and integrity verification.
  * macOS: brew install ffmpeg
  * Ubuntu/Debian: sudo apt install ffmpeg
  * Windows: winget install ffmpeg

### Installation

1. **Clone the repository:**
   git clone https://github.com/your-username/varta-saar.git
   cd varta-saar

2. **Create and activate a virtual environment:**
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies:**
   pip install -r requirements.txt

4. **Configure API Keys:**
   Create a .env file or .streamlit/secrets.toml with your credentials:
   ASSEMBLYAI_API_KEY = "your_assemblyai_key"
   OPENAI_API_KEY = "your_openai_key"
   GEMINI_API_KEY = "your_gemini_key"
   PERPLEXITY_API_KEY = "your_perplexity_key"
   GROQ_API_KEY = "your_groq_key"

5. **Run the Streamlit application:**
   streamlit run app.py

---

## 📂 Supported Formats

| Category | Supported Extension / Protocol |
| :--- | :--- |
| **Audio Files** | .mp3, .m4a, .wav, .ogg |
| **Video Files** | .mp4, .mov, .avi |
| **Streaming** | YouTube Video URLs, Direct Audio Links |

---

## ⚖️ License & Copyright

Distributed under the MIT License. See LICENSE for more information.

**© Copyright 2025–2026 by Madhav Agarwal. All rights reserved.**
