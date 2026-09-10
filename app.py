import streamlit as st
import platform
import os
import io
import time
import tempfile
import json
import base64
from pathlib import Path
import requests
import yt_dlp
import subprocess
import shutil
from openai import OpenAI
from groq import Groq
from huggingface_hub import InferenceClient
from bertopic import BERTopic
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
from xhtml2pdf import pisa

# =========================================================================
# === STEP 1: CONFIGURATION AND API CLIENT SETUP (Graceful Degradation) ===
# =========================================================================
st.set_page_config(
    page_title="Varta-Saar: The Ultimate AI Meeting Assistant",
    page_icon="🤖",
)

# Safely load keys from st.secrets or os.getenv without crashing the app
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY") or os.getenv("ASSEMBLYAI_API_KEY")

# Initialize Clients conditionally so missing/expired keys won't trigger hard crash screens
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
hf_client = InferenceClient(api_key=HUGGINGFACE_API_KEY) if HUGGINGFACE_API_KEY else None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        google_client = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        google_client = None
else:
    google_client = None

if not ASSEMBLYAI_API_KEY:
    st.warning("⚠️ AssemblyAI API key not found. Audio transcription features will be limited or disabled.")

# =========================================================================
# === STEP 2: HELPER FUNCTIONS ============================================
# =========================================================================

def get_audio_url(file_path):
    """
    Uploads a local file to a transcription service for processing.
    """
    if not ASSEMBLYAI_API_KEY:
        st.error("AssemblyAI API key is missing. Cannot upload audio for transcription.")
        st.stop()
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    response = requests.post(
        'https://api.assemblyai.com/v2/upload',
        headers=headers,
        data=read_file(file_path, 60000)
    )
    if response.status_code == 200:
        return response.json()['upload_url']
    else:
        st.error(f"Failed to upload audio for transcription: {response.status_code}")
        st.json(response.json())
        st.stop()
        return None

def read_file(file_path, chunk_size):
    """
    Reads a file in chunks for uploading to a service.
    """
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data

def transcribe_audio(audio_url):
    """
    Sends an audio URL for transcription with language detection enabled.
    """
    headers = {'authorization': ASSEMBLYAI_API_KEY, 'content-type': 'application/json'}
    data = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "sentiment_analysis": True,
        "language_detection": True
    }
    response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        return response.json()['id']
    else:
        st.error(f"Failed to submit transcription job: {response.status_code}")
        st.json(response.json())
        st.stop()

def get_transcription_result(transcript_id):
    """
    Polls a service for the transcription result.
    """
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    while True:
        response = requests.get(
            f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
            headers=headers
        )
        if response.status_code != 200:
            st.error(f"Failed to get transcription result: {response.status_code}")
            st.json(response.json())
            st.stop()
            return None

        result = response.json()
        if result['status'] == 'completed':
            return result
        elif result['status'] == 'failed':
            st.error(f"Transcription failed: {result.get('error')}")
            st.stop()
            return None
        time.sleep(1)

def get_summary_model_1(text):
    """
    Generates a summary using Groq (High Speed & Free Tier). Falls back gracefully.
    """
    if not groq_client:
        return ""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert meeting summarizer. Provide a concise and professional summary."},
                {"role": "user", "content": f"Please summarize the following meeting transcript:\n\n{text}"}
            ],
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[Groq model unavailable/expired: {e}]"

def get_summary_model_2(text):
    """
    Generates a summary using Google Gemini. Falls back gracefully.
    """
    if not google_client:
        return ""
    try:
        response = google_client.generate_content(
            f"You are an expert meeting summarizer. Your task is to provide a concise and professional summary of the following meeting transcript:\n\n{text}"
        )
        return response.text
    except Exception as e:
        return f"[Gemini model unavailable/expired: {e}]"

def get_summary_model_3(text):
    """
    Generates a summary using OpenAI or Hugging Face Inference API as backup. Falls back gracefully.
    """
    if openai_client:
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert meeting summarizer."},
                    {"role": "user", "content": f"Please summarize the following meeting transcript:\n\n{text}"}
                ]
            )
            return completion.choices[0].message.content
        except Exception:
            pass
            
    if hf_client:
        try:
            messages = [{"role": "user", "content": f"Summarize this meeting transcript concisely: {text}"}]
            response = hf_client.chat_completion(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Hugging Face model unavailable/expired: {e}]"
            
    return ""

def perform_topic_modeling(docs):
    """
    Performs topic modeling on the document list.
    """
    if len(docs) <= 1:
        return [{"topic": "Not enough data for topic modeling", "count": 1, "keywords": ""}]

    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    topics, probabilities = topic_model.fit_transform(docs)

    topic_info = topic_model.get_topic_info()
    topics_list = []

    for _, row in topic_info.iterrows():
        if row['Topic'] == -1:
            continue
        topics_list.append(
            {
                "topic": row['Name'],
                "count": row['Count'],
                "keywords": ", ".join([word for word, _ in topic_model.get_topic(row['Topic'])])
            }
        )
    return topics_list

def clean_text(text):
    """
    Cleans text to remove non-printable characters that can corrupt PDFs.
    """
    return ''.join(c for c in text if c.isprintable() or c in ('\n', '\t', '\r'))

def format_time(ms):
    """Converts milliseconds to HH:MM:SS format."""
    seconds = ms // 1000
    minutes = seconds // 60
    hours = minutes // 60
    seconds %= 60
    minutes %= 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def generate_pdf_report(report_data):
    """
    Generates a downloadable PDF report using xhtml2pdf for reliability.
    """
    try:
        report_data_cleaned = {
            "date": report_data['date'],
            "topic": clean_text(report_data['topic']),
            "consolidated_summary": clean_text(report_data['consolidated_summary']),
            "summary_1": clean_text(report_data['summary_1']),
            "summary_2": clean_text(report_data['summary_2']),
            "summary_3": clean_text(report_data['summary_3']),
            "diarization": clean_text(report_data['diarization']),
            "sentiment": clean_text(report_data['sentiment']),
            "topics": [{"topic": clean_text(t['topic']), "count": t['count'], "keywords": clean_text(t['keywords'])} for t in report_data['topics']]
        }

        pdf_content = f"""
            <html>
            <head>
                <title>Meeting Report</title>
                <style>
                    @page {{ size: A4; margin: 2cm; }}
                    body {{ font-family: 'Arial', sans-serif; }}
                    h1, h2, h3 {{ color: #1a237e; }}
                    .section {{ margin-bottom: 20px; border-left: 5px solid #3f51b5; padding-left: 15px; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; white-space: pre-wrap; }}
                    .summary {{ background-color: #e8eaf6; padding: 20px; border-radius: 8px; }}
                    .sentiment {{ font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Varta-Saar Meeting Report</h1>
                <p><strong>Date:</strong> {report_data_cleaned['date']}</p>
                <p><strong>Meeting Topic:</strong> {report_data_cleaned['topic']}</p>

                <div class="section summary">
                    <h2>Consolidated Summary</h2>
                    <pre>{report_data_cleaned['consolidated_summary']}</pre>
                </div>

                <div class="section">
                    <h2>AI Model Summaries</h2>
                    <h3>Summary from Groq Model</h3>
                    <pre>{report_data_cleaned['summary_1']}</pre>
                    <h3>Summary from Gemini Model</h3>
                    <pre>{report_data_cleaned['summary_2']}</pre>
                    <h3>Summary from Backup Model</h3>
                    <pre>{report_data_cleaned['summary_3']}</pre>
                </div>

                <div class="section">
                    <h2>Speaker Diarization</h2>
                    <pre>{report_data_cleaned['diarization']}</pre>
                </div>

                <div class="section">
                    <h2>Sentiment Analysis</h2>
                    <p>Overall Sentiment: <span class="sentiment">{report_data_cleaned['sentiment']}</span></p>
                </div>

                <div class="section">
                    <h2>Key Topics</h2>
                    <ul>
        """
        for topic in report_data_cleaned['topics']:
            pdf_content += f"<li><b>{topic['topic']}</b>: {topic['keywords']} (Documents: {topic['count']})</li>"

        pdf_content += """
                    </ul>
                </div>
            </body>
            </html>
        """

        pdf_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(pdf_content, dest=pdf_buffer)

        if pisa_status.err:
            raise Exception("PDF generation failed.")

        pdf_buffer.seek(0)
        b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')

        return f'<a href="data:application/pdf;base64,{b64_pdf}" download="meeting_report.pdf">Download Report as PDF</a>'
    except Exception as e:
        st.warning("An error occurred while generating the full PDF report. A simplified version has been created instead.")
        fallback_content = f"<html><body><h1>Simplified Meeting Report</h1><pre>{clean_text(report_data['consolidated_summary'])}</pre></body></html>"
        fallback_buffer = io.BytesIO()
        pisa.CreatePDF(fallback_content, dest=fallback_buffer)
        fallback_buffer.seek(0)
        b64_pdf_fallback = base64.b64encode(fallback_buffer.read()).decode('utf-8')
        return f'<a href="data:application/pdf;base64,{b64_pdf_fallback}" download="simplified_report.pdf">Download Simplified Report as PDF</a>'

# =========================================================================
# === STEP 3: MAIN APPLICATION PIPELINE ===================================
# =========================================================================

def run_full_pipeline(file_path, meeting_topic):
    st.markdown("---")
    st.header("Detailed Analysis 📊")

    st.subheader("1. Transcription and Analysis")
    try:
        with st.spinner("Uploading file and transcribing audio..."):
            audio_url = get_audio_url(file_path)
            transcript_id = transcribe_audio(audio_url)
            transcript_dict = get_transcription_result(transcript_id)

        if 'text' in transcript_dict and transcript_dict['text']:
            st.success("Transcription complete!")
            raw_transcript = transcript_dict.get('text', '')
            detected_language = transcript_dict.get('language_code', 'en')
            st.info(f"Detected language: **{detected_language}**")

            if detected_language != 'en' and google_client:
                with st.spinner(f"Translating the transcript from {detected_language} to English..."):
                    try:
                        translation_response = google_client.generate_content(f"Translate the following text into English:\n\n{raw_transcript}")
                        raw_transcript = translation_response.text
                        st.success("Translation complete!")
                    except Exception:
                        pass

            if not raw_transcript or len(raw_transcript.strip()) == 0:
                st.error("The transcription returned an empty text.")
                return 

            diarization_output = ""
            utterances_list = transcript_dict.get('utterances')
            if utterances_list:
                for utterance in utterances_list:
                    start_time = format_time(utterance.get('start', 0))
                    diarization_output += f"[{start_time}] Speaker {utterance['speaker']}: {utterance['text']}\n"
                st.markdown("### Speaker Diarization")
                st.text_area("Transcript with Speakers", diarization_output, height=200)
            else:
                diarization_output = "No speaker data available."
            
            if 'sentiment_analysis_results' in transcript_dict and transcript_dict['sentiment_analysis_results']:
                sentiment_counts = {}
                for sentiment in transcript_dict['sentiment_analysis_results']:
                    sentiment_counts[sentiment['sentiment']] = sentiment_counts.get(sentiment['sentiment'], 0) + 1
                dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                st.markdown("### Sentiment Analysis")
                st.write(f"The overall sentiment of the meeting is: **{dominant_sentiment.capitalize()}**")
            else:
                dominant_sentiment = "Not available"

            st.markdown("### Key Topics (powered by Topic Modeling)")
            if utterances_list:
                docs_for_topic_modeling = [u['text'] for u in utterances_list]
                try:
                    topics = perform_topic_modeling(docs_for_topic_modeling)
                    st.json(topics)
                except Exception:
                    topics = [{"topic": "Topic modeling skipped", "count": 0, "keywords": ""}]
            else:
                topics = [{"topic": "Not available", "count": 0, "keywords": ""}]

            st.subheader("2. AI-Powered Summaries")
            with st.spinner("Generating summaries with available AI models (Groq, Gemini, Backup)..."):
                summary_1 = get_summary_model_1(raw_transcript)
                summary_2 = get_summary_model_2(raw_transcript)
                summary_3 = get_summary_model_3(raw_transcript)
            
            st.success("Summaries generation step finished!")
            
            consolidated_summary_list = [s for s in [summary_1, summary_2, summary_3] if s and not s.startswith("[")]
            if summary_1:
                st.markdown("### Summary from Groq Model")
                st.text_area("Groq Summary", summary_1, height=120)
            if summary_2:
                st.markdown("### Summary from Gemini Model")
                st.text_area("Gemini Summary", summary_2, height=120)
            if summary_3:
                st.markdown("### Summary from Backup Model")
                st.text_area("Backup Summary", summary_3, height=120)

            if consolidated_summary_list:
                consolidated_summary = "\n\n".join(consolidated_summary_list)
            else:
                consolidated_summary = "All configured AI model endpoints failed or keys are missing/expired. Please check your credentials."

            st.markdown("---")
            st.header("Full Report 📋")
            report_data = {
                "date": time.strftime("%Y-%m-%d"),
                "topic": meeting_topic,
                "consolidated_summary": consolidated_summary,
                "summary_1": summary_1 or "N/A",
                "summary_2": summary_2 or "N/A",
                "summary_3": summary_3 or "N/A",
                "diarization": diarization_output,
                "sentiment": dominant_sentiment.capitalize(),
                "topics": topics
            }
            st.markdown(generate_pdf_report(report_data), unsafe_allow_html=True)
        else:
            st.error("Transcription failed to return a valid result.")
            
    except Exception as e:
        st.error(f"An unexpected error occurred during execution: {e}")

# =========================================================================
# === STEP 4: STREAMLIT UI AND LOGIC ======================================
# =========================================================================

st.title("Varta-Saar: The Ultimate AI Meeting Assistant")
st.markdown("Easily turn your meetings into a detailed, actionable report.")
st.markdown("---")

tab_upload, tab_youtube = st.tabs(["Upload File", "YouTube URL"])
with tab_upload:
    meeting_topic = st.text_input("Meeting Topic", placeholder="e.g., Q3 Marketing Strategy Review")
    uploaded_file = st.file_uploader(
        "Upload Meeting Audio/Video (.mp3, .m4a, .mp4, .mov)",
        type=["mp3", "m4a", "mp4", "mov"]
    )
    if st.button("Generate Report"):
        if not uploaded_file or not meeting_topic:
            st.error("Please provide both a file and a meeting topic.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name

        if audio_path.endswith((".mp4", ".mov")):
            st.info("Extracting audio from video using FFmpeg...")
            output_audio_path = tempfile.mktemp(suffix=".mp3")
            try:
                command = ['ffmpeg', '-i', audio_path, '-vn', '-acodec', 'libmp3lame', '-q:a', '2', output_audio_path]
                subprocess.run(command, check=True, capture_output=True, text=True)
                os.remove(audio_path)
                audio_path = output_audio_path
            except Exception as e:
                st.error(f"Failed to extract audio with FFmpeg: {e}")
                st.stop()

        try:
            run_full_pipeline(audio_path, meeting_topic)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

with tab_youtube:
    st.subheader("YouTube URL 🔗")
    input_url = st.text_input("Enter a YouTube video URL or a direct audio URL (.mp3, .m4a)")
    cookies_file = st.file_uploader("Upload your cookies.txt file (for restricted videos)", type=["txt"])
    meeting_topic_url = st.text_input("Enter the main topic of the content", placeholder="e.g., Apple WWDC 2024 Keynote")

    if st.button("Generate Report 🚀", key="url_button"):
        if not input_url or not meeting_topic_url:
            st.error("Please provide a valid URL and a meeting topic.")
            st.stop()

        video_path = tempfile.mktemp(suffix=".mp4")
        audio_path = None

        try:
            if "youtube.com" in input_url or "youtu.be" in input_url:
                st.info("YouTube URL detected. Downloading video via yt-dlp...")
                ydl_opts = {
                    'format': 'best[height<=720]/best',
                    'outtmpl': video_path,
                    'noplaylist': True,
                    'ignoreerrors': True,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android', 'web']
                        }
                    }
                }
                if cookies_file:
                    cookies_path = tempfile.mktemp(suffix=".txt")
                    with open(cookies_path, "wb") as f:
                        f.write(cookies_file.getbuffer())
                    ydl_opts['cookiefile'] = cookies_path

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([input_url])

                if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                    st.error("Video download failed. The video may be private or restricted.")
                    st.stop()

                st.info("Extracting audio from the video...")
                audio_path = tempfile.mktemp(suffix=".mp3")
                command = ['ffmpeg', '-i', video_path, '-vn', '-q:a', '0', audio_path]
                subprocess.run(command, check=True, capture_output=True, text=True)
            else:
                response = requests.get(input_url, stream=True)
                response.raise_for_status()
                audio_path = tempfile.mktemp(suffix=".mp3")
                with open(audio_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                run_full_pipeline(audio_path, meeting_topic_url)
            else:
                st.error("Extracted audio file is empty or missing.")
        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

# =========================================================================
# === COPYRIGHT NOTICE ====================================================
# =========================================================================
st.markdown("---")
st.markdown("© Copyright 2025 by Madhav Agarwal. All rights reserved.")
