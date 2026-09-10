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
from bertopic import BERTopic
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
from xhtml2pdf import pisa
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# =========================================================================
# === STEP 1: CONFIGURATION AND API CLIENT SETUP (Backend Handling) =======
# =========================================================================
st.set_page_config(
    page_title="Varta-Saar: The Ultimate AI Meeting Assistant",
    page_icon="🤖",
)

# Safe API Key retrieval with fallback warnings instead of crashing
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY") or os.getenv("ASSEMBLYAI_API_KEY")

if not ASSEMBLYAI_API_KEY:
    st.error("AssemblyAI API key not found. Please set it in your Streamlit secrets to enable transcription.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    google_client = genai.GenerativeModel("gemini-1.5-flash")
else:
    google_client = None

# =========================================================================
# === STEP 2: HELPER FUNCTIONS ============================================
# =========================================================================

def get_audio_url(file_path):
    """
    Uploads a local file to a transcription service for processing.
    """
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
    if not PERPLEXITY_API_KEY: return ""
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "sonar", "messages": [{"role": "user", "content": f"Summarize:\n\n{text}"}], "temperature": 0.2}
    try:
        res = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data)
        res.raise_for_status()
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        st.warning(f"Perplexity Skipped: {e}")
        return ""

def get_summary_model_2(text):
    if not openai_client: return ""
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize:\n\n{text}"}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.warning(f"OpenAI Skipped (Check billing quota): {e}")
        return ""

def get_summary_model_3(text):
    if not google_client: return ""
    try:
        # Use standard active Gemini model endpoint
        model = genai.GenerativeModel("gemini-3.5-flash")
        response = model.generate_content(f"Summarize:\n\n{text}")
        return response.text
    except Exception as e:
        st.warning(f"Gemini Skipped: {e}")
        return ""

def get_summary_model_groq(text):
    """Generates a summary using Groq AI with the correct active model ID."""
    if not groq_client:
        return ""
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",  # Valid active model on Groq free tier
            messages=[
                {"role": "system", "content": "You are an expert meeting summarizer."},
                {"role": "user", "content": f"Please summarize the following meeting transcript:\n\n{text}"}
            ],
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.warning(f"Groq API Error: {e}")
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

def clean_for_reportlab(text):
    """Removes unsupported markdown syntax and cleans line breaks for ReportLab."""
    if not text:
        return ""
    # Standardize or remove malformed br tags
    text = re.sub(r'<br\s*/?>', '<br/>', text)
    # Strip markdown table formatting rows that break XML parsing
    text = re.sub(r'\|.*?\|', '', text)
    # Strip raw markdown headers
    text = re.sub(r'#{1,6}\s*', '', text)
    return text

def generate_pdf_report(report_data):
    """
    Generates a clean, beautifully formatted downloadable PDF report 
    using native ReportLab flowables with Trebuchet MS styling.
    """
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=45, leftMargin=45,
            topMargin=45, bottomMargin=45
        )
        story = []
        
        # Styles using standard clean typography (Helvetica maps cleanly to system sans-serif like Trebuchet)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4
        )
        
        meta_style = ParagraphStyle(
            'MetaText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=2
        )
        
        section_heading = ParagraphStyle(
            'SectionHead',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=14,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            'BodyClean',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8
        )

        # Build Document Story Content
        story.append(Paragraph("Varta-Saar Meeting Report", title_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
        
        date_str = report_data.get('date', '2026-09-10')
        topic_str = report_data.get('topic', 'General')
        story.append(Paragraph(f"<b>Date:</b> {date_str}", meta_style))
        story.append(Paragraph(f"<b>Meeting Topic:</b> {topic_str}", meta_style))
        story.append(Spacer(1, 10))

        # Consolidated Summary
        story.append(Paragraph("Consolidated Summary", section_heading))
        consolidated_text = clean_for_reportlab(report_data.get('consolidated_summary', ''))
        story.append(Paragraph(consolidated_text.replace('\n', '<br/>'), body_style))
        
        # AI Summaries Section
        story.append(Paragraph("AI Model Summaries", section_heading))
        if report_data.get('summary_3'):
            story.append(Paragraph("<b>Summary from Gemini:</b>", body_style))
            story.append(Paragraph(clean_for_reportlab(report_data['summary_3']).replace('\n', '<br/>'), body_style))
            story.append(Spacer(1, 4))
            
        if report_data.get('summary_2'):
            story.append(Paragraph("<b>Summary from Groq AI:</b>", body_style))
            story.append(Paragraph(clean_for_reportlab(report_data['summary_2']).replace('\n', '<br/>'), body_style))
            story.append(Spacer(1, 4))

        # Speaker Diarization
        if report_data.get('diarization'):
            story.append(Paragraph("Speaker Diarization", section_heading))
            story.append(Paragraph(clean_for_reportlab(report_data['diarization']).replace('\n', '<br/>'), body_style))

        # Sentiment Analysis
        story.append(Paragraph("Sentiment Analysis", section_heading))
        sentiment_val = report_data.get('sentiment', 'Positive')
        story.append(Paragraph(f"Overall Sentiment: <b>{sentiment_val}</b>", body_style))

        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')

        return f'<a href="data:application/pdf;base64,{b64_pdf}" download="meeting_report.pdf">Download Report as PDF</a>'

    except Exception as e:
        st.warning(f"PDF Generation Error: {e}")
        return ""
        
def transcribe_audio(audio_url):
    """
    Sends an audio URL for transcription with language detection enabled.
    """
    headers = {'authorization': ASSEMBLYAI_API_KEY, 'content-type': 'application/json'}
    data = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "sentiment_analysis": True,
        "language_detection": True  # NEW: Enable automatic language detection
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
        
# =========================================================================
# === STEP 3: MAIN APPLICATION PIPELINE ===================================
# =========================================================================

def run_full_pipeline(file_path, meeting_topic):
    """
    Runs the full analysis pipeline from transcription to report generation.
    """
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
                        translation_response = google_client.generate_content(
                            f"Translate the following text into English:\n\n{raw_transcript}"
                        )
                        raw_transcript = translation_response.text
                        st.success("Translation complete!")
                    except Exception as e:
                        st.warning(f"Translation failed: {e}. Proceeding with original transcript.")

            if not raw_transcript or len(raw_transcript.strip()) == 0:
                st.error("The transcription failed or returned an empty transcript. Please check your API key and try a video with clear speech.")
                return 

            diarization_output = ""
            utterances_list = transcript_dict.get('utterances')
            if utterances_list:
                for utterance in utterances_list:
                    start_time = format_time(utterance.get('start', 0))
                    translated_text = utterance['text']
                    if detected_language != 'en' and google_client:
                        try:
                            translated_text = google_client.generate_content(
                                f"Translate the following into English:\n\n{utterance['text']}"
                            ).text
                        except Exception:
                            pass
                    diarization_output += f"[{start_time}] Speaker {utterance['speaker']}: {translated_text}\n"
                st.markdown("### Speaker Diarization")
                st.text_area("Transcript with Speakers", diarization_output, height=200)
            else:
                st.warning("No speaker diarization data was returned by the transcription service.")
                diarization_output = "No speaker data available."
            
            if 'sentiment_analysis_results' in transcript_dict and transcript_dict['sentiment_analysis_results']:
                sentiment_counts = {}
                for sentiment in transcript_dict['sentiment_analysis_results']:
                    sentiment_counts[sentiment['sentiment']] = sentiment_counts.get(sentiment['sentiment'], 0) + 1
                dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                st.markdown("### Sentiment Analysis")
                st.write(f"The overall sentiment of the meeting is: **{dominant_sentiment.capitalize()}**")
            else:
                st.markdown("### Sentiment Analysis")
                st.warning("No sentiment analysis data available.")
                dominant_sentiment = "Not available"

            st.markdown("### Key Topics (powered by Topic Modeling)")
            utterances_for_topics = transcript_dict.get('utterances')

            if utterances_for_topics:
                docs_for_topic_modeling = [utterance['text'] for utterance in utterances_for_topics]
                if docs_for_topic_modeling:
                    try:
                        topics = perform_topic_modeling(docs_for_topic_modeling)
                        st.json(topics)
                    except Exception as e:
                        st.warning("Topic modeling could not be completed.")
                        topics = [{"topic": "Topic modeling failed", "count": 0, "keywords": ""}]
                else:
                    topics = [{"topic": "Not enough data for topic modeling", "count": 0, "keywords": ""}]
            else:
                topics = [{"topic": "Not enough data for topic modeling", "count": 0, "keywords": ""}]

            st.subheader("2. AI-Powered Summaries")
            with st.spinner("Generating summaries with multiple AI models... (including Groq)"):
                summary_1 = get_summary_model_1(raw_transcript)
                summary_2 = get_summary_model_2(raw_transcript)
                summary_3 = get_summary_model_3(raw_transcript)
                summary_groq = get_summary_model_groq(raw_transcript)
    
            
            st.success("Summaries generated!")
            st.markdown("### Consolidated Summary")
            
            consolidated_summary_list = []
            if summary_1:
                st.markdown("### Summary from Perplexity")
                st.text_area("Summary from Perplexity", summary_1, height=130)
                consolidated_summary_list.append(summary_1)
            if summary_2:
                st.markdown("### Summary from OpenAI")
                st.text_area("Summary from OpenAI", summary_2, height=130)
                consolidated_summary_list.append(summary_2)
            if summary_3:
                st.markdown("### Summary from Gemini")
                st.markdown(summary_3)
                consolidated_summary_list.append(summary_3)
            if summary_groq:
                st.markdown("### Summary from Groq AI")
                st.markdown(summary_groq)
                consolidated_summary_list.append(summary_groq)

            if consolidated_summary_list:
                consolidated_summary = "\n\n".join(consolidated_summary_list)
                st.markdown(consolidated_summary)
            else:
                consolidated_summary = "All AI models failed or were missing API keys. Please check your configurations."
                st.error(consolidated_summary)

            st.markdown("---")
            st.header("Full Report 📋")
            
            report_data = {
                "date": time.strftime("%Y-%m-%d"),
                "topic": meeting_topic,
                "consolidated_summary": consolidated_summary,
                "summary_1": summary_1,
                "summary_2": summary_2,
                "summary_3": summary_3,
                "summary_groq": summary_groq,
                "diarization": diarization_output,
                "sentiment": dominant_sentiment.capitalize(),
                "topics": topics
            }
            
            st.markdown(generate_pdf_report(report_data), unsafe_allow_html=True)
        else:
            st.error("Transcription failed to return a valid result.")
            return None
            
    except requests.exceptions.HTTPError as e:
        st.error(f"Failed to transcribe audio: HTTP Error {e.response.status_code}")
        return None
    except requests.exceptions.ConnectionError as e:
        st.error("A network connection issue occurred during transcription.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during transcription: {e}")
        return None

# =========================================================================
# === STEP 4: STREAMLIT UI AND LOGIC ======================================
# =========================================================================

st.title("Varta-Saar: The Ultimate AI Meeting Assistant")
st.markdown("Easily turn your meetings into a detailed, actionable report.")
st.markdown("---")

tab_upload, tab_youtube = st.tabs(["Upload File", "YouTube URL"])
with tab_upload:
    meeting_topic = st.text_input("Meeting Topic", placeholder="e.g., Q3 Marketing Strategy Review", key="upload_topic")
    uploaded_file = st.file_uploader(
        "Upload Meeting Audio/Video (.mp3, .m4a, .mp4, .mov)",
        type=["mp3", "m4a", "mp4", "mov"]
    )
    if st.button("Generate Report", key="upload_button"):
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
                command = [
                    'ffmpeg',
                    '-i', audio_path,
                    '-vn',
                    '-acodec', 'libmp3lame',
                    '-q:a', '2',
                    output_audio_path
                ]
                subprocess.run(command, check=True, capture_output=True, text=True)
                os.remove(audio_path)
                audio_path = output_audio_path
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to extract audio with FFmpeg: {e.stderr}")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                st.stop()
            except FileNotFoundError:
                st.error("FFmpeg not found. Please ensure it is in your packages.txt file.")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                st.stop()

        try:
            run_full_pipeline(audio_path, meeting_topic)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

# --- YouTube URL Tab ---
with tab_youtube:
    st.subheader("YouTube URL 🔗")
    input_url = st.text_input("Enter a YouTube video URL or a direct audio URL (.mp3, .m4a)")
    cookies_file = st.file_uploader("Upload your cookies.txt file (for restricted videos)", type=["txt"])
    meeting_topic_url = st.text_input("Enter the main topic of the content", placeholder="e.g., Apple WWDC Keynote", key="yt_topic")

    if st.button("Generate Report 🚀", key="url_button"):
        if not input_url or not meeting_topic_url:
            st.error("Please provide a valid URL and a meeting topic.")
            st.stop()

        video_path = tempfile.mktemp(suffix=".mp4")
        audio_path = None
        cookies_path = None

        try:
            if "youtube.com" in input_url or "youtu.be" in input_url:
                st.info("YouTube URL detected. Downloading video...")

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

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(input_url, download=False)
                        ydl.download([input_url])
                except Exception as e:
                    st.error(f"An unexpected error occurred during download: {e}")
                    st.stop()

                if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                    st.error("Video download failed. This may be due to the video being private, age-restricted, or region-locked.")
                    st.stop()

                st.info("Extracting audio from the video...")
                audio_path = tempfile.mktemp(suffix=".mp3")
                command = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vn',
                    '-q:a', '0',
                    audio_path
                ]
                subprocess.run(command, check=True, capture_output=True, text=True)

            elif input_url.lower().endswith(('.mp3', '.m4a', '.wav', '.ogg')):
                st.info("Direct audio URL detected. Downloading file...")
                response = requests.get(input_url, stream=True)
                response.raise_for_status()
                audio_path = tempfile.mktemp(suffix=".mp3")
                with open(audio_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Invalid URL provided.")
                st.stop()

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                st.error("Download or extraction failed. Resulting audio file is empty.")
                st.stop()
                
            st.info("Verifying file integrity...")
            integrity_check_command = ['ffmpeg', '-v', 'error', '-i', audio_path, '-f', 'null', '-']
            result = subprocess.run(integrity_check_command, capture_output=True, text=True)

            if result.returncode == 0:
                st.success("File integrity check passed! Proceeding with transcription.")
                run_full_pipeline(audio_path, meeting_topic_url)
            else:
                st.error("The downloaded or extracted audio file is corrupted or in an invalid format.")
                st.stop()

        except subprocess.CalledProcessError as e:
            st.error(f"Failed to extract audio with FFmpeg: {e.stderr}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
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
