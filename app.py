import streamlit as st
import platform
import os
import io
import time
import tempfile
import json
import base64
import re
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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFonts

# =========================================================================
# === STEP 1: CONFIGURATION AND API CLIENT SETUP (Backend Handling) =======
# =========================================================================
st.set_page_config(
    page_title="Varta-Saar: The Ultimate AI Meeting Assistant",
    page_icon="🤖",
    layout="wide"
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
# === STEP 2: HELPER FUNCTIONS & FONT REGISTRATION ========================
# =========================================================================

def read_file(file_path, chunk_size):
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data

def get_audio_url(file_path):
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

def transcribe_audio(audio_url):
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
    data = {"model": "sonar", "messages": [{"role": "user", "content": f"Provide an extensive and detailed summary:\n\n{text}"}], "temperature": 0.2}
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
            messages=[{"role": "user", "content": f"Provide an extensive and structured professional summary with key takeaways:\n\n{text}"}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.warning(f"OpenAI Skipped: {e}")
        return ""

def get_summary_model_3(text):
    if not google_client: return ""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Provide a comprehensive structured summary:\n\n{text}")
        return response.text
    except Exception as e:
        st.warning(f"Gemini Skipped: {e}")
        return ""

def get_summary_model_groq(text):
    if not groq_client: return ""
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are an expert meeting summarizer delivering extensive and organized notes."},
                {"role": "user", "content": f"Please provide an extensive and comprehensive summary of this meeting transcript:\n\n{text}"}
            ],
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.warning(f"Groq API Error: {e}")
        return ""

def perform_topic_modeling(docs):
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
        topics_list.append({
            "topic": row['Name'],
            "count": row['Count'],
            "keywords": ", ".join([word for word, _ in topic_model.get_topic(row['Topic'])])
        })
    return topics_list

def format_time(ms):
    seconds = ms // 1000
    minutes = seconds // 60
    hours = minutes // 60
    seconds %= 60
    minutes %= 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def clean_for_reportlab(text):
    """Sanitizes text and markdown elements for strict PDF XML flowables."""
    if not text:
        return ""
    text = re.sub(r'<br\s*/?>', '<br/>', text)
    text = re.sub(r'\|.*?\|', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    return text

# --- Trebuchet MS Font Registration Handler ---
FONT_NAME = 'Helvetica'
FONT_BOLD = 'Helvetica-Bold'

try:
    font_url = "https://github.com/fogAndWhisky/TestCodeRepo/raw/master/src/fonts/Trebuchet%20MS.ttf"
    font_response = requests.get(font_url, timeout=10)
    if font_response.status_code == 200:
        font_stream = io.BytesIO(font_response.content)
        pdfmetrics.registerFont(TTFonts('TrebuchetMS', font_stream))
        FONT_NAME = 'TrebuchetMS'
        FONT_BOLD = 'TrebuchetMS'
except Exception as font_err:
    # Safe fallback if network stream drops during server boot
    pass

# =========================================================================
# === STEP 3: EXTENSIVE PDF REPORT BUILDER ================================
# =========================================================================

def generate_pdf_report(report_data):
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=40, bottomMargin=40
        )
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName=FONT_BOLD,
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4
        )
        
        meta_style = ParagraphStyle(
            'MetaText',
            parent=styles['Normal'],
            fontName=FONT_NAME,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=2
        )
        
        section_heading = ParagraphStyle(
            'SectionHead',
            parent=styles['Heading2'],
            fontName=FONT_BOLD,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=14,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            'BodyClean',
            parent=styles['Normal'],
            fontName=FONT_NAME,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8
        )

        # Header Section
        story.append(Paragraph("Varta-Saar Comprehensive Executive Meeting Report", title_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
        
        story.append(Paragraph(f"<b>Date:</b> {report_data.get('date', '2026-09-10')}", meta_style))
        story.append(Paragraph(f"<b>Meeting Topic / Context:</b> {report_data.get('topic', 'General')}", meta_style))
        story.append(Spacer(1, 10))

        # Consolidated Summary
        story.append(Paragraph("Consolidated Master Summary", section_heading))
        consolidated_text = clean_for_reportlab(report_data.get('consolidated_summary', ''))
        story.append(Paragraph(consolidated_text.replace('\n', '<br/>'), body_style))
        
        # Individual Successful AI Model Summaries Section
        story.append(Paragraph("Extensive AI Model Insights", section_heading))
        
        if report_data.get('summary_1'):
            story.append(Paragraph("<b>Perplexity AI Summary & Insights:</b>", body_style))
            story.append(Paragraph(clean_for_reportlab(report_data['summary_1']).replace('\n', '<br/>'), body_style))
            story.append(Spacer(1, 6))

        if report_data.get('summary_2'):
            story.append(Paragraph("<b>OpenAI (GPT-4o-mini) Summary:</b>", body_style))
            story.append(Paragraph(clean_for_reportlab(report_data['summary_2']).replace('\n', '<br/>'), body_style))
            story.append(Spacer(1, 6))

        if report_data.get('summary_3'):
            story.append(Paragraph("<b>Google Gemini Summary:</b>", body_style))
            story.append(Paragraph(clean_for_reportlab(report_data['summary_3']).replace('\n', '<br/>'), body_style))
            story.append(Spacer(1, 6))
            
        if report_data.get('summary_groq'):
            story.append(Paragraph("<b>Groq AI Summary:</b>", body_style))
            story.append(Paragraph(clean_for_reportlab(report_data['summary_groq']).replace('\n', '<br/>'), body_style))
            story.append(Spacer(1, 6))

        # Speaker Diarization Section
        if report_data.get('diarization'):
            story.append(Paragraph("Detailed Speaker Diarization & Transcript Logs", section_heading))
            story.append(Paragraph(clean_for_reportlab(report_data['diarization']).replace('\n', '<br/>'), body_style))

        # Sentiment Analysis Section
        story.append(Paragraph("Sentiment Analysis & Tone Metrics", section_heading))
        sentiment_val = report_data.get('sentiment', 'Positive')
        story.append(Paragraph(f"Overall Evaluated Meeting Sentiment: <b>{sentiment_val}</b>", body_style))

        doc.build(story)
        pdf_buffer.seek(0)
        b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')

        return f'<a href="data:application/pdf;base64,{b64_pdf}" download="varta_saar_extensive_report.pdf" style="font-family: TrebuchetMS, sans-serif; background-color: #0f172a; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">📥 Download Extensive Report as PDF</a>'

    except Exception as e:
        st.warning(f"PDF Generation Error: {e}")
        return ""

# =========================================================================
# === STEP 4: MAIN PIPELINE & STREAMLIT UI LOGIC ==========================
# =========================================================================

def run_full_pipeline(file_path, meeting_topic):
    st.markdown("---")
    st.header("Detailed Analysis Pipeline 📊")

    st.subheader("1. Transcription and AI Data Harvesting")
    try:
        with st.spinner("Uploading file and processing speech-to-text with AssemblyAI..."):
            audio_url = get_audio_url(file_path)
            transcript_id = transcribe_audio(audio_url)
            transcript_dict = get_transcription_result(transcript_id)

        if 'text' in transcript_dict and transcript_dict['text']:
            st.success("Transcription complete successfully!")
            
            raw_transcript = transcript_dict.get('text', '')
            detected_language = transcript_dict.get('language_code', 'en')
            st.info(f"Detected language code: **{detected_language}**")

            if detected_language != 'en' and google_client:
                with st.spinner(f"Translating source transcript from {detected_language} to English..."):
                    try:
                        translation_response = google_client.generate_content(
                            f"Translate the following text into English cleanly:\n\n{raw_transcript}"
                        )
                        raw_transcript = translation_response.text
                        st.success("Translation complete!")
                    except Exception as e:
                        st.warning(f"Translation step encountered an issue: {e}. Proceeding with original source.")

            if not raw_transcript or len(raw_transcript.strip()) == 0:
                st.error("The transcription job returned an empty text block.")
                return 

            diarization_output = ""
            utterances_list = transcript_dict.get('utterances')
            if utterances_list:
                for utterance in utterances_list:
                    start_time = format_time(utterance.get('start', 0))
                    translated_text = utterance['text']
                    diarization_output += f"[{start_time}] Speaker {utterance['speaker']}: {translated_text}\n"
                st.markdown("### Speaker Diarization Breakdown")
                st.text_area("Transcript with Speaker IDs", diarization_output, height=180)
            else:
                diarization_output = "No speaker metadata detected."
            
            if 'sentiment_analysis_results' in transcript_dict and transcript_dict['sentiment_analysis_results']:
                sentiment_counts = {}
                for sentiment in transcript_dict['sentiment_analysis_results']:
                    sentiment_counts[sentiment['sentiment']] = sentiment_counts.get(sentiment['sentiment'], 0) + 1
                dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            else:
                dominant_sentiment = "Neutral"

            st.subheader("2. Multi-Model AI Summary Aggregation")
            with st.spinner("Querying Perplexity, OpenAI, Gemini, and Groq models simultaneously..."):
                summary_1 = get_summary_model_1(raw_transcript)
                summary_2 = get_summary_model_2(raw_transcript)
                summary_3 = get_summary_model_3(raw_transcript)
                summary_groq = get_summary_model_groq(raw_transcript)

            st.success("All available model summaries collected!")
            
            consolidated_summary_list = []
            if summary_1:
                st.markdown("### Perplexity Insight Summary")
                st.write(summary_1)
                consolidated_summary_list.append(summary_1)
            if summary_2:
                st.markdown("### OpenAI Insight Summary")
                st.write(summary_2)
                consolidated_summary_list.append(summary_2)
            if summary_3:
                st.markdown("### Gemini Insight Summary")
                st.write(summary_3)
                consolidated_summary_list.append(summary_3)
            if summary_groq:
                st.markdown("### Groq AI Insight Summary")
                st.write(summary_groq)
                consolidated_summary_list.append(summary_groq)

            if consolidated_summary_list:
                consolidated_summary = "\n\n".join(consolidated_summary_list)
            else:
                consolidated_summary = "Summarization models did not return valid strings."

            st.markdown("---")
            st.header("Comprehensive Downloadable Report 📋")
            
            report_data = {
                "date": time.strftime("%Y-%m-%d"),
                "topic": meeting_topic,
                "consolidated_summary": consolidated_summary,
                "summary_1": summary_1,
                "summary_2": summary_2,
                "summary_3": summary_3,
                "summary_groq": summary_groq,
                "diarization": diarization_output,
                "sentiment": dominant_sentiment.capitalize()
            }
            
            st.markdown(generate_pdf_report(report_data), unsafe_allow_html=True)

        else:
            st.error("Transcription pipeline returned an invalid structural payload.")
            return None
            
    except Exception as e:
        st.error(f"Pipeline execution encountered an exception: {e}")
        return None

# Streamlit User Interface Layout
st.title("Varta-Saar: The Ultimate AI Meeting Assistant")
st.markdown("Generate deep, extensive, and multi-model synthesized reports from any audio or video stream.")
st.markdown("---")

tab_upload, tab_youtube = st.tabs(["Upload File", "YouTube URL"])

with tab_upload:
    meeting_topic = st.text_input("Meeting Topic / Title", placeholder="e.g., National Education Policy (NEP) Review", key="upload_topic")
    uploaded_file = st.file_uploader(
        "Upload Meeting File (.mp3, .m4a, .mp4, .mov)",
        type=["mp3", "m4a", "mp4", "mov"]
    )
    if st.button("Generate Extensive Report", key="upload_button"):
        if not uploaded_file or not meeting_topic:
            st.error("Please provide both a source file and a meeting topic title.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name

        if audio_path.endswith((".mp4", ".mov")):
            output_audio_path = tempfile.mktemp(suffix=".mp3")
            try:
                subprocess.run(['ffmpeg', '-i', audio_path, '-vn', '-acodec', 'libmp3lame', '-q:a', '2', output_audio_path], check=True, capture_output=True, text=True)
                os.remove(audio_path)
                audio_path = output_audio_path
            except Exception as e:
                st.error(f"FFmpeg audio extraction failed: {e}")
                if os.path.exists(audio_path): os.remove(audio_path)
                st.stop()

        try:
            run_full_pipeline(audio_path, meeting_topic)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

with tab_youtube:
    input_url = st.text_input("Enter YouTube Video URL")
    cookies_file = st.file_uploader("Upload cookies.txt (Optional for restricted files)", type=["txt"], key="yt_cookie")
    meeting_topic_url = st.text_input("Enter Content Topic Title", placeholder="e.g., NEP 2020 Anniversary Speeches", key="yt_topic")

    if st.button("Generate Extensive YouTube Report 🚀", key="url_button"):
        if not input_url or not meeting_topic_url:
            st.error("Please provide a valid YouTube URL and topic title.")
            st.stop()

        video_path = tempfile.mktemp(suffix=".mp4")
        audio_path = None
        cookies_path = None

        try:
            ydl_opts = {
                'format': 'best[height<=720]/best',
                'outtmpl': video_path,
                'noplaylist': True,
                'ignoreerrors': True,
                'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
            }
            if cookies_file:
                cookies_path = tempfile.mktemp(suffix=".txt")
                with open(cookies_path, "wb") as f:
                    f.write(cookies_file.getbuffer())
                ydl_opts['cookiefile'] = cookies_path

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([input_url])

            audio_path = tempfile.mktemp(suffix=".mp3")
            subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-q:a', '0', audio_path], check=True, capture_output=True, text=True)
            run_full_pipeline(audio_path, meeting_topic_url)

        except Exception as e:
            st.error(f"YouTube processing encountered an error: {e}")
        finally:
            if video_path and os.path.exists(video_path): os.remove(video_path)
            if audio_path and os.path.exists(audio_path): os.remove(audio_path)

st.markdown("---")
st.markdown("© Copyright 2025-2026 by Madhav Agarwal. All rights reserved.")
