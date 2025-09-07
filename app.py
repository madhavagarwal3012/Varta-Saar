import streamlit as st
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
from openai import OpenAI
from bertopic import BERTopic
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
from xhtml2pdf import pisa

# =========================================================================
# === STEP 1: CONFIGURATION AND API CLIENT SETUP (Backend Handling) =======
# =========================================================================
st.set_page_config(
    page_title="Varta-Saar: The Ultimate AI Meeting Assistant",
    page_icon="🤖",
)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    google_client = genai.GenerativeModel("gemini-1.5-flash")
    
except KeyError as e:
    st.error(f"Missing API key. Please ensure you have configured all required API keys in your Streamlit secrets.")
    st.stop()

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
    Sends an audio URL for transcription.
    """
    headers = {'authorization': ASSEMBLYAI_API_KEY, 'content-type': 'application/json'}
    data = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "sentiment_analysis": True,
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
        return None

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
    Generates a summary using a powerful language model (Perplexity).
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "llama-3.1-sonar-huge-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert meeting summarizer. Your task is to provide a concise and professional summary of the meeting transcript."
            },
            {
                "role": "user",
                "content": f"Please summarize the following meeting transcript:\n\n{text}"
            }
        ],
        "temperature": 0.2
    }
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return ""

def get_summary_model_2(text):
    """
    Generates a summary using a second powerful language model (OpenAI).
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert meeting summarizer. Your task is to provide a concise and professional summary of the meeting transcript."},
                {"role": "user", "content": f"Please summarize the following meeting transcript:\n\n{text}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return ""

def get_summary_model_3(text):
    """
    Generates a summary using a third powerful language model (Gemini).
    """
    try:
        response = google_client.generate_content(
            f"You are an expert meeting summarizer. Your task is to provide a concise and professional summary of the following meeting transcript:\n\n{text}"
        )
        return response.text
    except Exception as e:
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
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .neutral {{ color: orange; }}
                    .timestamp {{ color: #666; font-size: 0.9em; }}
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
                    <h3>Summary from AI Model 1</h3>
                    <pre>{report_data_cleaned['summary_1']}</pre>
                    <h3>Summary from AI Model 2</h3>
                    <pre>{report_data_cleaned['summary_2']}</pre>
                    <h3>Summary from AI Model 3</h3>
                    <pre>{report_data_cleaned['summary_3']}</pre>
                </div>

                <div class="section">
                    <h2>Speaker Diarization</h2>
                    <pre>{report_data_cleaned['diarization']}</pre>
                </div>

                <div class="section">
                    <h2>Sentiment Analysis</h2>
                    <p>Overall Sentiment: <span class="sentiment {report_data_cleaned['sentiment'].lower()}">{report_data_cleaned['sentiment']}</span></p>
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
        pisa_status = pisa.CreatePDF(
            pdf_content,
            dest=pdf_buffer
        )

        if pisa_status.err:
            raise Exception("PDF generation failed.")
            
        pdf_buffer.seek(0)
        b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')

        return f'<a href="data:application/pdf;base64,{b64_pdf}" download="meeting_report.pdf">Download Report as PDF</a>'
    except Exception as e:
        st.warning("An error occurred while generating the full PDF report. A simplified version has been created instead.")
        
        fallback_content = f"""
            <html>
            <head><title>Simplified Meeting Report</title></head>
            <body>
                <h1>Simplified Varta-Saar Meeting Report</h1>
                <h2>Consolidated Summary</h2>
                <pre>{clean_text(report_data['consolidated_summary'])}</pre>
            </body>
            </html>
        """
        
        fallback_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(
            fallback_content,
            dest=fallback_buffer
        )
        fallback_buffer.seek(0)
        b64_pdf_fallback = base64.b64encode(fallback_buffer.read()).decode('utf-8')

        return f'<a href="data:application/pdf;base64,{b64_pdf_fallback}" download="simplified_report.pdf">Download Simplified Report as PDF</a>'

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
    with st.spinner("Uploading file and transcribing audio..."):
        audio_url = get_audio_url(file_path)
        transcript_id = transcribe_audio(audio_url)
        transcript = get_transcription_result(transcript_id)

    if transcript:
        st.success("Transcription complete!")
        
        # New: Language Detection and Translation
        detected_language = transcript.get('language_code', 'en')
        st.info(f"Detected language: **{detected_language}**")

        raw_transcript = transcript.get('text', '')
        if detected_language != 'en':
            with st.spinner(f"Translating the transcript from {detected_language} to English..."):
                # Use one of your powerful LLMs to perform the translation
                try:
                    translation_response = google_client.generate_content(
                        f"Translate the following text into English:\n\n{raw_transcript}"
                    )
                    raw_transcript = translation_response.text
                    st.success("Translation complete!")
                except Exception as e:
                    st.warning(f"Translation failed: {e}. Proceeding with original transcript.")

        # Speaker Diarization with Timestamps
        diarization_output = ""
        utterances_list = transcript.get('utterances')
        if utterances_list:
            for utterance in utterances_list:
                start_time = format_time(utterance.get('start', 0))
                # New: Translate individual utterances for the diarization output
                if detected_language != 'en':
                    try:
                        translated_text = google_client.generate_content(
                            f"Translate the following into English:\n\n{utterance['text']}"
                        ).text
                    except Exception:
                        translated_text = utterance['text']
                else:
                    translated_text = utterance['text']

                diarization_output += f"[{start_time}] Speaker {utterance['speaker']}: {translated_text}\n"

            st.markdown("### Speaker Diarization")
            st.text_area("Transcript with Speakers", diarization_output, height=200)
        else:
            st.warning("No speaker diarization data was returned by the transcription service. This may be due to a transcription failure or very short audio.")
            diarization_output = "No speaker data available."
            
        # Sentiment Analysis
        # ... (This section remains the same, it will run on the translated text if applicable) ...
        if 'sentiment_analysis_results' in transcript and transcript['sentiment_analysis_results']:
            sentiment_counts = {}
            for sentiment in transcript['sentiment_analysis_results']:
                sentiment_counts[sentiment['sentiment']] = sentiment_counts.get(sentiment['sentiment'], 0) + 1
            
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            st.markdown("### Sentiment Analysis")
            st.write(f"The overall sentiment of the meeting is: **{dominant_sentiment.capitalize()}**")
        else:
            st.markdown("### Sentiment Analysis")
            st.warning("No sentiment analysis data available.")
            dominant_sentiment = "Not available"

        # Key Topics
        st.markdown("### Key Topics (powered by Topic Modeling)")
        utterances_for_topics = transcript.get('utterances')
        docs_for_topic_modeling = [utterance['text'] for utterance in utterances_for_topics] if utterances_for_topics else []
        
        if docs_for_topic_modeling:
            topics = perform_topic_modeling(docs_for_topic_modeling)
            st.json(topics)
        else:
            topics = [{"topic": "Not enough data for topic modeling", "count": 0, "keywords": ""}]
            st.warning("Not enough data to perform topic modeling. This may be due to a transcription failure.")

        # Summarization
        st.subheader("2. AI-Powered Summaries")
        with st.spinner("Generating summaries with multiple AI models..."):
            summary_1 = get_summary_model_1(raw_transcript)
            summary_2 = get_summary_model_2(raw_transcript)
            summary_3 = get_summary_model_3(raw_transcript)
        
        st.success("Summaries generated!")
        st.markdown("### Consolidated Summary")
        
        consolidated_summary_list = []
        if summary_1: 
            st.markdown("### Summary from Model 1")
            st.text_area("Summary from Model 1", summary_1, height=150)
            consolidated_summary_list.append(summary_1)
        if summary_2: 
            st.markdown("### Summary from Model 2")
            st.text_area("Summary from Model 2", summary_2, height=150)
            consolidated_summary_list.append(summary_2)
        if summary_3: 
            st.markdown("### Summary from Model 3")
            st.text_area("Summary from Model 3", summary_3, height=150)
            consolidated_summary_list.append(summary_3)

        if consolidated_summary_list:
            consolidated_summary = "\n\n".join(consolidated_summary_list)
            st.text_area("Final Consolidated Summary", consolidated_summary, height=300)
        else:
            consolidated_summary = "All AI models failed to generate a summary. Please check your API keys and try again."
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
            "diarization": diarization_output,
            "sentiment": dominant_sentiment.capitalize(),
            "topics": topics
        }
        
        st.markdown(generate_pdf_report(report_data), unsafe_allow_html=True)
    else:
        st.error("Transcription failed to return a valid result. The full pipeline cannot be executed.")    
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
                # Direct subprocess call to ffmpeg
                command = [
                    'ffmpeg',
                    '-i', audio_path,
                    '-vn', # no video
                    '-acodec', 'libmp3lame', # use libmp3lame for MP3
                    '-q:a', '2', # audio quality
                    output_audio_path
                ]
                subprocess.run(command, check=True, capture_output=True, text=True)
                os.remove(audio_path) # Clean up original video file
                audio_path = output_audio_path
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to extract audio with FFmpeg: {e.stderr}")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                if os.path.exists(output_audio_path):
                    os.remove(output_audio_path)
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

# --- YouTube Tab ---
with tab_youtube:
    st.subheader("YouTube URL")
    youtube_url = st.text_input("YouTube Video URL")
    st.subheader("Meeting Topic (YouTube)")
    meeting_topic_yt = st.text_input("Enter the main topic of the meeting for the YouTube video", placeholder="e.g., Apple WWDC 2024 Keynote")

    if st.button("Generate Report 🚀", key="youtube_button"):
        if not youtube_url or not meeting_topic_yt:
            st.error("Please provide both a YouTube URL and a meeting topic.")
            st.stop()
        
        st.info("Downloading and processing audio from YouTube video...")
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                temp_file_path = tmp_file.name
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
                    'outtmpl': temp_file_path,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            run_full_pipeline(temp_file_path, meeting_topic_yt)
        except Exception as e:
            st.error(f"Failed to download or process YouTube video: {e}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    st.warning(f"Failed to remove temporary file: {e}")
        
# =========================================================================
# === COPYRIGHT NOTICE ====================================================
# =========================================================================

st.markdown("---")
st.markdown("© Copyright 2025 by Madhav Agarwal. All rights reserved.")
