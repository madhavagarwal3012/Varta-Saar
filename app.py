import streamlit as st
import os
import io
import time
import tempfile
import json
import base64
from pathlib import Path
import pydub
import requests
import yt_dlp
from openai import OpenAI
from bertopic import BERTopic
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer

# =========================================================================
# === STEP 1: CONFIGURATION AND API CLIENT SETUP (Backend Handling) =======
# =========================================================================

# Set Streamlit page configuration
st.set_page_config(
    page_title="Varta-Saar: The Ultimate AI Meeting Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize API clients from Streamlit secrets
ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Fix for the Google Gemini API client
genai.configure(api_key=GEMINI_API_KEY)
google_client = genai.GenerativeModel("gemini-pro")

# =========================================================================
# === STEP 2: HELPER FUNCTIONS ============================================
# =========================================================================

def get_audio_url(file_path):
    """
    Uploads a local file to AssemblyAI's servers for transcription.
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
        st.error(f"Failed to upload audio to AssemblyAI: {response.status_code}")
        st.json(response.json())
        st.stop()
        return None

def read_file(file_path, chunk_size):
    """
    Reads a file in chunks for uploading to AssemblyAI.
    """
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data

def transcribe_audio(audio_url):
    """
    Sends an audio URL to AssemblyAI for transcription.
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
        st.error(f"Failed to submit transcription job to AssemblyAI: {response.status_code}")
        st.json(response.json())
        st.stop()
        return None

def get_transcription_result(transcript_id):
    """
    Polls AssemblyAI for the transcription result.
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

def get_perplexity_summary(text):
    """
    Generates a summary using the Perplexity API.
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "pplx-7b-online",
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
        st.error(f"Perplexity API error: {e}")
        st.stop()
        return None

def get_openai_summary(text):
    """
    Generates a summary using the OpenAI API.
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
        st.error(f"OpenAI API error: {e}")
        st.stop()
        return None

def get_google_summary(text):
    """
    Generates a summary using the Google Gemini API.
    """
    try:
        response = google_client.generate_content(
            f"You are an expert meeting summarizer. Your task is to provide a concise and professional summary of the following meeting transcript:\n\n{text}"
        )
        return response.text
    except Exception as e:
        st.error(f"Google Gemini API error: {e}")
        st.stop()
        return None

def perform_topic_modeling(docs):
    """
    Performs topic modeling on the document list.
    """
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

def generate_pdf_report(report_data):
    """
    Generates a downloadable PDF report.
    """
    pdf_content = f"""
        <html>
        <head>
            <title>Meeting Report</title>
            <style>
                body {{ font-family: 'Arial', sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #1a237e; }}
                .section {{ margin-bottom: 20px; border-left: 5px solid #3f51b5; padding-left: 15px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; white-space: pre-wrap; }}
                .summary {{ background-color: #e8eaf6; padding: 20px; border-radius: 8px; }}
                .sentiment {{ font-weight: bold; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .neutral {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>Varta-Saar Meeting Report</h1>
            <p><strong>Date:</strong> {report_data['date']}</p>
            <p><strong>Meeting Topic:</strong> {report_data['topic']}</p>

            <div class="section summary">
                <h2>Consolidated Summary</h2>
                <pre>{report_data['consolidated_summary']}</pre>
            </div>

            <div class="section">
                <h2>AI Model Summaries</h2>
                <h3>OpenAI Summary</h3>
                <pre>{report_data['openai_summary']}</pre>
                <h3>Google Gemini Summary</h3>
                <pre>{report_data['google_summary']}</pre>
                <h3>Perplexity Summary</h3>
                <pre>{report_data['perplexity_summary']}</pre>
            </div>

            <div class="section">
                <h2>Speaker Diarization</h2>
                <pre>{report_data['diarization']}</pre>
            </div>

            <div class="section">
                <h2>Sentiment Analysis</h2>
                <p>Overall Sentiment: <span class="sentiment {report_data['sentiment'].lower()}">{report_data['sentiment']}</span></p>
            </div>

            <div class="section">
                <h2>Key Topics</h2>
                <ul>
    """
    for topic in report_data['topics']:
        pdf_content += f"<li><b>{topic['topic']}</b>: {topic['keywords']} (Documents: {topic['count']})</li>"
    
    pdf_content += """
                </ul>
            </div>
        </body>
        </html>
    """
    b64_pdf = base64.b64encode(pdf_content.encode('utf-8')).decode('utf-8')
    return f'<a href="data:application/pdf;base64,{b64_pdf}" download="meeting_report.pdf">Download Report as PDF</a>'

# =========================================================================
# === STEP 3: MAIN APPLICATION PIPELINE ===================================
# =========================================================================

def run_full_pipeline(file_path, meeting_topic):
    """
    Runs the full analysis pipeline from transcription to report generation.
    """
    st.markdown("---")
    st.header("Detailed Analysis 📊")

    st.subheader("1. Transcription and Analysis (via AssemblyAI)")
    with st.spinner("Uploading file and transcribing audio..."):
        audio_url = get_audio_url(file_path)
        transcript_id = transcribe_audio(audio_url)
        transcript = get_transcription_result(transcript_id)

    if transcript:
        st.success("Transcription complete!")
        raw_transcript = transcript['text']
        
        # Speaker Diarization
        diarization_output = ""
        if transcript['speaker_labels']:
            for utterance in transcript['utterances']:
                diarization_output += f"Speaker {utterance['speaker']}: {utterance['text']}\n"
            st.markdown("### Speaker Diarization")
            st.text_area("Transcript with Speakers", diarization_output, height=200)

        # Sentiment Analysis
        sentiment_counts = {}
        for sentiment in transcript['sentiment_analysis_results']:
            sentiment_counts[sentiment['sentiment']] = sentiment_counts.get(sentiment['sentiment'], 0) + 1
        
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        st.markdown("### Sentiment Analysis")
        st.write(f"The overall sentiment of the meeting is: **{dominant_sentiment.capitalize()}**")

        # Key Topics
        st.markdown("### Key Topics (via BERTopic)")
        docs_for_topic_modeling = [utterance['text'] for utterance in transcript['utterances']]
        topics = perform_topic_modeling(docs_for_topic_modeling)
        st.json(topics)

        # Summarization
        st.subheader("2. AI-Powered Summaries")
        with st.spinner("Generating summaries with OpenAI, Google Gemini, and Perplexity..."):
            openai_summary = get_openai_summary(raw_transcript)
            google_summary = get_google_summary(raw_transcript)
            perplexity_summary = get_perplexity_summary(raw_transcript)
        
        st.success("Summaries generated!")
        st.markdown("### Consolidated Summary")
        consolidated_summary = openai_summary
        st.text_area("Final Consolidated Summary", consolidated_summary, height=300)

        st.markdown("---")
        st.header("Full Report 📋")
        
        report_data = {
            "date": time.strftime("%Y-%m-%d"),
            "topic": meeting_topic,
            "consolidated_summary": consolidated_summary,
            "openai_summary": openai_summary,
            "google_summary": google_summary,
            "perplexity_summary": perplexity_summary,
            "diarization": diarization_output,
            "sentiment": dominant_sentiment.capitalize(),
            "topics": topics
        }
        
        st.markdown(generate_pdf_report(report_data), unsafe_allow_html=True)
    
# =========================================================================
# === STEP 4: STREAMLIT APP UI ============================================
# =========================================================================

st.title("Varta-Saar: The Ultimate AI Meeting Assistant 🧠")
st.markdown("Easily turn your meetings into a detailed, actionable report.")

st.markdown("---")

tab_upload, tab_youtube = st.tabs(["Upload File", "YouTube URL"])

# --- File Upload Tab ---
with tab_upload:
    st.subheader("Meeting Topic")
    meeting_topic = st.text_input("Enter the main topic of the meeting", placeholder="e.g., Q3 Marketing Strategy Review")
    st.markdown("---")
    
    st.subheader("Upload Meeting Audio/Video (.mp3, .m4a, .mp4, .mov)")
    uploaded_file = st.file_uploader(
        "Drag and drop file here", 
        type=["mp3", "m4a", "mp4", "mov"],
        help="Limit 200MB per file"
    )
    
    if st.button("Generate Report", key="upload_button"):
        if not uploaded_file or not meeting_topic:
            st.error("Please provide both a file and a meeting topic.")
            st.stop()
        
        # Use a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name
        
        # Check if the file is a video and needs audio extraction
        if audio_path.endswith((".mp4", ".mov")):
            try:
                video_clip = VideoFileClip(audio_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_tmp:
                    # Removed the unsupported 'format' argument
                    video_clip.audio.write_audiofile(audio_tmp.name)
                
                video_clip.close() # Explicitly close the clip
                os.remove(audio_path) # Clean up video file
                audio_path = audio_tmp.name
            except Exception as e:
                st.error(f"Failed to extract audio from video: {e}")
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
        
        st.info("Downloading audio from YouTube video...")
        temp_video_path = None
        temp_audio_path = None
        try:
            # Download the YouTube video as an MP4
            temp_video_path = tempfile.mktemp(suffix=".mp4")
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': temp_video_path,
                'noplaylist': True,
                'continue': False,
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            # Convert the downloaded video file to MP3 audio using pydub
            temp_audio_path = tempfile.mktemp(suffix=".mp3")
            audio_segment = pydub.AudioSegment.from_file(temp_video_path)
            audio_segment.export(temp_audio_path, format="mp3")

            run_full_pipeline(temp_audio_path, meeting_topic_yt)

        except Exception as e:
            st.error(f"Failed to download or process YouTube video: {e}")
            st.error("Please ensure FFmpeg is correctly installed and accessible on your system or check the `packages.txt` file on Streamlit Cloud.")
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    st.warning("Failed to remove temporary video file.")
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    st.warning("Failed to remove temporary audio file.")

# =========================================================================
# === COPYRIGHT NOTICE ====================================================
# =========================================================================

st.markdown("---")
st.markdown("© Copyright 2025 by Madhav Agarwal. All rights reserved.")

