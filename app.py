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
        
        # Use a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name
        
        # Check if the file is a video and needs audio extraction
        if audio_path.endswith((".mp4", ".mov")):
            try:
                # Use pydub to extract the audio
                audio = AudioSegment.from_file(audio_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_tmp:
                    audio.export(audio_tmp.name, format="mp3")
                os.remove(audio_path) # Clean up the original video file
                audio_path = audio_tmp.name # Update the path to the new audio file
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
