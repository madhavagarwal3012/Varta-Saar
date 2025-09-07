with tab_youtube:
    st.subheader("YouTube URL")
    youtube_url = st.text_input("YouTube Video URL")
    st.subheader("Meeting Topic (YouTube)")
    meeting_topic_yt = st.text_input("Enter the main topic of the meeting for the YouTube video", placeholder="e.g., Apple WWDC 2024 Keynote")

    if st.button("Generate Report 🚀", key="youtube_button"):
        if not youtube_url or not meeting_topic_yt:
            st.error("Please provide both a YouTube URL and a meeting topic.")
            st.stop()

        st.info("Downloading YouTube video...")
        video_path = None
        audio_path = None
        temp_cookies_path = None
        try:
            # Get cookies from Streamlit secrets
            cookies_content = st.secrets["YOUTUBE_COOKIES"]
            
            # Write the cookies to a temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_cookies_file:
                temp_cookies_file.write(cookies_content)
                temp_cookies_path = temp_cookies_file.name

            # 1. Download the raw video file
            video_path = tempfile.mktemp(suffix=".mp4")
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': video_path,
                'noplaylist': True,
                'ignoreerrors': True,
                'cookiefile': temp_cookies_path, # Use the path to the temporary file
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            if not os.path.exists(video_path):
                st.error("Video download failed.")
                st.stop()

            st.info("Extracting audio from the video...")
            # 2. Use a direct FFmpeg subprocess call to extract the audio
            audio_path = tempfile.mktemp(suffix=".mp3")
            
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-q:a', '0',
                audio_path
            ]
            
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            if os.path.exists(audio_path):
                run_full_pipeline(audio_path, meeting_topic_yt)
            else:
                st.error("Audio extraction failed.")
                st.stop()

        except subprocess.CalledProcessError as e:
            st.error(f"Failed to extract audio with FFmpeg: {e.stderr}")
        except KeyError:
            st.error("Cookies are missing from your app's secrets. Please add them.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if temp_cookies_path and os.path.exists(temp_cookies_path):
                os.remove(temp_cookies_path)
