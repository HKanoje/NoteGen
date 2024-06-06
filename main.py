import streamlit as st
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
import tempfile
import whisper
import re

# Load environment variables
load_dotenv()

# Configure the API key for Google Gemini
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the prompt for the model
prompt = """
You are a professional note-taker with expertise in distilling key insights from video content. Your task is to generate a comprehensive, yet concise set of notes from the provided video transcript. Focus on the following:

1. Main points
2. Critical information
3. Key takeaways
4. Examples or case studies
5. Quotes or important statements
6. Actionable steps or recommendations

Make sure the notes are well-structured and formatted as bullet points. The total length should not exceed 1000 words. Please summarize the following text:


"""

# Function to extract transcript details from YouTube videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return None
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound):
        return generate_transcript_using_whisper(youtube_video_url)
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    video_id = None
    regex_patterns = [
        r"(?<=v=)[^#\&\?]*",
        r"(?<=be/)[^#\&\?]*",
        r"(?<=embed/)[^#\&\?]*",
        r"(?<=youtu.be/)[^#\&\?]*"
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(0)
            break
    return video_id

# Function to download audio from YouTube video and transcribe it using Whisper
def generate_transcript_using_whisper(youtube_video_url):
    try:
        yt = YouTube(youtube_video_url)
        stream = yt.streams.filter(only_audio=True).first()
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_audio_file:
            stream.download(filename=temp_audio_file.name)
            audio_path = temp_audio_file.name
        
        # Load the Whisper model
        model = whisper.load_model("base")

        # Transcribe the audio file
        result = model.transcribe(audio_path)
        transcript = result['text']
        
        return transcript
    except Exception as e:
        st.error(f"Error generating transcript: {e}")
        return None

# Function to generate summary using Google Gemini API
def generate_gemini_content(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Streamlit app interface
st.title("NoteGen: Automate Your Note-Taking Process.")
youtube_link = st.text_input("Enter Video Link:")

if youtube_link:
    video_id = extract_video_id(youtube_link)
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    else:
        st.error("Invalid YouTube URL. Unable to extract video ID.")

if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)

    if transcript_text:
        summary = generate_gemini_content(transcript_text, prompt)
        if summary:
            st.markdown("## Detailed Notes:")
            st.write(summary)
