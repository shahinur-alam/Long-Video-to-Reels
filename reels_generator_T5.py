import streamlit as st
import os
import torch
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Set up CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"Using device: {device}")

# Set up T5 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


tokenizer, model = load_model()


def extract_video_id(url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(?:embed\/)?(?:v\/)?(?:shorts\/)?(?:live\/)?(?:(?!videos)(?!channel)(?!user)(?!playlist)(?!@))([^\?&\n]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def download_video(video_url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        # Uncomment and modify the line below if FFmpeg is not in your system PATH
        # 'ffmpeg_location': r'C:\path\to\ffmpeg\bin',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
            return None
    return output_path


def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None


def summarize_text(text, max_length=150, min_length=50):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def create_highlight_reel(video_path, timestamps, output_path):
    try:
        video = VideoFileClip(video_path)
        clips = [video.subclip(start, end) for start, end in timestamps]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path)
        return output_path
    except Exception as e:
        st.error(f"Error creating highlight reel: {str(e)}")
        return None


def generate_video_summary(video_path, video_id):
    # Get transcript
    transcript = get_transcript(video_id)

    if transcript:
        # Generate summary
        summary = summarize_text(transcript)
    else:
        summary = "Unable to generate summary due to transcript unavailability."

    # Create highlight reel
    video = VideoFileClip(video_path)
    duration = video.duration
    timestamps = [(i * 60, i * 60 + 10) for i in range(int(duration / 60)) if i * 60 + 10 <= duration]

    highlight_path = os.path.join("output", "highlight.mp4")
    highlight_result = create_highlight_reel(video_path, timestamps, highlight_path)

    return summary, highlight_result


# Streamlit UI
st.title("Video Summarizer")

input_type = st.radio("Choose input type:", ("YouTube Link", "Upload Video"))

if input_type == "YouTube Link":
    video_url = st.text_input("Enter YouTube video URL:")
    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            output_path = os.path.join("output", "video.mp4")

            with st.spinner("Downloading video..."):
                result = download_video(video_url, output_path)

            if result:
                st.success("Video downloaded successfully!")

                with st.spinner("Generating summary and highlight reel..."):
                    summary, highlight_path = generate_video_summary(output_path, video_id)

                if summary:
                    st.subheader("Summary")
                    st.write(summary)

                if highlight_path:
                    st.subheader("Highlight Reel")
                    st.video(highlight_path)
                else:
                    st.error("Failed to create highlight reel.")
            else:
                st.error("Failed to download the video. Please check if FFmpeg is installed and in your system PATH.")
        else:
            st.error("Invalid YouTube URL. Please check and try again.")

else:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
    if uploaded_file is not None:
        video_path = os.path.join("output", "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Video uploaded successfully!")

        with st.spinner("Generating highlight reel..."):
            video = VideoFileClip(video_path)
            duration = video.duration
            timestamps = [(i * 60, i * 60 + 10) for i in range(int(duration / 60)) if i * 60 + 10 <= duration]

            highlight_path = os.path.join("output", "highlight.mp4")
            highlight_result = create_highlight_reel(video_path, timestamps, highlight_path)

        if highlight_result:
            st.subheader("Highlight Reel")
            st.video(highlight_result)
        else:
            st.error("Failed to create highlight reel.")

st.info(
    "Note: This is a simplified implementation. For a production-ready tool, consider adding more error handling, improving the UI, and implementing more sophisticated video analysis techniques.")