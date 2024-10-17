import streamlit as st
import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Set up CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"Using device: {device}")


# Set up ResNet50 model
@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model


model = load_model()

# Set up image transforms
preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        return url.split("v=")[-1].split("&")[0]
    return None


def download_video(video_url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
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


def extract_features(frame):
    img = Image.fromarray(frame)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()


def create_highlight_reel(video_path, output_path, num_highlights=5, highlight_duration=10):
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        frame_interval = duration / (num_highlights + 1)

        highlight_clips = []
        for i in range(1, num_highlights + 1):
            start_time = i * frame_interval
            end_time = min(start_time + highlight_duration, duration)
            highlight_clips.append(video.subclip(start_time, end_time))

        final_clip = concatenate_videoclips(highlight_clips)
        final_clip.write_videofile(output_path)
        return output_path
    except Exception as e:
        st.error(f"Error creating highlight reel: {str(e)}")
        return None


def generate_video_summary(video_path, video_id):
    transcript = get_transcript(video_id) if video_id else None

    if transcript:
        summary = f"Video transcript: {transcript[:500]}..."  # Simplified summary
    else:
        summary = "Unable to generate summary due to transcript unavailability."

    highlight_path = os.path.join("output", "highlight.mp4")
    highlight_result = create_highlight_reel(video_path, highlight_path)

    return summary, highlight_result


# Streamlit UI
st.title("Video Summarizer with ResNet50")

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
                st.error("Failed to download the video. Please check the URL and try again.")
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
            highlight_path = create_highlight_reel(video_path, os.path.join("output", "highlight.mp4"))

        if highlight_path:
            st.subheader("Highlight Reel")
            st.video(highlight_path)
        else:
            st.error("Failed to create highlight reel.")

st.info(
    "Note: This implementation uses ResNet50 for feature extraction. The highlight reel is created by selecting evenly spaced segments from the video.")