# Long-Video-to-Reels Generator

This project is a Streamlit app that lets users upload a video or input a YouTube link to generate a text summary and a highlight reel of the video. It uses T5 for summarization, MoviePy for video editing, and yt-dlp for downloading YouTube videos.

## Features

- Download YouTube videos or upload local videos.
- Generate transcripts for YouTube videos.
- Summarize transcripts with the T5 transformer model.
- Create highlight reels from video clips at 1-minute intervals.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shahinur-alam/Long-Video-to-Reels.git
   cd Long-Video-to-Reels
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run reels_generator_T5.py
   ```

## Usage

- **YouTube Link:** Enter a YouTube URL to download the video, retrieve the transcript, summarize it, and generate a highlight reel.
- **Upload Video:** Upload an MP4 video to create a highlight reel.

Both options generate a video summary and highlight reel, which are stored in the output folder.

## Key Functions

- load_model(): Loads the T5 model for summarization.
- extract_video_id(url): Extracts the YouTube video ID.
- download_video(url, output_path): Downloads a YouTube video.
- get_transcript(video_id): Retrieves the video transcript.
- summarize_text(text): Summarizes the text with T5.
- create_highlight_reel(video_path, timestamps, output_path): Creates a video highlight reel.
