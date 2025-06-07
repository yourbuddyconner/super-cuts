# Super Cuts

![Super Cuts](./hero.png)

Super Cuts is a powerful tool for video editors and creatives that automates finding and extracting key moments from long videos, such as wedding films or interviews. It streamlines the post-production workflow, saving hours of manual logging and review.

It uses OpenAI's Whisper API to create a time-stamped transcript of your video's audio. Then, a GPT model analyzes the transcript to identify significant moments like vows, toasts, and speeches. Finally, Super Cuts automatically generates separate video clips for each key event, ready for you to use in your edit.

## Key Features

-   **Automatic Audio Transcription**: Transcribes video audio using the Whisper API.
-   **Intelligent Moment Detection**: Uses a GPT model to identify key moments from the transcript.
-   **Automatic Clip Generation**: Creates individual video clips for each identified moment using FFmpeg.
-   **Large File Support**: Handles videos larger than Whisper's 25MB limit by automatically chunking the audio.
-   **Local First**: Runs entirely on your local machine, with no need for web servers or databases.

## Prerequisites

Before you begin, ensure you have the following installed:

-   **Python 3.8+**
-   **FFmpeg**: This must be installed on your system and accessible from your command line.
    -   On macOS (with [Homebrew](https://brew.sh/)): `brew install ffmpeg`
    -   On Windows (with [Chocolatey](https://chocolatey.org/)): `choco install ffmpeg`
    -   On Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`
-   **OpenAI API Key**: You need an active OpenAI account and an API key with credits.

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a `.env` File**:
    Create a file named `.env` in the root of the project directory. This file will store your OpenAI API key.
    ```
    OPENAI_API_KEY="your-api-key-here"
    ```

3.  **Install Dependencies**:
    Install the required Python packages using `pip`.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Script

The script is executed from the command line. You must provide the path to the video file you want to process.

**Basic Usage**:
```bash
python process_video.py /path/to/your/video.mp4
```

This will run the full pipeline and leave the intermediate files (extracted audio, transcript, etc.) in the `temp/` directory for you to inspect.

**Cleaning Up Temporary Files**:
If you want the script to automatically delete the `temp/` directory after it finishes, use the `--cleanup` flag.

```bash
python process_video.py /path/to/your/video.mp4 --cleanup
```

## Output

-   **`output_clips/`**: This directory is where the generated video clips will be saved. Each clip is named with a sequential number (e.g., `clip_001.mp4`, `clip_002.mp4`). A `moments.json` file is also saved here, containing the metadata for each clip, including its description and category.
-   **`temp/`**: This directory contains intermediate files created during processing. By default, these files are kept for debugging or inspection. It includes:
    -   `[video-name].mp3`: The extracted audio from the video.
    -   `transcript.json`: The full transcript from Whisper.

## Technologies Used

-   **Python**: The core language for the script.
-   **OpenAI API**:
    -   **Whisper**: For audio-to-text transcription.
    -   **GPT-4-Turbo**: For analyzing the transcript and identifying moments.
-   **FFmpeg**: For all video and audio processing, including extraction and clipping.
-   **`ffmpeg-python`**: A Python wrapper for FFmpeg.
-   **`python-dotenv`**: For managing environment variables. 