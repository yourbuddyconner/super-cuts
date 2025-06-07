# Super Cuts

![Super Cuts](./hero.png)

Super Cuts is a powerful tool for video editors and creatives that automates finding and extracting key moments from long videos, such as wedding films or interviews. It streamlines the post-production workflow, saving hours of manual logging and review.

It uses AI-powered transcription and analysis to create a time-stamped transcript of your video's audio, then intelligently identifies significant moments like vows, toasts, and speeches. Finally, Super Cuts automatically generates separate video clips for each key event and can export a timeline file for your video editor.

## Key Features

-   **Automatic Audio Transcription**: Transcribes video audio using either OpenAI's Whisper API or local models.
-   **Intelligent Moment Detection**: Uses AI models (GPT-4 or local vision models) to identify key moments from the transcript and video frames.
-   **Automatic Clip Generation**: Creates individual video clips for each identified moment using FFmpeg.
-   **Timeline Export**: Generates FCPXML files that can be imported into Final Cut Pro, DaVinci Resolve, or Adobe Premiere Pro.
-   **Local Processing Options**: Run entirely offline with local transcription and analysis models for privacy and cost savings.
-   **Large File Support**: Handles videos larger than Whisper's 25MB limit by automatically chunking the audio.
-   **Simple Installation**: Comes with a one-line command to install the tool on your system.

## Easy Installation (Recommended)

You can install Super Cuts with a single command. This will download the latest version for your operating system (macOS or Linux) and make it available system-wide.

```bash
# Downloads and runs the installer script.
curl -sSL https://raw.githubusercontent.com/yourbuddyconner/super-cuts/main/install.sh | sudo bash
```

After installation, you can run the tool from any directory using the `supercuts` command.

---

## Manual Setup

Before you begin, ensure you have the following installed:

-   **Python 3.8+**
-   **FFmpeg**: This must be installed on your system and accessible from your command line.
    -   On macOS (with [Homebrew](https://brew.sh/)): `brew install ffmpeg`
    -   On Windows (with [Chocolatey](https://chocolatey.org/)): `choco install ffmpeg`
    -   On Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`
-   **OpenAI API Key** (optional): Only needed if using OpenAI's cloud services. You can run entirely locally without an API key.

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourbuddyconner/super-cuts.git
    cd super-cuts
    ```

2.  **Create a `.env` File** (optional):
    Only needed if using OpenAI's services. Create a file named `.env` in the root of the project directory.
    ```
    OPENAI_API_KEY="your-api-key-here"
    ```

3.  **Install Dependencies**:
    Install the required Python packages using `pip`.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Script

If you've installed the binary, you can run it directly. If you're running from the source, use `python3`.

**Basic Usage**:
```bash
supercuts /path/to/your/video.mp4
```

**Running from Source**:
```bash
python3 -m supercuts /path/to/your/video.mp4
```

### Advanced Options

**Local Processing** (no internet required):
```bash
# Use local models for both transcription and analysis
supercuts video.mp4 --transcriber local --analyzer local

# Local analysis with video clips instead of keyframes (more accurate but slower)
supercuts video.mp4 --analyzer local --analysis-mode video_clip
```

**Export Timeline to Video Editor**:
```bash
# Generate an FCPXML file for Final Cut Pro, DaVinci Resolve, or Premiere Pro
supercuts video.mp4 --export-xml my_timeline.fcpxml
```

**Cleanup Temporary Files**:
```bash
supercuts video.mp4 --cleanup
```

### All Available Options

- `--transcriber [openai|local]`: Choose transcription engine (default: openai)
- `--analyzer [openai|local]`: Choose analysis engine (default: openai)
- `--analysis-mode [keyframes|video_clip]`: For local analyzer, how to provide visual context (default: keyframes)
- `--export-xml <filename>`: Generate an FCPXML timeline file
- `--cleanup`: Remove temporary files after processing

## Output

-   **`output_clips/`**: This directory contains:
    -   Individual video clips for each identified moment (e.g., `clip_001.mp4`)
    -   `moments.json`: Metadata about all identified moments including timestamps and descriptions
-   **`temp/`**: This directory contains intermediate files created during processing:
    -   `[video-name].mp3`: The extracted audio from the video
    -   `transcript.json`: The full transcript with timestamps
    -   `moments_raw.json`: Raw moment data before clip generation
-   **`.fcpxml` file** (if requested): Timeline file that can be imported into video editing software

## Technologies Used

-   **Python**: The core language for the script.
-   **AI Models**:
    -   **OpenAI Whisper**: Cloud-based audio transcription
    -   **GPT-4-Turbo**: Cloud-based transcript and video analysis
    -   **Distil-Whisper**: Local audio transcription (via Hugging Face)
    -   **Qwen2.5-VL**: Local vision-language model for video analysis
-   **FFmpeg**: For all video and audio processing, including extraction and clipping.
-   **OpenTimelineIO**: For generating timeline files compatible with video editors.
-   **`ffmpeg-python`**: A Python wrapper for FFmpeg.
-   **`python-dotenv`**: For managing environment variables.

## For Developers

### Building the Binary

To create a new distributable binary, run the build script. This requires `python3`, `pip`, and all the packages in `requirements.txt`.

```bash
bash build.sh
```

The final binary will be located in the `dist/` directory.

### Cutting a New Release

To create a new release, go to the **Actions** tab of the repository, select the **Create Release** workflow, and click **Run workflow**. You will be prompted to enter a new version tag (e.g., `v1.0.2`).

This single action will:

1.  Create and push a new Git tag with your specified version.
2.  Build binaries for both macOS and Linux.
3.  Create a new GitHub Release.
4.  Upload the `supercuts-macos` and `supercuts-linux` binaries to the release as assets.

The `install.sh` script will automatically download the correct binary for the user's platform. 