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
-   **OpenAI API Key**: You need an active OpenAI account and an API key with credits.

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourbuddyconner/super-cuts.git
    cd super-cuts
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

If you've installed the binary, you can run it directly. If you're running from the source, use `python3`.

**Installed Binary Usage**:
```bash
supercuts /path/to/your/video.mp4
```

**Running from Source**:
```bash
python3 process_video.py /path/to/your/video.mp4
```

This will run the full pipeline and leave the intermediate files (extracted audio, transcript, etc.) in the `temp/` directory for you to inspect.

**Cleaning Up Temporary Files**:
If you want the script to automatically delete the `temp/` directory after it finishes, use the `--cleanup` flag.

```bash
supercuts /path/to/your/video.mp4 --cleanup
```

## For Developers

### Building the Binary

To create a new distributable binary, run the build script. This requires `python3`, `pip`, and all the packages in `requirements.txt`.

```bash
bash build.sh
```

The final binary will be located in the `dist/` directory.

### Automated Releases

This repository is configured with a GitHub Action that automates the release process. When you push a new tag in the format `v*.*.*` (e.g., `v1.0.1`), the action will:

1.  Create a new GitHub Release.
2.  Build binaries for both macOS and Linux.
3.  Upload the `supercuts-macos` and `supercuts-linux` binaries to the release as assets.

The `install.sh` script will automatically download the correct binary for the user's platform.

### Cutting a New Release

To create a new release, go to the **Actions** tab of the repository, select the **Cut New Release Tag** workflow, and click **Run workflow**. You will be prompted to enter a new version tag (e.g., `v1.0.2`). This will create and push the new tag, which in turn will trigger the automated release build.

## Output

-   **`output_clips/`**: This directory is where the generated video clips are saved. Each clip is named with its category and description (e.g., `01_Toasts_A_toast_from_the_best_man.mp4`).
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