# Super Cuts - Technical Design

## 1. Goal & Philosophy

This document outlines a simplified Minimum Viable Product (MVP) for Super Cuts. The primary goal is to create a single, easy-to-run Python script that processes a local video file, identifies key moments, and generates separate video clips for those moments.

The philosophy is **simplicity and local execution**. This MVP deliberately omits production-grade features like web interfaces, databases, user authentication, and cloud infrastructure to focus on the core value proposition: automated clip generation from a long video.

## 2. Core Workflow

The script will execute the following pipeline from the command line:

1.  **Input**: The user provides a path to a local video file.
2.  **Audio Extraction**: The script uses `ffmpeg` to extract the audio track into a temporary MP3 file.
3.  **Transcription**: The audio is sent to OpenAI's Whisper API. To handle large videos, the audio is automatically split into manageable chunks before transcription. The timestamps from each chunk are correctly re-aligned.
4.  **Moment Analysis**: The complete transcript, with accurate timestamps, is sent to a GPT-4 model. A carefully crafted prompt instructs the model to identify key moments (like vows, toasts, funny parts) and return their start and end times in a structured JSON format.
5.  **Clip Generation**: The script uses `ffmpeg` again, this time to precisely cut the original video into smaller clips based on the timestamps identified by the AI.
6.  **Output**: The resulting clips are saved into a local `output_clips/` directory with descriptive filenames.

## 3. Technical Implementation

The MVP will be a single Python script (`process_video.py`). It will rely on a few key libraries for its operations.

### Dependencies

*   **`openai`**: To interact with the Whisper and GPT APIs.
*   **`ffmpeg-python`**: A Python wrapper for the `ffmpeg` command-line tool, used for all video and audio manipulation. (`ffmpeg` must be installed on the system).
*   **`python-dotenv`**: For managing environment variables from a `.env` file.
*   **`argparse`**: For parsing command-line arguments (part of Python's standard library).

A `requirements.txt` file will be provided:
```txt
openai
ffmpeg-python
python-dotenv
```

### Script Structure and Key Functions

The script will be organized into several functions, each responsible for one part of the pipeline. Below is the proposed structure and implementation for `process_video.py`.

```python
# process_video.py
import openai
import ffmpeg
from pathlib import Path
import argparse
import json
import os
import uuid
from math import ceil
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OUTPUT_DIR = Path("./output_clips")
TEMP_DIR = Path("./temp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_FILE_LIMIT_MB = 25

def setup_directories():
    """Creates temporary and output directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

def extract_audio(video_path: Path) -> Path:
    """Extracts audio from a video file and saves it as an MP3."""
    print("Extracting audio...")
    audio_path = TEMP_DIR / f"{video_path.stem}.mp3"
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(audio_path),
                acodec='mp3',
                ac=1,
                ar='16000',
                ab='64k'
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}")
        raise
    print(f"Audio extracted to {audio_path}")
    return audio_path

def transcribe_audio(audio_path: Path) -> dict:
    """Transcribes audio using OpenAI Whisper API. Handles large files by chunking."""
    print("Transcribing audio...")
    
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb < WHISPER_FILE_LIMIT_MB:
        with open(audio_path, "rb") as audio_file:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            return transcript
    else:
        print(f"Audio file is larger than {WHISPER_FILE_LIMIT_MB}MB. Chunking...")
        probe = ffmpeg.probe(str(audio_path))
        duration = float(probe['format']['duration'])
        
        # 20 minutes per chunk
        chunk_duration = 1200
        
        all_segments = []
        for start_time in range(0, int(duration), chunk_duration):
            chunk_path = TEMP_DIR / f"audio_chunk_{uuid.uuid4()}.mp3"
            print(f"Processing audio from {start_time}s to {start_time+chunk_duration}s")

            try:
                (
                    ffmpeg
                    .input(str(audio_path), ss=start_time, t=chunk_duration)
                    .output(str(chunk_path), acodec='copy')
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                print(f"Error creating audio chunk: {e.stderr.decode()}")
                continue

            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            with open(chunk_path, "rb") as audio_chunk_file:
                transcript_part = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_chunk_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            for seg in transcript_part.segments:
                seg['start'] += start_time
                seg['end'] += start_time
                all_segments.append(seg)
            
            chunk_path.unlink()
        
        full_text = " ".join([s['text'] for s in all_segments])
        return {"text": full_text, "segments": all_segments}


def analyze_transcript(transcript: dict) -> list[dict]:
    """Analyzes transcript to find key moments using a GPT model."""
    print("Analyzing transcript for key moments...")
    
    segments_text = "\n".join([
        f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}"
        for s in transcript['segments']
    ])
    
    analysis_prompt = f"""
    Analyze this wedding video transcript and identify key moments.
    
    CATEGORIES TO IDENTIFY:
    1. Wedding vows
    2. Thank you speeches
    3. Toasts
    4. Funny moments or jokes
    5. Emotional or heartfelt moments
    
    TRANSCRIPT:
    {segments_text}
    
    Return a JSON array of moments. Each moment should be an object with:
    - "category": (string, one of the categories above)
    - "start_time": (float, in seconds)
    - "end_time": (float, in seconds)
    - "description": (string, a brief description of the moment)
    
    Example response:
    [
        {{
            "category": "Toasts",
            "start_time": 120.5,
            "end_time": 185.2,
            "description": "A toast from the best man."
        }}
    ]
    """
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are an expert video analyst. Your task is to find meaningful moments in a video transcript and provide timestamps. Return a valid JSON array."},
            {"role": "user", "content": analysis_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )
    
    # The response is a JSON object which might have a key. Let's assume it has a 'moments' key.
    # If not, we'll try to load the content directly.
    try:
        content = json.loads(response.choices[0].message.content)
        moments_data = content.get('moments', content) # Handle both { "moments": [...] } and [...]
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error decoding moments from AI response: {e}")
        return []

    print(f"Identified {len(moments_data)} moments.")
    return moments_data

def generate_clips(video_path: Path, moments: list[dict]):
    """Generates video clips for each identified moment."""
    print(f"Generating {len(moments)} clips...")
    for i, moment in enumerate(moments):
        start_time = moment['start_time']
        end_time = moment['end_time']
        description = moment.get('description', f'moment_{i+1}').replace(" ", "_").replace("/", "_")
        category = moment.get('category', 'general').replace(" ", "_").replace("/", "_")
        
        safe_description = "".join(c for c in description if c.isalnum() or c in ('_','-')).rstrip()
        safe_category = "".join(c for c in category if c.isalnum() or c in ('_','-')).rstrip()
        
        output_filename = OUTPUT_DIR / f"{i+1:02d}_{safe_category}_{safe_description[:50]}.mp4"
        print(f"  - Creating clip: {output_filename}")
        
        duration = end_time - start_time
        if duration <= 0:
            print(f"    Skipping clip with invalid duration: {duration}")
            continue
        
        try:
            (
                ffmpeg
                .input(str(video_path), ss=start_time)
                .output(
                    str(output_filename),
                    t=duration,
                    vcodec='libx264',
                    acodec='aac',
                    pix_fmt='yuv420p',
                    movflags='faststart',
                    preset='medium',
                    crf=23
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(f"Error generating clip for moment {i+1}: {e.stderr.decode()}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Super Cuts MVP")
    parser.add_argument("video_path", type=Path, help="Path to the video file to process.")
    parser.add_argument("--cleanup", action="store_true", help="Clean up the temporary directory after processing.")
    args = parser.parse_args()

    if not args.video_path.exists():
        print(f"Error: Video file not found at {args.video_path}")
        return

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
        
    setup_directories()
    
    try:
        audio_path = extract_audio(args.video_path)
        transcript = transcribe_audio(audio_path)
        with open(TEMP_DIR / 'transcript.json', 'w') as f:
            json.dump(transcript, f, indent=2)
        
        moments = analyze_transcript(transcript)
        with open(TEMP_DIR / 'moments.json', 'w') as f:
            json.dump(moments, f, indent=2)

        generate_clips(args.video_path, moments)

        print("\nProcessing complete!")
        print(f"Clips saved in: {OUTPUT_DIR.resolve()}")
    
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
    
    finally:
        print("Cleaning up temporary files...")
        for f in TEMP_DIR.glob('*'):
            try:
                f.unlink()
            except OSError as e:
                print(f"Error removing temp file {f}: {e}")
        if args.cleanup:
            try:
                TEMP_DIR.rmdir()
            except OSError as e:
                print(f"Error removing temp directory {TEMP_DIR}: {e}")

if __name__ == "__main__":
    main()
```

### Example Prompt for Moment Analysis

The quality of moment detection heavily relies on the prompt sent to the GPT model. The script will use a structured prompt like this to ensure the model returns data in a predictable format that the script can easily parse.

```
Analyze this wedding video transcript and identify key moments.

CATEGORIES TO IDENTIFY:
1. Wedding vows
2. Thank you speeches
3. Toasts
4. Funny moments or jokes
5. Emotional or heartfelt moments

TRANSCRIPT:
[0.52-5.88] And now, the moment we've all been waiting for, the exchange of vows.
[6.20-15.50] John, do you take Mary to be your lawfully wedded wife?
...

Return a JSON array of moments. Each moment should be an object with:
- "category": (string, one of the categories above)
- "start_time": (float, in seconds)
- "end_time": (float, in seconds)
- "description": (string, a brief description of the moment)
```

## 4. Setup and Usage

### Prerequisites

1.  **Python 3.8+**
2.  **`ffmpeg`**: Must be installed and accessible in the system's PATH.
3.  **OpenAI API Key**: An active OpenAI account and API key are required.

### Setup Steps

1.  **Create Files**: Save the Python code above as `process_video.py`. Create a `requirements.txt` file with the specified dependencies.
2.  **Create `.env` file**: Create a file named `.env` in the same directory as the script and add your OpenAI API key to it:
    ```
    OPENAI_API_KEY="your-api-key-here"
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

The script is executed from the command line, pointing to the video file to be processed.

```bash
python process_video.py /path/to/my/wedding_video.mp4
```

To automatically clean up the temporary directory after processing, add the `--cleanup` flag:

```bash
python process_video.py /path/to/my/wedding_video.mp4 --cleanup
```

The script will print its progress to the console. Upon completion, the generated clips will be available in the `output_clips` directory. By default, temporary files like the transcript and extracted audio will be kept in the `temp` directory for inspection. 