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
            return transcript.model_dump()
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
                ).model_dump()
            
            for seg in transcript_part['segments']:
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
    
    try:
        content = json.loads(response.choices[0].message.content)
        moments_data = content.get('moments', content)

        if isinstance(moments_data, dict):
            moments_data = [moments_data]

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error decoding moments from AI response: {e}")
        return []

    if not isinstance(moments_data, list):
        print(f"Warning: AI returned an unexpected data type for moments: {type(moments_data)}")
        return []

    print(f"Identified {len(moments_data)} moments.")
    return moments_data

def generate_clips(video_path: Path, moments: list[dict]):
    """Generates video clips for each identified moment."""
    print(f"Generating {len(moments)} clips...")
    for i, moment in enumerate(moments):
        if not isinstance(moment, dict):
            print(f"    Skipping invalid moment (expected a dictionary, got {type(moment)}): {moment}")
            continue

        start_time = moment.get('start_time')
        end_time = moment.get('end_time')

        if start_time is None or end_time is None:
            print(f"    Skipping moment due to missing 'start_time' or 'end_time': {moment}")
            continue

        description = moment.get('description', f'moment_{i+1}').replace(" ", "_").replace("/", "_")
        category = moment.get('category', 'general').replace(" ", "_").replace("/", "_")
        
        safe_description = "".join(c for c in description if c.isalnum() or c in ('_','-')).rstrip()
        safe_category = "".join(c for c in category if c.isalnum() or c in ('_','-')).rstrip()
        
        output_filename = OUTPUT_DIR / f"{i+1:02d}_{safe_category}_{safe_description[:50]}.mp4"
        print(f"  - Creating clip: {output_filename}")
        
        try:
            duration = float(end_time) - float(start_time)
            if duration <= 0:
                print(f"    Skipping clip with invalid duration: {duration}")
                continue
        except (ValueError, TypeError):
            print(f"    Skipping moment with invalid time values: {moment}")
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
    parser = argparse.ArgumentParser(description="Super Cuts")
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
        print(f"Transcript saved to {TEMP_DIR / 'transcript.json'}")
        with open(TEMP_DIR / 'transcript.json', 'w') as f:
            json.dump(transcript, f, indent=2)
        
        moments = analyze_transcript(transcript)
        print(f"Identified moments saved to {TEMP_DIR / 'moments.json'}")
        with open(TEMP_DIR / 'moments.json', 'w') as f:
            json.dump(moments, f, indent=2)

        generate_clips(args.video_path, moments)

        print("\nProcessing complete!")
        print(f"Clips saved in: {OUTPUT_DIR.resolve()}")
    
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
    
    finally:
        if args.cleanup:
            print("Cleaning up temporary files...")
            for f in TEMP_DIR.glob('*'):
                try:
                    f.unlink()
                except OSError as e:
                    print(f"Error removing temp file {f}: {e}")
            try:
                TEMP_DIR.rmdir()
            except OSError as e:
                print(f"Error removing temp directory {TEMP_DIR}: {e}")
        else:
            print(f"Temporary files (transcript, etc.) are preserved in: {TEMP_DIR.resolve()}")

if __name__ == "__main__":
    main() 