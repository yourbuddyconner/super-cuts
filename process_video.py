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
import base64
import concurrent.futures

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

def encode_image(image_path: Path) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_frames(video_path: Path, start_time: float, end_time: float, probe) -> list[Path]:
    """Extracts frames from the start, middle, and end of a time range."""
    video_duration = float(probe['format']['duration'])

    def get_safe_time(t):
        return max(0, min(t, video_duration - 0.1))

    # Use a set to avoid duplicate frame times if the segment is short
    if end_time <= start_time:
        frame_times = {get_safe_time(start_time)}
    else:
        frame_times = {
            get_safe_time(start_time),
            get_safe_time(start_time + (end_time - start_time) / 2),
            get_safe_time(end_time)
        }
    
    frame_paths = []
    for i, t in enumerate(sorted(list(frame_times))):
        frame_path = TEMP_DIR / f"frame_{uuid.uuid4()}.jpg"
        try:
            (
                ffmpeg
                .input(str(video_path), ss=t)
                .output(str(frame_path), vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            frame_paths.append(frame_path)
        except ffmpeg.Error as e:
            print(f"Error extracting frame at {t}s: {e.stderr.decode()}")

    return frame_paths

def analyze_chunk(video_path: Path, start_time: float, end_time: float, segments: list[dict], probe) -> list[dict]:
    """Analyzes a single chunk of video/transcript with visual context."""
    print(f"Analyzing chunk from {start_time:.2f}s to {end_time:.2f}s...")

    frame_paths = extract_frames(video_path, start_time, end_time, probe)
    base64_images = [encode_image(p) for p in frame_paths]
    
    # Clean up frames immediately after encoding
    for p in frame_paths:
        p.unlink()

    segments_text = "\n".join([
        f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}"
        for s in segments
    ])

    analysis_prompt = f"""
    Analyze the provided transcript segment and corresponding video frames (start, middle, and end of the segment) to identify key moments.

    The transcript covers the time from {start_time:.2f}s to {end_time:.2f}s of the video. The frames provide visual context.

    SUGGESTED CATEGORIES (adapt or create new ones based on content):
    - Key Speech or Presentation
    - Toast or Tribute
    - Funny Moment or Joke
    - Emotional or Heartfelt Moment
    - Important Announcement
    - Audience Reaction

    TRANSCRIPT SEGMENT:
    {segments_text}

    Based on BOTH the transcript and images, return a JSON array of moments found ONLY within this segment.
    Timestamps must be within [{start_time:.2f}, {end_time:.2f}].

    Each moment must have: "category", "start_time", "end_time", "description".
    If no key moments are found, return an empty array [].
    """

    user_content = [{"type": "text", "text": analysis_prompt}]
    if base64_images:
        for img in base64_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "low"}
            })

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a video analyst finding key moments. Return a valid JSON array of moment objects."},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=4000
        )
        content = json.loads(response.choices[0].message.content)
        moments_data = content.get('moments', content)

        if isinstance(moments_data, dict):
            moments_data = [moments_data]

        valid_moments = []
        if isinstance(moments_data, list):
            for m in moments_data:
                if isinstance(m, dict) and 'start_time' in m and 'end_time' in m:
                    if m['start_time'] < end_time + 5 and m['end_time'] > start_time - 5:
                        valid_moments.append(m)
                    else:
                        print(f"Warning: AI returned a moment outside of chunk time range: {m}")
        return valid_moments

    except (json.JSONDecodeError, AttributeError, openai.APIError) as e:
        print(f"Error analyzing chunk {start_time}-{end_time}: {e}")
        return []

def analyze_transcript(transcript: dict, video_path: Path, probe) -> list[dict]:
    """Analyzes transcript in chunks with visual context to find key moments."""
    print("Analyzing transcript for key moments (with visual context)...")

    segments = transcript.get('segments')
    if not segments:
        return []

    video_duration = float(probe['format']['duration'])
    chunk_duration = 300  # seconds

    chunks = []
    for i in range(0, int(ceil(video_duration)), chunk_duration):
        chunk_start = float(i)
        chunk_end = min(float(i + chunk_duration), video_duration)
        
        chunk_segments = [s for s in segments if s['start'] < chunk_end and s['end'] > chunk_start]
        if not chunk_segments:
            continue
            
        actual_chunk_start = min(s['start'] for s in chunk_segments)
        actual_chunk_end = max(s['end'] for s in chunk_segments)
        chunks.append((actual_chunk_start, actual_chunk_end, chunk_segments))

    all_moments = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_chunk = {
            executor.submit(analyze_chunk, video_path, cs, ce, segs, probe): (cs, ce)
            for cs, ce, segs in chunks
        }
        for future in concurrent.futures.as_completed(future_to_chunk):
            cs, ce = future_to_chunk[future]
            try:
                moments = future.result()
                if moments:
                    print(f"Found {len(moments)} moments in chunk {cs:.2f}-{ce:.2f}s.")
                    all_moments.extend(moments)
            except Exception as exc:
                print(f'Chunk {cs:.2f}-{ce:.2f}s generated an exception: {exc}')
    
    if not all_moments:
        return []
        
    all_moments.sort(key=lambda x: x['start_time'])
    merged_moments = [all_moments[0]]
    for current_moment in all_moments[1:]:
        last_moment = merged_moments[-1]
        if current_moment['start_time'] < last_moment['end_time'] and \
           current_moment.get('category') == last_moment.get('category'):
            last_moment['end_time'] = max(last_moment['end_time'], current_moment['end_time'])
            last_moment['description'] += " " + current_moment.get('description', '')
        else:
            merged_moments.append(current_moment)

    print(f"Identified {len(merged_moments)} total moments after merging.")
    return merged_moments

def generate_clips(video_path: Path, moments: list[dict]):
    """Generates video clips for each identified moment."""
    print(f"Generating {len(moments)} clips...")
    
    # Update moments with clip IDs and filenames
    updated_moments = []
    
    for i, moment in enumerate(moments):
        if not isinstance(moment, dict):
            print(f"    Skipping invalid moment (expected a dictionary, got {type(moment)}): {moment}")
            continue

        start_time = moment.get('start_time')
        end_time = moment.get('end_time')

        if start_time is None or end_time is None:
            print(f"    Skipping moment due to missing 'start_time' or 'end_time': {moment}")
            continue

        # Generate simple clip ID and filename
        clip_id = f"clip_{i+1:03d}"
        clip_filename = f"{clip_id}.mp4"
        output_path = OUTPUT_DIR / clip_filename
        
        # Add clip metadata to moment
        moment['clip_id'] = clip_id
        moment['clip_filename'] = clip_filename
        
        print(f"  - Creating {clip_id}: {moment.get('category', 'general')} - {moment.get('description', '')[:50]}...")
        
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
                    str(output_path),
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
            moment['generated'] = True
            updated_moments.append(moment)
        except ffmpeg.Error as e:
            print(f"Error generating clip for {clip_id}: {e.stderr.decode()}")
            moment['generated'] = False
            moment['error'] = str(e.stderr.decode())
            updated_moments.append(moment)
            continue
    
    # Save updated moments with clip information to output directory
    moments_output_path = OUTPUT_DIR / 'moments.json'
    with open(moments_output_path, 'w') as f:
        json.dump(updated_moments, f, indent=2)
    print(f"\nMoments metadata saved to: {moments_output_path}")
    
    return updated_moments

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
    
    transcript_path = TEMP_DIR / 'transcript.json'

    try:
        probe = ffmpeg.probe(str(args.video_path))
        
        if transcript_path.exists():
            print(f"Found existing transcript at {transcript_path}, skipping transcription.")
            with open(transcript_path, 'r') as f:
                transcript = json.load(f)
        else:
            audio_path = extract_audio(args.video_path)
            transcript = transcribe_audio(audio_path)
            with open(transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2)
            print(f"Transcript saved to {transcript_path}")
        
        moments = analyze_transcript(transcript, args.video_path, probe)
        
        # Save initial moments to temp directory for debugging
        moments_temp_path = TEMP_DIR / 'moments_raw.json'
        with open(moments_temp_path, 'w') as f:
            json.dump(moments, f, indent=2)
        print(f"Raw moments saved to {moments_temp_path}")

        # Generate clips and get updated moments with clip metadata
        updated_moments = generate_clips(args.video_path, moments)

        print("\nProcessing complete!")
        print(f"Clips saved in: {OUTPUT_DIR.resolve()}")
        print(f"Moments metadata: {OUTPUT_DIR / 'moments.json'}")
        
        # Print summary
        successful_clips = sum(1 for m in updated_moments if m.get('generated', False))
        print(f"\nSummary: {successful_clips}/{len(moments)} clips generated successfully")
    
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
    
    finally:
        if args.cleanup:
            print("Cleaning up temporary files...")
            for f in TEMP_DIR.glob('*'):
                try:
                    # Don't delete the final JSON files if cleanup is enabled
                    if f.name not in ['transcript.json', 'moments_raw.json']:
                        f.unlink()
                except OSError as e:
                    print(f"Error removing temp file {f}: {e}")
            # The temp dir itself is not removed to preserve the JSON files
        else:
            print(f"Temporary files (transcript, etc.) are preserved in: {TEMP_DIR.resolve()}")

if __name__ == "__main__":
    main() 