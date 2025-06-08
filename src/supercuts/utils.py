import ffmpeg
from pathlib import Path
import base64
import uuid
import json


def extract_audio(video_path: Path, temp_dir: Path) -> Path:
    """Extracts audio from a video file and saves it as an MP3."""
    print("Extracting audio...")
    audio_path = temp_dir / f"{video_path.stem}.mp3"
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

def encode_image(image_path: Path) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_video_clip(video_path: Path, start_time: float, end_time: float, temp_dir: Path) -> Path:
    """Extracts a clip from a video file."""
    clip_path = temp_dir / f"clip_chunk_{uuid.uuid4()}.mp4"
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError("End time must be after start time for clip extraction.")
        
    print(f"Extracting video clip from {start_time:.2f}s to {end_time:.2f}s...")
    try:
        (
            ffmpeg
            .input(str(video_path), ss=start_time)
            .output(
                str(clip_path),
                t=duration,
                vcodec='libx264',
                acodec='aac',
                preset='fast', 
                strict='-2'
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        return clip_path
    except ffmpeg.Error as e:
        print(f"Error extracting video clip: {e.stderr.decode()}")
        raise

def extract_frames(video_path: Path, start_time: float, end_time: float, probe, temp_dir: Path) -> list[Path]:
    """Extracts frames from the start, middle, and end of a time range."""
    video_duration = float(probe['format']['duration'])

    def get_safe_time(t):
        return max(0, min(t, video_duration - 0.1))

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
        frame_path = temp_dir / f"frame_{uuid.uuid4()}.jpg"
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

def generate_clips(video_path: Path, moments: list[dict], output_dir: Path) -> list[dict]:
    """Generates video clips for each identified moment."""
    print(f"Generating {len(moments)} clips...")
    
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

        clip_id = f"clip_{i+1:03d}"
        clip_filename = f"{clip_id}.mp4"
        output_path = output_dir / clip_filename
        
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
    
    moments_output_path = output_dir / 'moments.json'
    with open(moments_output_path, 'w') as f:
        json.dump(updated_moments, f, indent=2)
    print(f"\nMoments metadata saved to: {moments_output_path}")
    
    return updated_moments 