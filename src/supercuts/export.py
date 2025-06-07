import opentimelineio as otio
from pathlib import Path
import math

def generate_fcpxml(moments: list[dict], video_path: Path, output_path: Path, probe: dict):
    """
    Generates a Final Cut Pro XML (.fcpxml) file from a list of moments.

    Args:
        moments: A list of moment dictionaries, each with 'start_time' and 'end_time'.
        video_path: The path to the original source video file.
        output_path: The path where the .fcpxml file will be saved.
        probe: The result of ffmpeg.probe() on the source video.
    """
    print(f"Generating FCPXML file at {output_path}...")

    # --- Get Frame Rate from Probe ---
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if not video_stream:
        print("Warning: No video stream found in probe. Defaulting to 24 fps.")
        rate = 24.0
    else:
        frame_rate_str = video_stream.get('r_frame_rate', '24/1') # Use r_frame_rate for precise value
        if '/' in frame_rate_str:
            num, den = map(int, frame_rate_str.split('/'))
            rate = float(num) / float(den) if den != 0 else float(num)
        else:
            rate = float(frame_rate_str)
            
    print(f"Using frame rate: {rate:.3f} fps")

    timeline = otio.schema.Timeline(name="Supercuts Moments")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)

    # FCPX expects a file URI, so we convert the absolute path.
    video_path_uri = video_path.resolve().as_uri()

    # Get the total duration from the probe to set the available range accurately
    video_duration_sec = float(probe.get('format', {}).get('duration', 1_000_000))
    total_frames = math.ceil(video_duration_sec * rate)
    
    available_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, rate),
        duration=otio.opentime.RationalTime(total_frames, rate)
    )

    # Create a single media reference to be shared by all clips from this video.
    # This prevents the "Invalid edit with no respective media" error.
    media_reference = otio.schema.ExternalReference(
        target_url=video_path_uri,
        available_range=available_range
    )

    for moment in moments:
        start_time_sec = moment.get('start_time')
        end_time_sec = moment.get('end_time')

        if start_time_sec is None or end_time_sec is None:
            continue

        # Quantize to the frame grid to avoid "not on an edit frame boundary" errors.
        # We use round() to get the nearest frame, which is more accurate than truncating.
        start_frame = round(start_time_sec * rate)
        end_frame = round(end_time_sec * rate)
        duration_frames = end_frame - start_frame
        
        # A clip must have a duration of at least one frame.
        if duration_frames <= 0:
            continue

        time_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(start_frame, rate),
            duration=otio.opentime.RationalTime(duration_frames, rate)
        )
        
        clip = otio.schema.Clip(
            name=moment.get('description', 'Untitled Moment')[:30],
            media_reference=media_reference, # Reuse the same media reference
            source_range=time_range
        )
        track.append(clip)

    # Make sure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        otio.adapters.write_to_file(timeline, str(output_path))
        print("FCPXML file generated successfully.")
    except Exception as e:
        print(f"Error writing FCPXML file: {e}") 