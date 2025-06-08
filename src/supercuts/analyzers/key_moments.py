import openai
import ffmpeg
from pathlib import Path
import json
import os
import uuid
from math import ceil
import base64
import concurrent.futures
from typing import Union, List, Dict, Any
from PIL import Image

from .base import BaseAnalyzer
from .. import utils

# --- Local LLM Analysis Imports ---
LOCAL_ANALYSIS_ERROR = None
try:
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    import torch
    LOCAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    LOCAL_ANALYSIS_AVAILABLE = False
    LOCAL_ANALYSIS_ERROR = e


# --- Configuration ---
# These might be better passed in via config, but for now, we'll keep them here
TEMP_DIR = Path("./temp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class KeyMomentsAnalyzer(BaseAnalyzer):
    """
    Analyzes a video to find key moments based on transcript and visual context.
    This is the primary analyzer that generates the initial list of moments.
    """

    def _analyze_chunk_openai(self, video_path: Path, start_time: float, end_time: float, segments: list[dict], probe) -> list[dict]:
        """Analyzes a single chunk of video/transcript with visual context using OpenAI."""
        print(f"Analyzing chunk with OpenAI from {start_time:.2f}s to {end_time:.2f}s...")

        frame_paths = utils.extract_frames(video_path, start_time, end_time, probe, TEMP_DIR)
        base64_images = [utils.encode_image(p) for p in frame_paths]
        
        for p in frame_paths:
            p.unlink()

        segments_text = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments])

        analysis_prompt = f"""
        Analyze the provided transcript segment and corresponding video frames to identify key moments.
        The transcript covers {start_time:.2f}s to {end_time:.2f}s.
        SUGGESTED CATEGORIES: Key Speech, Toast, Funny Moment, Emotional Moment, Important Announcement, Audience Reaction.
        Based on BOTH transcript and images, return a JSON array of moments within this segment.
        Timestamps must be within [{start_time:.2f}, {end_time:.2f}].
        Each moment must have: "category", "start_time", "end_time", "description".
        If no key moments, return an empty array [].
        TRANSCRIPT:
        {segments_text}
        """

        user_content = [{"type": "text", "text": analysis_prompt}]
        if base64_images:
            for img in base64_images:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "low"}})

        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a video analyst. Return a valid JSON array of moment objects."},
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

    def _analyze_chunk_local(self, start_time: float, end_time: float, segments: list[dict], media_input: Union[Path, list[Path]], media_type: str, model, processor) -> list[dict]:
        """Analyzes a single chunk with a local vision model."""
        print(f"Analyzing chunk locally from {start_time:.2f}s to {end_time:.2f}s using {media_type}...")
        
        segments_text = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments])

        analysis_prompt = f"""
        You are a video analyst. Analyze the transcript and video frames to find key moments.
        Transcript time: {start_time:.2f}s to {end_time:.2f}s.
        SUGGESTED CATEGORIES: Key Speech, Toast, Funny Moment, Emotional Moment, Important Announcement, Audience Reaction.
        Based on BOTH transcript and images, return a JSON array of moments ONLY within this segment.
        Timestamps must be within [{start_time:.2f}, {end_time:.2f}].
        Each moment needs: "category", "start_time", "end_time", "description".
        Your response must be ONLY the JSON array. If no moments, return [].
        TRANSCRIPT:
        {segments_text}
        """

        # Prepare conversation in LLaVA format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                ]
            }
        ]
        
        # Add images to the conversation
        if media_type == 'image' and isinstance(media_input, list):
            for _ in media_input:
                conversation[0]["content"].append({"type": "image"})
        
        # Apply chat template
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Load images
        images = []
        if media_type == 'image' and isinstance(media_input, list):
            for frame_path in media_input:
                img = Image.open(frame_path)
                images.append(img)

        try:
            # Process inputs
            inputs = processor(images=images if images else None, text=prompt, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
            # Decode output - skip the input tokens
            output_text = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Clean up JSON response
            output_text = output_text.strip()
            if output_text.startswith("```json"):
                output_text = output_text[7:]
                if output_text.endswith("```"):
                    output_text = output_text[:-3]
            elif output_text.startswith("```"):
                output_text = output_text[3:]
                if output_text.endswith("```"):
                    output_text = output_text[:-3]
            
            moments_data = json.loads(output_text)
            
            valid_moments = []
            if isinstance(moments_data, list):
                for m in moments_data:
                    if isinstance(m, dict) and 'start_time' in m and 'end_time' in m:
                        valid_moments.append(m)
            return valid_moments
        except (json.JSONDecodeError, Exception) as e:
            import traceback
            print(f"Error analyzing local chunk {start_time}-{end_time}: {e}")
            traceback.print_exc()
            print(f"Model output was: {output_text if 'output_text' in locals() else 'Not available'}")
            return []

    def analyze(self, video_path: Path, transcript: dict, probe: dict, moments: list[dict] = None) -> list[dict]:
        """Analyzes transcript in chunks with visual context to find key moments."""
        # Get config values
        analyzer_type = self.config.get("analyzer", "openai") # openai or local
        analysis_mode = self.config.get("analysis_mode", "keyframes") # keyframes or video_clip

        print(f"Finding key moments (using {analyzer_type} analyzer)...")

        segments = transcript.get('segments')
        if not segments:
            return []

        video_duration = float(probe['format']['duration'])
        chunk_duration = 300

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

        if analyzer_type == 'local':
            if not LOCAL_ANALYSIS_AVAILABLE:
                raise RuntimeError(f"Local analysis dependencies not installed: {LOCAL_ANALYSIS_ERROR}")
            
            print("Loading local analysis model (LLaVA-OneVision 0.5B)...")
            local_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            local_processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
            
            for cs, ce, segs in chunks:
                cleanup_paths = []
                try:
                    if analysis_mode == 'video_clip':
                        clip_path = utils.extract_video_clip(video_path, cs, ce, TEMP_DIR)
                        cleanup_paths.append(clip_path)
                        chunk_moments = self._analyze_chunk_local(cs, ce, segs, clip_path, 'video', local_model, local_processor)
                    else:
                        frame_paths = utils.extract_frames(video_path, cs, ce, probe, TEMP_DIR)
                        cleanup_paths.extend(frame_paths)
                        chunk_moments = self._analyze_chunk_local(cs, ce, segs, frame_paths, 'image', local_model, local_processor)

                    if chunk_moments:
                        print(f"Found {len(chunk_moments)} moments in chunk {cs:.2f}-{ce:.2f}s.")
                        all_moments.extend(chunk_moments)
                except Exception as exc:
                     print(f'Chunk {cs:.2f}-{ce:.2f}s generated an exception: {exc}')
                finally:
                    for p in cleanup_paths:
                        if p.exists():
                            p.unlink()
        else: # OpenAI
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_chunk = {
                    executor.submit(self._analyze_chunk_openai, video_path, cs, ce, segs, probe): (cs, ce)
                    for cs, ce, segs in chunks
                }
                for future in concurrent.futures.as_completed(future_to_chunk):
                    cs, ce = future_to_chunk[future]
                    try:
                        chunk_moments = future.result()
                        if chunk_moments:
                            print(f"Found {len(chunk_moments)} moments in chunk {cs:.2f}-{ce:.2f}s.")
                            all_moments.extend(chunk_moments)
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