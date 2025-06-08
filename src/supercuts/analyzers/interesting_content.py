import openai
import ffmpeg
from pathlib import Path
import json
import os
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
TEMP_DIR = Path("./temp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class InterestingContentAnalyzer(BaseAnalyzer):
    """
    Analyzes video content to identify highly-sharable clips for social media.
    Focuses on finding moments that work well as standalone clips for platforms like
    TikTok, Instagram Reels, YouTube Shorts, Twitter/X, etc.
    """

    def _analyze_chunk_openai(self, video_path: Path, start_time: float, end_time: float, segments: list[dict], probe) -> list[dict]:
        """Analyzes a chunk for social media worthy moments using OpenAI."""
        print(f"Analyzing for viral moments from {start_time:.2f}s to {end_time:.2f}s...")

        frame_paths = utils.extract_frames(video_path, start_time, end_time, probe, TEMP_DIR)
        base64_images = [utils.encode_image(p) for p in frame_paths]
        
        for p in frame_paths:
            p.unlink()

        segments_text = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments])

        analysis_prompt = f"""
        Analyze this video segment to identify moments that would make GREAT social media clips.
        The transcript covers {start_time:.2f}s to {end_time:.2f}s.
        
        Look for moments that are:
        - Funny or entertaining
        - Surprising or unexpected
        - Educational "aha" moments
        - Emotional or heartwarming
        - Controversial or thought-provoking
        - Contains quotable statements
        - Shows interesting reactions
        - Has visual appeal or interesting action
        
        IDEAL CLIP CHARACTERISTICS:
        - 15-90 seconds long (MINIMUM 15 seconds, no shorter clips)
        - Can stand alone without context
        - Has a clear beginning and end
        - Contains a "hook" in the first few seconds
        - Would make viewers want to share or comment
        
        Return a JSON array of shareable moments within [{start_time:.2f}, {end_time:.2f}].
        Each moment must have:
        - "platform": Suggested platform (tiktok, instagram, youtube_shorts, twitter, all)
        - "hook": The compelling hook or opening (first 3-5 seconds)
        - "category": Type of content (funny, educational, emotional, reaction, quote, surprising)
        - "start_time": Start time in seconds
        - "end_time": End time in seconds (MINIMUM duration 15 seconds, max 90 seconds)
        - "description": Why this would be shareable
        - "suggested_caption": A catchy caption for social media
        - "hashtags": Array of 3-5 relevant hashtags
        
        Only return moments that are truly shareable.
        If no highly shareable moments, return [].
        
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
                    {"role": "system", "content": "You are a social media expert who knows what content goes viral. Be selective - only identify truly shareable moments."},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
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
                        duration = m['end_time'] - m['start_time']
                        
                        # Skip clips that are too short
                        if duration < 15:
                            continue
                            
                        # Ensure clips aren't too long
                        if duration > 90:
                            m['end_time'] = m['start_time'] + 90
                            
                        if m['start_time'] < end_time + 5 and m['end_time'] > start_time - 5:
                            valid_moments.append(m)
            return valid_moments
        except (json.JSONDecodeError, AttributeError, openai.APIError) as e:
            print(f"Error analyzing chunk {start_time}-{end_time}: {e}")
            return []

    def _analyze_chunk_local(self, start_time: float, end_time: float, segments: list[dict], media_input: Union[Path, list[Path]], model, processor) -> list[dict]:
        """Analyzes a chunk for social media worthy moments using local model."""
        print(f"Analyzing locally for shareable content from {start_time:.2f}s to {end_time:.2f}s...")
        
        segments_text = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments])

        analysis_prompt = f"""
        Find shareable social media clips in this segment ({start_time:.2f}s to {end_time:.2f}s).
        
        Look for: funny, surprising, educational, emotional, or quotable moments.
        Clips should be 15-90 seconds (MINIMUM 15 seconds) and work standalone.
        
        Return JSON array with:
        - category: funny/educational/emotional/reaction/quote/surprising
        - start_time, end_time (in seconds, min duration 15s)
        - description: Why it's shareable
        - suggested_caption: Catchy social caption
        
        Return only the JSON array.
        
        TRANSCRIPT:
        {segments_text}
        """

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                ]
            }
        ]
        
        if isinstance(media_input, list):
            for _ in media_input:
                conversation[0]["content"].append({"type": "image"})
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        images = []
        if isinstance(media_input, list):
            for frame_path in media_input:
                img = Image.open(frame_path)
                images.append(img)

        try:
            inputs = processor(images=images if images else None, text=prompt, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
            output_text = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            output_text = output_text.strip()
            if output_text.startswith("```json"):
                output_text = output_text[7:]
                if output_text.endswith("```"):
                    output_text = output_text[:-3]
            
            moments_data = json.loads(output_text)
            
            valid_moments = []
            if isinstance(moments_data, list):
                for m in moments_data:
                    if isinstance(m, dict) and 'start_time' in m and 'end_time' in m:
                        duration = m['end_time'] - m['start_time']
                        
                        # Skip clips that are too short
                        if duration < 15:
                            continue
                            
                        # Add defaults for missing fields
                        m.setdefault('platform', 'all')
                        m.setdefault('hashtags', [])
                        m.setdefault('hook', m.get('description', '')[:50])
                        valid_moments.append(m)
            return valid_moments
        except Exception as e:
            print(f"Error analyzing local chunk {start_time}-{end_time}: {e}")
            return []

    def analyze(self, video_path: Path, transcript: dict, probe: dict, moments: list[dict] = None) -> list[dict]:
        """Analyzes video to find highly-shareable social media clips."""
        analyzer_type = self.config.get("analyzer", "openai")
        max_clip_duration = self.config.get("max_clip_duration", 90)
        min_clip_duration = self.config.get("min_clip_duration", 15)

        print(f"Finding shareable content for social media (using {analyzer_type} analyzer)...")
        print(f"Settings: clip_duration={min_clip_duration}-{max_clip_duration}s")

        segments = transcript.get('segments')
        if not segments:
            return []

        video_duration = float(probe['format']['duration'])
        chunk_duration = 600  # 10 minute chunks for better context

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
                    frame_paths = utils.extract_frames(video_path, cs, ce, probe, TEMP_DIR)
                    cleanup_paths.extend(frame_paths)
                    chunk_moments = self._analyze_chunk_local(cs, ce, segs, frame_paths, local_model, local_processor)

                    if chunk_moments:
                        print(f"Found {len(chunk_moments)} shareable moments in chunk {cs:.2f}-{ce:.2f}s.")
                        all_moments.extend(chunk_moments)
                except Exception as exc:
                    print(f'Chunk {cs:.2f}-{ce:.2f}s generated an exception: {exc}')
                finally:
                    for p in cleanup_paths:
                        if p.exists():
                            p.unlink()
        else:  # OpenAI
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
                            print(f"Found {len(chunk_moments)} shareable moments in chunk {cs:.2f}-{ce:.2f}s.")
                            all_moments.extend(chunk_moments)
                    except Exception as exc:
                        print(f'Chunk {cs:.2f}-{ce:.2f}s generated an exception: {exc}')
        
        # Filter by clip duration
        filtered_moments = []
        for m in all_moments:
            duration = m.get('end_time', 0) - m.get('start_time', 0)
            
            # Skip clips that are too short
            if duration < min_clip_duration:
                print(f"  Skipping clip ({duration:.1f}s) - too short (min {min_clip_duration}s)")
                continue
                
            if duration <= max_clip_duration:
                filtered_moments.append(m)
            else:
                # Trim clips that are too long
                m['end_time'] = m['start_time'] + max_clip_duration
                filtered_moments.append(m)
        
        # Sort by start time
        filtered_moments.sort(key=lambda x: x.get('start_time', 0))
        
        # Remove overlapping clips (keep first ones)
        final_moments = []
        for current in filtered_moments:
            overlaps = False
            for existing in final_moments:
                if (current['start_time'] < existing['end_time'] and 
                    current['end_time'] > existing['start_time']):
                    overlaps = True
                    break
            if not overlaps:
                final_moments.append(current)
        
        print(f"\nIdentified {len(final_moments)} highly-shareable clips:")
        for m in final_moments:
            print(f"  - {m.get('category', 'general')}: {m.get('suggested_caption', '')[:60]}...")
            print(f"    Duration: {m['end_time'] - m['start_time']:.1f}s, Platform: {m.get('platform', 'all')}")
        
        return final_moments 