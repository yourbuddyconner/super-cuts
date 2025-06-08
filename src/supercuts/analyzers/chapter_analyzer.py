import openai
import ffmpeg
from pathlib import Path
import json
import os
from math import ceil
import base64
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

class ChapterAnalyzer(BaseAnalyzer):
    """
    Analyzes long-form content to identify chapter boundaries and create a structured outline.
    Designed for podcasts, speeches, lectures, and other long-form content.
    """

    def _analyze_with_openai(self, video_path: Path, transcript: dict, probe: dict) -> list[dict]:
        """Analyzes the full transcript to identify chapter boundaries using OpenAI."""
        print("Analyzing content structure with OpenAI...")
        
        segments = transcript.get('segments', [])
        if not segments:
            return []
        
        # Create a condensed transcript with timestamps
        transcript_text = "\n".join([
            f"[{s['start']:.0f}s] {s['text']}" 
            for i, s in enumerate(segments) 
            if i % 5 == 0  # Sample every 5th segment to keep it manageable
        ])
        
        # Extract key frames at regular intervals
        video_duration = float(probe['format']['duration'])
        sample_interval = max(60, video_duration / 20)  # Sample every minute or 20 samples total
        
        frame_times = [i for i in range(0, int(video_duration), int(sample_interval))][:10]  # Max 10 frames
        frame_paths = []
        base64_images = []
        
        for time in frame_times:
            frames = utils.extract_frames(video_path, time, min(time + 1, video_duration), probe, TEMP_DIR)
            if frames:
                frame_paths.extend(frames)
                base64_images.append(utils.encode_image(frames[0]))
        
        # Clean up frames
        for p in frame_paths:
            p.unlink()
        
        analysis_prompt = f"""
        Analyze this long-form content (podcast/speech/lecture) and identify logical chapter boundaries.
        The content is {video_duration:.0f} seconds long.
        
        Based on the transcript and visual samples, identify 3-8 major chapters/sections.
        Look for:
        - Topic transitions
        - Speaker changes
        - Natural breaks in content
        - Shift in discussion themes
        - Introduction/conclusion sections
        
        Return a JSON array of chapters. Each chapter should have:
        - "title": Descriptive title for the chapter
        - "start_time": Start time in seconds
        - "end_time": End time in seconds  
        - "description": Brief summary of what's covered
        - "key_topics": Array of 2-3 main topics/themes
        
        The first chapter should start at 0 and the last should end at {video_duration:.0f}.
        Chapters should be substantial (usually 2-10 minutes each).
        
        TRANSCRIPT SAMPLE:
        {transcript_text}
        """
        
        user_content = [{"type": "text", "text": analysis_prompt}]
        if base64_images:
            for img in base64_images[:5]:  # Limit to 5 images
                user_content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "low"}
                })
        
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at analyzing long-form content and creating meaningful chapter divisions."
                    },
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000
            )
            
            content = json.loads(response.choices[0].message.content)
            chapters = content.get('chapters', content)
            
            if isinstance(chapters, dict):
                chapters = [chapters]
            
            # Validate and clean chapters
            valid_chapters = []
            if isinstance(chapters, list):
                for ch in chapters:
                    if isinstance(ch, dict) and all(k in ch for k in ['title', 'start_time', 'end_time']):
                        valid_chapters.append(ch)
            
            return sorted(valid_chapters, key=lambda x: x['start_time'])
            
        except Exception as e:
            print(f"Error analyzing with OpenAI: {e}")
            return []

    def _analyze_with_local(self, video_path: Path, transcript: dict, probe: dict) -> list[dict]:
        """Analyzes the full transcript to identify chapter boundaries using local model."""
        print("Loading local analysis model (LLaVA-OneVision 0.5B)...")
        
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        
        segments = transcript.get('segments', [])
        if not segments:
            return []
        
        video_duration = float(probe['format']['duration'])
        
        # Create condensed transcript
        transcript_text = "\n".join([
            f"[{s['start']:.0f}s] {s['text']}" 
            for i, s in enumerate(segments) 
            if i % 10 == 0  # Sample every 10th segment for local model
        ])
        
        # Extract key frames
        sample_interval = max(120, video_duration / 10)  # Fewer samples for local model
        frame_times = [i for i in range(0, int(video_duration), int(sample_interval))][:5]
        
        frame_paths = []
        for time in frame_times:
            frames = utils.extract_frames(video_path, time, min(time + 1, video_duration), probe, TEMP_DIR)
            if frames:
                frame_paths.extend(frames)
        
        analysis_prompt = f"""
        Analyze this content and identify chapter boundaries.
        Duration: {video_duration:.0f} seconds.
        
        Identify 3-6 major chapters with:
        - title: Chapter title
        - start_time: Start in seconds
        - end_time: End in seconds
        - description: What's covered
        
        Return ONLY a JSON array of chapters.
        
        TRANSCRIPT:
        {transcript_text[:2000]}  # Limit for local model
        """
        
        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                ]
            }
        ]
        
        # Add images
        for _ in frame_paths[:3]:  # Limit images for local model
            conversation[0]["content"].append({"type": "image"})
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Load images
        images = []
        for frame_path in frame_paths[:3]:
            img = Image.open(frame_path)
            images.append(img)
        
        try:
            inputs = processor(images=images if images else None, text=prompt, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=800, do_sample=False)
            
            output_text = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Clean up
            for p in frame_paths:
                if p.exists():
                    p.unlink()
            
            # Parse output
            output_text = output_text.strip()
            if output_text.startswith("```json"):
                output_text = output_text[7:]
                if output_text.endswith("```"):
                    output_text = output_text[:-3]
            
            chapters = json.loads(output_text)
            
            if isinstance(chapters, dict):
                chapters = [chapters]
            
            valid_chapters = []
            if isinstance(chapters, list):
                for ch in chapters:
                    if isinstance(ch, dict) and all(k in ch for k in ['title', 'start_time', 'end_time']):
                        valid_chapters.append(ch)
            
            return sorted(valid_chapters, key=lambda x: x['start_time'])
            
        except Exception as e:
            print(f"Error with local analysis: {e}")
            # Clean up on error
            for p in frame_paths:
                if p.exists():
                    p.unlink()
            return []

    def analyze(self, video_path: Path, transcript: dict, probe: dict, moments: list[dict] = None) -> list[dict]:
        """Analyzes content to identify chapter boundaries."""
        analyzer_type = self.config.get("analyzer", "openai")
        
        print(f"Analyzing content structure for chapters (using {analyzer_type})...")
        
        if analyzer_type == "local":
            if not LOCAL_ANALYSIS_AVAILABLE:
                raise RuntimeError(f"Local analysis dependencies not installed: {LOCAL_ANALYSIS_ERROR}")
            chapters = self._analyze_with_local(video_path, transcript, probe)
        else:
            chapters = self._analyze_with_openai(video_path, transcript, probe)
        
        if not chapters:
            # Fallback: Create basic chapters based on duration
            video_duration = float(probe['format']['duration'])
            chapter_duration = min(300, video_duration / 3)  # 5 min chapters or 3 equal parts
            
            chapters = []
            for i in range(0, int(video_duration), int(chapter_duration)):
                chapters.append({
                    "title": f"Part {len(chapters) + 1}",
                    "start_time": float(i),
                    "end_time": min(float(i + chapter_duration), video_duration),
                    "description": f"Content from {i//60}:{i%60:02d} to {min(i+chapter_duration, video_duration)//60:.0f}:{min(i+chapter_duration, video_duration)%60:02d}",
                    "key_topics": []
                })
        
        # Ensure chapters cover the full duration and don't overlap
        if chapters:
            chapters[0]['start_time'] = 0.0
            chapters[-1]['end_time'] = float(probe['format']['duration'])
            
            # Fix any gaps or overlaps
            for i in range(1, len(chapters)):
                if chapters[i]['start_time'] != chapters[i-1]['end_time']:
                    chapters[i]['start_time'] = chapters[i-1]['end_time']
        
        print(f"Identified {len(chapters)} chapters")
        for ch in chapters:
            print(f"  - {ch['title']}: {ch['start_time']:.0f}s - {ch['end_time']:.0f}s")
        
        return chapters 