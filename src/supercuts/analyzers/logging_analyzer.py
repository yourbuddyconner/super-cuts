from typing import List, Dict, Any
from pathlib import Path
from .base import BaseAnalyzer

class LoggingAnalyzer(BaseAnalyzer):
    """
    A simple analyzer that logs the moments it receives and passes them on.
    Useful for debugging the pipeline.
    """
    def analyze(self, video_path: Path, transcript: Dict[str, Any], probe: Dict[str, Any], moments: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        print("--- Logging Analyzer ---")
        if not moments:
            print("No moments received from the previous step.")
        else:
            print(f"Received {len(moments)} moments from the previous analyzer.")
            for i, moment in enumerate(moments):
                print(f"  Moment {i+1}:")
                print(f"    Category: {moment.get('category', 'N/A')}")
                print(f"    Time: {moment.get('start_time', 'N/A'):.2f}s - {moment.get('end_time', 'N/A'):.2f}s")
                print(f"    Description: {moment.get('description', 'N/A')[:80]}...")
        
        # This analyzer doesn't modify the moments, it just passes them through.
        return moments 