from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

class BaseAnalyzer(ABC):
    """
    Abstract base class for an analyzer.
    Analyzers are chained together in a pipeline, where the output of one
    (a list of 'moments') is the input to the next.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the analyzer with an optional configuration dictionary.

        :param config: Configuration parameters for the analyzer.
        """
        self.config = config or {}

    @abstractmethod
    def analyze(self, video_path: Path, transcript: Dict[str, Any], probe: Dict[str, Any], moments: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs analysis on the video, transcript, and prior moments.

        :param video_path: Path to the video file.
        :param transcript: The full transcript object from Whisper.
        :param probe: FFMPEG probe data for the video.
        :param moments: A list of moments from a previous analyzer step.
                        The first analyzer will receive an empty list.
        :return: A list of new or modified moment dictionaries.
        """
        pass 