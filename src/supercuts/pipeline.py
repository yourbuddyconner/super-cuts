import importlib
from pathlib import Path
from typing import List, Dict, Any
from .analyzers.base import BaseAnalyzer

def load_analyzers(analyzer_names: List[str], configs: Dict[str, Dict[str, Any]]) -> List[BaseAnalyzer]:
    """
    Dynamically loads analyzer classes from the 'analyzers' module.
    """
    instances = []
    for name in analyzer_names:
        try:
            # Convert snake_case name to CamelCase class name
            # If the name already ends with "_analyzer", don't append "Analyzer" again
            if name.endswith('_analyzer'):
                # Remove '_analyzer' suffix and convert to CamelCase
                base_name = name[:-9]  # Remove last 9 chars ('_analyzer')
                class_name = "".join(word.capitalize() for word in base_name.split('_')) + "Analyzer"
            else:
                class_name = "".join(word.capitalize() for word in name.split('_')) + "Analyzer"
            
            module = importlib.import_module(f"supercuts.analyzers.{name}")
            analyzer_class = getattr(module, class_name)
            
            analyzer_config = configs.get(name, {})
            instances.append(analyzer_class(config=analyzer_config))
            print(f"Successfully loaded analyzer: {class_name}")
        except (ImportError, AttributeError) as e:
            print(f"Error loading analyzer '{name}': {e}")
            raise
    return instances

def run_pipeline(
    analyzers: List[BaseAnalyzer],
    video_path: Path,
    transcript: Dict[str, Any],
    probe: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Runs the analysis pipeline by executing analyzers in sequence.
    """
    print(f"Starting analyzer pipeline with {len(analyzers)} steps...")
    moments = []  # Start with an empty list of moments
    
    for analyzer in analyzers:
        analyzer_name = analyzer.__class__.__name__
        print(f"--- Running analyzer: {analyzer_name} ---")
        try:
            moments = analyzer.analyze(
                video_path=video_path,
                transcript=transcript,
                probe=probe,
                moments=moments  # Pass the result of the previous step
            )
            print(f"--- Finished analyzer: {analyzer_name}, found {len(moments)} moments ---")
        except Exception as e:
            print(f"Error in analyzer {analyzer_name}: {e}")
            # Decide if you want to stop the pipeline on error or continue
            # For now, we'll stop.
            raise
            
    print("Analyzer pipeline finished.")
    return moments 