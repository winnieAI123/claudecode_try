"""
Tri-Model Deep Search Pipeline.

A strict 3-stage search pipeline that uses DeepSeek, Gemini, and ChatGPT
to perform comprehensive research and produce buy-side brief format output.
"""

from .pipeline import TriModelSearchPipeline
from .providers.base import ResearchPlan, StageResult

__all__ = [
    "TriModelSearchPipeline",
    "ResearchPlan",
    "StageResult"
]

__version__ = "1.0.0"
