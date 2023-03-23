"""Data processing modules"""

from pathlib import Path
from typing import List

from pydub import AudioSegment


class AudioSplitter:
    """Split the audio into chunks"""
