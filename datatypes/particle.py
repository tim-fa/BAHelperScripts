import datetime

from src.datatypes.track import Track


class Particle:
    """
    A particle is a recognized real world particle based on the underlying track
    """
    def __init__(self, label: str, confidence: float, timestamp: datetime, track: Track):
        self._label = label
        self._confidence = label
        self._timestamp = timestamp
        self._track = track




