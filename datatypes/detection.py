import os

import numpy as np


class Detection:
    """
    The detection class includes all necessary attributes for an object detection.
    An object detection is a region of interest in an image that is enriched during the processing
    with relevant data to determine a particle.
    """

    def __init__(self, source_id: int, source_path: str, bounding_box: np.array, image_segment: np.array,
                 features: np.array = None):
        self._source_id = source_id
        self._source_path = source_path
        self._filename, self._file_extension = os.path.splitext(os.path.basename(source_path))
        self._bb = bounding_box  # Used bounding box model (XYWH)
        self._image_segment = image_segment
        self._score = 0
        self._features = features

    @property
    def image_segment(self):
        return self._image_segment

    @property
    def bounding_box(self):
        return self._bb

    @property
    def score(self):
        return self._score

    @property
    def features(self):
        return self._features

    @property
    def path(self):
        return self._source_path

    @property
    def file_extension(self):
        return self._file_extension

    @property
    def filename(self):
        return self._filename

    @features.setter
    def features(self, value):
        self._features = value
