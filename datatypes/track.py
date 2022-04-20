from src.datatypes.detection import Detection


class Track:
    """
    A track consists of a list of similar detections, ideally containing the same particle
    """

    def __init__(self, id, detections: [Detection]):
        # meta data
        self._id = id
        self._detections = detections

    @property
    def id(self):
        return self._id

    @property
    def detections(self):
        return self._detections
