import os
from uuid import uuid4

import cv2

from src.datatypes.imageMetaData import ImageMetaData


class Image:
    '''
    An Image is a  photograph enriched with meta information like filename, GPS , ...
    '''

    def __init__(self, file_path: str):
        self._id = uuid4()
        self._file_path = file_path
        self._file_name = os.path.basename(file_path)
        self._data = cv2.imread(file_path)
        self._meta_info = ImageMetaData(file_path)

    @property
    def id(self):
        return self._id

    @property
    def data(self):
        return self._data

    @property
    def meta_info(self):
        return self._meta_info

    @property
    def path(self):
        return self._file_path
