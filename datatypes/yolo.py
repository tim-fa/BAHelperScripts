import math
import os

import cv2
import numpy as np


def cv2_box_to_yolo_box(cv2_bounding_box: tuple, image_size: np.shape) -> tuple:
    """
    Converts the cv2 bounding rect coordinates (x, y: top left) to normalized yolo coordinates (x, y: in the middle)
    """
    x = (cv2_bounding_box[0] + 0.5 * cv2_bounding_box[2]) / image_size[1]
    y = (cv2_bounding_box[1] + 0.5 * cv2_bounding_box[3]) / image_size[0]
    w = cv2_bounding_box[2] / image_size[1]
    h = cv2_bounding_box[3] / image_size[0]
    return x, y, w, h


def yolo_box_to_cv2_box(yolo_box: list, image_size: np.shape) -> tuple:
    """
    Converts the normalized yolo coordinates (x, y: in the middle) to cv2 bounding rect coordinates (x, y: top left)
    """
    x = int((yolo_box[0] - 0.5 * yolo_box[2]) * image_size[1])
    y = int((yolo_box[1] - 0.5 * yolo_box[3]) * image_size[0])
    w = int(yolo_box[2] * image_size[1])
    h = int(yolo_box[3] * image_size[0])
    return x, y, w, h


def yolo_text_label_to_box(yolo_label: str) -> []:
    """
    Converts a yolo string label to a tuple representing the box coordinates x, y, w, h
    """
    object_class, x, y, w, h = yolo_label.split(" ")
    return [float(x), float(y), float(w), float(h)]


def yolo_get_box_distance(yolo_box1: list, yolo_box2: list) -> float:
    """
    Get the distance between two yolo bounding boxes
    """
    return math.sqrt(math.pow(yolo_box2[0] - yolo_box1[0], 2) + math.pow(yolo_box2[1] - yolo_box1[1], 2))


def yolo_to_xyxy_relative(yolo_box: list) -> tuple:
    """
    Converts a yolo bounding box to a bounding box specified by a top left and bottom right point
    """
    x_top = max(yolo_box[0] - yolo_box[2] * 0.5, 0)
    y_top = max(yolo_box[1] - yolo_box[3] * 0.5, 0)
    x_bottom = min(yolo_box[0] + yolo_box[2] * 0.5, 1)
    y_bottom = min(yolo_box[1] + yolo_box[3] * 0.5, 1)
    return x_top, y_top, x_bottom, y_bottom


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def yolo_to_xyxy_absolute(yolo_box: list, image_size: np.shape) -> tuple:
    """
    Converts a yolo bounding box to a bounding box specified by a top left and bottom right point with absolute values
    """
    x_abs = int(yolo_box[0] * image_size[1])
    y_abs = int(yolo_box[1] * image_size[0])
    w_abs = int(yolo_box[2] * image_size[1])
    h_abs = int(yolo_box[3] * image_size[0])

    y_top = clamp(y_abs - int(h_abs / 2), 0, image_size[0])
    y_bottom = clamp(y_abs + int(h_abs / 2), 0, image_size[0])
    x_top = clamp(x_abs - int(w_abs / 2), 0, image_size[1])
    x_bottom = clamp(x_abs + int(w_abs / 2), 0, image_size[1])

    return x_top, y_top, x_bottom, y_bottom


def xyxy_to_yolo(sub_region: tuple, image_size: np.shape) -> []:
    """
    Convert a region specified by top left and bottom right point to yolo box coordinates.
    the image dimensions are used to normalize the values
    """
    # coordinates of top left and bottom right point of the bb
    width = image_size[1]
    height = image_size[0]
    relative_x_top = sub_region[0] / width
    relative_y_top = sub_region[1] / height
    relative_x_bottom = sub_region[2] / width
    relative_y_bottom = sub_region[3] / height

    # coordinates in the yolo format (x, y in the middle of the box)
    yolo_w = relative_x_bottom - relative_x_top
    yolo_h = relative_y_bottom - relative_y_top
    yolo_x = relative_x_top + yolo_w / 2
    yolo_y = relative_y_top + yolo_h / 2

    return [yolo_x, yolo_y, yolo_w, yolo_h]


def xyxy_to_yolo_subregion(sub_region: tuple, base_region: tuple) -> []:
    """
    Convert a region specified by top left and bottom right point to yolo box coordinates.
    The base_region is the reference region in which the sub_region should be placed.
    """
    # coordinates of top left and bottom right point of the bb
    base_width = base_region[2] - base_region[0]  # x bottom - x top
    base_height = base_region[3] - base_region[1]  # y bottom - y top
    relative_x_top = (sub_region[0] - base_region[0]) / base_width
    relative_y_top = (sub_region[1] - base_region[1]) / base_height
    relative_x_bottom = (sub_region[2] - base_region[0]) / base_width
    relative_y_bottom = (sub_region[3] - base_region[1]) / base_height

    # coordinates in the yolo format (x, y in the middle of the box)
    yolo_w = relative_x_bottom - relative_x_top
    yolo_h = relative_y_bottom - relative_y_top
    yolo_x = relative_x_top + yolo_w / 2
    yolo_y = relative_y_top + yolo_h / 2

    return [yolo_x, yolo_y, yolo_w, yolo_h]


def yolo_get_object_class(yolo_label: str) -> int:
    """
    Return the object class of a single label
    """
    return int(yolo_label.split(" ")[0])


def yolo_xyxy_area(box: tuple):
    return (box[2] - box[0]) * (box[3] - box[1])


def yolo_bb_area(box: list):
    return box[2] * box[3]


def yolo_xyxy_intersection(boxA: tuple, boxB: tuple):

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    return interArea


def yolo_bb_intersection_over_union(yolo_box1: list, yolo_box2: list):
    boxA = yolo_to_xyxy_relative(yolo_box1)
    boxB = yolo_to_xyxy_relative(yolo_box2)

    interArea = yolo_xyxy_intersection(boxA, boxB)
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class YoloLabel:
    """
    A yolo label consists of a file name and the normalized box coordinates x, y, w, h where x and y represent the
    middle of the box
    """

    def __init__(self, file_name: str = ''):
        # meta data
        self._name = file_name
        self._labels = []

    def add_label(self, class_id: int, bounding_box: tuple):
        self.labels.append(f"{class_id} {bounding_box[0]} {bounding_box[1]} {bounding_box[2]} {bounding_box[3]}")

    def add_text_label(self, text_label: str):
        self.labels.append(text_label)

    def save_to_file(self):
        if self._name == '':
            raise Exception('No file name was set')
        if os.path.exists(self._name):
            os.remove(self._name)
        label_file = open(self._name, "a")
        for result in self._labels:
            label_file.write(result + "\n")
        label_file.close()

    def load_from_file(self):
        if self._name == '':
            raise Exception('No file name was set')
        with open(self._name) as label_file:
            self._labels = [line.rstrip() for line in label_file]

    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @labels.setter
    def labels(self, value):
        self._labels = value


def plot_to_image(labels: list, image: np.array, color: tuple, size: tuple, show: bool = True):
    for label in labels:
        xyxy = yolo_to_xyxy_absolute(yolo_text_label_to_box(label), image.shape)
        cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 5)

    if show:
        resized = cv2.resize(image, size)
        cv2.imshow("YOLO Boxes Plot", resized)
        cv2.waitKey(0)
