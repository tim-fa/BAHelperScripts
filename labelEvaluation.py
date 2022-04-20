import argparse
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.datatypes import yolo
from src.datatypes.yolo import YoloLabel, yolo_text_label_to_box, \
    yolo_bb_intersection_over_union


def read_arguments(args):
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
        Compare a set of guessed bounding boxes e.g. by a segmentation algorithm or an object detector with the ground 
        truth boxes.
        The bounding boxes need to be in the yolo annotation format.
            """.strip()
    )

    parser.add_argument("-gt", "--ground-truth-dir", required=True, type=str,
                        help="The directory containing the correct labels")
    parser.add_argument("-gl", "--guessed-label-dir", required=True, type=str,
                        help="The directory containing the guessed labels")
    parser.add_argument("-iou", "--iou-threshold", required=True, type=float,
                        help="An IoU above the threshold is interpreted as a true positive")
    parser.add_argument("-c", "--class-id", required=False, default=-1, type=int,
                        help="If set only bounding boxes of the specified class are evaluated")

    options = parser.parse_args(args)

    return options


def get_confusion_matrix(predictions: YoloLabel, ground_truth: YoloLabel, iou_threshold: float):
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    for gt in ground_truth.labels:
        any_box_found = False
        actual_box = yolo_text_label_to_box(gt)
        for pr in predictions.labels:
            predicted_box = yolo_text_label_to_box(pr)
            iou = yolo_bb_intersection_over_union(actual_box, predicted_box)
            # true positive
            if iou >= iou_threshold:
                any_box_found = True
                true_positives += 1
        if not any_box_found:
            false_negatives += 1

    for pr in predictions.labels:
        is_tp = False
        predicted_box = yolo_text_label_to_box(pr)
        for gt in ground_truth.labels:
            actual_box = yolo_text_label_to_box(gt)
            iou = yolo_bb_intersection_over_union(actual_box, predicted_box)
            # false positive
            if iou >= iou_threshold:
                is_tp = True
        if not is_tp:
            false_positives += 1

    return true_positives, false_positives, false_negatives


def main(args):
    options = read_arguments(args)

    recalls = []
    precisions = []
    ious = np.arange(0.001, 1, 0.05)
    for iouval in ious:

        ttp, tfp, tfn = 0, 0, 0
        highest_iou = 0
        lowest_iou = 1
        total_predicted_boxes = 0
        total_ground_truth_boxes = 0
        iou_sum = 0

        for guess_file_name in os.listdir(options.guessed_label_dir):
            try:
                prediction = YoloLabel(os.path.join(options.guessed_label_dir, guess_file_name))
                ground_truth = YoloLabel(os.path.join(options.ground_truth_dir, guess_file_name))

                prediction.load_from_file()
                ground_truth.load_from_file()

                if options.class_id > -1:
                    prediction.labels = [label for label in prediction.labels if
                                         yolo.yolo_get_object_class(label) == options.class_id]
                    ground_truth.labels = [label for label in ground_truth.labels if
                                           yolo.yolo_get_object_class(label) == options.class_id]

                total_ground_truth_boxes += len(ground_truth.labels)
                total_predicted_boxes += len(prediction.labels)

                tp, fp, fn = get_confusion_matrix(prediction, ground_truth, iouval)
                ttp += tp
                tfp += fp
                tfn += fn

                print(f"Comparing labels for file {guess_file_name}")

                for idx, gl in enumerate(prediction.labels):
                    prediction_box = yolo_text_label_to_box(gl)
                    best_iou = 0
                    for gtl in ground_truth.labels:

                        actual_box = yolo_text_label_to_box(gtl)

                        iou = yolo_bb_intersection_over_union(prediction_box, actual_box)
                        if iou >= best_iou:
                            best_iou = iou

                    iou_sum += best_iou
                    if best_iou > highest_iou:
                        highest_iou = best_iou
                    if best_iou < lowest_iou:
                        lowest_iou = best_iou
                    print(f"Guessed box {idx} has an IoU of {best_iou} with the closest ground truth box")
            except Exception as e:
                print(e)
        print("****************************************************************")
        print(
            f"Compared {total_predicted_boxes} predicted bounding boxes with {total_ground_truth_boxes} ground truth boxes")
        print(f"\tAverage IoU: {iou_sum / total_predicted_boxes}")
        print(f"\tBest IoU: {highest_iou}")
        print(f"\tWorst IoU: {lowest_iou}")
        print("****************************************************************")
        print(f"\tPrecision: {ttp / (ttp + tfp)}")
        precisions.append(ttp / (ttp + tfp))
        print(f"\tRecall: {ttp / (ttp + tfn)}")
        recalls.append(ttp / (ttp + tfn))
        print(f"\tFalse Positives / Total Predicted boxes: {tfp}/{total_predicted_boxes}")
        print(f"\tTrue Positives / Total Ground truth boxes: {ttp}/{total_ground_truth_boxes}")
        print(f"\tFalse Negatives / Total Ground truth boxes: {tfn}/{total_ground_truth_boxes}")

    # line 1 points
    x1 = ious
    y1 = precisions
    # plotting the line 1 points
    plt.plot(x1, y1, label="Precision")
    # line 2 points
    x2 = ious
    y2 = recalls
    # plotting the line 2 points
    plt.plot(x2, y2, label="Recall")
    plt.xlabel('IoU Threshold Value')
    plt.yticks(np.arange(0, 1, 0.05))
    # Set a title of the current axes.
    plt.title('Precision und Recall bei verschiedenen IoU Schwellwerten')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
