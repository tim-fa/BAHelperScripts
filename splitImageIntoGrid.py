import argparse
import math
import os
import sys

import cv2

from datatypes.yolo import YoloLabel, yolo_text_label_to_box, yolo_to_xyxy_absolute, \
    clamp, xyxy_to_yolo_subregion

image_output_folder = "images"
label_output_folder = "labels"


def read_arguments(args):
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
                This script splits an labelled image into a specified grid size.
                New labels are created for each new image based on the original labels so that each object stays 
                labelled.
            """.strip()
    )

    parser.add_argument("-id", "--image_path", required=True, type=str,
                        help="The labeled image")
    parser.add_argument("-ld", "--label_path", required=True, type=str,
                        help="The labels of the image")
    parser.add_argument("-dd", "--dest_dir", required=True, type=str,
                        help="The directory in which to save the newly create training images. The images will be saved"
                             " in two separate folders: images/ and labels/")
    parser.add_argument('-cs', "--cell-size", required=True, type=int,
                        help="Minimum cell size in pixels")

    parser.add_argument('-v', '--verbose', action='store_true')

    options = parser.parse_args(args)

    return options


def main(args):
    options = read_arguments(args)

    if not os.path.exists(os.path.join(options.dest_dir, image_output_folder)):
        os.mkdir(os.path.join(options.dest_dir, image_output_folder))
    if not os.path.exists(os.path.join(options.dest_dir, label_output_folder)):
        os.mkdir(os.path.join(options.dest_dir, label_output_folder))

    cell_count = 0
    base_name = os.path.splitext(os.path.basename(options.image_path))[0]
    image_ext = os.path.splitext(os.path.basename(options.image_path))[1]
    label_file = YoloLabel(os.path.join(options.label_dir, base_name + ".txt"))
    label_file.load_from_file()
    img = cv2.imread(options.image_path)

    num_cells_x = math.floor(img.shape[1] / options.cell_size)
    cell_size_x = int(img.shape[1] / (num_cells_x if num_cells_x > 0 else 1))
    num_cells_y = math.floor(img.shape[0] / options.cell_size)
    cell_size_y = int(img.shape[0] / (num_cells_y if num_cells_y > 0 else 1))

    for x in range(0, img.shape[1], cell_size_x):
        for y in range(0, img.shape[0], cell_size_y):
            top_x = x
            top_y = y
            bottom_x = top_x + cell_size_x
            bottom_y = top_y + cell_size_y

            cropped_region = img[top_y:bottom_y, top_x:bottom_x]
            cv2.imwrite(
                os.path.join(options.dest_dir, image_output_folder, base_name + f"_{cell_count}" + image_ext),
                cropped_region)

            label_result = YoloLabel(
                os.path.join(options.dest_dir, label_output_folder, base_name + f"_{cell_count}.txt"))

            for label in label_file.labels:
                yolo_box = yolo_text_label_to_box(label)
                points = yolo_to_xyxy_absolute(yolo_box, img.shape)

                # calculate the intersection of the each bounding box with the region to be cut out
                dx = min(bottom_x, points[2]) - max(top_x, points[0])
                dy = min(bottom_y, points[3]) - max(top_y, points[1])

                if (dx >= 0) and (dy >= 0):
                    # if the other bounding box exceeds the current cropping region, clamp it
                    l_x_top = clamp(points[0], top_x, points[2])
                    l_x_bottom = clamp(points[2], points[0], bottom_x)
                    l_y_top = clamp(points[1], top_y, points[3])
                    l_y_bottom = clamp(points[3], points[1], bottom_y)

                    new_yolo_bb = xyxy_to_yolo_subregion((l_x_top, l_y_top, l_x_bottom, l_y_bottom),
                                                         (top_x, top_y, bottom_x, bottom_y))
                    label_result.add_label(0, new_yolo_bb)

            label_result.save_to_file()
            cell_count += 1


if __name__ == "__main__":
    main(sys.argv[1:])
