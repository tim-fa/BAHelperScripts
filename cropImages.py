import argparse
import os
import sys

import cv2


def read_arguments(args):
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
                Crop all images in the source directory. 
                The area which will be kept is defined as [x to x+width][y to y+height]
            """.strip()
    )

    parser.add_argument("-sd", "--src_dir", required=True, type=str, help="The directory in which to crop images")
    parser.add_argument("-dd", "--dest_dir", required=True, type=str,
                        help="The directory in which to save the cropped images")
    parser.add_argument('-x', required=True, type=int,
                        help="starting x coord")
    parser.add_argument('-y', required=True, type=int,
                        help="starting y coors")
    parser.add_argument('-wi', '--width', required=False, type=int, default=-1,
                        help="width starting from x. In case of -1 the whole width is used")
    parser.add_argument('-he', '--height', required=False, type=int, default=-1,
                        help="height starting from y. In case of -1 the whole height is used")
    parser.add_argument('-wip', '--width-percentage', required=False, type=float, default=0,
                        help="You might use this instead of the absolute width. Values between 0 and 1")
    parser.add_argument('-hep', '--height-percentage', required=False, type=float, default=0,
                        help="You might use this instead of the absolute height. Values between 0 and 1")
    parser.add_argument('-t', '--type', required=False, type=str,
                        default=".jpg", help="the file format which should be cropped")

    options = parser.parse_args(args)

    return options


def main(args):
    options = read_arguments(args)

    for filename in os.listdir(options.src_dir):
        if filename.endswith(options.type):

            img = cv2.imread(os.path.join(options.src_dir, filename))
            if options.height is -1:
                if options.height_percentage > 0:
                    options.height = int(img.shape[1] * options.height_percentage)
                else:
                    options.height = img.shape[1]
            if options.width is -1:
                if options.width_percentage > 0:
                    options.width = int(img.shape[0] * options.width_percentage)
                else:
                    options.width = img.shape[0]

            crop_img = img[options.y:options.y + options.height, options.x:options.x + options.width]
            cv2.imwrite(os.path.join(options.dest_dir, filename), crop_img)
            print("Saved image {}".format(filename))


if __name__ == "__main__":
    main(sys.argv[1:])

