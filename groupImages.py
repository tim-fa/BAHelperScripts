import argparse
import glob
import os
import shutil
import sys
from os.path import join
from shutil import copyfile
from subprocess import Popen


def read_arguments(args):
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
                Group files together whose timestamp difference is less than the specified time difference.
                The files are copied to output directories in the format <output-dir-name> + index
            """.strip()
    )

    parser.add_argument("-td", "--time-difference", required=True, type=int,
                        help="The maximum difference in seconds in between files to be grouped together")
    parser.add_argument("-sd", "--source-dir", required=True, type=str,
                        help="The directory in which the files are located")
    parser.add_argument("-od", "--output-dir", required=True, type=str,
                        help="The directory in which to save the grouped images")

    options = parser.parse_args(args)

    return options


def main(args):
    options = read_arguments(args)

    current_group = 0
    last_timestamp = None

    files = glob.glob(os.path.expanduser(options.source_dir + '/*'))
    sorted_file_list = sorted(files, key=lambda t: os.stat(t).st_mtime)

    for file_path in sorted_file_list:
        current_timestamp = os.path.getmtime(file_path)
        if last_timestamp is None:
            last_timestamp = current_timestamp
        difference_s = abs(last_timestamp - current_timestamp)
        if difference_s > options.time_difference:
            current_group += 1

        dest_dir_name = "{} {}".format(options.output_dir, current_group)
        print("Copying file {0} to {1}/{0}".format(os.path.basename(file_path), dest_dir_name))
        if not os.path.exists(dest_dir_name):
            os.mkdir(dest_dir_name)
        copyfile(file_path, os.path.join(dest_dir_name, os.path.basename(file_path)))

        last_timestamp = current_timestamp

if __name__ == "__main__":
    main(sys.argv[1:])
