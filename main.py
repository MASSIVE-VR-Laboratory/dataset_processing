"""
MIT License

Copyright (c) 2020 Ratnajit Mukherjee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

FUNCTION:
    1. explore a given dataset to obtain the total number of positives and negative samples
    2. obtain the number of categories of the dataset
    3. obtain the number of annotated positive instances in the dataset.
"""
import os
import argparse
import xmltodict
from tqdm import tqdm
from collections import Counter
from utils import (
    _conv_xml_files,
    parse_xml_annotation,
    parse_txt_annotation,
    display_statistics,
    _check_annotation_,
)
from txt2coco import TXT2JSON


class ExploreDataset:
    def __init__(
        self, root_dir, output_dir, conv_xml, image_format, ann_format, show_stats
    ):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.conv_xml = conv_xml
        self.image_format = image_format
        self.annotation_format = ann_format
        self.show_stats = show_stats

    def get_dataset_stats(self):
        """
        function to obtain dataset statistics
        :return:
        """
        # first check for XML files and convert them to TXT files

        if self.conv_xml:
            _conv_xml_files(self.root_dir)

        # positives contain at least 1 annotation else the image in a negative
        # negative images can be used for -ve example mining
        positive_images = list()
        negative_images = list()
        annotation_classes = list()  # this is going to be a list of lists
        total_instances = 0

        # get the list of JPG / EXR images in the dataset
        image_filelist = sorted(
            [
                os.path.join(root, filename)
                for root, subdirs, files in os.walk(self.root_dir)
                for filename in files
                if filename.endswith(self.image_format)
            ]
        )

        """
        iterate through the image dataset to explore the following:
        1. total number of classes
        2. total number of instances
        3. instances per class
        """
        for i, image_file in zip(
            tqdm(range(len(image_filelist)), ncols=100), image_filelist
        ):
            annotation_file = _check_annotation_(self.root_dir, image_file)
            if annotation_file is None:
                negative_images.append(image_file)
            else:
                positive_images.append(image_file)
                if self.annotation_format == "xml":
                    ann_data = xmltodict.parse(open(annotation_file).read())
                    parsed_class_names, num_instances = parse_xml_annotation(ann_data)
                    annotation_classes.append(parsed_class_names)
                    total_instances += num_instances

                elif self.annotation_format == "txt":
                    with open(annotation_file, "r") as ann_file:
                        ann_data = ann_file.read().splitlines()
                    parsed_class_names, num_instances = parse_txt_annotation(
                        ann_data=ann_data
                    )
                    annotation_classes.extend(parsed_class_names)
                    total_instances += num_instances

        occurences = dict(Counter(annotation_classes))
        if self.show_stats:
            display_statistics(self.output_dir, occurences)

        print(
            "**********<SUMMARY STATISTICS>**********\n"
            "Positive images: {0}\n"
            "Negative images: {1}\n"
            "Categories: {2}\n"
            "Total instances: {3}\n"
            "****************************************".format(
                len(positive_images),
                len(negative_images),
                len(occurences),
                total_instances,
            )
        )

        return occurences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Exploration of a dataset to convert annotated XMLs to JSON"
    )
    parser.add_argument(
        "--root_dir",
        "-r",
        type=str,
        required=True,
        help="Root directory where the images and XML files are located",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory where the statistics and JSON will be saved",
    )
    parser.add_argument(
        "--image_format",
        "-im",
        default="jpg",
        help="Input image format. Choices: [jpg, exr]",
    )
    parser.add_argument(
        "--annotation_format",
        "-af",
        default="txt",
        help="Annotation format. Choices: [txt, xml]",
    )
    parser.add_argument(
        "--xml2txt",
        action="store_true",
        help="Argument to convert XML files to text files. Default: False",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=False,
        help="Show dataset statistics. Categories and occurences",
    )
    parser.add_argument(
        "--txt2json",
        action="store_true",
        help="Argument to convert txt files to COCO json",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train to test ratio. Default: 0.8",
    )

    args = parser.parse_args()

    explore = ExploreDataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        conv_xml=args.xml2txt,
        image_format=args.image_format,
        ann_format=args.annotation_format,
        show_stats=args.stats,
    )
    occurences = explore.get_dataset_stats()

    if args.txt2json:
        txt2json = TXT2JSON(
            root_dir=args.root_dir,
            image_format=args.image_format,
            output_dir=args.output_dir,
            occurences=occurences,
            test_train_ratio=args.train_ratio,
        )
        txt2json.generate_json()
