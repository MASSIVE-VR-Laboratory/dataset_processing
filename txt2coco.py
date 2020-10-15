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
***************************************************************************
FUNCTION: to convert TEXT format annotations to COCO JSON
INPUT TEXT FORMAT: <category name> <xmin> <ymin> <xmax> <ymax>
OUTPUT FORMAT: COCO styled json (for reference see MS COCO website)
***************************************************************************
https://cocodataset.org/#format-data
"""
import os
import json
import random
from tqdm import tqdm
import cv2
from utils import _check_annotation_


class TXT2JSON:
    def __init__(self, root_dir, image_format, output_dir, occurences, train_ratio):
        self.root_dir = root_dir
        self.image_format = image_format
        self.output_dir = output_dir
        self.occurences = occurences
        self.train_ratio = train_ratio
        print("---Converting TXT files to JSON files---")

    def convert_txt2coco(self, image_filelist):
        # Starting the JSON creation
        attrDict = dict()
        images = list()
        annotations = list()

        attrDict["categories"] = list()

        # 1. Creating the category list from occurences
        category_list = self.occurences.keys()
        for id, category in enumerate(category_list):
            category_dict = dict()
            category_dict["supercategory"] = "none"
            category_dict["id"] = id + 1
            category_dict["name"] = category
            attrDict["categories"].append(category_dict)

        # 2. Creating the image and annotations
        image_id = 20200000000  # this number can start from anything (don't use 00000 because that becomes 0)
        id1 = 1
        for i, image_file in zip(
            tqdm(range(len(image_filelist)), ncols=100), image_filelist
        ):
            annotation_file = _check_annotation_(self.root_dir, image_file)
            if annotation_file is None:
                continue
            else:
                image = dict()
                # image quality check (can be loaded or not)
                try:
                    img = cv2.imread(
                        image_file, cv2.IMREAD_ANYCOLOR + cv2.IMREAD_ANYDEPTH
                    )
                    height, width = (img.shape[0], img.shape[1])
                except IOError as error:
                    print("IO ERROR: {0}".format(error))
                    continue

                # populate the images
                image_id += 1
                image["id"] = image_id
                image["file_name"] = os.path.basename(image_file)
                image["width"] = width
                image["height"] = height
                images.append(image)

                # populate the annotations
                with open(annotation_file, "r") as ann_file:
                    ann_data = ann_file.read().splitlines()
                for annotation in ann_data:
                    ann_string = annotation.split(" ")
                    obj_category = ann_string[0]
                    for value in attrDict["categories"]:
                        annotation = dict()
                        if obj_category in value["name"]:
                            # get the bounding box and calculate width and height
                            xmin = int(ann_string[1])
                            ymin = int(ann_string[2])
                            w = int(ann_string[3]) - xmin
                            h = int(ann_string[4]) - ymin

                            seg = []
                            # bbox[] is x,y,w,h
                            # left_top
                            seg.append(xmin)
                            seg.append(ymin)
                            # left_bottom
                            seg.append(xmin)
                            seg.append(ymin + h)
                            # right_bottom
                            seg.append(xmin + w)
                            seg.append(ymin + h)
                            # right_top
                            seg.append(xmin + w)
                            seg.append(ymin)

                            annotation["segmentation"] = []
                            annotation["segmentation"].append(seg)
                            annotation["area"] = round(float(w * h), 3)
                            annotation["iscrowd"] = 0
                            annotation["ignore"] = 0
                            annotation["image_id"] = image_id
                            annotation["bbox"] = [xmin, ymin, w, h]
                            annotation["category_id"] = value["id"]

                            annotation["id"] = id1
                            id1 += 1

                            annotations.append(annotation)
                            break
        attrDict["images"] = images
        attrDict["annotations"] = annotations
        attrDict["type"] = "instances"
        json_string = json.dumps(attrDict, indent=4)
        return json_string

    def generate_json(self):
        """
        Controlling function to generate ground truth json
        :return: None
        """
        # inner function for writing the JSON twice
        def _write_json_file(json_string, json_filename):
            output_json_file = os.path.join(self.output_dir, json_filename)
            with open(output_json_file, "w") as json_file_obj:
                json_file_obj.write(json_string)
            print("[INFO] - JSON file write completed..")

        image_filelist = sorted(
            [
                os.path.join(root, filename)
                for root, subdirs, files in os.walk(self.root_dir)
                for filename in files
                if filename.endswith(self.image_format)
            ]
        )

        # fixed the random seed and generate the JSONs
        random.seed(10)
        random.shuffle(image_filelist)
        ratio = int(len(image_filelist) * self.train_ratio)
        train_list = image_filelist[:ratio]
        test_list = image_filelist[ratio:]

        train_json_str = self.convert_txt2coco(train_list)
        _write_json_file(train_json_str, "instances_train2020.json")

        test_json_str = self.convert_txt2coco(test_list)
        _write_json_file(test_json_str, "instances_test2020.json")

        print('[INFO] JSON creation complete...')
        return
