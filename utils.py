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

FUNCTION: Helper functions for various utilities
"""
import os
from tqdm import tqdm
from shutil import move
from collections import OrderedDict
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt


def _check_annotation_(root_dir, image_file):
    annotation_dir = os.path.join(root_dir, "labels")
    annotation_file_basename = os.path.basename(image_file).split(".")[0] + ".txt"

    if os.path.exists(os.path.join(annotation_dir, annotation_file_basename)):
        annotation_filename = os.path.join(annotation_dir, annotation_file_basename)
        return annotation_filename
    else:
        return None


def _conv_xml_files(root_dir):
    xml_filelist = sorted(
        [
            os.path.join(root, filename)
            for root, subdirs, files in os.walk(root_dir)
            for filename in files
            if filename.endswith(".xml")
        ]
    )

    if len(xml_filelist) == 0:
        print("No XML files found. Conversion not required..")
        return
    else:
        for i, xml_file in zip(tqdm(range(len(xml_filelist)), ncols=100), xml_filelist):
            with open(xml_file.replace(".xml", ".txt"), "a") as txt_file:
                root = ET.parse(xml_file).getroot()
                for obj in root.findall("object"):
                    obj_name = obj.find("name").text
                    bndbox = obj.find("bndbox")
                    left = bndbox.find("xmin").text
                    top = bndbox.find("ymin").text
                    right = bndbox.find("xmax").text
                    bottom = bndbox.find("ymax").text
                    txt_file.write(
                        "%s %s %s %s %s\n" % (obj_name, left, top, right, bottom)
                    )
            # move the XML file to xml_backup directory
            os.makedirs(
                os.path.join(os.path.dirname(xml_file), "xml_backup"), exist_ok=True
            )
            move(
                xml_file,
                os.path.join(
                    os.path.dirname(xml_file), "xml_backup", os.path.basename(xml_file)
                ),
            )

        print("XML to TXT file conversion completed..")


def parse_xml_annotation(ann_data):
    if "object" not in ann_data["annotation"].keys():
        return None, None
    else:
        object_categories = list()
        if isinstance(ann_data["annotation"]["object"], OrderedDict):
            ann_data["annotation"]["object"] = [ann_data["annotation"]["object"]]

        num_instances = len(ann_data["annotation"]["object"])
        object_categories = [obj["name"] for obj in ann_data["annotation"]["object"]]
        return object_categories, num_instances


def parse_txt_annotation(ann_data):
    # we can get the num of instances from the length of the list
    num_instances = len(ann_data)
    object_categories = [annotation.split(" ")[0] for annotation in ann_data]
    return object_categories, num_instances


def display_statistics(output_dir, occurences):
    # generate bar plots to show statistics
    categories = list(occurences.keys())
    values = list(occurences.values())

    # tick_label does the some work as plt.xticks()
    plt.barh(range(len(occurences)), values, tick_label=categories)
    for i, v in enumerate(values):
        plt.text(v + 3, i, str(v), color="blue", fontweight="bold")
    plt.grid(which="both")
    plt.xlabel("Number of instances", fontsize=20, weight="bold", color="blue")
    plt.ylabel("Object Categories", fontsize=20, weight="bold", color="blue")
    plt.title("Dataset statistics", fontsize=25, weight="bold", color="blue")
    plt.savefig(os.path.join(output_dir, "dataset_stats.png"), dpi=300)
    plt.show()
