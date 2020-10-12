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

NOTE: Please manually remove the XML files. Auto removal of the XML files is NOT implemented as a safety feature.
FUNCTION: to convert VOC styled XML annotations to COCO json format
"""

import os
import json
import xmltodict
from collections import OrderedDict


class XML2JSON:
    def __init__(self, input_xml_dir, output_json_dir, json_filename):
        self.xmldir = input_xml_dir
        self.jsondir = output_json_dir
        self.json_name = json_filename
        print('****************************************')
        print('---Converting XML files to JSON files---')
        print('\n XML Directory: {0}'.format(self.xmldir))
        print('\n Output JSON directory {0}'.format(self.jsondir))
        print('\n JSON filename {0}'.format(self.json_name))
        print('****************************************')
        return

    def generateVOC2Json(self, rootDir, xmlFiles):
        attrDict = dict()
        """
        NOTE: this is specifically for BDD but you can change to any other dataset provided you have the pre-trained
        model on the dataset which you are going to predict the unlabelled data on. Replicate the categories of the
        pre-training dataset.
        """        

        attrDict["categories"] = [{"supercategory": "none", "id": 1, "name": "prohibitory"},
                                  {"supercategory": "none", "id": 2, "name": "mandatory"},
                                  {"supercategory": "none", "id": 3, "name": "danger"},
                                  {"supercategory": "none", "id": 4, "name": "other"}]

        images = list()
        annotations = list()

        image_id = 20190000000
        id1 = 1
        for root, dirs, files in os.walk(rootDir):

            for file in xmlFiles:
                image_id = image_id + 1
                if file in files:

                    annotation_path = os.path.abspath(os.path.join(root, file))
                    print(annotation_path)
                    image = dict()
                    doc = xmltodict.parse(open(annotation_path).read())
                    # check if there are any annotations in the first place
                    if 'object' not in doc['annotation'].keys():
                        print('Skipping the file: {0}'.format(file))
                        continue

                    image['id'] = image_id
                    image['file_name'] = str(doc['annotation']['filename'])
                    image['width'] = int(doc['annotation']['size']['width'])
                    image['height'] = int(doc['annotation']['size']['height'])
                    images.append(image)

                    if 'object' in doc['annotation']:
                        if isinstance(doc['annotation']['object'], OrderedDict):
                            doc['annotation']['object'] = [doc['annotation']['object']]

                        for obj in doc['annotation']['object']:
                            for value in attrDict["categories"]:
                                annotation = dict()
                                if str(obj['name']) in value["name"]:
                                    # get the bounding box and calculate width and height
                                    x1 = int(obj["bndbox"]["xmin"])
                                    y1 = int(obj["bndbox"]["ymin"])
                                    x2 = int(obj["bndbox"]["xmax"]) - x1
                                    y2 = int(obj["bndbox"]["ymax"]) - y1

                                    seg = []
                                    # bbox[] is x,y,w,h
                                    # left_top
                                    seg.append(x1)
                                    seg.append(y1)
                                    # left_bottom
                                    seg.append(x1)
                                    seg.append(y1 + y2)
                                    # right_bottom
                                    seg.append(x1 + x2)
                                    seg.append(y1 + y2)
                                    # right_top
                                    seg.append(x1 + x2)
                                    seg.append(y1)

                                    # start the structure of the JSON file
                                    annotation["segmentation"] = []
                                    annotation["segmentation"].append(seg)
                                    annotation["area"] = round(float(x2 * y2), 3)
                                    annotation["iscrowd"] = 0
                                    annotation["ignore"] = 0
                                    annotation["image_id"] = image_id
                                    annotation["bbox"] = [x1, y1, x2, y2]
                                    annotation["category_id"] = value["id"]

                                    annotation["id"] = id1
                                    id1 += 1

                                    annotations.append(annotation)
                                    break


                    else:
                        print("File: {} doesn't have any object".format(file))
                else:
                    print("File: {} not found".format(file))

        attrDict["images"] = images
        attrDict["annotations"] = annotations
        attrDict["type"] = "instances"

        json_string = json.dumps(attrDict, indent=4)

        output_json_file = os.path.join(self.jsondir, self.json_name)
        with open(output_json_file, "w") as json_file_obj:
            json_file_obj.write(json_string)

        print('[INFO] JSON creation complete...')
        return
