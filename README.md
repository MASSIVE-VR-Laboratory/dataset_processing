# dataset_processing

### Description: Repository to explore any annotated dataset and create COCO styled JSON files to train object detectors on the annotated dataset. The repository supports XML and TXT files.
![alt text](https://github.com/MASSIVE-VR-Laboratory/dataset_processing/blob/main/DatasetSpecifications.png?raw=true "Dataset Specifications")

##### Usage for statistics only
``
python main.py --root_dir /data/annotated_dataset --output_dir /data/annotated_dataset/annotations --image_format jpg --annotation_format txt --xml2txt --stats
``
##### Usage for txt to json conversion
``
python main.py --root_dir /data/annotated_dataset --output_dir /data/annotated_dataset/annotations --image_format jpg --annotation_format txt --xml2txt --txt2json --train_ratio 0.8
``