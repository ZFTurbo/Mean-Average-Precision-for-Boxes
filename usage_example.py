"""
Author: Roman Solovyev, IPPM RAS
URL: https://github.com/ZFTurbo
"""

from map_boxes import mean_average_precision_for_boxes
import pandas as pd

if __name__ == '__main__':
    # Version 1
    annotations_file = 'example/annotations.csv'
    detections_file = 'example/detections.csv'
    mean_ap, average_precisions = mean_average_precision_for_boxes(annotations_file, detections_file)

    # Version 2
    ann = pd.read_csv('example/annotations.csv')
    det = pd.read_csv('example/detections.csv')
    ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
    det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det)
