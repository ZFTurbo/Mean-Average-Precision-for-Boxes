Function to calculate mean average precision (mAP) for set of boxes. Useful for object detection pipelines.

# Requirements

python 3.*, numpy, pandas

# Installation

```
pip install map-boxes
```

## Usage example:

You can provide paths to CSV-files:

```python
from map_boxes import mean_average_precision_for_boxes

annotations_file = 'example/annotations.csv'
detections_file = 'example/detections.csv'
mean_ap, average_precisions = mean_average_precision_for_boxes(annotations_file, detections_file)
```

or you can pass directly numpy arrays of shapes **(N, 6)** and **(M, 7)**. **Be careful about order of variables in arrays!**:

```python
from map_boxes import mean_average_precision_for_boxes
import pandas as pd

ann = pd.read_csv('example/annotations.csv')
det = pd.read_csv('example/detections.csv')
ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det)
```


## Input files format

Boxes must be in normalized form e.g. coordinates must be in range: `[0, 1]`. To normalize pixel values you need to recalculate them as: `x_norm = x / width`, `y_norm = y / height`

* Annotation CSV-file:

```csv
ImageID,LabelName,XMin,XMax,YMin,YMax
i0.jpg,Shellfish,0.0875,0.8171875,0.35625,0.8958333
i0.jpg,Seafood,0.0875,0.8171875,0.35625,0.8958333
i1.jpg,Tin can,0.1296875,0.3375,0.31875,0.68958336
i1.jpg,Drink,0.4234375,0.546875,0.58958334,0.92083335
i1.jpg,Drink,0.5375,0.7375,0.16666667,0.575
...
```

* Detection CSV-file:

```csv
ImageID,LabelName,Conf,XMin,XMax,YMin,YMax
i0.jpg,Turtle,0.41471,0.1382,0.7440,0.3585,0.8951
i0.jpg,Reptile,0.32093,0.1391,0.7439,0.3582,0.8944
i0.jpg,Seahorse,0.11860,0.1393,0.7434,0.3589,0.8943
i0.jpg,Caterpillar,0.11275,0.1390,0.7438,0.3588,0.8948
i1.jpg,Personal care,0.42326,0.2624,0.5473,0.1112,0.7274
i1.jpg,Personal care,0.31120,0.1318,0.3381,0.3149,0.6863
i1.jpg,Personal care,0.34866,0.4277,0.5446,0.5861,0.9211
i1.jpg,Blender,0.10578,0.7678,0.9476,0.2674,0.5847
...
```
