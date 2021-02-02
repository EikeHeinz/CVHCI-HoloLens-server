# Requirements Installation

* Python 3.8
* CUDA(version >= 10.1)

## Manually installation

### (optional)activate virtualenv

```
source virtual-env/bin/activate
```

### install pytorch

Please refere to [Pytorch official website](https://pytorch.org/)

pytorch version need to be >= 1.5.0 and should mathch the CUDA version on you machine.

### install opencv and scipy

```shell
python -m pip install opencv-python
python -m pip install scipy
```

### install detectron2

```shell

# pip install specific branch of forked detectron2 repo

## for Linux/MacOS
pip install 'git+https://github.com/merlinz165/detectron2.git@ting_dev'

## for Windows
pip install 'git+https://github.com/merlinz165/detectron2.git@ting_dev_win'

```

### install pytorch3d

```shell

# pip install specific branch of forked pytorch3d repo
## for Linux/MacOS and Windows
pip install 'git+https://github.com/merlinz165/pytorch3d.git@ting_dev'

```

### install meshrcnn

```shell

# pip install specific branch of meshrcnn repo
## for Linux/MacOS and Windows
pip install 'git+https://github.com/merlinz165/meshrcnn.git@ting_dev'

```

# Usage

in the project root path:

## Command Line
```shell
# --image: input image path
# --roi: result[roi] of instance segmentation output. rois: [N, (y1, x1, y2, x2)] detection bounding boxes
python shape_estimation/inference.py --image ./shape_estimation/0007.png --roi "95, 25, 390, 470"

```
## Using `ShapeEstimationModel` Class
```
...
from shape_estimation.inference import ShapeEstimationModel
...
estimation_model = ShapeEstimationModel(input_image_path, roi)
detections = estimation_model.get_detection()
## detections is a dict. object file path will be detections['object_file']
...

```
