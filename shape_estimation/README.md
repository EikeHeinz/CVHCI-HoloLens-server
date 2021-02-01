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

### (necessary for virtual env)install opencv and scipy

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

git clone -b ting_dev https://github.com/merlinz165/pytorch3d.git
```

### install meshrcnn (Locally)

```shell
# clone specific branch of meshrcnn repo
git clone -b ting_dev https://github.com/merlinz165/meshrcnn.git

# install meshrcnn
cd meshrcnn
pip install -e .

```

# Usage

in the `shape_estimation` directory:

Simply run `python inference.py` will infer all sample images.

run `python inference.py --help` for more options (e.g., single image, output path, focal length).
