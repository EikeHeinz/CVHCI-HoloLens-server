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
# clone specific branch of forked detectron2 repo

## for Linux/MacOS
git clone -b ting_dev https://github.com/merlinz165/detectron2.git

## for Windows
git clone -b ting_dev_win https://github.com/merlinz165/detectron2.git

# install detectron2
cd detectron2
python setup.py build develop
```

### install pytorch3d

```shell
cd ..

# clone specific branch of forked pytorch3d repo

git clone -b ting_dev https://github.com/merlinz165/pytorch3d.git

# install pytorch3d
cd pytorch3d
export FORCE_CUDA=1
python setup.py build develop
```

# Usage

in the `shape_estimation` directory:

Simply run `python inference.py` will infer all sample images.

run `python inference.py --help` for more options (e.g., single image, output path, focal length).
