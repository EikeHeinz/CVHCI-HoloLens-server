# Requirements Installation

* Python 3.8
* CUDA(version >= 10.1)

## Manually installation

### (optional)activate virtualenv

`source virtual-env/bin/activate`

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
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.3'
```

### install pytorch3d

```shell
export FORCE_CUDA=1
python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.2.5'
```

## Using requirements

*Install Pytorch before running `pip install -r requirements`*
