# LaDeco: A Tool to Analyze Visual Landscape Elements
Citation: Li-Chih Ho (2024). LaDeco: Atool to Analyze Visual Landscape Elemnts. Ecological Informatics,78,102289.
link: https://www.sciencedirect.com/science/article/pii/S1574954123003187?dgcid=author

## Abstract
  The assessment of visual landscape elements plays a crucial role in landscape change studies, aesthetic evaluation, and visual impact assessment. The proportions and statistical distributions of these elements are key factors that significantly influence these domains. Historically, the analysis has been performed manually, a process that is both labor intensive and time consuming, particularly when dealing with large assessment regions. To address this limitation, this study employs cutting-edge artificial intelligence technology to introduce an automated tool called LaDeco (Landscape Decoder). This tool enables researchers, planners, and evaluators to rapidly and objectively calculate the proportions of visual elements in images, thereby streamlining the assessment process.

## Installation

### Linux or MaxOS

The easiest way to install LaDeco's dependencies is installing via `Conda` package manager:
```bash
$ conda env create
```

### Windows

#### RTX 30 series and newer

Latest version of MXNet are not currently support Windows platform well.  
It is recommanded to install LaDeco in [WSL2][] via `Conda` package manager:
```console
>conda env create
```

#### RTX 20 series and older

If your VGA card are RTX 20 series or older ones, it is possible to install the dependencies
directly on your Windows machine by following commands.

```console
>conda create -n ladeco python==3.8
>conda activate ladeco
>conda install cudatoolkit=10.2
>pip install mxnet-cu102==1.7.0
>pip install gluoncv
```

Optionally, you may want to set environment variable `MXNET_CUDNN_AUTOTUNE_DEFAULT` to `0`.


```console
>conda env config vars set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
```

Sometimes, the auto-tune process will cause a problem and leads to a much slower initialization.


[WSL2]: https://learn.microsoft.com/en-us/windows/wsl/install

## Usage

`ladeco-v11.py` is a command line program to analysis all images in a folder recursively. Following commands shows how to use it.

Activate virtual environment:
```bash
$ conda activate ladeco
```

Run `ladeco-v11.py`:
```bash
(ladeco) $ python ladeco-v11.py "PATH/TO/IMAGE/DIRECTORY"
```

Suppose we have a directory like this:
```bash
$ tree /data/images
/data/images
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── subdir
    ├── img4.jpg
    └── img5.jpg
```

To analyze the directory:
```bash
(ladeco) $ python ladeco-v11.py /data/images
```

The result will be save to a directory under current working directory with timestamp (format of `yyyy_mmdd_HHMMSS_LADECO`).  
The result CSV file will be like:

| fid                          | L1_Nature | L1_man-made | L2_landform | ... | others | LC_NFI
|------------------------------|-----------|-------------|-------------|-----|--------|---------
| /data/images/img1.jpg        | 0.999     | 0.0         | 0.452       | ... | 0.001  | 1.0
| /data/images/img2.jpg        | 0.991     | 0.0         | 0.078       | ... | 0.009  | 1.0
| /data/images/img3.jpg        | 1.0       | 0.0         | 0.313       | ... | 0.0    | 1.0
| /data/images/subdir/img4.jpg | 1.0       | 0.0         | 0.397       | ... | 0.0    | 1.0
| /data/images/subdir/img5.jpg | 1.0       | 0.0         | 0.583       | ... | 0.0    | 1.0

Other options are listed below:
```bash
$ python ladeco-v11.py -h
usage: ladeco-v11.py [-h] [-m MODEL] [-t THRESHOLD] [-d DEVICE] IMG_FOLDER

Calculate the proportion of landscape elements in each image within the folder, with each image being computed independently.

positional arguments:
  IMG_FOLDER            path to image folder.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The semantic segmantation model.
                        Choose one from below:
                        fcn_resnet50_ade
                        fcn_resnet101_ade
                        psp_resnet50_ade
                        psp_resnet101_ade
                        deeplab_resnet50_ade
                        deeplab_resnet101_ade
                        deeplab_resnest50_ade
                        deeplab_resnest101_ade
                        deeplab_resnest200_ade
                        deeplab_resnest269_ade (default)
  -t THRESHOLD, --threshold THRESHOLD
                        The thresold to round an element to zero. (default 0.01)
  -d DEVICE, --device DEVICE
                        Use CPU or GPU.
                        Choice are:
                        - literal "auto": program will attempt to utilize GPU. If failed, use CPUs. (default)
                        - literal "cpu": use CPUs.
                        - literal "gpu": use the first GPU device.
                        - a digit like 0, 1, or 2: Use the GPU device specified by the index.
```
