# LaDeco: A Tool to Analyze Visual Landscape Elements
Citation: Li-Chih Ho (2023). LaDeco: A Tool to Analyze Visual Landscape Elements. Ecological Informatics,78,102289.

link: https://doi.org/10.1016/j.ecoinf.2023.102289

## CHANGELOG
- 2025/6/3 The GloneCV Deeplab v3 + RestNext backbone, which functioned as LaDeco’s ADE20K semantic segmentation engine, has been removed due to its deprecated status.
- 2023/10/6 We offer a new version of LaDeco using the state-of-the-art OneFormer engine. Easier to install and more accurate.

## Abstract
  The assessment of visual landscape elements plays a crucial role in landscape change studies, aesthetic evaluation, and visual impact assessment. The proportions and statistical distributions of these elements are key factors that significantly influence these domains. Historically, the analysis has been performed manually, a process that is both labor intensive and time consuming, particularly when dealing with large assessment regions. To address this limitation, this study employs cutting-edge artificial intelligence technology to introduce an automated tool called LaDeco (Landscape Decoder). This tool enables researchers, planners, and evaluators to rapidly and objectively calculate the proportions of visual elements in images, thereby streamlining the assessment process.


![LaDeco-v11](https://github.com/lichihho/LaDeco/assets/35607785/4d846545-25dc-488c-a900-fe29e7ecf8d4)



## Installation

Installing via [Anaconda][] or [Miniconda][] is the most convenient method.

### Common

Run those commands before heading into OS-specific guide:
```console
git clone https://github.com/lichihho/LaDeco.git
cd LaDeco
```

### Windows & Linux

```console
conda env create
```

For CPU only inference, use `environ_cpu.yml` to install
```console
conda env create --file environ_cpu.yml
```

### MacOS

```bash
conda env create --file environ_MacOS.yml
```

* *MPS acceleration is available on MacOS 12.3+.


[Anaconda]: https://www.anaconda.com/download
[Miniconda]: https://docs.conda.io/projects/miniconda/en/latest/

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
                        shi-labs/oneformer_ade20k_swin_tiny
                        shi-labs/oneformer_ade20k_swin_large (default)
                        shi-labs/oneformer_ade20k_dinat_large
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

# Acknowledgment  

Our implementation relies on [OneFormer](https://arxiv.org/abs/2211.06220), whose foundational research was conducted by its original developers; we express our deep appreciation for their efforts.
```
@misc{jain2022oneformer,
      title={OneFormer: One Transformer to Rule Universal Image Segmentation}, 
      author={Jitesh Jain and Jiachen Li and MangTik Chiu and Ali Hassani and Nikita Orlov and Humphrey Shi},
      year={2022},
      eprint={2211.06220},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
