# LaDeco in OneFormer segmentation engine

Since MXNet is not user friendly enough at installation for Microsoft Windows© user even today, here we provide another choice which is implemented on PyTorch.

# Installation

It is the most convient to install using [Anaconda][] or [Miniconda][].

## Windows & Linux

```console
conda env create
```

For CPU only inference, use `environ_cpu.yml` to install
```console
conda env create --file environ_cpu.yml
```

## MacOS

```bash
conda env create --file environ_MacOS.yml
```

Note, MPS acceleration is available on MacOS 12.3+.


[Anaconda]: https://www.anaconda.com/download
[Miniconda]: https://docs.conda.io/projects/miniconda/en/latest/

# Usage

Please refer to [upper README.md](../README.md#Usage)


# Citation

[OneFormer](https://arxiv.org/abs/2211.06220) comes from their great work, big thanks!
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
