# <p align=center>  `CrossKD: Cross-Head Knowledge Distillation for Dense Object Detection` </p>

![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.0-blue.svg) 

This repository contains the official implementation of the following paper:
> **CrossKD: Cross-Head Knowledge Distillation for Dense Object Detection**<br>
> [Jiabao Wang](https://scholar.google.co.uk/citations?hl=en&user=S9ErhhEAAAAJ)<sup>\*</sup>, [Yuming Chen](https://github.com/FishAndWasabi/)<sup>\*</sup>, [Zhaohui Zheng](https://scholar.google.co.uk/citations?hl=en&user=0X71NDYAAAAJ)，[Xiang Li](http://implus.github.io/), [Ming-Ming Cheng](https://mmcheng.net/cmm), [Qibin Hou](https://houqb.github.io/)<sup>\*</sup>  <br>
> (\* denotes equal contribution) <br>
> VCIP, School of Computer Science, Nankai University <br>
> In ICCV 2023 (underreview) <br>


[[Arxiv Paper](https://arxiv.org/abs/2306.11369)]
[中文版 (TBD)]
[Website Page]

## Introduction

Knowledge Distillation (KD) has been validated as an effective model compression technique for learning compact object detectors.
Existing state-of-the-art KD methods for object detection are mostly based on feature imitation, which is generally observed to be better than prediction mimicking. 
In this paper, we show that the inconsistency of the optimization objectives between the ground-truth signals and distillation targets is the key reason for the inefficiency of prediction mimicking.
To alleviate this issue, we present a simple yet effective distillation scheme, termed CrossKD, which delivers the intermediate features of the student's detection head to the teacher's detection head. 
The resulting cross-head predictions are then forced to mimic the teacher's predictions.
Such a distillation manner relieves the student's head from receiving contradictory supervision signals from the ground-truth annotations and the teacher's predictions, greatly improving the student's detection performance.
On MS COCO, with only prediction mimicking losses applied, our CrossKD boosts the average precision of GFL ResNet-50 with 1x training schedule from 40.2 to 43.7, outperforming all existing KD methods for object detection.



![struture](assets/structure.png)


## Dependencies and Installation

- Ubuntu >= 20.04
- CUDA >= 11.3
- mmdetection
- mmcv
- Other required packages in `requirements.txt`






## Get Started
### Prepare pretrained models & dataset


## Citation
If you find our repo useful for your research, please cite us:
```
@misc{wang2023crosskd,
      title={CrossKD: Cross-Head Knowledge Distillation for Dense Object Detection}, 
      author={Jiabao Wang and Yuming Chen and Zhaohui Zheng and Xiang Li and Ming-Ming Cheng and Qibin Hou},
      year={2023},
      eprint={2306.11369},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

## Contact

For technical questions, please contact `jiabaowang[AT]mail.nankai.edu.cn` and `chenyuming[AT]mail.nankai.edu.cn`.

For commercial licensing, please contact `cmm[AT]nankai.edu.cn` and `andrewhoux[AT]gmail.com`


## Acknowledgement


This repo is based on mmDetection.
