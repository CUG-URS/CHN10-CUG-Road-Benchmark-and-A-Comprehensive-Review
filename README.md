# CHN10-CUG-Road-Benchmark-and-A-Comprehensive-Review
Advancing large-scale road mapping with deep learning: A comprehensive review and the CHN10-CUG benchmark

# CHN10-CUG-Roads-Dataset
Version 1.0

## 1.Overview

<img width="1061" height="537" alt="image" src="https://github.com/user-attachments/assets/5b2652af-bc7d-45ca-9f31-332687eb6a2f" />
Fig. 1. Difficulties in road extraction from high resolution remote sensing imagery.
<img width="1036" height="527" alt="image" src="https://github.com/user-attachments/assets/16a3787d-d627-45c4-803e-d0c842f80f77" />
Fig. 2. CHN10-CUG: a high-resolution Chinese road dataset with richer annotations, greater intra-class diversity, and finer details.
<img width="717" height="601" alt="image" src="https://github.com/user-attachments/assets/9b2f30bc-ffba-4dbc-96d2-c6b590f381b4" />
Fig. 3. Urban distribution in CHN10-CUG Roads Dataset.


## 2.Dataset

dataset tree:

```
CHN10-CUG
└───train
│   └───gt
│   └───images
└───val
│   └───gt
│   └───images
└───test
│   └───gt
│   └───images
└───Graph

```

---------

We collected typical areas of ten cities in China from [Google Earth](http://earth.google.com). The images are with 50 cm/pixel spatial resolution. 

**Please note that we do not own the copyrights to these original satellite images. Their use is RESTRICTED to non-commercial research and educational purposes.**

### 3.Road review


Code：

Mmsegmentation https://github.com/open-mmlab/mmsegmentation

DeepRoadMapper 2017 https://github.com/mitroadmaps/roadtracer/tree/master/deeproadmapper

Roadtracer 2018 https://github.com/mitroadmaps/roadtracer

D-LinkNet 2018 https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge

SIINet 2019 https://github.com/ErenTuring/SIINet

Seg-Orientation 2019 https://github.com/anilbatra2185/road_connectivity

Sat2Graph 2020 https://github.com/songtaohe/Sat2Graph

BSNet 2020 https://github.com/astro-ck/Road-Extraction

CoANet 2021 https://github.com/mj129/CoANet

ScRoadExtractor 2021 https://github.com/weiyao1996/ScRoadExtractor

DiResNet 2021 https://github.com/ggsDing/DiResNet

SGCNNet 2021 https://github.com/tist0bsc/SGCN

RNGDet 2022 https://github.com/TonyXuQAQ/RNGDetPlusPlus

RNGDet++ 2023 https://github.com/TonyXuQAQ/RNGDetPlusPlus

SemiRoadExNet 2023 https://github.com/hchen118/SemiRoadExNet

MCMCNet 2024 https://github.com/zhouyiqingzz/MCMCNet

OARENet 2024 https://github.com/WanderRainy/OARENet

MSMDFF-Net 2024 https://github.com/wycloveinfall/MSMDFF-NET

SAM-Road 2024 https://github.com/htcr/sam_road


Dataset：

DeepGlobe Road Extraction Challenge https://competitions.codalab.org/competitions/18467#participate-get_data

SpaceNet 3: Road Network Detection https://spacenet.ai/spacenet-roads-dataset/

Roadtracer https://roadmaps.csail.mit.edu/roadtracer/

Massachusetts Roads Dataset https://www.cs.toronto.edu/~vmnih/data/

LSRV: The Large-Scale Road Validation Dataset https://rsidea.whu.edu.cn/resource_LSRV_sharing.htm

CityScale https://github.com/songtaohe/Sat2Graph

CHN6-CUG https://github.com/CUG-URS/CHN6-CUG-Roads-Dataset

WHU-Road https://github.com/fightingMinty/WHU-road-dataset?tab=readme-ov-file

GRSet https://github.com/xiaoyan07/GRNet_GRSet

### 4.Download

Download link: 

 - [BaiduYun]( https://pan.baidu.com/s/1uEMawOsHjn88q8uMqpaIvw?pwd=4nfp)（Password: 4nfp）
 - [Google Drive]( https://drive.google.com/drive/folders/1CyqtgqnoQaqHt7b_EF7ubCQ8zl_dtk4t?usp=drive_link)

## 5.Reference

Please cite this paper if you use this dataset:

```
@article{zhu2021global,
  title={A Global Context-aware and Batch-independent Network for road extraction from VHR satellite imagery},
  author={Zhu, Qiqi and Zhang, Yanan and Wang, Lizeng and Zhong, Yanfei and Guan, Qingfeng and Lu, Xiaoyan and Zhang, Liangpei and Li, Deren},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={175},
  pages={353--365},
  year={2021},
  publisher={Elsevier}
}
```
## 6.Contact
If you have any problem in using the CHN10-CUG Roads Dataset, please contact: pss@cug.edu.cn

For any possible research collaboration, please contact Prof. Qiqi Zhu (zhuqq@cug.edu.cn).

The homepage of our academic group is: http://grzy.cug.edu.cn/zhuqiqi/en/index.htm.

