# CHN10-CUG-Road-Benchmark-and-A-Comprehensive-Review
Advancing large-scale road mapping with deep learning: A comprehensive review and the CHN10-CUG benchmark

# CHN10-CUG-Roads-Dataset
Version 1.0

## 1.Overview
Achieving accurate road mapping across large-scale, geographically dispersed areas remains a major challenge in the field of remote sensing. While deep learning has achieved remarkable progress in road extraction in recent years, the continued enhancement of its performance largely depends on the availability of high-quality and diverse training data. However, existing public datasets are limited in diversity, typically including only a few road categories. Moreover, roads and backgrounds in these datasets are often easily distinguishable, failing to reflect real-world challenges such as occlusions, complex backgrounds, and radiation differences in high-resolution remote sensing imagery. These limitations make it difficult to achieve accurate and consistent road mapping at a national scale, especially given the diversity of geographical and urban landscapes. In this paper, we present CHN10-CUG, a new high-resolution remote sensing road dataset that covers roads across 10 representative cities in China. The dataset includes detailed annotations of both road surfaces and centerlines. Compared to existing datasets, CHN10-CUG provides more diverse spectral, textural, geometric, and topological features, making it particularly suitable for multi-task road extraction tasks such as surface segmentation and centerline extraction. Furthermore, this paper presents a systematic review of existing deep learning-based road extraction methods. Specifically, we categorize and describe road methods, then we reproduce and rigorously evaluate 15 widely used semantic segmentation models alongside 23 representative road extraction methods on the CHN10-CUG dataset.
<img width="709" height="594" alt="image" src="https://github.com/user-attachments/assets/c2b97340-88d4-459e-93f0-7373eec095e9" />
<img width="709" height="361" alt="image" src="https://github.com/user-attachments/assets/6dd0faab-70cd-4b89-82be-c8d38b3afa01" />
<img width="709" height="358" alt="image" src="https://github.com/user-attachments/assets/64dfeff5-0618-416a-ba32-6b6074a2bd54" />



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
mmsegmentation
https://github.com/open-mmlab/mmsegmentation

DeepRoadMapper 2017 
https://github.com/mitroadmaps/roadtracer/tree/master/deeproadmapper

Roadtracer 2018 
https://github.com/mitroadmaps/roadtracer

D-LinkNet 2018 
https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge

SIINet 2019
https://github.com/ErenTuring/SIINet

Seg-Orientation 2019  https://github.com/anilbatra2185/road_connectivity

Sat2Graph 2020 
https://github.com/songtaohe/Sat2Graph

BSNet 2020
https://github.com/astro-ck/Road-Extraction

CoANet 2021
https://github.com/mj129/CoANet

ScRoadExtractor 2021 
https://github.com/weiyao1996/ScRoadExtractor

DiResNet 2021
https://github.com/ggsDing/DiResNet

SGCNNet 2021
https://github.com/tist0bsc/SGCN

RNGDet 2022
https://github.com/TonyXuQAQ/RNGDetPlusPlus

RNGDet++ 2023 
https://github.com/TonyXuQAQ/RNGDetPlusPlus

SemiRoadExNet 2023 
https://github.com/hchen118/SemiRoadExNet

MCMCNet 2024
https://github.com/zhouyiqingzz/MCMCNet

OARENet 2024
https://github.com/WanderRainy/OARENet

MSMDFF-Net 2024 
https://github.com/wycloveinfall/MSMDFF-NET

SAM-Road 2024
https://github.com/htcr/sam_road


Dataset：
DeepGlobe Road Extraction Challenge
https://competitions.codalab.org/competitions/18467#participate-get_data

SpaceNet 3: Road Network Detection
https://spacenet.ai/spacenet-roads-dataset/

Roadtracer
https://roadmaps.csail.mit.edu/roadtracer/

Massachusetts Roads Dataset
https://www.cs.toronto.edu/~vmnih/data/

LSRV: The Large-Scale Road Validation Dataset
https://rsidea.whu.edu.cn/resource_LSRV_sharing.htm

CityScale
https://github.com/songtaohe/Sat2Graph

CHN6-CUG
https://github.com/CUG-URS/CHN6-CUG-Roads-Dataset

WHU-Road
https://github.com/fightingMinty/WHU-road-dataset?tab=readme-ov-file

GRSet
https://github.com/xiaoyan07/GRNet_GRSet

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
If you have any problem in using the CHN10-CUG Roads Dataset, please contact: 1946675524@qq.com

For any possible research collaboration, please contact Prof. Qiqi Zhu (zhuqq@cug.edu.cn).

The homepage of our academic group is: http://grzy.cug.edu.cn/zhuqiqi/en/index.htm.

