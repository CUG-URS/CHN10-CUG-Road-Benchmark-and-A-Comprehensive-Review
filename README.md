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
In this code framework, we have integrated mainstream fully supervised road extraction methods from 2017 to 2024. Based on the Dinknet base framework, the complete reproduction of all the aforementioned methods can be achieved. Meanwhile, we have also open-sourced the GCBNet network framework proposed in the paper *A Global Context-aware and Batch-independent Network for road extraction from VHR satellite imagery*; for specific implementation details, please refer to the code under the `networks/GCBNet` directory.
|Type            | Year | Methods              | Paper | Code                                                                                                            |
|----------------|------|:---------------------|-------|:----------------------------------------------------------------------------------------------------------------|
|Fully Supervised| 2017 | CASNet               | (https://doi.org/10.1109/TGRS.2017.2669341) | [Pytorch](https://github.com/CUG-URS/CHN10-CUG-Road-Benchmark-and-A-Comprehensive-Review)  
|Fully Supervised| 2017 | DeepRoadMapper       | (https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf) | [Tensorflow](https://github.com/mitroadmaps/roadtracer/tree/master/deeproadmapper)     |
|Fully Supervised| 2018 | D-LinkNet            | (https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf) | [Pytorch](https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge)         |
|Fully Supervised| 2018 | RoadCNN              | (https://roadmaps.csail.mit.edu/roadtracer.pdf) | [Tensorflow](https://github.com/mitroadmaps/roadtracer)                                                         |
|Fully Supervised| 2019 | ImprovedConnectivity | (https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf) | [Pytorch](https://github.com/anilbatra2185/road_connectivity)                                                   |
|Fully Supervised| 2019 | SIINet               | (https://doi.org/10.1016/j.isprsjprs.2019.10.001) | [Pytorch](https://github.com/ErenTuring/SIINet/tree/master?tab=readme-ov-file)                                                   |
|Fully Supervised| 2020 | BSNet+Fusion         | (https://ieeexplore.ieee.org/document/9094008) | [Pytorch](https://github.com/astro-ck/Road-Extraction)                                                          |
|Fully Supervised| 2021 | GAMSNet              | (https://doi.org/10.1016/j.isprsjprs.2021.03.008) | [Pytorch](https://github.com/CUG-URS/CHN10-CUG-Road-Benchmark-and-A-Comprehensive-Review)                                                          |
|Fully Supervised| 2021 | CoANet               | (https://ieeexplore.ieee.org/document/9563125) | [Pytorch](https://github.com/mj129/CoANet)                                                                      |
|Fully Supervised| 2021 | DiResNet             | (https://arxiv.org/pdf/2005.07232) | [Pytorch](https://github.com/ggsDing/DiResNet)                                                                      |
|Fully Supervised| 2021 | SGCN                 | (https://doi.org/10.1109/TGRS.2021.3128033) | [Pytorch](https://github.com/tist0bsc/SGCN)                                                                      |
|Fully Supervised| 2021 | GCB-Net              | (https://doi.org/10.1016/j.isprsjprs.2021.03.016) | [Pytorch](https://github.com/CUG-URS/CHN10-CUG-Road-Benchmark-and-A-Comprehensive-Review)                                                                      |
|Fully Supervised| 2024 | MSMDFF-Net           | (https://ieeexplore.ieee.org/document/10477437) | [Pytorch](https://github.com/wycloveinfall/MSMDFF-NET)                                                          |
|Fully Supervised| 2024 | OARENet              | (https://doi.org/10.1109/TGRS.2024.3387945) | [Pytorch](https://github.com/WanderRainy/OARENet)                                                          |
|Weekly Supervised| 2021 | ScRoadExtractor      | (https://ieeexplore.ieee.org/document/9372390) | [Pytorch](https://github.com/weiyao1996/ScRoadExtractor)                                                        |
|Semi Supervised  | 2023 | SemiRoadExNet        |  (https://www.sciencedirect.com/science/article/pii/S0924271623000722) | [Pytorch](https://github.com/hchen118/SemiRoadExNet)                                                            |
|Semi Supervised  | 2024 | MCMCNet              |  (https://doi.org/10.1109/TGRS.2024.3426561) | [Pytorch](https://github.com/zhouyiqingzz/MCMCNet)                                                            |
|Graph| 2018 | RoadTracer           | (https://roadmaps.csail.mit.edu/roadtracer.pdf) | [Tensorflow](https://github.com/mitroadmaps/roadtracer)                                                         |
|Graph| 2020 | Sat2Graph            | (https://arxiv.org/pdf/2007.09547) | [Tensorflow](https://github.com/songtaohe/Sat2Graph)                                                            |
|Graph| 2020 | VecRoad              | (https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_VecRoad_Point-Based_Iterative_Graph_Exploration_for_Road_Graphs_Extraction_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/tansor/VecRoad) |
|Graph| 2022 | RNGDet               | (https://ieeexplore.ieee.org/abstract/document/9810294) | [Pytorch](https://github.com/TonyXuQAQ/RNGDetPlusPlus)                                                          |
|Graph| 2023 | RNGDet++             | (https://ieeexplore.ieee.org/abstract/document/10093124) | [Pytorch](https://github.com/TonyXuQAQ/RNGDetPlusPlus)                                                          |
|Graph| 2024 | SAM-Road             | (https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/papers/Hetang_Segment_Anything_Model_for_Road_Network_Graph_Extraction_CVPRW_2024_paper.pdf) | [Pytorch](https://github.com/htcr/sam_road)              |
|Mmsegmentation|  |             |   | (https://github.com/open-mmlab/mmsegmentation) |

| Datasets                                                                                   | Year    | Resolution (m/pixel)| Annotation Type         | Size (pixels)| Images(train/val/test) | Paper                                                                            |
|:-------------------------------------------------------------------------------------------|:--------|---------------------|-------------------------|--------------|------------------------|----------------------------------------------------------------------------------|
| [Massachusetts](https://www.cs.toronto.edu/~vmnih/data/)                                   | 2013    | 1                   | Surface                 | 1500×1500    | 1108/14/49             | [Paper](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf)    |
| [RoadTracer ](https://roadmaps.csail.mit.edu/roadtracer/)                                  | 2018    | 0.6                 | Centerline	Graph        | 1024×1024    | 2880/-/1920            | [Paper](https://roadmaps.csail.mit.edu/roadtracer.pdf)                           |
| [SpaceNet 3: Road Network Detection](https://spacenet.ai/spacenet-roads-dataset/)          | 2018    | 0.3                 | Centerline	Graph        | 1300×1300    | 2213/-/567             | [Paper](https://arxiv.org/pdf/1807.01232)                                        |
| [DeepGlobe](https://competitions.codalab.org/competitions/18467#participate-get_data)      | 2018    | 0.5                 | Surface                 | 1024×1024    | 4696/-/1530            | [Paper](https://arxiv.org/pdf/1805.06561)                                        |
| [CityScale](https://github.com/songtaohe/Sat2Graph)                                        | 2020    | 1                   | Centerline	Graph        | 2048×2048    | 144/9/27               | [Paper](https://arxiv.org/pdf/2007.09547)                                        |
| [WHU-Road](https://github.com/fightingMinty/WHU-road-dataset?tab=readme-ov-file)           | 2020    | 0.8~2.0             | Surface                 | 512×512      | -/-/-                  | [Paper](https://doi.org/10.1016/j.isprsjprs.2020.08.019)                         |
| [CHN6-CUG](https://github.com/CUG-URS/CHN6-CUG-Roads-Dataset)                              | 2021    | 0.5                 | Surface                 | 512×512      | 3608/-/903             | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271621000873) |
| [LSRV](http://rsidea.whu.edu.cn/resource_LSRV_sharing.htm)                                 | 2021    | 0.3~0.6             | Surface                 | 16640~23552  | -/-/3                  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271621000770) |
| [GSRV](https://github.com/xiaoyan07/GRNet_GRSet)                                           | 2024    | 0.3~1.2             | Surface                 | 1024~36,864  | -/-/5743               | [Paper](https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2362760?src=) |
| [GRSet](https://github.com/xiaoyan07/GRNet_GRSet)                                          | 2024    | 1                   | Centerline              | 1024×1024    | 47,210/-/-             | [Paper](https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2362760?src=) |
| [Global-Scale](https://github.com/earth-insights/samroadplus)                              | 2024    | 1                   | Centerline	Graph        | 1024×1024    | 2375/339/624+130       | [Paper](https://arxiv.org/pdf/2411.16733)                                        |
| [CHN10-CUG](https://github.com/CUG-URS/CHN10-CUG-Road-Benchmark-and-A-Comprehensive-Review)| 2025    | 0.5                 | Surface Centerline	Graph| 512×512      | 6015/962/1043          | [Paper]                                                                          |

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

