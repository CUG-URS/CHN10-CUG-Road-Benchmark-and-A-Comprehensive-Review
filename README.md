# CHN10-CUG-Road-Benchmark-and-A-Comprehensive-Review
Advancing large-scale road mapping with deep learning: A comprehensive review and the CHN10-CUG benchmark

# CHN10-CUG-Roads-Dataset
Version 1.0

## 1.Overview
Achieving accurate road mapping across large-scale, geographically dispersed areas remains a major challenge in the field of remote sensing. While deep learning has achieved remarkable progress in road extraction in recent years, the continued enhancement of its performance largely depends on the availability of high-quality and diverse training data. However, existing public datasets are limited in diversity, typically including only a few road categories. Moreover, roads and backgrounds in these datasets are often easily distinguishable, failing to reflect real-world challenges such as occlusions, complex backgrounds, and radiation differences in high-resolution remote sensing imagery. These limitations make it difficult to achieve accurate and consistent road mapping at a national scale, especially given the diversity of geographical and urban landscapes. In this paper, we present CHN10-CUG, a new high-resolution remote sensing road dataset that covers roads across 10 representative cities in China. The dataset includes detailed annotations of both road surfaces and centerlines. Compared to existing datasets, CHN10-CUG provides more diverse spectral, textural, geometric, and topological features, making it particularly suitable for multi-task road extraction tasks such as surface segmentation and centerline extraction. Furthermore, this paper presents a systematic review of existing deep learning-based road extraction methods. Specifically, we categorize and describe road methods, then we reproduce and rigorously evaluate 15 widely used semantic segmentation models alongside 23 representative road extraction methods on the CHN10-CUG dataset.
<img width="764" height="640" alt="image" src="https://github.com/user-attachments/assets/b8559df7-c81c-4e58-88aa-28dd2ae65ed4" />
<img width="418" height="213" alt="image" src="https://github.com/user-attachments/assets/314ecc2d-974a-4c5c-81dd-7fb3f1f9bf24" /><img width="420" height="213" alt="image" src="https://github.com/user-attachments/assets/e29617ce-e3e4-43a7-a403-b42919e3af2f" />


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

### 3.Download

Download link: 

 - [BaiduYun]( https://pan.baidu.com/s/1uEMawOsHjn88q8uMqpaIvw?pwd=4nfp)（Password: 4nfp）
 - [Google Drive]( https://drive.google.com/drive/folders/1CyqtgqnoQaqHt7b_EF7ubCQ8zl_dtk4t?usp=drive_link)

## 4.Reference

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
## 5.Contact
If you have any problem in using the CHN10-CUG Roads Dataset, please contact: 1946675524@qq.com

For any possible research collaboration, please contact Prof. Qiqi Zhu (zhuqq@cug.edu.cn).

The homepage of our academic group is: http://grzy.cug.edu.cn/zhuqiqi/en/index.htm.

