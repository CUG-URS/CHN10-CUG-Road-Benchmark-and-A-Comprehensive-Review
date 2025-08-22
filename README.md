# CHN10-CUG-Road-Benchmark-and-A-Comprehensive-Review
Advancing large-scale road mapping with deep learning: A comprehensive review and the CHN10-CUG benchmark
1.Overview
 
  
Achieving accurate road mapping across large-scale, geographically dispersed areas remains a major challenge in the field of remote sensing. While deep learning has achieved remarkable progress in road extraction in recent years, the continued enhancement of its performance largely depends on the availability of high-quality and diverse training data. However, existing public datasets are limited in diversity, typically including only a few road categories. Moreover, roads and backgrounds in these datasets are often easily distinguishable, failing to reflect real-world challenges such as occlusions, complex backgrounds, and radiation differences in high-resolution remote sensing imagery. These limitations make it difficult to achieve accurate and consistent road mapping at a national scale, especially given the diversity of geographical and urban landscapes. In this paper, we present CHN10-CUG, a new high-resolution remote sensing road dataset that covers roads across 10 representative cities in China. The dataset includes detailed annotations of both road surfaces and centerlines. Compared to existing datasets, CHN10-CUG provides more diverse spectral, textural, geometric, and topological features, making it particularly suitable for multi-task road extraction tasks such as surface segmentation and centerline extraction. Furthermore, this paper presents a systematic review of existing deep learning-based road extraction methods. Specifically, we categorize and describe road methods, then we reproduce and rigorously evaluate 15 widely used semantic segmentation models alongside 23 representative road extraction methods on the CHN10-CUG dataset.
2.Dataset
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

We collected typical areas of six cities in China from Google Earth. The images are with 50 cm/pixel spatial resolution.
3.Download
Download link:
BaiduYun
https://pan.baidu.com/s/1uEMawOsHjn88q8uMqpaIvw?pwd=4nfp 提取码: 4nfp 
Google Drive
https://drive.google.com/drive/folders/1CyqtgqnoQaqHt7b_EF7ubCQ8zl_dtk4t?usp=drive_link
4.Reference

5.Contact
For any possible research collaboration, please contact Prof. Qiqi Zhu (zhuqq@cug.edu.cn).
The homepage of our academic group is: http://grzy.cug.edu.cn/zhuqiqi/en/index.htm.

