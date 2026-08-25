# <font color=#5dbe8a>SoftFormer</font>
## Intorduction
SoftFormer is a deep learning network designed for urban land cover and land use classification based on SAR and Optical data fusion. The rationale behind SoftFormer is:<br> 

(1) Feature Extraction: balancing the local and global feature extraction capabilities of the network.<br>

(2) Feature Fusion: enhancing the classification accuracy of heterogenous remote sensing data by multi-level (feature level and decision level) fusion. When fusing, the complementarity and redundancy of heterogeneous source features need to be considered, as well as the issue of the credibility of the classification results of different modalities.<br>

SoftFormer is a patch based **classification** network rather than **segmentation** network. When the labeled samples are prepared by ENVI or GEE, you can use SoftFormer to do classification.

## Citations
If this work is helpful to you, please consider citing the paper via the following BibTex entry.
```bibtex
@article{LIU2024277,
title = {SoftFormer: SAR-optical fusion transformer for urban land use and land cover classification},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {218},
pages = {277-293},
year = {2024},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2024.09.012},
author = {Rui Liu and Jing Ling and Hongsheng Zhang},
}
```


