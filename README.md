## TS-AI: a deep learning pipeline for multimodal subject-specific parcellation with task contrasts synthesis
Accurate mapping of brain functional subregions at an individual level is crucial. Task-based functional MRI (tfMRI) captures subject-specific activation patterns during various functions and behaviors, facilitating the individual localization of functionally distinct subregions. However, acquiring high-quality tfMRI is time-consuming and resource-intensive in both scientific and clinical settings. TS-AI is a two-stage network model that individualizes an atlas on cortical surfaces through the prediction of tfMRI data. TS-AI first synthesizes a battery of task contrast maps for each individual by leveraging tract-wise anatomical connectivity and resting-state networks. These synthesized maps, along with feature maps of tract-wise anatomical connectivity and resting-state networks, are then fed into an end-to-end deep neural network to individualize an atlas. TS-AI enables the synthesized task contrast maps to be used in individual parcellation without the acquisition of actual task fMRI scans. In addition, a novel feature consistency loss is designed to assign vertices with similar features to the same parcel, which increases individual specificity and mitigates overfitting risks caused by the absence of individual parcellation ground truth.

![Fig1](https://github.com/user-attachments/assets/cd8a7a4e-43f0-4e4c-8e43-29d7920b0274)

### Data release

- [Download the pretrained TS-AI weights](https://github.com/YueCui-Labs/TS-AI/releases/download/weights-v1.0.0/trained_models.zip) (1.19 GB).
- [Download the individualized HCP parcellations](https://github.com/YueCui-Labs/TS-AI/releases/download/weights-v1.0.0/TSAI_indiv_parcellation_HCP1200.zip) (52.37 MB). This archive contains Glasser and Brainnetome parcellations for both hemispheres of 968 HCP participants.

See the [release notes](https://github.com/YueCui-Labs/TS-AI/releases/tag/weights-v1.0.0) for archive contents and SHA-256 checksums.

### :books: Citation
Please cite the [following paper](https://doi.org/10.1016/j.media.2024.103297) when using TS-AI:

Chengyi Li, Yuheng Lu, Shan Yu, Yue Cui. TS-AI: A deep learning pipeline for multimodal subject-specific parcellation with task contrasts synthesis, Medical Image Analysis, 103297, 2024
