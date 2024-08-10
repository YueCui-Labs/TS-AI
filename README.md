## TS-AI: a deep learning pipeline for multimodal subject-specific parcellation with task contrasts synthesis
Accurate mapping of brain functional subregions at an individual level is crucial. Task-based functional MRI (tfMRI) captures subject-specific activation patterns during various functions and behaviors, facilitating the individual localization of functionally distinct subregions. However, acquiring high-quality tfMRI is time-consuming and resource-intensive in both scientific and clinical settings. TS-AI is a two-stage network model that individualizes an atlas on cortical surfaces through the prediction of tfMRI data. TS-AI first synthesizes a battery of task contrast maps for each individual by leveraging tract-wise anatomical connectivity and resting-state networks. These synthesized maps, along with feature maps of tract-wise anatomical connectivity and resting-state networks, are then fed into an end-to-end deep neural network to individualize an atlas. TS-AI enables the synthesized task contrast maps to be used in individual parcellation without the acquisition of actual task fMRI scans. In addition, a novel feature consistency loss is designed to assign vertices with similar features to the same parcel, which increases individual specificity and mitigates overfitting risks caused by the absence of individual parcellation ground truth.

![Fig1](https://github.com/user-attachments/assets/cd8a7a4e-43f0-4e4c-8e43-29d7920b0274)

### Data release
Download the weights of TS-AI from: 
https://drive.google.com/file/d/1dfo2HjdynruXH7pH7D6P1jz1I7Doxpwi/view?usp=drive_link .

TS-AI-derived individualized parcellations based on Glasser atlas and Brannetome atlas for HCP subjects is available on 
https://drive.google.com/file/d/109_Yc7YhHkdq2CPYow-Jo1vtygF6E5QZ/view?usp=drive_link .

### :books: Citation
Please cite the [following paper](https://doi.org/10.1016/j.media.2024.103297) when using TS-AI:

Chengyi Li, Yuheng Lu, Shan Yu, Yue Cui. TS-AI: A deep learning pipeline for multimodal subject-specific parcellation with task contrasts synthesis, Medical Image Analysis, 103297, 2024
