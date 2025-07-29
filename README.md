# TransFlow

This is the official PyTorch implementation of our paper:

> **TransFlow: Motion Knowledge Transfer from Video Diffusion Models to Video Salient Object Detection**, *ICCVW 2025*\
> Suhwan Cho, Minhyeok Lee, Jungho Lee, Sunghun Yang, Sangyoun Lee\
> Link: [[arXiv]](https://arxiv.org/abs/2507.19789)

<img src="https://github.com/user-attachments/assets/4466342a-4eed-4b55-a13d-f85f0972db9f" width=800>

You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
Leveraging large-scale image datasets is a common strategy in video processing tasks. However, motion-guided 
approaches are limited in this regard, as spatial distortions cannot effectively simulate realistic motion dynamics. 
In this study, we demonstrate that **image-to-video generation models** can generate realistic optical flows by appropriately 
transforming static images. Our synthetic video dataset, **DUTS-Video**, offers valuable potential for future research.



## Setup
1\. Download the datasets:
[DUTS](http://saliencydetection.net/duts/#org3aad434), 
[DAVIS16](https://davischallenge.org/davis2016/code.html),
[DAVSOD](https://github.com/DengPingFan/DAVSOD),
[FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets),
[ViSal](https://github.com/wenguanwang/ViSal-dataset-for-video-salient-object-detection).

2\. Estimate and save optical flow maps from the videos using [RAFT](https://github.com/princeton-vl/RAFT).

3\. For DUTS, simulate future frames and optical flow maps using [Stable Video Diffusion](https://github.com/Stability-AI/generative-models).

4\. I also provide the pre-processed datasets:
[DUTS-Video](https://drive.google.com/file/d/1e2CFX4QIzlRAhjoEkEhxsLwqNOGSMXnC/view?usp=drive_link),
[DAVIS16](https://drive.google.com/file/d/10qzxi6bNkM44SXYjjX2BDra7qQMUdbot/view?usp=drive_link),
[FBMS](https://drive.google.com/file/d/1arrBPuhjnwlKlgmCZveUsKw5uForr4JN/view?usp=drive_link),
[DAVSOD](https://drive.google.com/file/d/1zJxuejRJOXD9Wd7D5jH3_KkoUikvpYH3/view?usp=drive_link),
[ViSal](https://drive.google.com/file/d/19FNPm-kPS_x_IoPBHSHaLW18L3xMK1VN/view?usp=drive_link).



##  Running 

### Training
Start TransFlow training with:
```
python run.py --train
```

Verify the following before running:\
✅ Training dataset selection and configuration\
✅ GPU availability and configuration\
✅ Backbone network selection


### Testing
Run TransFlow with:
```
python run.py --test
```

Verify the following before running:\
✅ Testing dataset selection\
✅ GPU availability and configuration\
✅ Backbone network selection\
✅ Pre-trained model path


## Attachments
[Pre-trained model (mitb0)](https://drive.google.com/file/d/1by9qDMr-sKL7J8MzVJdDAjysUCH5hutM/view?usp=drive_link)\
[Pre-trained model (mitb1)](https://drive.google.com/file/d/1yp_bT6ABl0uZ92s27bVVEOwUd-MjnY3T/view?usp=drive_link)\
[Pre-trained model (mitb2)](https://drive.google.com/file/d/1ha_zzleAe7UbREf8zGxXNV3Plhtwz9UK/view?usp=drive_link)\
[Pre-computed results](https://drive.google.com/file/d/1VD4IWBPJai-WUR8wQnDq_6ms2HCccx0F/view?usp=drive_link)


## Contact
Code and models are only available for non-commercial research purposes.\
For questions or inquiries, feel free to contact:
```
E-mail: suhwanx@gmail.com
```

