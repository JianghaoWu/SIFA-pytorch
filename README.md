# SIFA-pytorch
This is a PyTorch implementation of SIFA for 'Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation.'

If you find this code useful, please consider citing my UDA research: [*FPL-UDA: Filtered Pseudo Label-Based Unsupervised Cross-Modality Adaptation for Vestibular Schwannoma Segmentation*](https://ieeexplore.ieee.org/abstract/document/9761706). You can access the research paper [here](https://ieeexplore.ieee.org/abstract/document/9761706), and the code is also available [here](https://github.com/JianghaoWu/FPL-UDA).

### 1. Dataset

If you wish to utilize the provided UnpairedDataset, please prepare your dataset in the following format. Please note that each individual data unit should be stored in an NPZ file, where '[arr_0]' contains the image data, and '[arr_1]' contains the corresponding labels:
```
your/data_root/
       source_domain/
          s001.npz
            ['arr_0']:imgae_arr
            ['arr_1']:label_arr
          s002.npz
          ...

       target_domain/
          t001.npz
            ['arr_0']:imgae_arr
            ['arr_1']:label_arr
          t002.npz
          ...
       test/
          t101.npz
            ['arr_0']:imgae_arr
            ['arr_1']:label_arr
          t102.npz
          ...
```

### 2. Perform experimental settings in ```config/train.cfg```

### 3. Train SIFA
```
CUDA_LAUNCH_BLOCKING=0 python train.py
```

### 4. Test SIFA
```
CUDA_LAUNCH_BLOCKING=0 python test.py
```


#### If you find the code useful, please consider citing the following article:

```bibtex
@inproceedings{wu2022fpl,
  title={FPL-UDA: Filtered Pseudo Label-Based Unsupervised Cross-Modality Adaptation for Vestibular Schwannoma Segmentation},
  author={Wu, Jianghao and Gu, Ran and Dong, Guiming and Wang, Guotai and Zhang, Shaoting},
  booktitle={2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
@article{dorent2023crossmoda,
  title={CrossMoDA 2021 challenge: Benchmark of cross-modality domain adaptation techniques for vestibular schwannoma and cochlea segmentation},
  author={Dorent, Reuben and Kujawa, Aaron and Ivory, Marina and Bakas, Spyridon and Rieke, Nicola and Joutard, Samuel and Glocker, Ben and Cardoso, Jorge and Modat, Marc and Batmanghelich, Kayhan and others},
  journal={Medical Image Analysis},
  volume={83},
  pages={102628},
  year={2023},
  publisher={Elsevier}
}
```
