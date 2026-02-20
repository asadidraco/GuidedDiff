# Guided Diffusion for Precise Skin Lesion Segmentation: Integrating Spatial and Semantic Priors

This repository contains the official implementation of the manuscript:

**"Guided Diffusion for Precise Skin Lesion Segmentation: Integrating Spatial and Semantic Priors"**  
Submitted to *The Visual Computer*.

The proposed framework integrates pretrained U-Net-based spatial guidance and CLIP-based semantic guidance within a conditional denoising diffusion model for accurate skin lesion segmentation.

Users of this code are kindly requested to cite the corresponding paper.

---

## Model Architecture

The proposed framework consists of three main components:

1. A pretrained U-shaped network for coarse lesion localization  
2. A CLIP-based textual encoder for semantic guidance  
3. A denoising diffusion probabilistic model for refined segmentation  

<p align="center">
  <img src="model_architecture.png" alt="Guided Diffusion Framework Architecture" width="800"/>
  <br>
  <em>Figure 1: Overview of the proposed guided diffusion framework with spatial and semantic priors.</em>
</p>
---

## Requirements

We recommend using Python 3.8 or later.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Download the ISIC dataset from:

https://challenge.isic-archive.com/data/

Your dataset directory should be organized as follows:
```bash
data
└── ISIC
    ├── Test
    │   ├── ISBI2016_ISIC_Part1_Test_GroundTruth.csv
    │   ├── ISBI2016_ISIC_Part1_Test_Data
    │   │   ├── ISIC_0000003.jpg
    │   │   └── ...
    │   └── ISBI2016_ISIC_Part1_Test_GroundTruth
    │       ├── ISIC_0000003_Segmentation.png
    │       └── ...
    └── Train
        ├── ISBI2016_ISIC_Part1_Training_GroundTruth.csv
        ├── ISBI2016_ISIC_Part1_Training_Data
        │   ├── ISIC_0000000.jpg
        │   └── ...
        └── ISBI2016_ISIC_Part1_Training_GroundTruth
            ├── ISIC_0000000_Segmentation.png
            └── ...
```
Ensure that images and corresponding masks are correctly matched.

---

## Training

To train the guided diffusion segmentation model, run:
```bash
python scripts/segmentation_train.py \
--data_name ISIC \
--data_dir <input_data_directory> \
--out_dir <output_directory> \
--image_size 256 \
--num_channels 128 \
--class_cond False \
--num_res_blocks 2 \
--num_heads 1 \
--learn_sigma True \
--use_scale_shift_norm False \
--attention_resolutions 16 \
--diffusion_steps 1000 \
--noise_schedule linear \
--rescale_learned_sigmas False \
--rescale_timesteps False \
--lr 1e-4 \
--batch_size 8
```

Trained models will be saved in the specified output directory.

---

## Sampling (Inference)

To generate segmentation predictions:
```bash
python scripts/segmentation_sample.py \
--data_name ISIC \
--data_dir <input_data_directory> \
--out_dir <output_directory> \
--model_path <saved_model_path> \
--image_size 256 \
--num_channels 128 \
--class_cond False \
--num_res_blocks 2 \
--num_heads 1 \
--learn_sigma True \
--use_scale_shift_norm False \
--attention_resolutions 16 \
--diffusion_steps 1000 \
--noise_schedule linear \
--rescale_learned_sigmas False \
--rescale_timesteps False \
--num_ensemble 5
```
By default, generated samples are saved in:
```bash
./results/
```

---

## Evaluation

To evaluate predicted segmentation masks:
```bash
python scripts/segmentation_env.py \
--inp_pth <prediction_folder> \
--out_pth <ground_truth_folder>
```
This script computes Dice and Jaccard metrics.

---

## Reproducibility

This repository provides:

Complete training and inference scripts

Dataset preprocessing instructions

Dependency specifications

Model configuration details

All experiments reported in the manuscript can be reproduced using this code.

---

## Code and Data Archiving

An archived version of this repository will be released on Zenodo with a permanent DOI for long-term accessibility.

The DOI will be added after publication.


--- 
## Related Publication

This code is directly related to the following manuscript submitted to The Visual Computer:

"Guided Diffusion for Precise Skin Lesion Segmentation: Integrating Spatial and Semantic Priors"

If you use this repository, please cite the corresponding paper.

Citation

If you use this code, please cite:
```bash

(The final BibTeX entry will be updated after publication.)
```
---

## License

This project is released under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub.

Repository: https://github.com/asadidraco/GuidedDiff
