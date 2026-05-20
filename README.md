# Hierarchical Waste Classification Using Deep Learning

MSc Dissertation project implementing a two-stage hierarchical CNN pipeline for automated waste classification. Stage 1 performs binary classification (organic vs non-organic/recyclable) and Stage 2 performs fine-grained 
13-class waste classification.

---

## Models Evaluated
- MobileNetV2, ResNet34, DenseNet121 (ImageNet pretrained)
- Custom residual CNN trained from scratch

---

## Dataset
Consolidated multi-source dataset combining 7 existing benchmark datasets

Trashnet (Yang & Thung, 2016) -  https://github.com/garythung/trashnet 
Realwaste (Single et al., 2023) - https://doi.org/10.3390/info14120633
Garbage classification (Mohamed, 2020) - https://www.kaggle.com/datasets/mostafaabla/garbage-classification 
Mendeley data GC (Sekar, 2019) - https://www.kaggle.com/techsash/waste-classification-data 
TACO (Proença & Simões, 2020) - https://doi.org/10.48550/arXiv.2003.06975
WaRP dataset (Yudin et al., 2024) - https://doi.org/10.1016/j.engappai.2023.107542
Tricascade waste dataset (Nahiduzzaman et al., 2025) - https://doi.org/10.1016/j.knosys.2025.113028


Consolidated dataset present in : https://huggingface.co/datasets/Sanky1309/HWC
---

## Input data setup

Home directory - waste_classification

Input image files organized into respective stage folders(organic/non-organic or 13 stage-2 folders) inside parent_dir/waste_classification/data/consolidated. Metadata file for training in parent_dir/waste_classification/data/metadata. Output file generated in parent_dir/output. Metadata file to be generated from Stage-1 and Stage-2 folders with columns : filepath, Stage1_label, Stage2_label, source, type

## Project Structure

app/
- app.py                   # HF spaces application code
- model_loader.py          # Model loader code 
- utils.py                 # Utility classes/functions
- requirements.txt             # Python dependencies for HF app with Gradio UI
notebooks/
- dataset_creation.ipynb          # Stage-1, Stage-2 foldern creation and metadata file creation
- datacleaning-dupicates.ipynb    # Cleaning duplicate images
src/training
- cnn_stg1.py               # Stage-1 custom CNN training and evaluation pipeline
- cnn_stg2.py               # Stage-1 Pretrained models training and evaluation pipeline
- pre-trained-stg1.py       # Stage-2 custom CNN training and evaluation pipeline
- pre-trained-stg2.py       # Stage-2 Pretrained models training and evaluation pipeline
requirements.txt              # Python dependencies
README.md

## Requirements
```bash
pip install -r requirements.txt
```

---

## Hardware
- Development: Apple Silicon (MPS backend)
- Training: HPC cluster with NVIDIA L40S GPU (48GB)

---

## Acknowledgements
Developed as part of MSc dissertation — University of Hull, May 2026.