# Monocular Depth Estimation with DINOv3

## Quick Start (Easiest Way)
Use the Google Colab notebook to run the analysis immediately without local setup.
* **[Click Here to Run in Colab](https://colab.research.google.com/drive/1ofS54sy6byKLzyMebWKhkJZSdZEIbDef?usp=sharing)**
* *Note: This notebook automatically clones the code, downloads the DINOv3 backbone, and pulls the custom trained weights from Hugging Face. No manual file uploads are required.*

## 1. Overview
This project explores **Transfer Learning for Dense Prediction** by adapting a Foundation Model (**DINOv3**) for Monocular Depth Estimation. It compares a standard CNN baseline against a modern Transformer-based approach.

The repository implements three distinct modeling phases:
1.  **Baseline (UNet):** A standard CNN trained from scratch on the NYU Depth v2 dataset. The architecture is adapted from [DikshaMeghwal/unet-depth-prediction](https://github.com/DikshaMeghwal/unet-depth-prediction).
2.  **Phase 1 (Linear Probe):** A frozen DINOv3 backbone with a lightweight decoder, testing the raw quality of self-supervised features.
3.  **Phase 3 (Hybrid Adapter):** A **DPT (Dense Prediction Transformer)** head on top of DINOv3, using multi-scale feature fusion for state-of-the-art detail recovery.

## 2. Setup Instructions

### Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Data Preparation
The code is designed to pull data automatically from Hugging Face. However, if running locally, organize data as follows:
```text
.
├── da3_adapter.py        # Phase 3: The Hybrid DPT Model
├── tiny_unet.py          # Baseline: Simple CNN
├── model_phase1.py       # Phase 1: Linear Probe
├── dinov3/               # Cloned Facebook Research Backbone
└── data/
    ├── nyu_depth_v2/     # Training Data
    └── eth3d_sample/     # Generalization Test Data
```

## 3. How to Run

### Step 1: Download Weights & Data
The project uses the `huggingface_hub` library to fetch the 1.2GB backbone and custom checkpoints:
```bash
# Download Backbone and Trained Weights
huggingface-cli download Junming111/task3 dinov3_vitl16.pth --local-dir .
huggingface-cli download Junming111/task3 hybrid_best.pth --local-dir .
```

### Step 2: Run Inference
The `inference.py` script runs a comparative study on a provided image:

```bash
python inference.py --image_path ./sample_image.jpg
```

## 4. Expected Results & Interpretation

The results demonstrate the "Foundation Model Advantage," particularly in generalizing to complex scenes.

### A. Visual Comparison (Qualitative)
The output visualization typically includes three panels:

1.  **UNet (Baseline):** Often blurry or "foggy." Edges of objects (tables, chairs) may bleed into the background.
2.  **Phase 1 (Linear Probe):** Blocky or grid-like artifacts due to the fixed patch size.
3.  **Phase 3 (Hybrid - Our Model):** Sharp, crisp boundaries. The DPT adapter fuses high-level semantics with low-level details.

### B. Generalization Test (ETH3D)
* **Goal:** Testing the model on a dataset it *never saw during training*.
* **Observation:** The **Hybrid Phase 3** model maintains structural consistency on ETH3D images (outdoor/industrial scenes), whereas the **UNet** baseline often fails, proving that DINOv3 learns robust, universal features.