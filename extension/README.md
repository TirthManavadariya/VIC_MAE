# ViC-MAE Temporal Extension: Cross-Frame Reconstruction & Feature Preservation

This folder contains the **Temporal Extension** for ViC-MAE, expanding the baseline image/video masked autoencoder with cross-frame conditional reconstruction and teacher feature preservation.

---

## 📌 Key Architectural Features

1. **SiamMAE-Style Cross-Frame Decoder (`models/cross_decoder.py`):**
   - Given a context frame $I_i$ and a heavily masked target frame $I_j$ (~95% masking ratio), the decoder uses cross-attention conditioned on context tokens and continuous time-gap $\Delta t$ to predict the masked target patches.

2. **Continuous Time-Gap Embedding ($\Delta t$):**
   - Projects normalized inter-frame intervals into sinusoidal time embeddings added to query tokens.

3. **Feature Preservation Loss ($L_{pres}$):**
   - Penalizes representation drift from a frozen pretrained ViC-MAE teacher model using cosine similarity distance.

4. **LoRA Fine-tuning (`models/lora.py`):**
   - Low-Rank Adaptation ($r=16, \alpha=32$) on query and value projections of the ViT encoder to enable lightweight fine-tuning on consumer/cloud GPUs (T4 / RTX 3050).

---

## 📂 Extension Structure

```text
extension/
├── models/
│   ├── extended_vicmae.py          # Unified wrapper & loss computation
│   ├── cross_decoder.py            # SiamMAE cross-attention decoder
│   ├── lora.py                     # LoRA linear modules
│   └── __init__.py
├── datasets/
│   ├── video_image_dataset.py      # Video pair & static image dataset loader
│   └── __init__.py
├── notebooks/
│   ├── vicmae_extension_training_kaggle.ipynb                   # Master end-to-end Kaggle training
│   ├── vicmae_extension_training_kaggle_resume_epoch10_to29.ipynb # Kaggle resume run (epochs 10-29)
│   └── vicmae_extension_training_kaggle_resume_epoch21_to31.ipynb # Kaggle resume run (epochs 21-31)
├── demo_extension.py               # Interactive web demo for video reconstruction
├── train_extended.py               # Main training script with mixed-precision (AMP)
├── engine_extended_pretrain.py     # Per-epoch training & validation loop
└── test_extension.py               # Unit tests verifying shapes and loss backpropagation
```

---

## 🚀 Quick Start

### 1. Run Unit Tests
```bash
python extension/test_extension.py
```

### 2. Launch the Interactive Demo
```bash
python extension/demo_extension.py
```
Open [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.

### 3. Training on Kaggle GPU
Import any of the notebooks from `extension/notebooks/` into Kaggle, enable GPU accelerator, and run top-to-bottom.
