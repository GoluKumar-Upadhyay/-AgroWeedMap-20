<div align="center">

# 🌾 WeedVision-20: Attention-Enhanced EfficientNet for Weed–Crop Classification

**A CBAM-augmented EfficientNetB1 framework for fine-grained, explainable weed and crop identification in precision agriculture.**

Field-collected in Bihar, India · 20 classes · 100,000 images · 99.98% validation accuracy

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active--development-yellow)](#-project-status--roadmap)

[Overview](#-overview) •
[Results](#-results) •
[Architecture](#-proposed-architecture) •
[Dataset](#-dataset) •
[Quickstart](#-quickstart) •
[Explainability](#-explainability-grad-cam) •
[Limitations](#-honest-limitations) •
[Citation](#-citation)

</div>

---

## 📌 Overview

Weeds compete with crops for light, water, and nutrients, and manual field identification is slow and error-prone — especially when weed and crop seedlings look nearly identical at early growth stages. **WeedVision-20** is a deep-learning pipeline that classifies **20 weed and crop species** from a single field photo, built around:

- 🧠 **EfficientNetB1 backbone** + **CBAM** (Convolutional Block Attention Module) for channel + spatial attention
- 🔀 **GAP + GMP dual-statistic pooling fusion** head
- 🌗 **CLAHE preprocessing** (LAB colour space) to normalise uneven outdoor lighting
- ⚙️ **Two-phase training**: frozen-backbone head warmup → partial fine-tuning
- 🔍 **Grad-CAM explainability** to verify the model attends to leaf morphology, not background soil
- 📊 A controlled, identical-pipeline comparison against **ResNet50** and a **plain EfficientNetB1** baseline

> 🇮🇳 **Dataset provenance:** ~15–20 agricultural fields around **Chandragarh village, Motihari, Purvi Champaran district, Bihar**, captured with a **Realme smartphone** under natural field conditions — no studio setup, real sun angle, real shadows, real background clutter.

---

## 🏆 Results

<div align="center">

| Model | Backbone Params | Accuracy | Macro F1 | Mean AUC | Unbatched Latency |
|:---|---:|---:|---:|---:|---:|
| ResNet50 (baseline) | 24.1 M | 99.94% | 0.9994 | 1.0000 | 141.6 ms/img |
| EfficientNetB1 (no attention) | ~6.9 M | 99.90% | 0.9990 | 1.0000 | n/b |
| **🌟 Proposed: EfficientNetB1 + CBAM** | **8.85 M** | **99.98%** | **0.9998** | **1.0000** | **107.5 ms/img** |



**Why this matters, not just the accuracy number:** the proposed model uses **63.3% fewer parameters than ResNet50** and is **24.1% faster** on unbatched CPU inference — a meaningfully better accuracy-per-parameter trade-off, even though all three models sit near the accuracy ceiling of this dataset (see [honest limitations](#-honest-limitations) before treating 99.98% as proof of generalisation).

<details>
<summary>📈 <b>Confusion matrix</b> — only 4 misclassifications across 20,000 validation images</summary>
<br>
<div align="center">
<img width="1126" height="989" alt="image" src="https://github.com/user-attachments/assets/044f7af3-3eda-4484-b843-d46cda0920dd" />

</div>

All errors occur between visually similar species pairs: 2× Potato→Tomato, 1× Chenopodium album→Potato, 1× Goosegrass→Bermuda.
</details>

<details>
<summary>📊 <b>Per-class precision / recall / F1</b></summary>
<br>
<div align="center">
<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/c343da56-5995-4c3f-9172-8a18ea2e4429" />

</div>
</details>

<details>
<summary>📉 <b>ROC-AUC curves (one-vs-rest, 20 classes)</b></summary>
<br>
<div align="center">
<img width="857" height="701" alt="image" src="https://github.com/user-attachments/assets/1a7f3e77-4569-4ead-b279-18d6318934f8" />

</details>



---

## 🧬 Proposed Architecture


```
Input (224×224×3)
   │
   ▼
CLAHE Preprocessing (LAB L*-channel, clip=2.0, tile=8×8)
   │
   ▼
EfficientNetB1 Backbone (ImageNet-pretrained)
   │
   ▼
CBAM: Channel Attention → Spatial Attention
   │
   ├──────────────┬──────────────┐
   ▼              ▼
GlobalAvgPool  GlobalMaxPool
   │              │
   └──────┬───────┘
          ▼
     Concatenate (2560-d)
          │
          ▼
BatchNorm → Dense(512) → BatchNorm → ReLU → Dropout(0.4)
          │
          ▼
     Dense(256, ReLU) → Dropout(0.3)
          │
          ▼
     Softmax (20 classes)
```

**Training strategy — two phases, not one:**

| Phase | Backbone | Epochs | LR | Purpose |
|---|---|---|---|---|
| 1 — Warmup | Frozen | 5 | 1e-3 | Stabilise randomly-initialised head before touching backbone weights |
| 2 — Fine-tune | Last 50 layers unfrozen | up to 25 (early-stop) | 1e-4 (wd 1e-5) | Adapt pretrained features to field imagery |

Loss: categorical cross-entropy with **label smoothing (ε=0.05)** to prevent over-confidence at near-ceiling accuracy.

---

## 🔍 Explainability: Grad-CAM

<div align="center">
<img width="1129" height="2749" alt="image" src="https://github.com/user-attachments/assets/ebc2163b-5423-4f9d-9421-ecbd9eabf79b" />

</div>

Grad-CAM is computed on the **CBAM-refined feature map** (not the raw backbone output), so the heat-map reflects what the attention mechanism itself prioritises. Across sampled classes, activation concentrates on **leaf blade, margin, and venation** — not background soil — which is the qualitative signature expected if CBAM is doing its job.

> ⚠️ This is currently **qualitative, visual-inspection evidence**. A quantitative CAM-occlusion agreement metric is planned — see [limitations](#-honest-limitations).

---

## 🌱 Dataset

<details open>
<summary><b>Class list (20 classes, 5,000 images each)</b></summary>
<br>

| Weed Species | Crop Species |
|---|---|
| Bermuda grass | Maize |
| Boerhavia erecta | Mustard |
| Broadleaf plantain | Potato |
| Cannabis sativa | Tomato |
| Chenopodium album | |
| Common cocklebur | |
| Creeping woodsorrel | |
| Dhaniya (coriander, volunteer) | |
| Goosegrass | |
| Launaea sarmentosa | |
| Parthenium hysterophorus | |
| Pigweed (Amaranthus) | |
| Prostrate spurge | |
| Purple nutsedge | |
| Sesbania bispinosa | |
| Sowthistle | |

</details>

<details>
<summary><b>Collection details</b></summary>
<br>

- **Location:** ~15–20 agricultural fields around Chandragarh village, Motihari, Purvi (East) Champaran district, Bihar, India
- **Device:** Realme smartphone camera
- **Conditions:** Natural outdoor illumination, no studio setup — variable sun angle, cast shadow, background soil/foliage clutter
- **Balance:** Exactly 5,000 images per class (perfectly balanced)
- **Current split:** 80% train / 20% validation *(a 70/15/15 train/val/test split has been prepared — see [Project Status](#-project-status--roadmap))*

</details>

<details>
<summary><b>Expected folder structure</b></summary>
<br>

```
dataset/
├── train/
│   ├── Bermuda/
│   ├── Maize/
│   └── ... (20 class folders)
├── val/
│   ├── Bermuda/
│   └── ...
└── test/            # new held-out split, in progress
    ├── Bermuda/
    └── ...
```

</details>

---

## 🚀 Quickstart

<details>
<summary><b>1. Environment setup</b></summary>

```bash
git clone https://github.com/<your-username>/weedvision-20.git
cd weedvision-20
pip install -r requirements.txt
```

`requirements.txt`:
```
tensorflow>=2.15
keras>=3.0
numpy
pandas
scikit-learn
opencv-python
matplotlib
seaborn
imagehash
statsmodels
Pillow
```
</details>

<details>
<summary><b>2. Run inference on a single image</b></summary>

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np, cv2

def apply_clahe(image):
    image = np.clip(image, 0, 255).astype('uint8')
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB).astype('float32')

def preprocess(img):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(apply_clahe(img))

model = load_model("weights/best_ProposedModel.keras", custom_objects={'tf': tf}, safe_mode=False)

img = img_to_array(load_img("sample.jpg", target_size=(224, 224)))
inp = np.expand_dims(preprocess(img), 0)

pred = model.predict(inp)
class_idx = np.argmax(pred[0])
print(f"Predicted class: {CLASS_NAMES[class_idx]}  (confidence: {pred[0][class_idx]:.2%})")
```
</details>

<details>
<summary><b>3. Train from scratch</b></summary>

```bash
jupyter notebook notebooks/WeedVision20_Final_Complete.ipynb
```

Update `SRC_DIR`, `BASE_DIR`, and `SAVE_DIR` at the top of the notebook to match your paths, then run Sections 1–7 in order (dataset split → preprocessing → model build → two-phase training).

> ⚠️ **Hardware note:** training was developed on a 4-core CPU-only laptop; epochs took 90–120 minutes each. A GPU (even a free Colab T4) will cut this to minutes. See `notebooks/colab_setup.md` if training on Colab.
</details>

---

## 🛰️ Proposed Deployment Architecture

<div align="center">
<img src="assets/deployment_pipeline.png" width="620" alt="Deployment pipeline">
</div>

> **Note:** this is a **conceptual, not yet field-tested** architecture — see limitations below. No UAV flights, edge-hardware benchmarks, or real spraying hardware have been used in this project yet.

---

## ⚠️ Honest Limitations

This project is under active revision after a rigorous review pass. Rather than hide the gaps, here's exactly where it stands:

- [ ] **No independent test-set result yet** — current accuracy is validation-set accuracy, which was also used for early stopping/model selection. A 70/15/15 split is built; retraining is in progress.
- [ ] **Plant/session-level independence unverified** — image-wise split; can't rule out the same plant appearing on both sides.
- [ ] **CBAM's individual contribution not isolated** — proposed model changes 5 things vs. the plain baseline simultaneously; a component-wise ablation is planned.
- [ ] **Single training run per model** — no multi-seed variance or significance testing yet (McNemar's test planned).
- [ ] **Grad-CAM evidence is qualitative** — a quantitative CAM-occlusion agreement score is planned.
- [ ] **CLAHE ablation and MobileNetV2/VGG16 baselines** — started, interrupted by CPU compute limits.
- [ ] **Single-region, single-device dataset** — no cross-region, cross-camera, or UAV-altitude validation yet.
- [ ] **No edge-hardware benchmarking** — all latency figures are CPU-only; real-time UAV feasibility is not demonstrated.

See the full academic writeup for the complete, itemised discussion.

---

## 🗺️ Project Status & Roadmap

| Status | Item |
|:---:|---|
| ✅ | Three-way controlled comparison (ResNet50 / EfficientNetB1 / Proposed) on original 80/20 split |
| ✅ | Grad-CAM qualitative explainability |
| ✅ | Near-duplicate leakage audit |
| ✅ | 70/15/15 train/val/test split prepared |
| 🔄 | Retraining all three models on new split (in progress, GPU) |
| ⬜ | Held-out test-set evaluation |
| ⬜ | Component-wise ablation (CBAM / GAP+GMP / label smoothing / two-phase) |
| ⬜ | Multi-seed statistical significance testing |
| ⬜ | Quantitative Grad-CAM (CAM-occlusion agreement) |
| ⬜ | Edge-hardware (Jetson/Coral) latency benchmarking |

---

## 📁 Repository Structure

```
weedvision-20/
├── notebooks/
│   ├── Weed_Crop_Classification_Complete_Pipeline.ipynb   # ResNet50 baseline
│   ├── 20_classes.ipynb                                    # Plain EfficientNetB1 baseline
│   └── WeedVision20_Final_Complete.ipynb                   # Proposed CBAM model
├── assets/                # figures used in this README
├── weights/                # trained .keras checkpoints (not tracked in git — see Releases)
├── docs/
│   └── WeedVision20_IEEE_Paper.docx
├── requirements.txt
└── README.md
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{weedvision20_2026,
  title   = {An Attention-Enhanced EfficientNet Framework with Explainable AI for
             Fine-Grained Weed--Crop Classification in Precision Agriculture},
  author  = {Abhijeet},
  year    = {2026},
  note    = {Department of Computer Science and Engineering,
             Jodhpur Institute of Engineering and Technology}
}
```

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Dataset field-collected in Purvi Champaran, Bihar, India
- Built with TensorFlow / Keras, CBAM ([Woo et al., ECCV 2018](https://doi.org/10.1007/978-3-030-01234-2_1)), and EfficientNet ([Tan & Le, ICML 2019](https://arxiv.org/abs/1905.11946))

<div align="center">

**⭐ If this project is useful to you, consider starring the repo!**

</div>
