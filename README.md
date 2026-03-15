# Aircraft Classification

A PySpark-based image classification pipeline that predicts the **manufacturer**, **family**, and **variant** of aircraft from photos, using Vision Transformers (ViT) for feature extraction and PyTorch for classification.

---

## Project Structure

```
aircraft_project/
│
├── data/
│   └── aircraft_data/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── fgvc-aircraft-2013b/
│           └── fgvc-aircraft-2013b/
│               └── data/
│                   ├── images/                        # 10,000 aircraft photos
│                   ├── images_manufacturer_train.txt
│                   ├── images_manufacturer_val.txt
│                   ├── images_manufacturer_test.txt
│                   ├── images_family_train.txt
│                   ├── images_family_val.txt
│                   ├── images_family_test.txt
│                   ├── images_variant_train.txt
│                   ├── images_variant_val.txt
│                   └── images_variant_test.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py               # paths and global parameters
│   ├── spark_session.py        # SparkSession setup
│   ├── parsing.py              # data parsing and joining
│   ├── embeddings.py           # ViT feature extraction + PCA
│   ├── training.py             # PyTorch linear probe training
│   ├── scoring.py              # model evaluation
│   └── predict.py              # prediction on new images
│
├── models/                     # saved models after training
│   ├── manufacturer_model.pt
│   ├── manufacturer_meta.pkl
│   ├── family_model.pt
│   ├── family_meta.pkl
│   ├── variant_model.pt
│   └── variant_meta.pkl
│
├── vit_model/                  # ViT weights 
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors       
│
├── app.py                      # Flask web interface
├── main.py                     # runs the full pipeline
├── download_model.py           # downloads ViT model
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**FGVC-Aircraft** — Fine-Grained Visual Classification of Aircraft
- Source: [Kaggle - seryouxblaster764/fgvc-aircraft](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft)
- 10,000 aircraft images
- 3-level hierarchy:

```
Manufacturer (30 classes)
    └── Family (70 classes)
            └── Variant (100 classes)
```

| File | Content |
|---|---|
| `images_manufacturer_*.txt` | image_id → manufacturer label |
| `images_family_*.txt` | image_id → family label |
| `images_variant_*.txt` | image_id → variant label |
| `train/val/test.csv` | filename, Classes, Labels (0-99) |

---

## Installation

### Prerequisites
- Python 3.11+
- Java JDK 19 ([download here](https://www.oracle.com/java/technologies/downloads/))
- WinUtils (Windows only) — place `winutils.exe` and `hadoop.dll` in `C:/Hadoop/bin/` and copy `hadoop.dll` to `C:/Windows/System32/`

### Install dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pyspark
pillow
numpy
pandas
matplotlib
transformers
torch
psutil
flask
huggingface_hub
scikit-learn
joblib
```

### Download ViT model (first time only)
```bash
python download_model.py
```
Or download manually and place in `./vit_model/`:
- `config.json`
- `preprocessor_config.json`
- `model.safetensors` from [HuggingFace](https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.safetensors)

---

## How to Run

### Option 1 — Run full pipeline
```bash
python main.py
```

### Option 2 — Run step by step
```bash
cd src

# Step 1 — Parse annotations and images (Spark)
python parsing.py

# Step 2 — Extract ViT embeddings + PCA (~20-30 min)
python embeddings.py

# Step 3 — Train 3 PyTorch models (~10 min)
python training.py

# Step 4 — Evaluate models
python scoring.py

# Step 5 — Test prediction on one image
python predict.py
```

### Option 3 — Launch web app
```bash
python app.py
```
Then open your browser at `http://localhost:5000`

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     PARSING  (Spark)                    │
│  .txt files + CSV  →  Spark DataFrame (10,000 rows)     │
│  columns: image_id, manufacturer, family, variant,      │
│           split, variant_label                          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 EMBEDDINGS  (Transformers)               │
│  image.jpg  →  ViT-B/16  →  768 features                │
│           →  PCA (Spark MLlib)  →  256 features         │
│           saved to embeddings_vit.pkl                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 TRAINING  (PyTorch)                     │
│  768 features  →  Linear Probe (MLP)                    │
│  3 separate models:                                     │
│    - manufacturer  (30 classes)                         │
│    - family        (70 classes)                         │
│    - variant       (100 classes)                        │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    SCORING                              │
│  Accuracy + F1-Score for each model                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   WEB APP  (Flask)                      │
│  Upload image  →  predict manufacturer/family/variant   │
│  Shows top 3 predictions with confidence scores         │
└─────────────────────────────────────────────────────────┘
```

---

## Model Details

### Feature Extraction — ViT-B/16 (Vision Transformer)
- Pre-trained on ImageNet-21k (14 million images)
- Splits each image into 196 patches of 16×16 pixels
- Analyzes relationships between all patches simultaneously
- Outputs 768 features per image (CLS token)

### Dimensionality Reduction — PCA (Spark MLlib)
- Reduces from 768 → 256 dimensions
- Keeps the most important information
- Saved as `pca_model` for reuse on new images

### Classification — Linear Probe (PyTorch)
```
Input:   768 features
Layer 1: 512 neurons  + ReLU + Dropout(0.3)
Layer 2: 256 neurons  + ReLU + Dropout(0.3)
Output:  30 / 70 / 100 neurons (depending on the model)
```

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 50 |
| Scheduler | StepLR (step=10, gamma=0.5) |
| Weight decay | 1e-4 |

---

## Results

| Model | Classes | Accuracy | F1-Score |
|---|---|---|---|
| Manufacturer | 30 | ~72% | ~71% |
| Family | 70 | ~64% | ~63% |
| Variant | 100 | ~51% | ~51% |

These results are strong considering the dataset has only ~67 images per class on average. Random baseline would be 1/100 = 1% for variant.

---

## Web App

The Flask web interface allows users to:
1. Upload any aircraft photo
2. Get instant predictions for manufacturer, family and variant
3. See confidence scores and top 3 predictions per level

```bash
python app.py
# open http://localhost:5000
```

---

## Windows Setup Notes

Spark on Windows requires additional configuration:

1. **Java** — Install JDK 19 and set `JAVA_HOME`
2. **WinUtils** — Place `winutils.exe` + `hadoop.dll` in `C:/Hadoop/bin/`
3. **hadoop.dll** — Also copy to `C:/Windows/System32/`
4. These environment variables are set automatically in `spark_session.py`

---

## Authors

BAKHOUCHE Rachel 
