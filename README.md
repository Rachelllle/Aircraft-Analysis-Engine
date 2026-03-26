# ✈️ Aircraft Classification

A PySpark-based image classification pipeline that predicts the **manufacturer**, **family**, and **variant** of aircraft from photos, using Vision Transformers (ViT) for feature extraction and PyTorch for classification.

---

## 📁 Project Structure

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
├── vit_model/                  # ViT weights (not pushed to GitHub)
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors       # ~350MB
│
├── app.py                      # Flask web interface
├── main.py                     # runs the full pipeline
├── download_model.py           # downloads ViT model
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Dataset

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

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- Java JDK 19 ([download here](https://www.oracle.com/java/technologies/downloads/))

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download ViT model (first time only)
```bash
python download_model.py
```
> ⚠️ If you get an SSL error on Windows, download `model.safetensors` manually from [HuggingFace](https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.safetensors) and place it in `./vit_model/`

---

## 🖥️ Setup by OS

### 🪟 Windows
1. Install Java JDK 19
2. Download `winutils.exe` and `hadoop.dll` → place them in `C:/Hadoop/bin/`
3. Copy `hadoop.dll` to `C:/Windows/System32/`
4. Environment variables are set automatically in `spark_session.py`

### 🍎 Mac
1. Install Java JDK 19
2. Install Homebrew if not already installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Install Hadoop via Homebrew:
```bash
brew install hadoop
```
4. Add to your `~/.zshrc` or `~/.bash_profile`:
```bash
export JAVA_HOME=$(/usr/libexec/java_home)
export HADOOP_HOME=/opt/homebrew/opt/hadoop
```

### 🐧 Linux
1. Install Java JDK 19:
```bash
sudo apt install openjdk-19-jdk
```
2. Add to your `~/.bashrc`:
```bash
export JAVA_HOME=/usr/lib/jvm/java-19-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

---

## 🚀 How to Run

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

## 🧠 Pipeline Architecture

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

## 🌐 Web App

The Flask web interface allows users to:
1. Upload any aircraft photo
2. Get instant predictions for manufacturer, family and variant
3. See confidence scores and top 3 predictions per level

```bash
python app.py
# open http://localhost:5000
```

---

## 👥 Authors

BAKHOUCHE Rachel — 4th year AIBD