# Aircraft Classification

A PySpark-based image classification pipeline that predicts the **manufacturer**, **family**, and **variant** of aircraft from photos, using Vision Transformers (ViT) for feature extraction and PyTorch for classification.

---

## Project Structure
```bash
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
│                   ├── images/        # 10,000 aircraft photos
│                   └── *.txt files
│
├── src/
│   ├── __init__.py
│   ├── config.py                # paths and global parameters
│   ├── spark_session.py         # SparkSession setup
│   ├── parsing.py               # data parsing and joining
│   ├── embeddings.py            # ViT feature extraction + PCA
│   ├── training.py              # PyTorch linear probe training
│   ├── scoring.py               # model evaluation
│   ├── predict.py               # prediction on new images
│   └── logger_config.py         # logging configuration
│
├── models/                      # saved models after training
│   ├── manufacturer_model.pt
│   ├── manufacturer_meta.pkl
│   ├── family_model.pt
│   ├── family_meta.pkl
│   ├── variant_model.pt
│   └── variant_meta.pkl
│
├── vit_model/                   # ViT weights
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors
│
├── web_interface/               # Web applications
│   ├── app.py                   # Flask app — predictions
│   └── streamlit_app.py         # Streamlit dashboard — results
│
├── outputs/                     # Generated files
│   ├── scores.json              # model evaluation results
│   └── predictions.json         # prediction history
│
├── logs/                        # log files
├── main.py                      # runs the full pipeline
├── download_model.py            # downloads ViT model
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

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download ViT model
```bash
python download_model.py
```
> If you get an SSL error on Windows, download `model.safetensors` manually from [HuggingFace](https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.safetensors) and place it in `./vit_model/`

---

## Setup by OS

### Windows
1. Install Java JDK 19
2. Download `winutils.exe` and `hadoop.dll` → place them in `C:/Hadoop/bin/`
3. Copy `hadoop.dll` to `C:/Windows/System32/`
4. Environment variables are set automatically in `spark_session.py`

### Mac
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

### Linux
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

## How to Run

### Option 1 — Run full pipeline
```bash
python main.py
```

### Option 2 — Run step by step
```bash
# Step 1 — Parse annotations and images (Spark)
python src/parsing.py

# Step 2 — Extract ViT embeddings + PCA (~20-30 min)
python src/embeddings.py

# Step 3 — Train 3 PyTorch models (~10 min)
python src/training.py

# Step 4 — Evaluate models and save scores.json
python src/scoring.py

# Step 5 — Test prediction on one image
python src/predict.py
```

### Option 3 — Launch Flask web app
```bash
python web_interface/app.py
```
Then open your browser at `http://localhost:5000`

### Option 4 — Launch Streamlit dashboard
```bash
streamlit run web_interface/streamlit_app.py
```
Then open your browser at `http://localhost:8501`

> Note: Run `scoring.py` before launching Streamlit to generate `scores.json`

---

## Authors

BAKHOUCHE Rachel
BELGUEDJ Nassilya
DIENG Mamadou
BEN OUSSAID Arezki
VAN Hugo