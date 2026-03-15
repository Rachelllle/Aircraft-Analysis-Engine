import os
import sys
sys.path.append(os.path.dirname(__file__))

from src.spark_session import get_spark
from src.parsing import build_full_dataset
from src.embeddings import load_vit_model, extract_embeddings, load_into_spark
from src.training import train_all_models
from src.scoring import score_all_models

def main():
    print('='*50)
    print('  AIRCRAFT CLASSIFICATION PIPELINE')
    print('='*50)

    # ── STEP 1 : Spark Session ──
    print('\n[1/5] starting spark session...')
    spark = get_spark()

    # ── STEP 2 : Parsing ──
    print('\n[2/5] parsing annotations and images...')
    build_full_dataset(spark)

    # ── STEP 3 : Embeddings ──
    print('\n[3/5] extracting ViT embeddings...')
    # comment out extract_embeddings if already done
    feature_extractor, vit, device = load_vit_model()
    extract_embeddings(spark, feature_extractor, vit, device)
    load_into_spark(spark)
    spark.stop()

    # ── STEP 4 : Training ──
    print('\n[4/5] training models...')
    train_all_models()

    # ── STEP 5 : Scoring ──
    print('\n[5/5] evaluating models...')
    score_all_models()

    print('\n' + '='*50)
    print('  PIPELINE COMPLETE!')
    print('  run: python app.py to launch the web app')
    print('='*50)

if __name__ == '__main__':
    main()
