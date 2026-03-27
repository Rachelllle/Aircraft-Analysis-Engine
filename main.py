import os
import sys
from src.logger_config import logger
sys.path.append(os.path.dirname(__file__))

from src.spark_session import get_spark
from src.parsing import build_full_dataset
from src.embeddings import load_vit_model, extract_embeddings, load_into_spark
from src.training import train_all_models
from src.scoring import score_all_models

def main():
    logger.info('='*50)
    logger.info('  AIRCRAFT CLASSIFICATION PIPELINE')
    logger.info('='*50)

    # ── STEP 1 : Spark Session ──
    logger.info('\n[1/5] starting spark session...')
    spark = get_spark()

    # ── STEP 2 : Parsing ──
    logger.info('\n[2/5] parsing annotations and images...')
    build_full_dataset(spark)

    # ── STEP 3 : Embeddings ──
    logger.info('\n[3/5] extracting ViT embeddings...')
    # comment out extract_embeddings if already done
    feature_extractor, vit, device = load_vit_model()
    extract_embeddings(spark, feature_extractor, vit, device)
    load_into_spark(spark)
    spark.stop()

    # ── STEP 4 : Training ──
    logger.info('\n[4/5] training models...')
    train_all_models()

    # ── STEP 5 : Scoring ──
    logger.info('\n[5/5] evaluating models...')
    score_all_models()

    logger.info('\n' + '='*50)
    logger.info('  PIPELINE COMPLETE!')
    logger.info('  run: python app.py to launch the web app')
    logger.info('='*50)

if __name__ == '__main__':
    main()