import ssl
import os
from src.logger_config import logger

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import requests
from requests.adapters import HTTPAdapter
requests.packages.urllib3.disable_warnings()

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import torch
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import PCA
from pyspark.sql.functions import udf

from src.config import DATA_PATH, BASE_PATH, BATCH_SIZE, PCA_K
from src.spark_session import get_spark


def load_vit_model():
    logger.info('loading ViT-B/16 (first download ~350MB)...')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    vit_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vit_model = vit_model.to(device)
    logger.info(f'ViT loaded on {device}')
    return feature_extractor, vit_model, device

def extract_embeddings(spark, feature_extractor, vit_model, device):
    # load metadata parsed in previous step
    logger.info('load metadata parsed in previous step')
    df_meta = spark.read.parquet(BASE_PATH + '/ms_parsed_full.parquet').toPandas()
    images_dir = DATA_PATH + 'images/'
    print(f'extracting embeddings for {len(df_meta)} images...')

    all_embeddings = []
    for i in range(0, len(df_meta), BATCH_SIZE):
        batch = df_meta.iloc[i:i+BATCH_SIZE]
        imgs = []
        for _, row in batch.iterrows():
            try:
                img = Image.open(images_dir + row['image_id'] + '.jpg').convert('RGB')
                imgs.append(img)
            except:
                imgs.append(Image.new('RGB', (224, 224)))

        inputs = feature_extractor(images=imgs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = vit_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_embeddings.extend(embeddings.tolist())

        if i % 160 == 0:
            logger.info(f'  {min(i+BATCH_SIZE, len(df_meta))}/{len(df_meta)} done')

    df_meta['embedding'] = all_embeddings
    df_meta.to_pickle(BASE_PATH + '/embeddings_vit.pkl')
    logger.info('embeddings saved : 768 features per image')
    return df_meta

def load_into_spark(spark, df_meta=None):
    if df_meta is None:
        df_meta = pd.read_pickle(BASE_PATH + '/embeddings_vit.pkl')

    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import PCA
    from pyspark.sql.types import StructType, StructField, StringType

    # convert embeddings to vectors directly in pandas (no UDF)
    logger.info('converting embeddings to vectors...')
    df_meta['features_raw'] = df_meta['embedding'].apply(lambda x: Vectors.dense(x))
    df_meta = df_meta.drop(columns=['embedding'])

    # load into spark
    logger.info('loading into spark...')
    df_emb = spark.createDataFrame(df_meta)
    df_emb = df_emb.repartition(10)

    # apply PCA : 768 → 256
    logger.info(f'applying PCA 768 -> {PCA_K}...')
    pca = PCA(k=PCA_K, inputCol='features_raw', outputCol='features')
    pca_model = pca.fit(df_emb)
    df_emb = pca_model.transform(df_emb).drop('features_raw')

    pca_model.write().overwrite().save(BASE_PATH + '/pca_model')
    df_emb.write.mode('overwrite').parquet(BASE_PATH + '/ms_embeddings_vit.parquet')
    logger.info(f'PCA done : 768 -> {PCA_K} dimensions')
    return df_emb, pca_model

if __name__ == '__main__':
    spark = get_spark()
    feature_extractor, vit_model, device = load_vit_model()
    df_meta = extract_embeddings(spark, feature_extractor, vit_model, device)
    
    df_emb, pca_model = load_into_spark(spark)
    df_emb.printSchema()
    logger.info('embeddings done!')
    spark.stop()