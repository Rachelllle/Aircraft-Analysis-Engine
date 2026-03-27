import os
import sys
from src.logger_config import logger
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pyspark.sql.functions import col, split, element_at, lit, regexp_replace
from src.config import DATA_PATH, CSV_PATH, BASE_PATH
from src.spark_session import get_spark

def read_annotation_file(spark, filepath, label_name):
    df = spark.read.text(filepath)
    df = df.select(
        split(col('value'), ' ', 2).getItem(0).alias('image_id'),
        split(col('value'), ' ', 2).getItem(1).alias(label_name)
    )
    return df

def build_annotation_df(spark, split_name):
    df_manuf   = read_annotation_file(spark, DATA_PATH + f'images_manufacturer_{split_name}.txt', 'manufacturer')
    df_family  = read_annotation_file(spark, DATA_PATH + f'images_family_{split_name}.txt', 'family')
    df_variant = read_annotation_file(spark, DATA_PATH + f'images_variant_{split_name}.txt', 'variant')
    df = df_manuf.join(df_family, 'image_id').join(df_variant, 'image_id')
    df = df.withColumn('split', lit(split_name))
    return df

def parse_annotations(spark):
    logger.info('combine train, val and test annotations into one dataframe')
    df = build_annotation_df(spark, 'train') \
        .union(build_annotation_df(spark, 'val')) \
        .union(build_annotation_df(spark, 'test'))
    logger.info(f'annotations ok : {df.count()} rows')
    return df

def parse_csv(spark):
    logger.info('read kaggle csv files and extract image_id from filename')
    df = spark.read.csv(CSV_PATH + 'train.csv', header=True).withColumn('split', lit('train')) \
        .union(spark.read.csv(CSV_PATH + 'val.csv', header=True).withColumn('split', lit('val'))) \
        .union(spark.read.csv(CSV_PATH + 'test.csv', header=True).withColumn('split', lit('test')))
    df = df.withColumn('image_id', regexp_replace(col('filename'), '\\.jpg$', ''))
    logger.info(f'csv ok : {df.count()} rows')
    return df

def build_full_dataset(spark):
    logger.info('Load images as binary files')
    df_images = spark.read.format('binaryFile') \
        .option('pathGlobFilter', '*.jpg') \
        .load(DATA_PATH + 'images/') \
        .withColumn('image_id', split(element_at(split(col('path'), '/'), -1), '\\.').getItem(0)) \
        .select('image_id', 'content')

    df_annotations = parse_annotations(spark)
    df_csv         = parse_csv(spark)

    df_parsed = df_images \
        .join(df_annotations, 'image_id') \
        .join(df_csv.select('image_id', 'Labels'), 'image_id') \
        .withColumnRenamed('Labels', 'variant_label')

    output_path = BASE_PATH + '/ms_parsed_full.parquet'
    df_parsed.drop('content').write.mode('overwrite').parquet(output_path)
    
    logger.info(f'parsing complete : {df_parsed.count()} images saved to {output_path}')
    
    logger.info('\n sample of the parsed data :')
    df_parsed.drop('content').show(5, truncate=True)
    
    logger.info('\n counts per split :')
    df_parsed.groupBy('split').count().show()
    
    logger.info('\n sample manufacturers :')
    df_parsed.select('manufacturer').distinct().show(10)
    
    return df_parsed

if __name__ == '__main__':
    spark = get_spark()
    df = build_full_dataset(spark)
    df.printSchema()
    spark.stop()