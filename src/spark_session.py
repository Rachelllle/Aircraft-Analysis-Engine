import os
from src.logger_config import logger
os.environ['HADOOP_HOME'] = 'C:/Hadoop'
os.environ['JAVA_HOME']   = 'C:/Program Files/Java/jdk-19'
os.environ['PYSPARK_PYTHON'] = 'python'

from pyspark.sql import SparkSession

def get_spark():
    logger.info('Create Spark session')
    spark = SparkSession.builder \
        .appName('aircraft_classification') \
        .config('spark.driver.memory', '8g') \
        .config('spark.executor.memory', '8g') \
        .config('spark.sql.shuffle.partitions', '4') \
        .getOrCreate()
    return spark

if __name__ == '__main__':
    spark = get_spark()
    logger.info('spark ok, version :', spark.version)
    spark.stop()