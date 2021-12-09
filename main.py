import sys
import pickle
import json
import requests
import time
from pyspark import SparkConf,SparkContext
from pyspark.sql.functions import col
from pyspark.sql import SparkSession,SQLContext,Row
from pyspark.streaming import StreamingContext
from pyspark.sql.types import StructType,StructField,StringType
from preprocess import pre_process
from models import PAClassifier

my_schema = StructType([StructField('Sentiment', StringType(), False), StructField('Tweet', StringType(), False)])


config = SparkConf()
config.setAppName("Sentiment")
sc = SparkContext(conf=config)


def create_dataframe(rdd):
	sp = SparkContext.getOrCreate()
	sqlContext = SQLContext(sp)
	collection = rdd.collect()
	records = []
	
	for i in collection:
		for j in json.loads(i).values():
			records.append(j)
	
    
	rows = map(lambda x: Row(**x),records) 
	Dataframe = sqlContext.createDataFrame(rows,my_schema)
	if Dataframe.count() >0:
		Dataframe = pre_process(Dataframe)
		Model1 = PAClassifier(Dataframe)
		pickle.dump(Model1,open("PAC.pickle",'wb'))

ssc = StreamingContext(sc,5)

ssc.checkpoint("checkpoint_Sentiment")

dataStream = ssc.socketTextStream("localhost",6100)

dataStream.foreachRDD(create_dataframe)

ssc.start()    
ssc.awaitTermination()
ssc.stop()

