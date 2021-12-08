import sys
import pickle
import json
import requests
import time
import pickle
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
from pyspark.sql.types import StructType,StructField,StringType
from pyspark.sql import SparkSession,SQLContext,Row
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import col
from preprocess import pre_process

acc = 0
pac = pickle.load(open("PAC.pickle",'rb'))

config = SparkConf()
config.setAppName("Sentiment")
sc = SparkContext(conf=config)

my_schema = StructType([StructField('Sentiment id', StringType(), False),StructField('Tweet', StringType(), False)])

#opening of files
accuracy = open("accuracy.txt","w+")
f1 = open("f1.txt","w+")
recal = open("recall.txt","w+")
prec = open("precision.txt","w+")
conc = open("conmat","w+")

#creating a dataframe
def create_dataframe(rdd):
	sp = SparkContext.getOrCreate()
	sqlContext = SQLContext(sp)
	collection = rdd.collect()
	records=[]
	#loading the values into list
	for i in collection:
		for j in json.loads(i).values():
			records.append(j)
	
    
	rows = map(lambda x: Row(**x),records) 
	Dataframe = sqlContext.createDataFrame(rows,my_schema)
	if Dataframe.count() >0:
		Dataframe = preprocess(Dataframe)
		feature = np.array(Dataframe.select("features ").collect())
		sentiment = np.array(Dataframe.select("Sentiment id").collect())
		new_feature = np.squeeze(feature)
		new_sentiment = np.squeeze(sentiment)
		model = pac.predict(new_feature)
		
		acc = accuracy_score(new_sentiment, model)
		concat = confusion_matrix(new_sentiment, model)
		precision = precision_score(new_sentiment, model, average = 'weighted')
		fone = f1_score(new_sentiment, model, average = 'weighted')
		recall = recall_score(new_sentiment, model, average = 'weighted')
		
		accuracy.write(f'{acc}\n')
		prec.write(f'{precision}\n')
		conc.write(f'{concat}\n')
		f1.write(f'{fone}\n')
		recal.write(f'{recall}\n')
        
ssc = StreamingContext(sc,5)

ssc.checkpoint("checkpoint_Sentiment")

dataStream = ssc.socketTextStream("localhost",6100)

dataStream.foreachRDD(create_dataframe)

ssc.start()    
ssc.awaitTermination()
ssc.stop()

