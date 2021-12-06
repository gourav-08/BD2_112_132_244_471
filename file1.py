from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql import SparkSession,Row,Column
import json
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
import sys
import json

sc = SparkContext("local[2]", "NetworkWordCount")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

lines = ssc.socketTextStream("localhost", 6100)

def get_prediction(tweet_text):
	try:
                # remove the blank tweets
		tweet_text = tweet_text.filter(lambda x: len(x) > 0)
                # create the dataframe with each row contains a tweet text
		#y=json.loads(tweet_text)
		rowRdd = tweet_text.map(lambda w: Row(tweet=w))
		wordsDataFrame = spark.createDataFrame(rowRdd)
		# get the sentiments for each row
		wordsDataFrame.show()
                #pipelineFit.transform(wordsDataFrame).select('tweet','prediction').show()
	except : 
		print('No data')

    # define the schema
my_schema = tp.StructType([
				tp.StructField(name= 'id',          dataType= tp.IntegerType(),  nullable= True),   				
    				tp.StructField(name= 'tweet',       dataType= tp.StringType(),   nullable= True)    
    			      ])		
		
#print('\n\nReading the dataset...........................\n')
#my_data = spark.read.csv('/home/pes2ug19cs013/Desktop/project/sentiment/test.csv', schema=my_schema, header=True)
#my_data.show(2)

#my_data.printSchema()


words = lines.flatMap(lambda line : line.split(" "))
lines.foreachRDD(get_prediction)

ssc.start()             # Start the computation
ssc.awaitTermination()
