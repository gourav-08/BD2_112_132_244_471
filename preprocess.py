from pyspark.sql.functions import col, udf,regexp_replace,concat_ws
from pyspark.ml.feature import Tokenizer, RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.types import IntegerType

def tokens(dataframe):
	
	token = Tokenizer(inputCol="Tweet",outputCol="words")
	tokenize = token.transform(dataframe)
	return tokenize.select("Sentiment","words")
	
def stop_words(dataframe):
	rem = StopWordsRemover(inputCol="words", outputCol="Tweet")
	remove = rem.transform(dataframe)
	return remove.select("Sentiment","Tweet")

def idf(dataframe):
	token = Tokenizer(inputCol="Tweet", outputCol="words")
	wordsData = token.transform(dataframe)
	count = CountVectorizer (inputCol="words", outputCol="rawFeatures",vocabSize = 600)
	model = count.fit(wordsData)
	featurizedData = model.transform(wordsData)
	idf = IDF(inputCol="rawFeatures", outputCol="features")
	idfModel = idf.fit(featurizedData)
	rescaledData = idfModel.transform(featurizedData)
	return rescaledData.select("Sentiment", "features")

def pre_process(df):
	df = df.withColumn("Tweet",regexp_replace("Tweet",r'[#,\d,\?,\!,\;,\-,\*,\.,\+,\&,\_,\$,\%,\^,\(,\),\<,\>,\/,\|,\},\{,\\,\~,\',\[,\],\:,\~,\`,","]',""))
	df = df.withColumn("Tweet",regexp_replace("Tweet",r'https?://\S+|www\.\S+',""))
	df = df.withColumn("Tweet",regexp_replace("Tweet",r'@\w+',""))
    df = tokens(df)
	df = stop_words(df)
	df = df.withColumn("Tweet",concat_ws(" ",col("Tweet")))
	df = idf(df)
	df = df.withColumn("Sentiment",col("Sentiment").cast('int'))
	return df
