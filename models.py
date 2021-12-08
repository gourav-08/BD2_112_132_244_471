from sklearn.linear_model import PassiveAggressiveClassifier
import numpy as np

def PAClassifier(df):
	classifier = PassiveAggressiveClassifier()
	sentiment = np.array(df.select("Sentiment id").collect())
	features = np.array(df.select("features").collect())
	new_sentiment = np.squeeze(sentiment)
	new_feature = np.squeeze(features)
	classifier.partial_fit(new_feature,new_sentiment,classes = np.unique(new_sentiment))
	return classfier
