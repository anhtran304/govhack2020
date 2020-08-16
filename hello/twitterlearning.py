from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.externals import joblib
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from decouple import config

import tweepy
import numpy as np
import pandas as pd
import re
import nltk

# Convert sparse matrix into dense data


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def learning():
  print('Ok')
  # Reading dataset from csv file
  dataset = pd.read_csv(os.path.join(BASE_DIR, "/static/data/data.csv", delimiter=";", names=['content', 'tag'])

  # Dowload stopwords function from nlkt package to remove all stopwords
  nltk.download('stopwords')

  # Variable to store all words have been steammed
  # corpus will include all words have been steammed
  corpus = []

  # Cleaning and Steaming words
  for i in range(0, len(dataset.index)):
      content = re.sub("[^a-zA-z]", ' ', dataset["content"][i])
      content = content.lower()
      content = content.split()
      ps = PorterStemmer()
      content = [ps.stem(word) for word in content if not word in set(
          stopwords.words('english'))]
      content = " ".join(content)
      corpus.append(content)

  if len(dataset.index) > 0:
      # y include all tag
      y = dataset.iloc[:, 1].values

      # Devide training data and testing data
      X_train, X_test, y_train, y_test = train_test_split(
          corpus, y, test_size=0.2, random_state=0)

      # Creating Pipeline of all steps needs to be perform in ML learning
      pipeline = Pipeline([
          ('vect', CountVectorizer()),
          ('to_dense', DenseTransformer()),
          ('classifier', GaussianNB()),
      ])

      # Fitting the model with training dataset
      pipeline.fit(X_train, y_train)

      # Evaluating on testing dataset
      y_pred = pipeline.predict(X_test)

      # Dump pipeline to improve efficientcy
      joblib.dump(pipeline, 'vectorizer_and_to_dense_and_classifier.pkl')

      # Calculating accuracy score
      accuracy = accuracy_score(y_pred, y_test)
      print(f"Accuracy score: {accuracy}")

  #Pre processing the new testing unseen data
  # corpus1 = []
  # new_text = "It is very hard to learn JavaScript"
  # new_text = new_text.split()
  # ps = PorterStemmer()
  # processed_text = [ps.stem(word) for word in new_text if not word in set(stopwords.words('english'))]

  # processed_text = " ".join(processed_text)
  # corpus1.append(processed_text)

  # result = pipeline.predict(corpus1)

  # if result == 1:
  #     answear = "Positive"
  # else:
  #     answear = "Negative"

  # print(answear)

  # Setting up API authentication for Twitter APIs
  consumer_key = config('TWITTER_CONSUMER_KEY')
  consumer_secret = config('TWITTER_CONSUMER_SECRET')
  access_token = config('TWITTER_ACCESS_TOKEN')
  access_token_secret = config('TWITTER_ACCESS_TOKEN_SECRET')

  # Creating the authentication object
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

  # Setting access token and secret
  auth.set_access_token(access_token, access_token_secret)

  # Creating the API object with authentication information
  api = tweepy.API(auth)

  # The Twitter user who we want to get tweets from
  name = "tuananh2809"
  # Number of tweets to pull
  tweetCount = 20

  # Calling the user_timeline function with our parameters
  tweets = api.user_timeline(id=name, count=tweetCount)

  # Variable to store all words have been steammed
  # corpus2 will include all words have been steammed
  corpus2 = []

  # Cleaning and Steaming words
  for tweet in tweets:
      tweet_content = re.sub("[^a-zA-z]", ' ', tweet.text)
      tweet_content = tweet_content.lower()
      tweet_content = tweet_content.split()
      if 'tasmania' in tweet_content:
          ps = PorterStemmer()
          tweet_content = [ps.stem(word) for word in tweet_content if not word in set(
              stopwords.words('english'))]
          tweet_content = " ".join(tweet_content)
          corpus2.append(tweet_content)

  # Predicting on new dataset only when dataset is not empty
  predicted_results = []
  if len(corpus2) > 0:
      predicted_results = pipeline.predict(corpus2)

  if len(predicted_results) > 0:
      # Printing out any Negative tweet
      for index, res in enumerate(predicted_results):
          if res == 0 and tweets[index].retweeted is False:
              print("Negative comment:")
              print(tweets[index].text)
              api.retweet(tweets[index].id)
          elif res == 1 and tweets[index].favorited is False:
              print("Positive comment:")
              print(tweets[index].text)
              api.create_favorite(tweets[index].id)
