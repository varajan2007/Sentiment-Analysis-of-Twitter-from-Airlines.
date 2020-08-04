# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:21:40 2019

@author: Vani Rajan
"""
""""\Using naive bayes classifier
make sure to do the pip intall nltk
nltk must get imported in the python shell before you can proceed
with this configuration. 

"""""

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from stop_words import get_stop_words
from nltk.corpus import stopwords
 
from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
import string
import re

import matplotlib.pyplot as mp
%matplotlib inline


tweets = pd.read_csv('d:/res papers/sentiments ana/TweetsAirlines.csv', header=0)

df = tweets[['airline_sentiment', 'text']]

n_class = 2
n_tweet = 2363

# Divide into number of classes
if n_class == 2:
    df_pos = df.copy()[df.airline_sentiment == 'positive'][:n_tweet]
    df_neg = df.copy()[df.airline_sentiment == 'negative'][:n_tweet]
    df_neu = pd.DataFrame()
    df = pd.concat([df_pos, df_neg], ignore_index=True).reset_index(drop=True)
elif n_class == 3:
    df_pos = df.copy()[df.airline_sentiment == 'positive'][:n_tweet]
    df_neg = df.copy()[df.airline_sentiment == 'negative'][:n_tweet]
    df_neu = df.copy()[df.airline_sentiment == 'neutral'][:n_tweet]
    df = pd.concat([df_pos, df_neg, df_neu], ignore_index=True).reset_index(drop=True)

def ProTweets(tweet):
    tweet = ''.join(c for c in tweet if c not in string.punctuation)
    tweet = re.sub('((www\S+)|(http\S+))', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = re.sub(' +',' ', tweet)
    tweet = tweet.lower().strip()
    return tweet

def rmStopWords(tweet, stop_words):
    text = tweet.split()
    text = ' '.join(word for word in text if word not in stop_words)
    return text


stop_words = get_stop_words('english')
stop_words = stopwords.words('english')
stop_words = [''.join(c for c in s if c not in string.punctuation) for s in stop_words]
stop_words = [t.encode('utf-8') for t in stop_words]

pro_tweets = []
for tweet in df['text']:
    processed = ProTweets(tweet)
    pro_stopw = rmStopWords(processed, stop_words)
    pro_tweets.append(pro_stopw)

df['text'] = pro_tweets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['airline_sentiment'], test_size=0.33, random_state=0)

df_train = pd.DataFrame()
df_test = pd.DataFrame()

df_train['text'] = X_train
df_train['airline_sentiment'] = y_train
df_train = df_train.reset_index(drop=True)

df_test['text'] = X_test
df_test['airline_sentiment'] = y_test
df_test = df_test.reset_index(drop=True)


class TweetNBClassifier(object):

    def __init__(self, df_train):
        self.df_train = df_train
        self.df_pos = df_train.copy()[df_train.airline_sentiment == 'positive']
        self.df_neg = df_train.copy()[df_train.airline_sentiment == 'negative']
        self.df_neu = df_train.copy()[df_train.airline_sentiment == 'neutral']

    def fit(self):
        Pr_pos = df_pos.shape[0]/self.df_train.shape[0]
        Pr_neg = df_neg.shape[0]/self.df_train.shape[0]
        Pr_neu = df_neu.shape[0]/self.df_train.shape[0]
        self.Prior  = (Pr_pos, Pr_neg, Pr_neu)

        self.pos_words = ' '.join(self.df_pos['text'].tolist()).split()
        self.neg_words = ' '.join(self.df_neg['text'].tolist()).split()
        self.neu_words = ' '.join(self.df_neu['text'].tolist()).split()

        all_words = ' '.join(self.df_train['text'].tolist()).split()

        self.vocab = len(Counter(all_words))

        wc_pos = len(' '.join(self.df_pos['text'].tolist()).split())
        wc_neg = len(' '.join(self.df_neg['text'].tolist()).split())
        wc_neu = len(' '.join(self.df_neu['text'].tolist()).split())
        self.word_count = (wc_pos, wc_neg, wc_neu)
        return self

    def predict(self, df_test):
        class_choice = ['positive', 'negative', 'neutral']

        classification = []
        for tweet in df_test['text']:
            text = tweet.split()

            val_pos = np.array([])
            val_neg = np.array([])
            val_neu = np.array([])
            for word in text:
                tmp_pos = np.log(self.pos_words.count(word)+1)
                tmp_neg = np.log(self.neg_words.count(word)+1)
                tmp_neu = np.log(self.neu_words.count(word)+1)
                val_pos = np.append(val_pos, tmp_pos)
                val_neg = np.append(val_neg, tmp_neg)
                val_neu = np.append(val_neu, tmp_neu)

            denom_pos = len(text)*np.log(self.word_count[0]+self.vocab)
            denom_neg = len(text)*np.log(self.word_count[1]+self.vocab)
            denom_neu = len(text)*np.log(self.word_count[2]+self.vocab)

            val_pos = np.log(self.Prior[0]) + np.sum(val_pos) - denom_pos
            val_neg = np.log(self.Prior[1]) + np.sum(val_neg) - denom_neg
            val_neu = np.log(self.Prior[2]) + np.sum(val_neu) - denom_neu

            probability = (val_pos, val_neg, val_neu)
            classification.append(class_choice[np.argmax(probability)])
        return classification

    def score(self, feature, target):
        comp_c, comp_i = (0,0)
        tp, tn, fp, fn = (0,0,0,0)
        for i in range(0,len(feature)):
            if feature[i] == target[i]:
                comp_c += 1
                if (target[i] == 'positive') & (feature[i] == 'positive'): tp += 1
            else:
                comp_i += 1
                if (target[i] == 'positive') & (feature[i] == 'negative'): fn += 1
                if (target[i] == 'negative') & (feature[i] == 'positive'): fp += 1

        accuracy  = comp_c/(comp_c + comp_i)
        precision = tp/(tp + fp)
        recall    = tp/(tp + fn)
        return accuracy, precision, recall
    
tnb = TweetNBClassifier(df_train)
tnb = tnb.fit()
predict = tnb.predict(df_test)
score = tnb.score(predict, df_test.airline_sentiment.tolist())


cm = confusion_matrix(df_test['airline_sentiment'], predict).T
cm = cm.astype('float')/cm.sum(axis=0)

fig, ax = mp.subplots()
sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
print(score)

