''' This python script is used to preprocess the text with a class that could be used in a machine learning pipeline
and then upload to an AWS database.  I also save the data to a .csv file since it isn't too bulky.  
Once this script is ran, we can begin doing EDA. '''

# Imports
import pandas as pd
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt
import seaborn as sns

from spacy.lang.en import English
import spacy

import gensim
from gensim.utils import simple_preprocess

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from textblob import TextBlob 
from nltk.stem.porter import PorterStemmer

import csv
import re
from functools import reduce
import itertools
import psycopg2 as ps

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import normalize
import tensorflow as tf
import keras

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from sklearn import metrics


''' This class performs the following preprocessing steps:
    1. Tokenize each article and return a list of words.
    2. Remove stopwords such as 'in', 'and', etc.
    3. Lemmatize each word to group similar words together to aid analysis.
    
    During that process it continuously writes to a csv file.'''
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = stopwords.words('English')
        # Articles often contain the newspaper name so we want to get rid of them
        self.stopwords.extend(['reuters', 'fox', 'tass', 'cnn', 'ukrinform'])
        self.lemmatizer = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    
   
    def write_csv(self, publish_date, content, newspaper):
        with open ('preprocessed_ukraine_war_news.csv', 'a', newline='', encoding="utf-8") as outfile:
            writer  = csv.writer(outfile)
            writer.writerow([publish_date, content, newspaper])
    
    def tokenize_text(self, text, newspaper, df_index):
        ''' First we use gensim to preprocess text.  Then, we remove tokens that don't contribute to the overall text in each newspaper'''
    
        
        tokenized_text = simple_preprocess(text, deacc=True)
        
        # For tass, fox and cnn there is an intro that ends with the name of the newspaper so we exclude that.
        if(newspaper == "tass"):
            if('tass' in tokenized_text):
                new_text = tokenized_text[3:]
            else:
                new_text = tokenized_text
            
        if(newspaper == "fox"):
            if(('fox' in tokenized_text) and (tokenized_text[0] == "first")):
                new_text = tokenized_text[3:]
            else:
                new_text = tokenized_text
        if(newspaper == "cnn"):
            if('cnn' in tokenized_text):
                index = tokenized_text.index('cnn')
                new_text = tokenized_text[index:]
            else:
                new_text = tokenized_text
        # Ukrinform just has a useless string at the end.
        if(newspaper == "ukrinform"):
            new_text = tokenized_text[:len(tokenized_text)-1]
            
        # Reuters is complicated but has a pattern of inserting "Register now for FREE unlimited access to Reuters.com Register" in the text which we remove
        # with the following.  register_indices tells us where "register" appears so we exclude the text sandwiched by it.  We also exclude text after the LAST
        # "for" since that only contains information on the authors which is not relevant to the text as a whole.
        if(newspaper == "reuters"):
            
            register_indices = [i for i, e in enumerate(tokenized_text) if e == 'register']
            for_indices = [i for i, e in enumerate(tokenized_text) if e == 'for']
            [i for i, e in enumerate(tokenized_text) if e == 'for']
            
            if(len(register_indices) > 0):
                if(len(for_indices) > 0):
                    max_for_indices = max(for_indices)
                    new_text = tokenized_text[:register_indices[0]]
                    for i in range(1, len(register_indices), 2):
                        if (i < len(register_indices)-1):
                            new_text += tokenized_text[register_indices[i]+1:register_indices[i+1]]
                        else:
                            new_text += tokenized_text[register_indices[len(register_indices)-1]+1:max_for_indices]
                else:
                    new_text = tokenized_text[:register_indices[0]]
                    for i in range(1, len(register_indices), 2):
                        if (i < len(register_indices)-1):
                            new_text += tokenized_text[register_indices[i]+1:register_indices[i+1]]
                        else:
                            new_text += tokenized_text[register_indices[len(register_indices)-1]+1]
            else:
                new_text = tokenized_text
        return new_text
        
    def lemmatize_text(self, text):
        ''' Lemmatize each word in the text'''
        return [str(self.lemmatizer(word)) for word in text]

    def remove_stopwords(self, text):
        ''' Remove stopwords from each text'''
        return [word for word in text if word not in self.stopwords]
    
    def fit(self, *argv):
        return self

    def transform(self, df):
        ''' Performs all the preprocessing steps and writes to csv in a for loop over all entries in dataframe.'''
        new_df = df.copy()
        new_texts = []
        for df_index in df.index:
            print(df_index)
            
            text = df['content'].iloc[df_index]
            newspaper = df['newspaper'].iloc[df_index]
            publish_date = df['publish_date'].iloc[df_index]

            new_text = self.tokenize_text(text, newspaper, df_index)
            new_text = self.remove_stopwords(new_text)
            new_text = self.lemmatize_text(new_text)
            new_text = ' '.join(new_text)
            self.write_csv(publish_date, new_text, newspaper)
            
            new_texts.append(new_text)
        new_df['content'] = new_texts

        return new_df


''' This class was developed by Maarten Grootendorst taken from this https://github.com/MaartenGr/cTFIDF
    We use it to analyze how each newspaper differs in their word choice.'''

class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) """
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF """
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm='l1', copy=False)
        return X

''' Filters out articles that aren't relevant.'''
def filter_articles(df):
    irrelevant_terms = ['musk', 'coronavirus', 'fiona', 'puerto', 'griner']
    for ind in df.index:
        if (("russia" and "ukraine" not in df.loc[ind, 'content']) or (any(terms in df.loc[ind, 'content'] for terms in irrelevant_terms))):
            df.drop(ind, axis=0, inplace=True)
    return df

''' Used to stem when using Tfidf vectorizer'''
def nltk_stemmer(text):
    porter_stemmer = PorterStemmer()
    #text = re.sub(r"[^A-Za-z0-9\-]", " ", text).split()
    # We also want to homogenize the spelling of zelensky
    
    
    stemmed_text = [porter_stemmer.stem(token) for token in text]
    return stemmed_text

''' The next few functions below are used to connect to an AWS database and upload data into a table.'''

def connect_to_database(host_name, dbname, username, password, port):
    try:
        conn = ps.connect(host=host_name, database=dbname, user=username, password=password, port=port)
    except ps.OperationalError as e:
        raise e
    else:
        print("Connected successfully.")
    return conn

def create_table(curr):
    sql_create_table = ( """ CREATE TABLE IF NOT EXISTS preprocessed_ukraine_war_news (
        publish_date DATE NOT NULL,
        content TEXT NOT NULL,
        newspaper TEXT NOT NULL
        )""")
    curr.execute(sql_create_table)

def insert_into_table(curr, publish_date, content, newspaper):
    insert_into_ukrwar = ("""INSERT INTO preprocessed_ukraine_war_news (publish_date, content, newspaper)
    VALUES(%s,%s,%s);""")
    row_to_insert = (publish_date, content, newspaper)
    curr.execute(insert_into_ukrwar, row_to_insert)

def df_to_db(curr, df):
    for df_index in df.index:
        insert_into_table(curr, df['publish_date'].iloc[df_index], df['content'].iloc[df_index], df['newspaper'].iloc[df_index])
        print(" At row = ", df_index)

''' Set database name and other parameters for AWS connection below'''
username = '____'
password = '____'
host_name = '____.rds.amazonaws.com'
port = '5432'
dbname = '____'

''' The following functions are utility functions used in the main notebook for analysis purposes'''

''' This function groups articles together into a single string based on newspaper and returns a dataframe with the articles in "content". 
    Assumes df has the column "publish_date", "newspaper", and "content"  '''
def unify_text_by_newspaper(df):
    words_grouped_by_newspaper = []
    newspapers = []
    for newspaper in df['newspaper'].unique():
        words = list(pd.Series(reduce(lambda x, y: x+y, df[df['newspaper']== newspaper]['content'])))
        newspapers.append(newspaper)
        # Need to do this in order for TDIDF to work
        words_grouped_by_newspaper.append(' '.join(words))
    newspaper_df = pd.DataFrame()
    newspaper_df['newspaper'] = newspapers
    newspaper_df['content'] = words_grouped_by_newspaper
    newspaper_df['publish_date'] = df['publish_date']
    return newspaper_df

def get_sentiment_score(df):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in df['content']:
        score = analyzer.polarity_scores(text)
        print(score)
        scores.append(score['compound'])
    
    df['sentiment_score'] = scores
    return df
''' Gets the keywords for each newspaper and returns a df with keywords stored in a series.  Needs to pass in a Tfidf vectorizer.'''
def get_newspaper_keywords(df, tfidf_vectorizer):
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])
    terms = tfidf_vectorizer.get_feature_names()
    
    all_keywords = []
    for row in range(tfidf_matrix.shape[0]):
        keywords = []
        for i in tfidf_matrix[row].toarray()[0].argsort()[::-1][:10]:
            keywords.append(terms[i])
        all_keywords.append(keywords)
    print(len(all_keywords))
    df['keywords'] = all_keywords
    return df

''' Returns a list of ngrams from a given text.  Assumes the text is already tokenized.'''
def create_ngrams(text, n=2):
    return [' '.join(text[i:i+n]) for i in range(len(text)-n)]

''' Splits ngrams back into unigrams.'''
def retokenize_from_ngrams(text, n=2):
    text = list(itertools.chain(*[text[i].split() for i in range(len(text))]))
    return [e for i, e in enumerate(text) if ((i+1) % 2 != 0)]

''' Removes unwanted unigrams and bigrams from list of tokens.  Based on results from determining keywords for each newspaper'''
def remove_unwanted_unigrams_bigrams(text):
    unwanted_bigrams=['click get', 'news app', 'click', 'get news', 'next image', 'image prev', 
    'news digital', 'getty images', 'ad feedback', 'feedback source', 'video ad', 
    'afp getty', 'opinion join', 'continue reading', 'image image', 'live updates', 'told digital', 'end video', 'begin video', 
    'video clip']
    unwanted_unigrams=['newsletter', 'prev', 'inbox', 'weekly', 'read']
    text = [word for word in text if word not in unwanted_unigrams]
    alternate_zelenski = ['zelensky', 'zelenskyy', 'zelenskiy']
    alternate_kiev = ['kyiv']
    for spelling in alternate_zelenski:
        text = [sub.replace(spelling, 'zelenski') for sub in text]
    
    for spelling in alternate_kiev:
        text = [sub.replace(spelling, 'kiev') for sub in text]

    text = ' '.join(text)
    pattern = re.compile(r' | '.join(unwanted_bigrams))
    text = re.sub(pattern,' ',text).split()
    return text

def xgboost_fit_evaluate(xgboost, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5):
    
    if useTrainCV:
        xgb_param = xgboost.get_xgb_params()
        xgb_param['num_class']=5
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgboost.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc')
        xgboost.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    xgboost.fit(X_train, y_train,eval_metric='auc')
        
    #Predict training set:
    predictions = xgboost.predict(X_test)
    #dtrain_predprob = xgboost.predict_proba(X_test)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print(metrics.classification_report(y_test, predictions, target_names=['cnn', 'fox', 'reuters', 'tass', 'ukrinform']) )

def xgboost_plot_importance(xgboost, num_important_features, feature_names):
    # Get feature importance                
    feature_important = xgboost.get_booster().get_score(importance_type='weight')
    # Get the index and values
    keys = list(feature_important.keys())
    
    values = list(feature_important.values())

    fig, ax = plt.subplots(1, 1, figsize=(10,7))

    # Preprocessing to get index of important words for plotting
    keys_index = [int(word.replace('f', '')) for word in keys]
    important_features = [feature_names[ind] for ind in keys_index]

    # Plot n most important features
    data = pd.DataFrame(data=values, index=important_features, columns=["score"]).sort_values(by = "score", ascending=False)
    nlargest_data = data.nlargest(num_important_features, columns="score")
    sns.barplot(x=nlargest_data['score'].values, y=nlargest_data.index, ax=ax, palette=sns.color_palette("hls", num_important_features))
    ax.set_xlabel("Importance", fontdict={'fontsize' : 15})

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

'''Generates batches of data that convert sparse matrices of TFIDF values to an array which are then passed into the neural net
    Code based on code presented in the article https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'''
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_data, y_data, batch_size=32, dim=(17862, 50000), n_channels=1,
                 n_classes=5, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.X_data = X_data
        self.y_data = y_data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.X_data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        X_batch = self.X_data[indexes].toarray()
        y_batch = self.y_data[indexes]
        

        return X_batch, y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X_data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    # Read in the raw unprocessed data
    articles = pd.read_csv('ukraine_war_news.csv')
    # Preprocess texts
    processed_articles = TextPreprocessor().transform(articles)
    # Filter out irrelvant texts that aren't about the Ukraine War.  I could put this in the TextPreprocessor class but I wanted that class
    # to preserve all the texts.  I drop in place since the unfiltered text is safefully stored in a csv file.
    for index in processed_articles.index:
        if ("ukraine" not in processed_articles['content'].iloc[index]):
            processed_articles.drop(index=index, inplace=True)
    print("hi")
    # Set up connection
   # conn = None
   # conn = connect_to_database(host_name, dbname, username, password, port)
   # curr = conn.cursor()
    # Upload data to table
    #df_to_db(curr, processed_articles)
    #conn.commit()

    # Also save filtered articles to new csv
   # processed_articles.to_csv('relevant_preprocessed_ukraine_war_news.csv', index=False)