import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
#!python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

# Read the CSV file into a Pandas DataFrame
train_df = pd.read_csv('./data/updated_output_classification.csv')
train_df = train_df.dropna()
print(train_df.shape)
# Split the data into a training set and a validation set
train_df, test_df= train_test_split(train_df, test_size=0.05, random_state=42)


def tokenize(sentence,method='spacy'):
# Tokenize and lemmatize text, remove stopwords and punctuation

    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)

    if method=='nltk':
        # Tokenize
        tokens = nltk.word_tokenize(sentence,preserve_line=True)
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
        # Lemmatize
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
        tokens = " ".join([i for i in tokens])
    return tokens


tqdm.pandas()
train_df['processed_text'] = train_df['Question'].progress_apply(lambda x: tokenize(x,method='nltk'))

# Process the test set text
tqdm.pandas()
test_df['processed_text'] = test_df['Question'].progress_apply(lambda x: tokenize(x,method='nltk'))

def build_features(train_data, test_data, ngram_range, method='count'):
    if method == 'tfidf':
        # Create features using TFIDF
        vec = TfidfVectorizer(ngram_range=ngram_range)
        X_train = vec.fit_transform(train_df['processed_text'])
        X_test = vec.transform(test_df['processed_text'])

    else:
        # Create features using word counts
        vec = CountVectorizer(ngram_range=ngram_range)
        X_train = vec.fit_transform(train_df['processed_text'])
        X_test = vec.transform(test_df['processed_text'])

    return X_train, X_test

method = 'tfidf'
ngram_range = (1, 2)
X_train,X_test = build_features(train_df['processed_text'],test_df['processed_text'],ngram_range,method)
y_train = train_df['Question class']
logreg_model = LogisticRegression(solver='saga')
logreg_model.fit(X_train,y_train)
preds = logreg_model.predict(X_train)
acc = sum(preds==y_train)/len(y_train)
print('Accuracy on the training set is {:.3f}'.format(acc))


y_test = test_df['Question class']
test_preds = logreg_model.predict(X_test)
test_acc = sum(test_preds==y_test)/len(y_test)
print('Accuracy on the test set is {:.3f}'.format(test_acc))