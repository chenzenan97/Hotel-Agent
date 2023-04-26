import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

def cosine_similarity_top_k(file_path, query, k=10):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    file = pd.read_csv(file_path)

    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    vocabulary = set()
    for tokens in file['Question']:
        vocabulary.update(tokens.split())
    vocabulary = list(vocabulary)

    tfidf = TfidfVectorizer(vocabulary=vocabulary)
    tfidf.fit(file['Question'])
    tfidf_matrix = tfidf.transform(file['Question'])

    def cosine_sim(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            cos_sim = 0
        else:
            cos_sim = np.dot(a, b) / (norm_a * norm_b)
        return cos_sim

    def gen_vector_T(tokens):
        Q = np.zeros((len(vocabulary)))
        x = tfidf.transform(tokens)
        for token in tokens[0].split():
            try:
                ind = vocabulary.index(token)
                Q[ind]  = x[0, tfidf.vocabulary_[token]]
            except:
                pass
        return Q

    preprocessed_query = re.sub("\W+", " ", query).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    q_df.loc[0, 'q_clean'] = tokens
    q_df['q_clean'] = q_df['q_clean'].apply(lambda x: [wordnet_lemmatizer.lemmatize(word) for word in x])
    q_df['q_clean'] = q_df['q_clean'].apply(lambda x: " ".join(x))
    d_cosines = []

    query_vector = gen_vector_T(q_df['q_clean'])
    for d in tfidf_matrix.A:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    result_df = pd.DataFrame()
    for i, index in enumerate(out):
        result_df.loc[i, 'index'] = str(index)
        result_df.loc[i, 'input'] = str(query)
        result_df.loc[i, 'Question'] = file['Question'][index]
        result_df.loc[i, 'Answer'] = file['Answer'][index]

    for j, simScore in enumerate(d_cosines[-k:][::-1]):
        result_df.loc[j, 'Score'] = simScore

    return result_df
