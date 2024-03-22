'''A py file to build pipeline for preprocessing'''

"""
Some of the common text preprocessing / cleaning steps are:
- Tokenization: v
- Lower casing: v
- Removal of Numbers: v
- Removal of Punctuations:  v
- Removal of Stopwords: v
- Removal of Frequent words: Optional
- Removal of Rare words: Optional
- Stemming
- Lemmatization
- Spelling correction: 

"""
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import string
from nltk.corpus import stopwords
# from spellchecker import Spellchecker
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

def preprocess_data(data):
    # Choose text >= 20
    data['Num_words_text'] = data['Text'].apply(lambda x:len(str(x).split())) 
    df_filtered_reviews = data[(data['Num_words_text'] >=20)]

    # Balacing the review
    df_sampled = df_filtered_reviews.groupby('Score').apply(lambda x: x.sample(n=10000, random_state = 17)).reset_index(drop = True)
    return df_sampled

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Removing numbers
    text = ' '.join(w for w in text.split() if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3)))

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ['i\'m', 'i\'ll']
    stop_words.update(custom_stopwords)
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    # # Stemming
    # stemmer = PorterStemmer()
    # text = " ".join([stemmer.stem(word) for word in text.split()])
    
    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    
    # Tokenizing
    tokens = word_tokenize(text)
    
    return tokens