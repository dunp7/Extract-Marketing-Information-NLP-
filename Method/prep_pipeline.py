'''A py file to build pipeline for preprocessing'''

"""
Some of the common text preprocessing / cleaning steps are:
- Tokenization: v
- Lower casing: v
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



def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
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