import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import json 

import acquire

import warnings
warnings.filterwarnings('ignore')

def basic_clean(string):
    '''this function takes in a string
    and makes everything lowercase
    normalizes, encodes, decodes
    and removes non-alpha-numerics, whitespace, and single quotes
    '''
    # make everything lowercase
    string = string.lower()
    
    # normalize
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8')
    
    # clulnky character removal
    string = re.sub('[^a-z0-9\'\s]', '', string)
    
    return string

def tokenize(string, charms = True):
    ''' This function takes a string and returns a tokenizes version.
    If set to false, returns a list of tokenized strings'''
    
    # create tokenize object
    tokenize = nltk.tokenize.ToktokTokenizer()
    # apply the tokenizer to the string
    string = tokenize.tokenize(string, return_str = charms)
    return string

def stem(string):
    '''
    This function takes a string and 
    returns a string of words stemmed. '''
    
    ps = nltk.porter.PorterStemmer()
    
    stems = [ps.stem(word) for word in string.split()]
    stems = ' '.join(stems)
    return stems

def lemmatize(string):
    '''
    This function takes a string and 
    returns a string of words lemmatized'''
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    lemmas = ' '.join(lemmas)
    return lemmas

