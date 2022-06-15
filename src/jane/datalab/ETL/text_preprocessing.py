import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import string
from typing import List


def apply_preprocessing(df: pd.DataFrame, len_threshold: int, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Apply classic preprocessing steps to dataframe containing a "text" column

    Args:
        df: dataframe containing a column named "text"
        len_threshold: minimum size of the token to keep
        test_size: proportion of the training set (ex: 0.2 for 20%)

    Returns:
        Cleaned train and test dataframe
    """

    # =========== Just for test with old format =============== 
    # df = df.reset_index().rename(columns={"index": "id"})
    # ========================================================= 

    # Init objects before applying preprocessing
    l_stopwords = nltk.corpus.stopwords.words('english')
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    # Apply text preprocessing
    df = df.assign(text=df.text.str.lower())
    df = df.assign(text=df.text.apply(lambda x: tokenization(x)))
    df = df.assign(text=df.text.apply(lambda x: remove_stopwords(x, l_stopwords)))
    df = df.assign(text=df.text.apply(lambda x: stemming(x, porter_stemmer)))
    df = df.assign(text=df.text.apply(lambda x: lemmatizer(x, wordnet_lemmatizer)))
    df = df.assign(text=df.text.apply(lambda x: filter_len(x, len_threshold)))

    # Adapt format for sklearn's TF-IDF
    df = df.assign(text=df.text.apply(lambda x: " ".join(x)))

    # Train/test split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)

    return df_train, df_test


def remove_punctuation(text: str) -> str:
    """Remove punctuation in a string"""
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    return punctuation_free


def tokenization(text: str) -> List[str]:
    """Create a list of token from a string"""
    # tokens = re.split('W+',text)
    l_tokens = re.findall(r'(?i)((?:[a-z]|\')+)', text)
    return l_tokens


def remove_stopwords(l_tokens: List[str], l_stopwords: List[str]) -> List[str]:
    """Remove the english stopwords in a list of token"""
    l_output = [token for token in l_tokens if token not in l_stopwords]
    return l_output


def stemming(l_tokens: List[str], porter_stemmer: PorterStemmer) -> List[str]:
    """Apply stemmer to a list of tokens"""
    l_stem_text = [porter_stemmer.stem(word) for word in l_tokens]
    return l_stem_text


def lemmatizer(l_tokens: List[str], wordnet_lemmatizer: WordNetLemmatizer) -> List[str]:
    """Apply lemmatizer to list of tokens"""
    l_lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in l_tokens]
    return l_lemm_text


def filter_len(l_tokens: List[str], limit: int) -> List[str]:
    """Select only the token in a list having their lenght above a certain threshold"""
    l_filtered = [token for token in l_tokens if len(token) > limit]
    return l_filtered
