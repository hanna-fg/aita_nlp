# normalizers shrink the vocabulary by merging or removing tokens. each takes a list of tokens and returns a list
from functools import lru_cache
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as english_stop_words

stem_word = lru_cache(maxsize=None)(SnowballStemmer("english").stem)   # cached because the same words repeat constantly


def none(tokens):
    return tokens                                  # leaves the tokens as they are


def stopwords(tokens):
    return [t for t in tokens if t not in english_stop_words]   # removes very common words such as "the" and "and"


def stem(tokens):
    return [stem_word(t) for t in tokens]          # cuts words to a stem, e.g. "yelling" and "yelled" become "yell"


def stopwords_and_stem(tokens):
    return stem(stopwords(tokens))


normalizers = {"none": none, "stopwords": stopwords, "stem": stem, "stopwords+stem": stopwords_and_stem}   # run.py compares every entry
