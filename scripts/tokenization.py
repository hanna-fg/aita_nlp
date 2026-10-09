# tokenizers split a post into words (tokens). each one takes a string and returns a list of tokens
import re
from nltk.tokenize import TreebankWordTokenizer


def whitespace(doc):
    return doc.split()                             # splits on spaces only, so "me." and "me" count as different tokens


def regex(doc):
    return re.findall(r"[a-z']+", doc)             # keeps letters and apostrophes, drops punctuation and digits


def treebank(doc):
    return TreebankWordTokenizer().tokenize(doc)   # penn treebank rules, e.g. "don't" becomes "do" and "n't"


tokenizers = {"whitespace": whitespace, "regex": regex, "treebank": treebank}   # run.py compares every entry
