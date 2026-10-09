# representations turn the text of a post into numbers a classifier can use
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

# the tokens of each post are joined with spaces, so the vectorizers only need to split on spaces.
# words in fewer than 5 posts are ignored, and at most 50,000 words are kept
settings = dict(token_pattern=r"\S+", lowercase=False, min_df=5, max_features=50_000)


def sparse(kind):
    # one column per word. "tfidf" weights each word by how rare it is across posts
    if kind == "tfidf":
        return TfidfVectorizer(**settings)
    # "count" is the raw number of times each word appears. the counts are rescaled to 0-1 per word, because
    # long posts otherwise produce very large values that make the logistic regression slow to converge
    return make_pipeline(CountVectorizer(**settings), MaxAbsScaler())


def embed(docs, cache):
    # sentence embeddings: each post becomes a vector of 384 numbers that capture its meaning. the model does its
    # own tokenization, and only reads roughly the first 256 word pieces of each post. computed once, then cached
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    vectors = SentenceTransformer("all-MiniLM-L6-v2").encode(list(docs), batch_size=64, show_progress_bar=True)
    np.save(cache, vectors)
    return vectors
