# topic model: latent dirichlet allocation (lda) on word counts. it groups words that tend to appear in the same
# posts into topics, without using the verdict or the narrator's gender
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as english_stop_words, CountVectorizer

# fragments left over from contractions in the cleaned text ("didn't" became "didn"), plus filler words that
# would otherwise form a topic of their own. these are ignored on top of the standard english stop words
extra_stop = {"didn", "don", "doesn", "wasn", "isn", "couldn", "wouldn", "ve", "ll", "re", "just", "like",
              "really", "know", "got", "said", "told", "want", "wanted", "think", "going", "say", "feel"}

# optional hand-written topic names, e.g. {2: "pets", 5: "money"}. topics not listed are named by their number
# and three most likely words, e.g. "4: dog/cat/vet"
rename = {}


def fit_topics(docs, k=15, seed=0):
    # words in fewer than 10 posts or more than half of all posts are ignored
    vec = CountVectorizer(stop_words=list(english_stop_words | extra_stop), min_df=10, max_df=0.5, max_features=20_000)
    x = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, n_jobs=-1)
    doc_topic = lda.fit_transform(x)                     # for each post, a probability for each of the k topics
    words = vec.get_feature_names_out()
    top = [", ".join(words[row.argsort()[::-1][:8]]) for row in lda.components_]   # 8 most likely words per topic
    table = pd.DataFrame({"topic": range(k), "top_words": top})
    table["name"] = [rename.get(i, f"{i}: " + "/".join(t.split(", ")[:3])) for i, t in enumerate(top)]
    return table, doc_topic.argmax(axis=1)               # each post is assigned its most likely topic
