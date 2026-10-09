# runs the full analysis and writes the results to results/
#
#   python -m scripts.run                        full run
#   python -m scripts.run --n 3000 --no-embed    quick test (overwrites results/)
#   python -m scripts.run --labels fault         nah counted as nta and esh as yta, written to results_fault/
#
# terms used below:
#   tokenizer       splits a post into words
#   normalizer      shrinks the vocabulary, for example by removing common words or cutting words to a stem
#   representation  turns the words into numbers: word counts, tf-idf weights, or a sentence embedding
#
# steps, and the files each one writes:
#   1. compares 25 ways of preparing the text for the same classifier (3 tokenizers x 4 normalizers x 2 word
#      representations, plus sentence embeddings), scored by cross-validation with 95% intervals
#                                                                -> grid.csv
#   2. keeps the predictions of the best of the 25 for every post
#   3. fits a topic model on the posts                             -> topics.csv
#   4. verdict and error rates by group, and tests comparing men and women
#                                                                -> by_*.csv, hypotheses.csv, topic_mix.csv, topic_verdicts.csv, topic_only.csv
#   5. the 40 most confident mistakes, and every prediction        -> errors.csv, predictions.csv
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

from scripts import analysis, data, representation, topics
from scripts.model import classifier, evaluate
from scripts.normalization import normalizers
from scripts.tokenization import tokenizers

np.seterr(all="ignore")   # hides harmless floating-point warnings that some numpy builds (e.g. apple accelerate) raise

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/aita_cleaned_dataset.csv", help="posts file, downloaded from huggingface if missing")
ap.add_argument("--n", type=int, default=50_000, help="number of posts to sample")
ap.add_argument("--k", type=int, default=15, help="number of topics")
ap.add_argument("--labels", choices=list(data.labelings), default="strict",
                help="strict: nta vs yta only. fault: also counts nah as nta and esh as yta")
ap.add_argument("--no-embed", action="store_true", help="skip the sentence embeddings (the slowest step)")
args = ap.parse_args()
out = "results/" if args.labels == "strict" else f"results_{args.labels}/"
Path(out).mkdir(exist_ok=True)

# load: posts with a verdict in the chosen labelling. adds doc (title + text), y (1 = yta), n_words, and age and gender where stated
df = data.load(args.csv, n=args.n, labels=args.labels)
y = df.y.values
print(f"{len(df)} posts, {y.mean():.0%} yta, gender known for {df.gender.notna().mean():.0%}")

# 1. set up every combination. each entry maps (tokenizer, normalizer, representation) to (input, model),
# where the model is the word representation followed by the shared classifier. nothing is trained in this loop
pipelines = {}
for tok_name, tok in tokenizers.items():
    tokens = df.doc.apply(tok)                                     # each post becomes a list of words
    for norm_name, norm in normalizers.items():
        text = tokens.apply(norm).str.join(" ")                    # normalized words joined back into one string per post
        for rep in ["count", "tfidf"]:
            pipelines[tok_name, norm_name, rep] = (text, make_pipeline(representation.sparse(rep), classifier()))
if not args.no_embed:
    # the embedding model reads the raw post and does its own tokenizing, so the tokenizer and normalizer are "-".
    # embeddings are slow to compute the first time, so they are saved to embeddings_<n>.npy in the output folder
    pipelines["-", "-", "embedding"] = (representation.embed(df.doc, f"{out}embeddings_{args.n}.npy"), classifier())

# train and score every combination. evaluate() uses the same cross-validation folds each time, and returns the
# scores with 95% intervals plus the predicted probability of yta for every post
rows, predictions = [], {}
for (tok_name, norm_name, rep), (x, model) in pipelines.items():
    scores, predictions[tok_name, norm_name, rep] = evaluate(model, x, y)
    rows.append({"tokenizer": tok_name, "normalizer": norm_name, "representation": rep, **scores})
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
grid = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)   # best first
grid.to_csv(out + "grid.csv", index=False)

# 2. the combination with the highest macro-f1 supplies each post's predicted probability of yta (p).
# these come from cross-validation, so no post is predicted by a model that was trained on it
best = grid.iloc[0]
df["p"] = predictions[best.tokenizer, best.normalizer, best.representation]
print("best combination:", best.tokenizer, best.normalizer, best.representation)

# 3. topics are found from the words alone. each post is assigned its most likely topic
table, df["topic"] = topics.fit_topics(df.doc, k=args.k)
table["share"] = table.topic.map(df.topic.value_counts(normalize=True)).round(3)   # share of posts in each topic
df["topic_name"] = df.topic.map(table.set_index("topic").name)                    # topic names are defined in topics.py
table.to_csv(out + "topics.csv", index=False)

# 4. groups for the breakdowns: post length in quartiles and narrator age in bands
df["length"] = pd.qcut(df.n_words, 4, labels=["short", "mid-short", "mid-long", "long"])
df["age_band"] = pd.cut(df.age, [12, 19, 24, 29, 39, 90]).astype(str).replace("nan", np.nan)
for col in ["topic_name", "gender", "length", "age_band"]:
    analysis.by(df, col).to_csv(out + f"by_{col}.csv")             # yta rate and model accuracy per group
# tests (two-sided, only posts where the narrator stated a gender, except h4):
#   h1 men and women write about different topics      h2 they get yta at different rates
#   h3 h2 still holds within the same topic            h4 topics differ in yta rate (all posts)
summary, mix, verdicts = analysis.hypotheses(df)
summary.to_csv(out + "hypotheses.csv", index=False)
mix.to_csv(out + "topic_mix.csv", index=False)                     # per topic: share of men's and of women's posts
verdicts.to_csv(out + "topic_verdicts.csv", index=False)           # per topic: yta rate of men and of women
print(summary.round(4).to_string(index=False))
analysis.topic_only(df).to_csv(out + "topic_only.csv", index=False)   # auc of a topic-only model vs the full text model

# 5. mistakes the model made with the most confidence, for qualitative inspection, and every prediction
wrong = df[(df.p > 0.5) != (df.y == 1)]
worst = wrong.assign(confidence=(wrong.p - 0.5).abs()).nlargest(40, "confidence")
worst.assign(doc=worst.doc.str[:600])[["id", "verdict", "p", "topic_name", "title", "doc"]].to_csv(out + "errors.csv", index=False)
df.drop(columns=["doc", "title", "text"]).to_csv(out + "predictions.csv", index=False)
