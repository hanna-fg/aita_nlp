# the classifier and the scores. the classifier is the same for every comparison in the study
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def classifier():
    return LogisticRegression(class_weight="balanced", max_iter=2000)   # "balanced" corrects for nta posts being about 4 times as common as yta


def score(y, p):
    # y is the true verdict (1 = yta) and p the predicted probability of yta. a post counts as predicted yta when p > 0.5
    guess = (p > 0.5).astype(int)
    return {"accuracy": accuracy_score(y, guess),
            "macro_f1": f1_score(y, guess, average="macro"),    # average of the f1 for nta and for yta, so both count equally
            "auc": roc_auc_score(y, p)}                         # chance that a random yta post gets a higher p than a random nta post (0.5 = guessing)


def bootstrap(y, p, n=200, seed=0):
    # estimates how much the scores would change with a different set of posts: the posts are resampled with
    # replacement n times, each resample is scored, and the middle 95% of those scores is returned (low, high)
    rng = np.random.default_rng(seed)
    runs = pd.DataFrame([score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))])
    return runs.quantile(.025), runs.quantile(.975)


def evaluate(model, x, y):
    # 5-fold cross-validation: the posts are split into 5 parts, and each part is predicted by a model trained on the
    # other 4. every post is therefore predicted by a model that never saw it. returns the scores with 95% intervals,
    # and the predicted probability for every post
    folds = StratifiedKFold(5, shuffle=True, random_state=0)         # the same folds for every model, so scores are comparable
    p = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    low, high = bootstrap(y, p)
    scores = score(y, p)
    for name in ["auc", "macro_f1"]:
        scores[name + "_low"], scores[name + "_high"] = low[name], high[name]
    return scores, p
