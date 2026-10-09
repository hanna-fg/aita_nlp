# breakdowns of verdicts and model errors by group, and statistical tests comparing men and women
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from scripts.model import bootstrap, classifier


def by(df, col):
    # for each group in df[col]: number of posts, share judged yta, and how often the model is right or wrong.
    # false_pos = share of nta posts the model called yta, false_neg = share of yta posts the model called nta
    guess = (df.p > 0.5).astype(int)
    d = df.assign(accuracy=(guess == df.y).astype(float),
                  false_pos=guess.where(df.y == 0),
                  false_neg=(1 - guess).where(df.y == 1))
    g = d.groupby(col, observed=True)
    out = g[["y", "accuracy", "false_pos", "false_neg"]].mean().rename(columns={"y": "yta_rate"})
    out.insert(0, "n", g.size())
    return out.round(3)


def diff(m_hits, m_n, f_hits, f_n):
    # men's share minus women's share (m_hits out of m_n men, f_hits out of f_n women),
    # with a 95% confidence interval and a two-sided two-proportion z-test
    low, high = confint_proportions_2indep(m_hits, m_n, f_hits, f_n, compare="diff")
    p = proportions_ztest([m_hits, f_hits], [m_n, f_n])[1]
    return dict(men=m_hits / m_n, women=f_hits / f_n, diff=m_hits / m_n - f_hits / f_n, ci_low=low, ci_high=high, p=p)


def per_topic(rows):
    # one row per topic, sorted by the difference. p_adj corrects the p-values for testing every topic at once
    # (benjamini-hochberg false discovery rate), so a topic is only flagged if it still stands out after that
    t = pd.DataFrame(rows).T.reset_index(names="topic_name")
    t["p_adj"] = multipletests(t.p.astype(float), method="fdr_bh")[1]
    return t.sort_values("diff")


def hypotheses(df):
    # the comparisons between men and women use only posts where the narrator stated their gender.
    # every test is two-sided: it asks whether the groups differ, not in which direction
    d = df.dropna(subset=["gender"])
    men, women = d[d.gender == "m"], d[d.gender == "f"]
    rows = []

    # h1: men and women write about different topics (chi-square test on the gender x topic table)
    counts = pd.crosstab(d.topic_name, d.gender)
    chi2, p, dof, _ = stats.chi2_contingency(counts)
    rows.append(["h1", "men and women write about different topics", f"chi-square, {dof} df; estimate is cramer's v",
                 np.sqrt(chi2 / len(d)), np.nan, np.nan, p])

    # h2: men and women get yta at different rates
    r = diff(men.y.sum(), len(men), women.y.sum(), len(women))
    rows.append(["h2", "men and women get yta at different rates", "difference in yta rate, men minus women",
                 r["diff"], r["ci_low"], r["ci_high"], r["p"]])

    # h3: the yta difference remains when comparing posts about the same topic (logistic regression with topic as a control)
    fit = smf.logit("y ~ C(gender) + C(topic_name)", data=d).fit(disp=0)
    ci = np.exp(fit.conf_int().loc["C(gender)[T.m]"])
    rows.append(["h3", "the yta difference remains after controlling for topic", "odds ratio, men vs women, same topic",
                 np.exp(fit.params["C(gender)[T.m]"]), ci[0], ci[1], fit.pvalues["C(gender)[T.m]"]])

    # h4: topics differ in how often they end in yta. gender is not needed here, so all posts are used
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df.topic_name, df.y))
    rows.append(["h4", "yta rates differ across topics", f"chi-square, {dof} df; estimate is cramer's v",
                 np.sqrt(chi2 / len(df)), np.nan, np.nan, p])
    summary = pd.DataFrame(rows, columns=["id", "hypothesis", "test", "estimate", "ci_low", "ci_high", "p"])

    # what each gender writes about: for each topic, the share of men's posts and of women's posts that fall in it
    mix = per_topic({t: diff(counts.loc[t, "m"], counts["m"].sum(), counts.loc[t, "f"], counts["f"].sum()) for t in counts.index})

    # how each gender is judged within each topic: the yta rate of men and of women
    verdicts = per_topic({t: diff(x[x.gender == "m"].y.sum(), (x.gender == "m").sum(), x[x.gender == "f"].y.sum(), (x.gender == "f").sum())
                          for t, x in d.groupby("topic_name")})
    return summary, mix, verdicts


def topic_only(df):
    # how much of the prediction comes from topic alone? compares a model that sees only which of the topics a post
    # belongs to (no words, no gender) with the full text model. both predict every post by 5-fold cross-validation
    x = pd.get_dummies(df.topic_name).values.astype(float)                       # one yes/no column per topic
    y = df.y.values
    p_topic = cross_val_predict(classifier(), x, y, cv=5, method="predict_proba")[:, 1]
    rows = {}
    for name, p in [("topic alone", p_topic), ("full text model", df.p.values)]:
        low, high = bootstrap(y, p)
        rows[name] = {"auc": roc_auc_score(y, p), "auc_low": low["auc"], "auc_high": high["auc"]}
    out = pd.DataFrame(rows).T.reset_index(names="model")
    out["share_of_lift"] = (out.auc - .5) / (out.auc.iloc[1] - .5)               # skill above guessing (auc 0.5), as a share of the text model's
    return out
