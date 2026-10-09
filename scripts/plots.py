# the figures in notebooks/compare.ipynb. each function takes tables from results/ and draws one figure
import numpy as np
import matplotlib.pyplot as plt

# colours: blue = men, red = women. grey and dark are used for everything else
men, women, grey, ink, faint = "#2a78d6", "#e34948", "#9a9a95", "#222222", "#dddddd"
false_pos, false_neg, light = "#4a3aa7", "#008300", "#b5b5b0"   # purple and green for the two error types, so they are not read as gender

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": grey, "axes.labelcolor": ink, "text.color": ink,
                     "xtick.color": ink, "ytick.color": ink, "figure.dpi": 110})


def pipelines(grid):
    # auc of every text-preparation combination, with 95% intervals, coloured by representation
    g = grid.sort_values("auc").reset_index(drop=True)
    rep_colour = {"tfidf": ink, "count": light, "embedding": false_pos}
    colours = g.representation.map(rep_colour)
    labels = [f"{t} / {n} / {r}" if r != "embedding" else "sentence embedding" for t, n, r in zip(g.tokenizer, g.normalizer, g.representation)]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.hlines(range(len(g)), g.auc_low, g.auc_high, color=colours, lw=2)
    ax.scatter(g.auc, range(len(g)), color=colours, s=26, zorder=3)
    ax.set_yticks(range(len(g)), labels, fontsize=8); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("auc, with 95% interval"); ax.grid(axis="x", color=faint, lw=.7, zorder=0)
    for name, col in rep_colour.items():
        ax.scatter([], [], color=col, s=26, label={"tfidf": "tf-idf", "count": "word counts", "embedding": "sentence embedding"}[name])
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, title="tokenizer / normalizer / representation", title_fontsize=8)
    plt.tight_layout()


def macro_f1_grid(grid):
    # macro-f1 for each tokenizer x normalizer, one panel per word representation, on a shared colour scale
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    sparse = grid[grid.representation != "embedding"]
    lo, hi = sparse.macro_f1.min(), sparse.macro_f1.max()
    for ax, rep in zip(axes, ["count", "tfidf"]):
        t = sparse[sparse.representation == rep].pivot(index="tokenizer", columns="normalizer", values="macro_f1")
        t = t[["none", "stopwords", "stem", "stopwords+stem"]]
        ax.imshow(t.values, cmap="Blues", vmin=lo, vmax=hi, aspect="auto")
        ax.set_xticks(range(4), t.columns, rotation=20); ax.set_yticks(range(3), t.index)
        ax.set_title({"count": "word counts", "tfidf": "tf-idf"}[rep], loc="left", fontsize=10)
        for (i, j), v in np.ndenumerate(t.values):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > (lo + hi) / 2 else ink)
        for s in ax.spines.values():
            s.set_visible(False)
    plt.tight_layout()


def yta_by_topic(by_topic, overall):
    # share of posts judged yta in each topic, against the rate over all posts (dashed line)
    t = by_topic.sort_values("yta_rate")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(t.topic_name, t.yta_rate * 100, color=grey, height=.65)
    ax.axvline(overall * 100, color=ink, lw=1, ls="--", zorder=1)
    ax.text(overall * 100 + .3, len(t) - .4, f"all posts {overall:.0%}", fontsize=8.5, va="center")
    for y, (r, n) in enumerate(zip(t.yta_rate, t.n)):
        ax.text(r * 100 + .4, y, f"{r:.0%}  (n={n:,})", va="center", fontsize=8.5, zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xlim(0, t.yta_rate.max() * 100 + 10); ax.set_ylim(-.6, len(t) - .1)
    ax.set_xlabel("share of posts judged yta (%)"); ax.tick_params(axis="y", length=0)
    plt.tight_layout()


def topic_only(topic_only):
    # auc of a model that sees only the topic, against the full text model
    fig, ax = plt.subplots(figsize=(7, 2.2))
    for i, r in enumerate(topic_only.itertuples()):
        col = grey if r.model == "topic alone" else ink
        ax.plot([r.auc_low, r.auc_high], [i, i], color=col, lw=2.5)
        ax.scatter(r.auc, i, s=50, color=col, zorder=3)
        ax.text(r.auc_high + .004, i, f"{r.auc:.3f}", va="center")
    ax.axvline(.5, color=ink, lw=1, ls="--"); ax.text(.502, 1.45, "guessing", fontsize=8.5)
    ax.set_yticks([0, 1], topic_only.model); ax.set_ylim(-.5, 1.7); ax.set_xlim(.48, .75)
    ax.set_xlabel("auc, with 95% interval"); ax.tick_params(axis="y", length=0)
    plt.tight_layout()


def topic_mix(mix):
    # share of men's and of women's posts in each topic. * marks topics that differ after correction
    m = mix.sort_values("diff")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = range(len(m))
    ax.hlines(y, m.women * 100, m.men * 100, color=faint, lw=3, zorder=1)
    ax.scatter(m.women * 100, y, s=45, color=women, zorder=2, label="women")
    ax.scatter(m.men * 100, y, s=45, color=men, zorder=2, label="men")
    for i, r in enumerate(m.itertuples()):
        if r.p_adj < .05:
            ax.text(max(r.men, r.women) * 100 + .6, i, "*", va="center", fontsize=13)
    ax.set_yticks(y, m.topic_name); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("share of that gender's posts in the topic (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    plt.tight_layout()


def yta_by_gender(pred):
    # yta rate of men and women overall (with 95% intervals) and by narrator age
    g = pred.dropna(subset=["gender"])
    rate = g.groupby("gender").y.agg(["mean", "size"])
    rate["ci"] = 1.96 * np.sqrt(rate["mean"] * (1 - rate["mean"]) / rate["size"])   # normal approximation
    age = g.dropna(subset=["age_band"]).groupby(["age_band", "gender"]).y.mean().unstack()
    age = age.loc[sorted(age.index, key=lambda s: float(s.strip("()[]").split(",")[0]))]
    age.index = ["13-19", "20-24", "25-29", "30-39", "40+"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    for i, (who, col) in enumerate([("f", women), ("m", men)]):
        a.bar(i, rate.loc[who, "mean"] * 100, yerr=rate.loc[who, "ci"] * 100, color=col, width=.55, capsize=4, ecolor=ink)
        a.text(i, rate.loc[who, "mean"] * 100 + 3, f"{rate.loc[who, 'mean']:.1%}", ha="center")
    a.set_xticks([0, 1], ["women", "men"]); a.set_ylabel("share judged yta (%)"); a.set_ylim(0, 30)
    a.set_title("overall (95% interval)", loc="left", fontsize=10)

    for who, col, name in [("f", women, "women"), ("m", men, "men")]:
        b.plot(age.index, age[who] * 100, color=col, lw=2, marker="o", ms=5)
        b.text(4.1, age[who].iloc[-1] * 100, name, color=col, va="center")
    b.set_ylim(0, 30); b.set_xlim(-.2, 4.9); b.set_ylabel("share judged yta (%)"); b.set_xlabel("narrator age")
    b.set_title("by narrator age", loc="left", fontsize=10)
    plt.tight_layout()


def gender_gap_by_topic(verdicts, hyp):
    # men-minus-women yta rate inside each topic, with 95% intervals. filled dots are significant after correction,
    # and the grey band is the overall difference (h2) with its interval
    f = verdicts.assign(sig=verdicts.p_adj < .05)
    h2 = hyp.set_index("id").loc["h2"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axvspan(h2.ci_low * 100, h2.ci_high * 100, color=faint, zorder=0)
    ax.axvline(0, color=ink, lw=1)
    for y, r in enumerate(f.itertuples()):
        col = ink if r.sig else grey
        ax.plot([r.ci_low * 100, r.ci_high * 100], [y, y], color=col, lw=1.6)
        ax.scatter(r.diff * 100, y, s=38, color=col if r.sig else "white", edgecolor=col, zorder=3)
    ax.set_yticks(range(len(f)), f.topic_name); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("yta rate: men minus women (percentage points)")
    ax.text(h2.estimate * 100, len(f) - .1, "overall", ha="center", fontsize=8.5, color=grey)
    plt.tight_layout()


def errors_by_topic(by_topic):
    # false positive and false negative rates of the classifier in each topic
    t = by_topic.sort_values("yta_rate")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = range(len(t))
    ax.hlines(y, t.false_neg * 100, t.false_pos * 100, color=faint, lw=3, zorder=1)
    ax.scatter(t.false_pos * 100, y, s=45, color=false_pos, zorder=2, label="false positives (nta predicted as yta)")
    ax.scatter(t.false_neg * 100, y, s=45, color=false_neg, zorder=2, label="false negatives (yta predicted as nta)")
    ax.set_yticks(y, t.topic_name); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("share of posts misjudged (%)"); ax.set_xlim(0, 80)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False, fontsize=8.5)   # above the plot, clear of the data
    plt.tight_layout()


def by_length(by_length):
    # yta rate and classifier errors by post-length quartile
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3))
    a.bar(by_length.length, by_length.yta_rate * 100, color=grey, width=.6)
    a.set_ylabel("share judged yta (%)"); a.set_title("verdicts", loc="left", fontsize=10)
    b.plot(by_length.length, by_length.false_pos * 100, color=false_pos, lw=2, marker="o")
    b.plot(by_length.length, by_length.false_neg * 100, color=false_neg, lw=2, marker="o")
    b.text(3.1, by_length.false_pos.iloc[-1] * 100, "false positives", color=false_pos, va="center", fontsize=8.5)
    b.text(3.1, by_length.false_neg.iloc[-1] * 100, "false negatives", color=false_neg, va="center", fontsize=8.5)
    b.set_xlim(-.3, 4.2); b.set_ylabel("share misjudged (%)"); b.set_title("model errors", loc="left", fontsize=10)
    for ax in (a, b):
        ax.set_xlabel("post length (quartile)")
    plt.tight_layout()
