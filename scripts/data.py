# loads the reddit posts and prepares them for analysis
import pandas as pd
from pathlib import Path

repo, fname = "OsamaBsher/AITA-Reddit-Dataset", "cleaned_dataset.csv"

# the dataset has four verdicts. "strict" keeps the two clear-cut ones. "fault" keeps every post and asks whether
# the narrator is at fault: "nah" (no assholes here) counts as nta, "esh" (everyone sucks here) counts as yta
labelings = {"strict": {"nta": 0, "yta": 1},
             "fault": {"nta": 0, "nah": 0, "yta": 1, "esh": 1}}

# narrators often state their age and gender in the title or first line, e.g. "22f" or "m 35"
tag = r"\b(?:(\d{2})\s?([mf])|([mf])\s?(\d{2}))\b"


def fetch(path):
    # returns the path to the csv, downloading it from huggingface first if it is missing (about 680mb)
    path = Path(path)
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path.parent.mkdir(exist_ok=True)
        hf_hub_download(repo, fname, repo_type="dataset", local_dir=path.parent)
        (path.parent / fname).rename(path)
    return path


def load(path, n=None, seed=0, labels="strict"):
    # only the title and text are read. the comments are left out because they contain the verdict itself
    df = pd.read_csv(fetch(path), usecols=["id", "title", "text", "verdict"])
    df = df[df.verdict.isin(labelings[labels])].dropna(subset=["text"])
    if n:
        df = df.sample(n, random_state=seed)                           # random sample of n posts
    df = df.reset_index(drop=True)
    df["doc"] = df.title + ". " + df.text                              # the full post
    df["y"] = df.verdict.map(labelings[labels])                        # 1 = yta, 0 = nta
    df["n_words"] = df.doc.str.split().str.len()

    # age and gender are only known when the narrator stated them (about a third of posts)
    t = df.doc.str[:250].str.extract(tag)
    age = pd.to_numeric(t[0].fillna(t[3]))
    plausible = age.between(13, 90)
    df["age"] = age.where(plausible)
    df["gender"] = t[1].fillna(t[2]).where(plausible)                  # "m" or "f"
    return df
