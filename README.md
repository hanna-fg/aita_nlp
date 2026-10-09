# aita_nlp

if you have ever found yourself reading strangers argue in a comment section at 3 am, you probably know
[r/amitheasshole](https://www.reddit.com/r/AmItheAsshole/). people post intense personal conflicts and ask whether
they were the asshole in the situation, and thousands of others take time out of their day to give a verdict:
**yta** ("you're the asshole") or **nta** ("not the asshole").

example post title:

> AITA for declining a wedding invitation after doing the math and realizing my 'friends' are incredibly cheap?

this project looks at those verdicts and asks the following questions:

- do men and women write about different topics?
- do they get different verdicts, even within the same topic?
- how much does the way the text is prepared change how well the verdict can be predicted from the post alone?

## run it

run everything from inside this folder.

```bash
conda env create -f environment.yml    # once
conda activate aita_nlp
python -m scripts.run                  # the whole study, writes results/
jupyter lab notebooks/compare.ipynb    # figures and explanations
```

`python -m scripts.run --n 3000 --no-embed` is a quick test (it overwrites `results/`).
`python -m scripts.run --labels fault` reruns the study with every post, asking whether the narrator is at fault: "nah" (no assholes here) counts as nta and "esh" (everyone sucks here) as yta. it writes to `results_fault/`. `--n 0` uses all posts instead of a sample.
the data (about 680mb, from huggingface `OsamaBsher/AITA-Reddit-Dataset`) downloads into `data/` on the first run.

## what is where

| file | what it does |
|---|---|
| `scripts/run.py` | runs every step in order |
| `scripts/data.py` | loads nta/yta posts, reads age and gender from tags like "22f" |
| `scripts/tokenization.py` | tokenizers: how a post is split into words (whitespace, regex, treebank) |
| `scripts/normalization.py` | normalizers: how the vocabulary is shrunk (none, stopwords, stem, both) |
| `scripts/representation.py` | representations: word counts, tf-idf, sentence embeddings |
| `scripts/model.py` | the classifier (logistic regression), scores, and confidence intervals |
| `scripts/topics.py` | lda topic model, with an optional `rename` dict for topic names |
| `scripts/analysis.py` | breakdowns by group and the statistical tests |
| `scripts/plots.py` | one function per figure in the notebook |
| `notebooks/compare.ipynb` | reads `results/` and explains the findings with figures |

to add a variant, write one function and add it to the dictionary at the bottom of that file
(`tokenizers`, `normalizers`); `run.py` compares every entry.

## data

270,709 r/amitheasshole posts from 2013 to april 2023, collected through the reddit pushshift archive. each post comes with its title, text, community verdict, two top-voted comments and score.

> Bsher, O., & Sabri, A. (2023). *AITA Generating Moral Judgements of the Crowd with Reasoning*. arXiv:2310.18336. https://arxiv.org/abs/2310.18336
> dataset: https://huggingface.co/datasets/OsamaBsher/AITA-Reddit-Dataset
