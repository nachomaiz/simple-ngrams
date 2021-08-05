from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import sys


def ngram_frequency(
    text: list,
    n_min: int = 2,
    n_max: int = 5,
    clean_stopwords: bool = False,
    clean_text: bool = False,
) -> pd.DataFrame:

    n_grams = []
    for i in range(n_min, n_max + 1):
        for line in text:
            for sentence in sent_tokenize(line):
                if clean_stopwords:
                    stop_words = stopwords.words("english")
                    token = [
                        word.casefold()
                        for word in word_tokenize(sentence)
                        if word not in stop_words
                    ]
                else:
                    token = [word.casefold() for word in word_tokenize(sentence)]
                ngram = list(ngrams(token, i))
                n_grams.extend(ngram)
    counts = Counter(x for x in n_grams)
    res = pd.DataFrame(counts.most_common(), columns=["n-gram", "freq"])
    res["word_count"] = res["n-gram"].apply(len)
    if clean_text:
        res["n-gram"] = res["n-gram"].str.join(" ")
    return res


if __name__ == "__main__":

    args = [i for i in sys.argv if not i.startswith("-")]
    kwargs = [i for i in sys.argv if i.startswith("-")]

    file = str(args[1])
    n_min = int(args[2])
    n_max = int(args[3])
    clean_stopwords = True if "-s" in kwargs else False
    clean_text = True if "-t" in kwargs else False

    print(f"Reading {file}...")

    with open(file, encoding="utf8") as f:
        text = f.readlines()

    print(f"Processing file with {len(text)} lines...")

    result = ngram_frequency(text, n_min, n_max, clean_stopwords, clean_text)

    print(
        f"Found {result.shape[0]} n-grams with sizes {n_min} to {n_max}. Saving to Excel..."
    )

    result.to_excel("result.xlsx")
    print("Done")
