from typing import List
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import sys


def ngram_frequency(
    text: List[str],
    n_min: int,
    n_max: int,
    clean_stopwords: bool = False,
    clean_text: bool = False,
) -> pd.DataFrame:
    """Find n-grams in text.

    Parameters
    ----------
    text: List[str]
        Text grouped by line into list. For example, from opening a file with .readlines()

    n_min, n_max: int
        The minimum and maximum number of words in each n-gram

    clean_stopwords: bool
        Whether to ignore stopwords when building n-grams
        This is helpful to compress the text further by removing non-meaningful words

    clean_text: bool
        Whether to convert the resulting n-grams from tuples to full strings

    Returns
    -------
    ngram_frequency: DataFrame
        Frequencies of all n-grams found in the entire text.
    """

    n_grams = []
    stop_words = stopwords.words("english") if clean_stopwords else []

    for i in range(n_min, n_max + 1):  # + 1 since range is exclusive of stop
        for line in text:
            for sentence in sent_tokenize(line):
                token = [
                    word.casefold()  # like str.lower() but supports weird language issues
                    for word in word_tokenize(sentence)
                    if word not in stop_words
                ]
                ngram = list(ngrams(token, i))
                n_grams.extend(ngram)  # add n-grams to final list
    counts = Counter(x for x in n_grams)  # stores n-grams and calculates their frequency
    res = pd.DataFrame(counts.most_common(), columns=["n-gram", "freq"])
    res["word_count"] = res["n-gram"].apply(len)
    if clean_text:
        res["n-gram"] = res["n-gram"].str.join(" ")
    return res


def main():
    args = [i for i in sys.argv if not i.startswith("-")]
    kwargs = [i for i in sys.argv if i.startswith("-")]
    if len(args) != 4:
        raise SyntaxError(
            "Correct syntax: python ngrams.py path-to-text.txt min-size(integer) max-size(integer) (-s and/or -t optional)"
        )

    file = str(args[1])  # 0th arg is this .py file
    n_min = int(args[2])
    n_max = int(args[3])
    clean_stopwords = True if "-s" in kwargs else False
    clean_text = True if "-t" in kwargs else False

    print(f"Reading {file}...")

    with open(file, encoding="utf8") as f:
        text = f.readlines()  # split text and load into a list of lines

    print(f"Processing file with {len(text)} lines...")

    result = ngram_frequency(text, n_min, n_max, clean_stopwords, clean_text)

    print(
        f"Found {result.shape[0]} n-grams with sizes {n_min} to {n_max}. Saving to Excel..."
    )

    result.to_excel("result.xlsx", encoding="utf8")
    print("Done")


if __name__ == "__main__":
    main()
