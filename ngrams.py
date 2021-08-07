from typing import List, Dict, Tuple, NamedTuple
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter, namedtuple
import pandas as pd
import sys
import os

import cli_msg

__doc__ = cli_msg.help


def parse_args(argv: List[str]) -> Tuple[NamedTuple, Dict[str, str]]:
    args = [i for i in argv if not i.startswith("-")]
    kwargs = [i for i in argv if i.startswith("-")]

    if "-h" in kwargs:
        sys.exit(__doc__)
    elif len(args) != 4 or len(kwargs) > 2:
        sys.exit(cli_msg.usage)

    Args = namedtuple("Args", ["path", "n_min", "n_max"])
    args = Args(str(args[1]), int(args[2]), int(args[3]))
    kwargs = {
        "clean_stopwords": True if "-s" in kwargs else False,
        "tuples": True if "-t" in kwargs else False,
    }

    return args, kwargs


def open_file(path: str) -> List[str]:
    print(f"Reading {path}...")

    with open(path, encoding="utf8") as f:
        return f.readlines()  # split text and load into a list of lines


def make_dir(dir: str) -> None:
    if not os.path.exists(dir):
        os.mkdir(dir)


def save_file(df: pd.DataFrame, args: NamedTuple, dir: str = "result") -> None:
    print(
        f"Found {df.shape[0]} n-grams with sizes {args.n_min} to {args.n_max}. Saving to Excel..."
    )

    _, filename = os.path.split(args.path)

    make_dir(dir)

    df.to_excel(
        f"{dir}\\ngrams_{filename}_{args.n_min}-{args.n_max}.xlsx", encoding="utf8"
    )
    print("Done")


def ngram_frequency(
    text: List[str],
    n_min: int,
    n_max: int,
    clean_stopwords: bool = False,
    tuples: bool = False,
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
    counts = Counter(x for x in n_grams)  # stores & calculates n-gram frequency
    res = pd.DataFrame(counts.most_common(), columns=["n-gram", "freq"])
    res["word_count"] = res["n-gram"].apply(len)
    if not tuples:
        res["n-gram"] = res["n-gram"].str.join(" ")
    return res


def main(argv: List[str]):
    args, kwargs = parse_args(argv)

    text = open_file(args.path)

    print(f"Processing file with {len(text)} lines...")

    result = ngram_frequency(text, *args[1:], **kwargs)

    save_file(result, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
