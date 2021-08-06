from typing import List, Dict, Tuple, NamedTuple
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter, namedtuple
import pandas as pd
import sys

import cli_msg


def parse_args(argv: List[str]) -> Tuple[NamedTuple, Dict[str, str]]:
    args = [i for i in argv if not i.startswith("-")]
    kwargs = [i for i in argv if i.startswith("-")]

    if "-h" in kwargs:
        print(cli_msg.help)
        sys.exit()
    elif len(args) != 4 or len(kwargs) > 2:
        print(cli_msg.usage)
        sys.exit()
    
    Args = namedtuple('Args', ['text', 'n_min', 'n_max'])
    args = Args(str(args[1]), int(args[2]), int(args[3]))
    kwargs = {
        "clean_stopwords" : True if "-s" in kwargs else False,
        "tuples" : True if "-t" in kwargs else False
    }

    return args, kwargs

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
    counts = Counter(x for x in n_grams)  # stores n-grams and calculates their frequency
    res = pd.DataFrame(counts.most_common(), columns=["n-gram", "freq"])
    res["word_count"] = res["n-gram"].apply(len)
    if not tuples:
        res["n-gram"] = res["n-gram"].str.join(" ")
    return res


def main(argv: List[str]):
    args, kwargs = parse_args(argv)

    print(f"Reading {args.text}...")

    with open(args.text, encoding="utf8") as f:
        text = f.readlines()  # split text and load into a list of lines

    print(f"Processing file with {len(text)} lines...")

    result = ngram_frequency(text, *args[1:], **kwargs)

    print(
        f"Found {result.shape[0]} n-grams with sizes {args.n_min} to {args.n_max}. Saving to Excel..."
    )

    result.to_excel("result.xlsx", encoding="utf8")
    print("Done")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
