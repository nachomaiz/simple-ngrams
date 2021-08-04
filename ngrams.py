from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import sys

def ngram_frequency(text: list, n_min: int = 2, n_max:int = 5) -> pd.DataFrame:
    n_grams = []
    stop_words = stopwords.words('english')
    if n_min < 2:
        raise ValueError('n-grams must be at least 2 words long! n_min must be equal or higher than 2.')
    for i in range(n_min, n_max + 1):
        for line in text:
            token = [word.casefold() for word in word_tokenize(line) if word not in stop_words]
            ngram = list(ngrams(token, i))
            n_grams.extend(ngram)

    counts = Counter(x for x in n_grams)
    res = pd.DataFrame(counts.most_common(), columns=['n-gram', 'freq'])
    res['word_count'] = res['n-gram'].apply(lambda x: len(x))
    return res

if __name__ == "__main__":
    file = str(sys.argv[1])
    n_min = int(sys.argv[2])
    n_max = int(sys.argv[3])

    with open(file, encoding='utf8') as f:
        text = f.readlines()

    print(f"Processing file with {len(text)} lines.")
    
    result = ngram_frequency(text, n_min, n_max)

    print(f"Found {result.shape[0]} n-grams with sizes {n_min} to {n_max}. Saving to Excel...")

    result.to_excel('result.xlsx')
    print("Done.")