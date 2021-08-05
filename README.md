# Find N-grams in text file

N-grams are collections of n number of words that appear together within a text.

With this script you can split the text into n-grams of specified sizes and get their frequencies in the whole text.

The script does not mix n-grams across lines of text.

Make sure you have installed the two required packages:
```
pandas==1.3.1
nltk==3.6.2
```

And make sure you have downloaded the required nltk corpora:
```
stopwords
punkt
```

To run this script, navigate to this folder in the terminal within an activated python environment and run the following:
```
python ngrams.py text-to-analyze.txt min-size max-size
```

Optional arguments:
- `-s` : clean stopwords (is, and, the, a, to...)
- `-t` : return a clean string instead of tuples of words

This would look like this:
```
python ngrams.py testtweets.txt 2 5 -s -t
```

Finally, the script will save the result as an Excel file:
|       | n-gram        | freq | word_count |
|-------|---------------|------|------------|
| 0     | ('a','b')     | 5    | 2          |
| 1     | ('c','d')     | 4    | 2          |
| 2     | ('e','f','g') | 3    | 3          |

If you use `-t` the result will look like this:
|       | n-gram        | freq | word_count |
|-------|---------------|------|------------|
| 0     | a b           | 5    | 2          |

It's possible that Excel will throw an error when opening if `-t` is passed and any n-gram starts with `=`, `+` or `-`: Excel will treat it as a function.