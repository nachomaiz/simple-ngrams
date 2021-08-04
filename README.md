# Find N-grams in text file

N-grams are collections of n words that appear together within a text.

With this script you can split the text into n-grams of specified sizes and get their frequencies in the whole text.

To run this script, paste the txt file into this folder and open a terminal with an activated python environment.

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

To run this script, navigate to this folder in the terminal and run the following:
```
python ngrams.py text-to-analyze.txt min-size max-size
```

This would look like this:
```
python ngrams.py testtweets.txt 2 5
```

And it will save an Excel file with the results:
|       | n-gram        | freq | word_count |
|-------|---------------|------|------------|
| 0     | ('a','b')     | 5    | 2          |
| 1     | ('c','d')     | 4    | 2          |
| 2     | ('e','f','g') | 3    | 1          |