import unittest
from pandas.testing import assert_frame_equal

import pandas as pd
import ast

from ngrams import ngram_frequency

test_text = [
    "This is a test. For n-gram frequencies.\n",
    "This is a test that should support multiple lines.",
]


class TestNgrams(unittest.TestCase):
    def setUp(self) -> None:
        self.target_data = pd.read_csv(
            "test/test_result.csv",
            index_col=0,
            converters={"n-gram": ast.literal_eval}
        )

    def tearDown(self) -> None:
        self.target_data = None

    def test_tuple_ngrams(self) -> None:
        """
        Test standard command with no optional args. Should return df with tuples.
        """
        result = ngram_frequency(test_text, 2, 3)

        assert_frame_equal(result, self.target_data)

    def test_str_ngrams(self) -> None:
        """
        Test standard command with string arg. Should return df with strings.
        """
        result = ngram_frequency(test_text, 2, 3, clean_text=True)
        self.target_data["n-gram"] = self.target_data["n-gram"].agg(" ".join)

        assert_frame_equal(result, self.target_data)


class TestNgramsStopword(unittest.TestCase):
    def setUp(self) -> None:
        self.target_data = pd.read_csv(
            "test/test_stopword_result.csv",
            index_col=0,
            converters={"n-gram": ast.literal_eval},
        )

    def tearDown(self) -> None:
        self.target_data = None

    def test_tuple_ngrams_stopword(self) -> None:
        """
        Test standard command with stopwords arg. Should return df with tuples.
        """
        result = ngram_frequency(test_text, 2, 3, clean_stopwords=True)

        assert_frame_equal(result, self.target_data)

    def test_str_ngrams_stopword(self) -> None:
        """
        Test standard command with stopwords and string args. Should return df with strings.
        """
        result = ngram_frequency(test_text, 2, 3, clean_stopwords=True, clean_text=True)
        self.target_data["n-gram"] = self.target_data["n-gram"].agg(" ".join)

        assert_frame_equal(result, self.target_data)


if __name__ == "__main__":
    unittest.main()
