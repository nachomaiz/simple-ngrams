import io
import unittest
import unittest.mock
from pandas.testing import assert_frame_equal

import pandas as pd
import ast

import ngrams
import cli_msg

test_text = [
    "This is a test. For n-gram frequencies.\n",
    "This is a test that should support multiple lines.",
]


class ScriptTests(unittest.TestCase):
    
    # asserts print to console
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assertStdout(self, func, attrs, expected_output, mock_stdout):
        func(attrs)
        self.assertEqual(mock_stdout.getvalue(), expected_output)
    
    def test_parse_args(self) -> None:
        args, kwargs = ngrams.parse_args('ngrams.py text.txt 1 2 -s -t'.split())
        self.assertEqual(args.text, 'text.txt')
        self.assertEqual(args.n_min, 1)
        self.assertEqual(args.n_max, 2)
        self.assertTrue(kwargs['clean_stopwords'])
        self.assertTrue(kwargs['tuples'])

    def test_help(self) -> None:
        with self.assertRaises(SystemExit):
            self.assertStdout(ngrams.parse_args, 'ngrams.py -h'.split(), cli_msg.help)

    def test_usage(self) -> None:
        with self.assertRaises(SystemExit):
            self.assertStdout(ngrams.parse_args, ['ngrams.py'], cli_msg.usage)
            
    def test_main(self) -> None:
        with self.assertRaises(SystemExit):
            ngrams.main(['ngrams.py'])

class NgramCommonTests(object):
    def setUp(self) -> None:
        self.target_data = pd.read_csv(
            self.path,
            index_col=0,
            converters={"n-gram": ast.literal_eval}
        )

    def tearDown(self) -> None:
        self.args = None
        self.path = None
        self.target_data = None

    def test_ngrams(self) -> None:
        result = ngrams.ngram_frequency(*self.args[:-1])
        self.target_data["n-gram"] = self.target_data["n-gram"].agg(" ".join)
        assert_frame_equal(result, self.target_data)

    def test_ngrams_tuples(self) -> None:
        result = ngrams.ngram_frequency(*self.args)
        assert_frame_equal(result, self.target_data)

class TestRaw(NgramCommonTests, unittest.TestCase):
    """Test output with stopwords"""
    def setUp(self) -> None:
        self.path = "test/test_result.csv"
        self.args = [test_text, 2, 3, False, True]
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

class TestsStopwords(NgramCommonTests, unittest.TestCase):
    """Test output without stopwords"""
    def setUp(self) -> None:
        self.path = "test/test_stopword_result.csv"
        self.args = [test_text, 2, 3, True, True]
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

if __name__ == "__main__":
    unittest.main()
