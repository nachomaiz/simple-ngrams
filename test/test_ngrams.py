import io
import sys
import unittest
import unittest.mock
from pandas.testing import assert_frame_equal
from tempfile import mkdtemp

import pandas as pd
import ast

import ngrams
import cli_msg

test_text = [
    "This is a test. For n-gram frequencies.\n",
    "This is a test that should support multiple lines.",
]


class _AssertStdoutContext:
    def __init__(self, testcase, expected):
        self.testcase = testcase
        self.expected = expected
        self.captured = io.StringIO()

    def __enter__(self):
        sys.stdout = self.captured
        return self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = sys.__stdout__
        captured = self.captured.getvalue()
        self.testcase.assertEqual(captured, self.expected)


class ScriptTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args, kwargs = ngrams.parse_args("ngrams.py text.txt 1 2 -s -t".split())
        self.assertEqual(args.path, "text.txt")
        self.assertEqual(args.n_min, 1)
        self.assertEqual(args.n_max, 2)
        self.assertTrue(kwargs["clean_stopwords"])
        self.assertTrue(kwargs["tuples"])

    def test_help(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            ngrams.parse_args("ngrams.py -h".split())
        self.assertEqual(cm.exception.args[0], cli_msg.help)

    def test_usage(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            ngrams.parse_args(["ngrams.py"])
        self.assertEqual(cm.exception.args[0], cli_msg.usage)

    def test_main(self) -> None:
        with self.assertRaises(SystemExit):
            ngrams.main(["ngrams.py"])

    def test_nonint_input(self) -> None:
        with self.assertRaises(ValueError):
            ngrams.parse_args("ngrams.py test.txt a b".split())


class FileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "test/test_text.txt"
        self.test_dir = mkdtemp()
        self.test_df = pd.DataFrame({"Numbers": [1, 2, 3, 4, 5]})
        self.args, _ = ngrams.parse_args("ngrams.py text.txt 1 2".split())

    def assertStdout(self, expected_output):
        return _AssertStdoutContext(self, expected_output)

    # as a bonus, this syntactical sugar becomes possible:
    def assertPrints(self, *expected_output):
        expected_output = "\n".join(expected_output) + "\n"
        return _AssertStdoutContext(self, expected_output)

    def test_open_file(self):
        mock_open = unittest.mock.mock_open(read_data="".join(test_text))
        with unittest.mock.patch("builtins.open", mock_open):
            with self.assertStdout("Reading test/test_text.txt...\n"):
                result = ngrams.open_file(self.path)
            self.assertEqual(result, test_text)

    def test_write_file(self):
        with unittest.mock.patch.object(self.test_df, "to_excel") as mock_to_excel:
            with self.assertPrints(
                "Found 5 n-grams with sizes 1 to 2. Saving to Excel...", "Done"
            ):
                ngrams.save_file(self.test_df, self.args, self.test_dir)
            filepath = f"{self.test_dir}\\ngrams_{self.args.path}_{self.args.n_min}-{self.args.n_max}.xlsx"
            mock_to_excel.assert_called_with(filepath, encoding="utf8")


class NgramCommonTests(object):
    def setUp(self) -> None:
        self.target_data = pd.read_csv(
            self.path, index_col=0, converters={"n-gram": ast.literal_eval}
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
    unittest.main()  # pragma: no cover
