from pandas.testing import assert_frame_equal
import unittest
import unittest.mock
import pandas as pd
import ast
import io
import sys

import ngrams

test_text = [
    "This is a test. For n-gram frequencies.\n",
    "This is a test that should support multiple lines.",
]


class _AssertStdoutContext:
    def __init__(self, testcase: unittest.TestCase, expected: str):
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


class PrintTestCase(unittest.TestCase):
    """Enables asserting print to cli."""
    
    def assertStdout(self, expected_output):
        return _AssertStdoutContext(self, expected_output)

    def assertPrints(self, *expected_output):
        expected_output = "\n".join(expected_output) + "\n"
        return _AssertStdoutContext(self, expected_output)


class ScriptTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args, kwargs = ngrams.parse_args("ngrams.py text.txt 1 2 -s -t".split())
        expected_output = ["text.txt", 1, 2, True, True]
        self.assertEqual(list(args) + list(kwargs.values()), expected_output)

    def test_help(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            ngrams.parse_args("ngrams.py -h".split())
        self.assertEqual(cm.exception.args[0], ngrams.__doc__)

    def test_usage(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            ngrams.parse_args(["ngrams.py"])
        self.assertEqual(cm.exception.args[0], ngrams.usage)

    def test_main(self) -> None:
        with self.assertRaises(SystemExit):
            ngrams.main(["ngrams.py"])

    def test_nonint_input(self) -> None:
        with self.assertRaises(ValueError):
            ngrams.parse_args("ngrams.py test.txt a b".split())


class FileTests(PrintTestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({"Numbers": [1, 2, 3, 4, 5]})
        self.args, _ = ngrams.parse_args("ngrams.py text.txt 1 2".split())
    
    @unittest.mock.mock_open(read_data="text")
    def test_open_file(self, mock_open):
        with self.assertStdout("Reading test.txt...\n"):
            result = ngrams.open_file("test.txt")
        mock_open.assert_called_with("test.txt", encoding="utf8")
        self.assertEqual(result, "text")

    @unittest.mock.patch("pandas.DataFrame.to_excel")
    def test_save_file(self, mock_to_excel):
        expected = f"Found {self.df.shape[0]} n-grams with sizes {self.args.n_min} to {self.args.n_max}. Saving to Excel..."
        with self.assertPrints(expected, "Done"):
            ngrams.save_file(self.df, self.args, "path")
        filepath = f"path\\ngrams_{self.args.path}_{self.args.n_min}-{self.args.n_max}.xlsx"
        mock_to_excel.assert_called_with(filepath, encoding="utf8")

    @unittest.mock.patch("ngrams.os.path")
    @unittest.mock.patch("ngrams.os")
    def test_make_dir(self, mock_os, mock_path):
        mock_path.isdir.return_value = False
        ngrams.make_dir("path")
        mock_os.mkdir.assert_called_with("path")


class NgramCommonTests(object):
    def setUp(self) -> None:
        self.target_data = pd.read_csv(
            self.target_path, index_col=0, converters={"n-gram": ast.literal_eval}
        )

    def tearDown(self) -> None:
        self.args = None
        self.target_path = None
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
        self.target_path = "test/test_result.csv"
        self.args = [test_text, 2, 3, False, True]
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()


class TestsStopwords(NgramCommonTests, unittest.TestCase):
    """Test output without stopwords"""

    def setUp(self) -> None:
        self.target_path = "test/test_stopword_result.csv"
        self.args = [test_text, 2, 3, True, True]
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
