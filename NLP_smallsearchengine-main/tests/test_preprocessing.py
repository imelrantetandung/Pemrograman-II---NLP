import unittest
from unittest.mock import patch

from preprocessing import StopwordsUnavailableError, TextPreprocessor


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor(stop_words={"the", "and", "is"})

    def test_case_stopwords_stemming_and_whitespace(self):
        self.assertEqual(
            self.preprocessor.process(" THE\n grading\tAND answers  is RUNNING "),
            ["grade", "answer", "run"],
        )

    def test_punctuation_numbers_and_hyphens_are_separators(self):
        self.assertEqual(
            self.preprocessor.process("short-answer score42model test/model"),
            ["short", "answer", "score", "model", "test", "model"],
        )

    def test_repeated_tokens_are_preserved_for_raw_tf(self):
        self.assertEqual(self.preprocessor.process("GRADE grade grades"), ["grade"] * 3)

    def test_empty_and_stopword_only_input(self):
        self.assertEqual(self.preprocessor.process(""), [])
        self.assertEqual(self.preprocessor.process("123 !?"), [])
        self.assertEqual(self.preprocessor.process("the and is"), [])

    def test_explicit_empty_stopword_set_does_not_load_resource(self):
        self.assertEqual(TextPreprocessor(stop_words=[]).process("the"), ["the"])

    def test_missing_stopwords_has_installation_instruction(self):
        # Replace the lazy NLTK corpus object so this test works without downloads.
        with patch("preprocessing.stopwords") as resource:
            resource.words.side_effect = LookupError("not installed")
            with self.assertRaisesRegex(StopwordsUnavailableError, "nltk.downloader stopwords"):
                TextPreprocessor()


if __name__ == "__main__":
    unittest.main()
