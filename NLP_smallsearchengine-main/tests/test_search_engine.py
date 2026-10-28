"""Contoh hitung kecil; fixture ini bukan artikel tugas di folder corpus."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from corpus import load_corpus
from preprocessing import TextPreprocessor
from search_engine import SearchIndex, cosine_similarity


class SearchEngineTests(unittest.TestCase):
    def setUp(self):
        # Uji matematis deterministik tanpa resource unduhan; uji integrasi
        # program secara terpisah memakai stopword NLTK bahasa Inggris asli.
        self.preprocessor = TextPreprocessor(stop_words={"the"})
        corpus = load_corpus(Path(__file__).parent / "data", self.preprocessor)
        self.index = SearchIndex(corpus.documents)

    def test_raw_tf_preserves_repetition(self):
        position = self.index.term_positions["grade"]
        self.assertEqual(self.index.tf[:, position].tolist(), [2, 1, 0])

    def test_df_counts_documents_once_not_occurrences(self):
        position = self.index.term_positions["grade"]
        self.assertEqual(int(self.index.df[position]), 2)
        self.assertEqual(int(self.index.tf[:, position].sum()), 3)

    def test_log10_idf_without_smoothing(self):
        expected = {"answer": math.log10(3 / 2), "common": 0.0,
                    "grade": math.log10(3 / 2), "rubric": math.log10(3)}
        self.assertEqual(self.index.vocabulary, sorted(expected))
        for term, value in expected.items():
            self.assertAlmostEqual(self.index.idf[self.index.term_positions[term]], value)

    def test_tfidf_is_raw_tf_times_idf(self):
        pos = self.index.term_positions["grade"]
        self.assertAlmostEqual(self.index.tfidf[0, pos], 2 * math.log10(3 / 2))

    def test_query_uses_same_preprocessing_vocabulary_and_raw_tf(self):
        terms = self.preprocessor.process("GRADING grades unknown")
        self.assertEqual(terms, ["grade", "grade", "unknown"])
        vector = self.index.query_vector(terms)
        self.assertEqual(vector.shape, (len(self.index.vocabulary),))
        self.assertAlmostEqual(vector[self.index.term_positions["grade"]], 2 * math.log10(3 / 2))
        self.assertEqual(np.count_nonzero(vector), 1)

    def test_query_does_not_change_corpus_or_idf(self):
        idf_before = self.index.idf.copy()
        df_before = self.index.df.copy()
        self.index.search("unknown grading grading", self.preprocessor)
        np.testing.assert_array_equal(self.index.idf, idf_before)
        np.testing.assert_array_equal(self.index.df, df_before)
        self.assertEqual(len(self.index.documents), 3)

    def test_identical_nonzero_vectors_have_cosine_one(self):
        self.assertAlmostEqual(cosine_similarity([2, 0, 3], [2, 0, 3]), 1.0)
        response = self.index.search("common grading grading rubric", self.preprocessor)
        self.assertEqual(response.results[0].document.code, "D01")
        self.assertAlmostEqual(response.results[0].score, 1.0)

    def test_document_norm_uses_all_document_features(self):
        response = self.index.search("grading", self.preprocessor)
        scores = {result.document.code: result.score for result in response.results}
        grade = math.log10(3 / 2)
        rubric = math.log10(3)
        expected = (2 * grade) / math.sqrt((2 * grade) ** 2 + rubric ** 2)
        self.assertAlmostEqual(scores["D01"], expected)
        self.assertAlmostEqual(scores["D02"], 1 / math.sqrt(2))
        self.assertLess(scores["D01"], 1.0)

    def test_ranking_descending_with_consistent_codes_for_ties(self):
        response = self.index.search("grading", self.preprocessor)
        self.assertEqual([r.document.code for r in response.results], ["D02", "D01", "D03"])
        scores = [r.score for r in response.results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        tied = self.index.search("common", self.preprocessor)
        self.assertEqual([r.document.code for r in tied.results], ["D01", "D02", "D03"])

    def test_zero_vectors_and_unknown_query_are_finite(self):
        for a, b in (([0, 0], [1, 2]), ([1, 2], [0, 0]), ([], [])):
            self.assertEqual(cosine_similarity(a, b), 0.0)
        for query, status in (("common", "no_weighted_match"), ("unknown", "unknown_terms")):
            response = self.index.search(query, self.preprocessor)
            self.assertEqual(response.status, status)
            self.assertTrue(all(r.score == 0.0 and math.isfinite(r.score) for r in response.results))

    def test_single_document_has_zero_idf_and_operational_score_zero(self):
        index = SearchIndex(self.index.documents[:1])
        self.assertTrue(np.all(index.idf == 0))
        response = index.search("grading rubric", self.preprocessor)
        self.assertEqual(response.status, "no_weighted_match")
        self.assertEqual(response.results[0].score, 0.0)

    def test_empty_query_stopword_query_and_unbuilt_index(self):
        self.assertEqual(self.index.search(" \n", self.preprocessor).status, "empty_query")
        self.assertEqual(self.index.search("the 123 !", self.preprocessor).status, "no_query_tokens")
        self.assertEqual(SearchIndex([]).search("grading", self.preprocessor).status, "index_missing")

    def test_export_transposes_to_terms_by_documents_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.index.export(temporary)
            self.assertEqual(len(paths), 4)
            with (Path(temporary) / "tf.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["term", "DF", "IDF", "D01", "D02", "D03"])
            grade_row = next(row for row in rows if row[0] == "grade")
            self.assertEqual(grade_row[1], "2")
            self.assertEqual(grade_row[3:], ["2", "1", "0"])
            with (Path(temporary) / "tfidf.csv").open(encoding="utf-8-sig", newline="") as handle:
                weights = list(csv.reader(handle))
            grade_row = next(row for row in weights if row[0] == "grade")
            self.assertAlmostEqual(float(grade_row[3]), 2 * math.log10(3 / 2))
            metadata = json.loads((Path(temporary) / "index_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["document_count"], 3)
            self.assertEqual(metadata["documents"][0]["original_text"], self.index.documents[0].text)


if __name__ == "__main__":
    unittest.main()
