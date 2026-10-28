import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from corpus import load_corpus
from preprocessing import TextPreprocessor


def write_text_pdf(path, page_texts):
    """PDF uji sementara; bukan artikel atau bagian corpus pengumpulan."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(("BT /F1 12 Tf 72 720 Td (%s) Tj ET" % text).encode("ascii"))
        page[NameObject("/Contents")] = stream
    with path.open("wb") as handle:
        writer.write(handle)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.preprocessor = TextPreprocessor(stop_words={"the", "and"})

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_and_empty_folders_are_explained(self):
        for folder in (self.folder / "missing", self.folder):
            result = load_corpus(folder, self.preprocessor)
            self.assertEqual(result.documents, [])
            self.assertEqual(result.candidate_count, 0)
            self.assertTrue(result.warnings)

    def test_sorted_codes_metadata_and_original_text(self):
        text = "Grading answers.\nORIGINAL formatting!"
        (self.folder / "b.txt").write_text("Models", encoding="utf-8")
        (self.folder / "A.TXT").write_text(text, encoding="utf-8")
        (self.folder / "metadata.csv").write_text(
            'filename,title,source_url\nA.TXT,"Judul, dari metadata",https://example.org/source\n',
            encoding="utf-8-sig",
        )
        (self.folder / "ignore.md").write_text("ignored", encoding="utf-8")
        result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(result.candidate_count, 2)
        first, second = result.documents
        self.assertEqual((first.code, first.filename), ("D01", "A.TXT"))
        self.assertEqual(first.title, "Judul, dari metadata")
        self.assertEqual(first.source_url, "https://example.org/source")
        self.assertEqual(first.text, text)
        self.assertEqual(second.code, "D02")
        self.assertEqual(second.title, "b.txt")
        self.assertEqual(second.source_url, "")

    def test_invalid_empty_and_no_token_documents_are_excluded(self):
        (self.folder / "a.txt").write_bytes(b"\xff\xfe")
        (self.folder / "b.txt").write_text("  \n", encoding="utf-8")
        (self.folder / "c.txt").write_text("the and 123 !", encoding="utf-8")
        (self.folder / "d.txt").write_text("grading", encoding="utf-8")
        result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(result.candidate_count, 4)
        self.assertEqual([doc.code for doc in result.documents], ["D04"])
        self.assertEqual(len(result.warnings), 3)
        for name, warning in zip(("a.txt", "b.txt", "c.txt"), result.warnings):
            self.assertIn(name, warning)

    def test_pdf_scan_or_empty_pdf_explains_ocr(self):
        (self.folder / "scan.pdf").write_bytes(b"test placeholder")
        with patch("corpus._extract_pdf", return_value="\n \t"):
            result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(result.documents, [])
        self.assertIn("scan.pdf", result.warnings[0])
        self.assertIn("OCR", result.warnings[0])

    def test_pdf_text_is_one_document_even_with_multiple_pages(self):
        (self.folder / "article.pdf").write_bytes(b"test placeholder")
        with patch("corpus._extract_pdf", return_value="page one\n\npage two"):
            result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.documents[0].text, "page one\n\npage two")

    def test_failed_pdf_mentions_filename_and_continues(self):
        (self.folder / "broken.pdf").write_bytes(b"test placeholder")
        (self.folder / "valid.txt").write_text("grading", encoding="utf-8")
        with patch("corpus._extract_pdf", side_effect=ValueError("invalid PDF")):
            result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(len(result.documents), 1)
        self.assertIn("broken.pdf", result.warnings[0])
        self.assertIn("invalid PDF", result.warnings[0])

    def test_actual_multipage_pdf_extracts_text_as_one_document(self):
        write_text_pdf(self.folder / "article.pdf", ["Grading answers", "Models scoring"])
        result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.warnings, [])
        document = result.documents[0]
        self.assertEqual(document.code, "D01")
        self.assertIn("Grading answers", document.text)
        self.assertIn("Models scoring", document.text)
        self.assertEqual(document.tokens, ["grade", "answer", "model", "score"])

    def test_actual_blank_pdf_is_excluded_with_ocr_instruction(self):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with (self.folder / "blank.pdf").open("wb") as handle:
            writer.write(handle)
        result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(result.documents, [])
        self.assertEqual(result.candidate_count, 1)
        self.assertIn("blank.pdf", result.warnings[0])
        self.assertIn("OCR", result.warnings[0])

    def test_actual_corrupt_pdf_is_excluded_and_other_files_still_load(self):
        (self.folder / "broken.pdf").write_bytes(b"%PDF-1.7\ncorrupt PDF\n%%EOF\n")
        (self.folder / "valid.txt").write_text("grading answers", encoding="utf-8")
        result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual([doc.filename for doc in result.documents], ["valid.txt"])
        self.assertEqual(result.documents[0].code, "D02")
        self.assertIn("broken.pdf", result.warnings[0])
        self.assertIn("gagal dibaca", result.warnings[0])

    def test_bad_metadata_does_not_discard_article(self):
        (self.folder / "article.txt").write_text("grading", encoding="utf-8")
        (self.folder / "metadata.csv").write_text("wrong,headers\na,b\n", encoding="utf-8")
        result = load_corpus(self.folder, self.preprocessor)
        self.assertEqual(result.documents[0].title, "article.txt")
        self.assertEqual(result.documents[0].source_url, "")
        self.assertIn("metadata.csv", result.warnings[0])

    def test_subfolders_are_not_read_as_extra_documents(self):
        nested = self.folder / "nested"
        nested.mkdir()
        (nested / "article.txt").write_text("grading", encoding="utf-8")
        self.assertEqual(load_corpus(self.folder, self.preprocessor).candidate_count, 0)


if __name__ == "__main__":
    unittest.main()
