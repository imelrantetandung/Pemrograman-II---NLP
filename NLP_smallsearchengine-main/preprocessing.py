"""Preprocessing yang sama untuk teks dokumen dan kueri berbahasa Inggris."""

import re
from typing import Iterable, List, Optional

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class StopwordsUnavailableError(RuntimeError):
    """Resource stopword NLTK perlu dipasang sekali oleh pengguna."""


class TextPreprocessor:
    def __init__(self, stop_words: Optional[Iterable[str]] = None):
        if stop_words is None:
            try:
                stop_words = stopwords.words("english")
            except LookupError as error:
                raise StopwordsUnavailableError(
                    "Resource stopword bahasa Inggris NLTK belum tersedia. "
                    "Jalankan sekali dengan interpreter/virtual environment yang sama: "
                    "python -m nltk.downloader stopwords. "
                    "Setelah selesai, jalankan kembali aplikasi."
                ) from error
        self.stop_words = set(stop_words)
        self.stemmer = PorterStemmer()

    def process(self, text: str) -> List[str]:
        """Lowercase -> whitespace -> pemisah nonhuruf -> stopword -> stemming.

        Hanya huruf ASCII a-z dipertahankan. Angka, tanda hubung, tanda baca,
        dan karakter lain menjadi pemisah, sehingga kata tidak menyatu.
        """
        normalized = " ".join(text.lower().split())
        normalized = re.sub(r"[^a-z]+", " ", normalized)
        words = normalized.split()
        return [
            self.stemmer.stem(word)
            for word in words
            if word not in self.stop_words
        ]
