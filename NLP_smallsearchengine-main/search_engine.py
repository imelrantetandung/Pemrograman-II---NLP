"""Index TF-IDF manual dan pencarian cosine tanpa smoothing."""

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np

from corpus import Document


@dataclass
class SearchResult:
    rank: int
    document: Document
    score: float


@dataclass
class SearchResponse:
    results: List[SearchResult]
    message: str
    status: str
    query_terms: List[str]
    unknown_terms: List[str]


def cosine_similarity(vector_a, vector_b) -> float:
    """Panjang dihitung dari SELURUH fitur; vektor nol diberi skor 0."""
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Kedua vektor harus satu dimensi dengan ukuran sama.")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    # Cegah nilai 1.0000000000000002 akibat pembulatan floating point.
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


class SearchIndex:
    def __init__(self, documents: Iterable[Document]):
        # Pembaca korpus sudah menyaring dokumen kosong; ini pengaman tambahan.
        self.documents = [doc for doc in documents if doc.tokens]
        self.vocabulary = sorted({term for doc in self.documents for term in doc.tokens})
        self.term_positions = {term: pos for pos, term in enumerate(self.vocabulary)}
        self.tf = np.zeros((len(self.documents), len(self.vocabulary)), dtype=np.int64)
        for row, document in enumerate(self.documents):
            for term, count in Counter(document.tokens).items():
                self.tf[row, self.term_positions[term]] = count
        self.df = np.count_nonzero(self.tf, axis=0)
        if self.documents:
            self.idf = np.log10(len(self.documents) / self.df)
        else:
            self.idf = np.zeros(len(self.vocabulary), dtype=float)
        self.tfidf = self.tf * self.idf
        self.document_norms = np.linalg.norm(self.tfidf, axis=1)

    def query_vector(self, query_terms: Iterable[str]) -> np.ndarray:
        """Proyeksi ke vocabulary korpus; kueri tidak mengubah TF/DF/IDF."""
        vector = np.zeros(len(self.vocabulary), dtype=float)
        for term, count in Counter(query_terms).items():
            position = self.term_positions.get(term)
            if position is not None:
                vector[position] = count * self.idf[position]
        return vector

    def search(self, query: str, preprocessor) -> SearchResponse:
        if not self.documents:
            return SearchResponse([], "Index belum dibuat atau tidak memiliki dokumen valid.",
                                  "index_missing", [], [])
        if not query.strip():
            return SearchResponse([], "Kueri kosong. Masukkan frasa pencarian.",
                                  "empty_query", [], [])
        terms = preprocessor.process(query)
        if not terms:
            return SearchResponse([], "Tidak ada token kueri setelah preprocessing "
                                  "(hanya stopword, angka, atau tanda baca).",
                                  "no_query_tokens", [], [])
        unknown = sorted(set(terms) - self.term_positions.keys())
        vector = self.query_vector(terms)
        scores = [cosine_similarity(vector, row) for row in self.tfidf]
        # Skor sama diurutkan berdasarkan kode dokumen agar deterministik.
        ordered = sorted(zip(self.documents, scores), key=lambda pair: (-pair[1], pair[0].code))
        results = [SearchResult(rank, document, score)
                   for rank, (document, score) in enumerate(ordered, start=1)]
        if len(unknown) == len(set(terms)):
            return SearchResponse(results, "Tidak ditemukan kecocokan berbobot: semua term "
                                  "kueri berada di luar vocabulary korpus. Skor operasional 0.",
                                  "unknown_terms", terms, unknown)
        if not any(score > 0 for score in scores):
            return SearchResponse(results, "Tidak ditemukan kecocokan berbobot. Vektor kueri "
                                  "atau dokumen bernorma nol diberi skor operasional 0 "
                                  "(misalnya term kueri memiliki IDF 0).",
                                  "no_weighted_match", terms, unknown)
        message = "{} dokumen diurutkan menurut cosine similarity.".format(len(results))
        if unknown:
            message += " Term di luar vocabulary diabaikan: " + ", ".join(unknown) + "."
        return SearchResponse(results, message, "ok", terms, unknown)

    def export(self, output_dir) -> List[Path]:
        """CSV baris=term, kolom=dokumen; simpan teks asli dan token untuk audit."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        # Matriks perhitungan dokumen x term ditranspose untuk tabel tugas.
        for name, matrix in (("tf.csv", self.tf.T), ("tfidf.csv", self.tfidf.T)):
            path = output_dir / name
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["term", "DF", "IDF"] + [doc.code for doc in self.documents])
                for position, term in enumerate(self.vocabulary):
                    writer.writerow([term, int(self.df[position]), float(self.idf[position])]
                                    + matrix[position].tolist())
            paths.append(path)
        document_path = output_dir / "documents.csv"
        with document_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["code", "title", "filename", "source_url", "token_count"])
            for doc in self.documents:
                writer.writerow([doc.code, doc.title, doc.filename, doc.source_url, len(doc.tokens)])
        paths.append(document_path)
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "document_count": len(self.documents),
            "term_count": len(self.vocabulary),
            "matrix_orientation": "CSV: term x document; internal: document x term",
            "tf": "raw count",
            "idf": "log10(N / DF), without smoothing",
            "zero_vector_policy": "operational cosine score 0",
            "documents": [{"code": doc.code, "title": doc.title, "filename": doc.filename,
                           "source_url": doc.source_url, "original_text": doc.text,
                           "tokens": doc.tokens} for doc in self.documents],
        }
        metadata_path = output_dir / "index_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        paths.append(metadata_path)
        return paths
