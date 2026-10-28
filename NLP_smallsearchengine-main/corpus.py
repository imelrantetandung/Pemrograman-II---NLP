"""Pembacaan satu artikel per file TXT/PDF dan metadata sumbernya."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from preprocessing import TextPreprocessor


@dataclass
class Document:
    code: str
    title: str
    filename: str
    path: Path
    source_url: str
    text: str
    tokens: List[str]


@dataclass
class CorpusLoadResult:
    documents: List[Document]
    warnings: List[str]
    candidate_count: int


def _read_metadata(path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Metadata opsional; judul/URL yang belum diisi tetap kosong."""
    records = {}  # type: Dict[str, Dict[str, str]]
    warnings = []  # type: List[str]
    if not path.exists():
        return records, warnings
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            required = {"filename", "title", "source_url"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("kolom wajib: filename,title,source_url")
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("jumlah kolom tidak sesuai pada baris %d" % line_number)
                filename = row["filename"].strip()
                if not filename:
                    continue
                if filename in records:
                    warnings.append(
                        "%s: metadata ganda untuk %s; baris terakhir dipakai."
                        % (path.name, filename)
                    )
                records[filename] = {
                    "title": row["title"].strip(),
                    "source_url": row["source_url"].strip(),
                }
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        warnings.append(
            "%s: metadata tidak dapat dibaca (%s). "
            "Nama file digunakan sebagai judul sementara; URL perlu diisi."
            % (path.name, error)
        )
        records = {}
    return records, warnings


def _extract_pdf(path: Path) -> str:
    # Import hanya diperlukan ketika ada PDF; kegagalan dilaporkan per file.
    from pypdf import PdfReader

    with path.open("rb") as handle:
        reader = PdfReader(handle)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_corpus(
    folder: Union[str, Path],
    preprocessor: TextPreprocessor,
    metadata_path: Optional[Union[str, Path]] = None,
) -> CorpusLoadResult:
    """Baca semua TXT/PDF di folder, tanpa batas jumlah atau subfolder.

    Kode mengikuti urutan semua kandidat file. File gagal dilewati sehingga
    kode dapat berlubang; file lain tidak berganti kode hanya karena satu
    kandidat gagal dibaca. Teks/token kosong tidak masuk perhitungan N.
    """
    folder = Path(folder).expanduser().resolve()
    documents = []  # type: List[Document]
    warnings = []  # type: List[str]
    if not folder.is_dir():
        warnings.append("Folder corpus belum ada atau bukan folder: %s" % folder)
        return CorpusLoadResult(documents, warnings, 0)

    try:
        candidates = sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except OSError as error:
        warnings.append("Folder corpus tidak dapat dibaca: %s (%s)" % (folder, error))
        return CorpusLoadResult(documents, warnings, 0)

    metadata, metadata_warnings = _read_metadata(
        Path(metadata_path) if metadata_path is not None else folder / "metadata.csv"
    )
    warnings.extend(metadata_warnings)
    if not candidates:
        warnings.append(
            "Folder corpus belum berisi artikel .txt atau .pdf. "
            "Masukkan artikel asli berbahasa Inggris tentang ASAG ke %s." % folder
        )

    for position, path in enumerate(candidates, start=1):
        try:
            text = _extract_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
        except Exception as error:
            warnings.append("%s: dokumen gagal dibaca (%s)." % (path.name, error))
            continue

        if not text.strip():
            if path.suffix.lower() == ".pdf":
                warnings.append(
                    "%s: PDF kosong atau berupa scan tanpa teks yang dapat diekstrak. "
                    "Dokumen perlu OCR atau versi PDF berbasis teks/TXT; dilewati."
                    % path.name
                )
            else:
                warnings.append("%s: dokumen kosong; dilewati." % path.name)
            continue

        tokens = preprocessor.process(text)
        if not tokens:
            warnings.append(
                "%s: tidak memiliki token setelah preprocessing; dilewati." % path.name
            )
            continue

        entry = metadata.get(path.name, {})
        documents.append(
            Document(
                code="D%02d" % position,
                title=entry.get("title") or path.name,
                filename=path.name,
                path=path,
                source_url=entry.get("source_url", ""),
                text=text,
                tokens=tokens,
            )
        )

    return CorpusLoadResult(documents, warnings, len(candidates))
