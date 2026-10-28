"""Buat ulang ZIP tugas tanpa environment, cache, rahasia, atau ZIP lain."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVE_NAME = "tugas_nlp_pencarian_asag.zip"


def collect_files(project_dir):
    project_dir = Path(project_dir)
    # Daftar berkas aplikasi yang diketahui menghindari berkas pribadi ikut ZIP.
    for name in ("main.py", "ui.py", "corpus.py", "preprocessing.py", "search_engine.py",
                 "build_zip.py", "requirements.txt", "README.txt", ".gitignore"):
        path = project_dir / name
        if path.is_file() and not path.is_symlink():
            yield path
    policies = {
        "corpus": lambda path: path.suffix.lower() in {".txt", ".pdf"} or path.name == "metadata.csv",
        "tests": lambda path: path.suffix.lower() in {".py", ".txt"},
        "output": lambda path: path.name in {"tf.csv", "tfidf.csv", "documents.csv", "index_metadata.json"},
    }
    for folder, allowed in policies.items():
        base = project_dir / folder
        if not base.is_dir() or base.is_symlink():
            continue
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(base)
            if any(part.startswith(".") or part in {"__pycache__", "venv"} for part in relative.parts):
                continue
            if any(parent.is_symlink() for parent in (path,) + tuple(path.parents) if parent != project_dir):
                continue
            if path.is_file() and allowed(path):
                yield path


def build_archive(project_dir=PROJECT_DIR):
    project_dir = Path(project_dir)
    destination = project_dir / ARCHIVE_NAME
    prefix = "tugas_nlp_pencarian_asag"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for directory in ("corpus", "output"):
            archive.writestr(prefix + "/" + directory + "/", "")
        for path in collect_files(project_dir):
            archive.write(path, prefix + "/" + path.relative_to(project_dir).as_posix())
    return destination


if __name__ == "__main__":
    print("Arsip dibuat: " + str(build_archive()))
    print("Periksa kelengkapan artikel, metadata sumber, dan identitas anggota di README.txt "
          "sebelum mengumpulkan.")
