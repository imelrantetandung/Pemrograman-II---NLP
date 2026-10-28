"""Jalankan UI, atau periksa index tanpa UI melalui opsi baris perintah."""

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Pencarian Artikel ASAG dengan TF-IDF manual.")
    parser.add_argument("--corpus", type=Path, default=PROJECT_DIR / "corpus",
                        help="Folder artikel TXT/PDF dan metadata.csv.")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "output",
                        help="Folder hasil index CSV/JSON untuk mode baris perintah.")
    parser.add_argument("--build-index", action="store_true", help="Bangun index tanpa membuka UI.")
    parser.add_argument("--query", help="Bangun index lalu jalankan kueri tanpa UI.")
    args = parser.parse_args()

    try:
        from corpus import load_corpus
        from preprocessing import StopwordsUnavailableError, TextPreprocessor
        from search_engine import SearchIndex
    except ImportError as error:
        print("Dependensi belum tersedia: {}. Jalankan: python -m pip install -r requirements.txt"
              .format(error), file=sys.stderr)
        return 1

    if args.build_index or args.query is not None:
        try:
            preprocessor = TextPreprocessor()
            loaded = load_corpus(args.corpus, preprocessor)
            for warning in loaded.warnings:
                print("Peringatan: " + warning)
            index = SearchIndex(loaded.documents)
            if not index.documents:
                print("Index tidak dibuat: belum ada dokumen valid. Isi folder {}."
                      .format(args.corpus.resolve()))
                return 1
            index.export(args.output)
            print("Index: {} dokumen valid, {} term. Hasil: {}"
                  .format(len(index.documents), len(index.vocabulary), args.output.resolve()))
            if args.query is not None:
                response = index.search(args.query, preprocessor)
                print(response.message)
                for result in response.results:
                    print("{}\t{}\t{:.4f}\t{}".format(result.rank, result.document.code,
                                                     result.score, result.document.title))
            return 0
        except (StopwordsUnavailableError, OSError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1

    try:
        import tkinter as tk
        from ui import SearchApp
    except ImportError as error:
        print("Tkinter belum tersedia pada instalasi Python ini: {}. "
              "Gunakan Python yang menyertakan Tkinter; jangan pip install tkinter."
              .format(error), file=sys.stderr)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print("UI tidak dapat dibuka: {}. Pastikan tersedia sesi desktop dan Python dengan "
              "Tkinter. Untuk memeriksa logika: python -m unittest discover -s tests -v"
              .format(error), file=sys.stderr)
        return 1
    app = SearchApp(root, PROJECT_DIR, corpus_dir=args.corpus, output_dir=args.output)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
