"""Uji alur widget opsional: python -m tests.ui_smoke (memerlukan Tk desktop)."""

from pathlib import Path
import tempfile
import time
import tkinter as tk

from ui import SearchApp


def wait_for_build(root, app):
    deadline = time.monotonic() + 20
    while app._busy and time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)
    assert not app._busy, "Worker index tidak selesai dalam 20 detik."
    root.update()


def main():
    project_dir = Path(__file__).resolve().parents[1]
    root = tk.Tk()
    # Uji widget secara otomatis; ini bukan inspeksi tampilan secara visual.
    root.withdraw()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            app = SearchApp(root, project_dir, corpus_dir=project_dir / "tests" / "data",
                            output_dir=temporary / "output")
            assert root.title() == "Pencarian Artikel ASAG"
            app.search_button.invoke()
            assert "Index belum dibuat" in app.status_var.get()

            app.build_button.invoke()
            wait_for_build(root, app)
            assert app.index is not None, app.notes_text.get("1.0", "end")
            assert len(app.index.documents) == 3
            assert len(app.index_tree.get_children()) == len(app.index.vocabulary)
            assert (temporary / "output" / "tfidf.csv").is_file()

            app.query_var.set("grading")
            app.search_button.invoke()
            assert app.results_tree.get_children() == ("D02", "D01", "D03")
            assert app.results_tree.item("D02", "values")[3] == "0.7071"
            assert app.selected_document.code == "D02"
            assert "common grading answer" in app.preview_text.get("1.0", "end")
            assert app.open_button.instate(["!disabled"])

            app.index_mode_var.set("TF")
            app._render_index()
            rows = [app.index_tree.item(row, "values") for row in app.index_tree.get_children()]
            grade = next(row for row in rows if row[0] == "grade")
            assert grade[1] == "2" and grade[3:] == ("2", "1", "0")

            for query in ("common", "unknown"):
                app.query_var.set(query)
                app.search_button.invoke()
                assert "Tidak ditemukan kecocokan berbobot" in app.status_var.get()
                assert all(app.results_tree.item(row, "values")[3] == "0.0000"
                           for row in app.results_tree.get_children())

            empty = temporary / "empty"
            empty.mkdir()
            app.folder_var.set(str(empty))
            app.build_button.invoke()
            wait_for_build(root, app)
            assert app.index is None
            assert "0 dari 0" in app.count_var.get()
            assert not app.results_tree.get_children()
            assert "belum berisi artikel" in app.notes_text.get("1.0", "end")
            app._closed = True
        print("Uji alur widget Tk lulus: build, pencarian, pratinjau, TF, vektor nol, korpus kosong.")
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
