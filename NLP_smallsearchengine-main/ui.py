"""Antarmuka desktop sederhana; semua perubahan widget dilakukan di thread Tk."""

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk
from typing import Optional

from corpus import Document, load_corpus
from preprocessing import TextPreprocessor
from search_engine import SearchIndex


class SearchApp:
    def __init__(self, root: tk.Tk, project_dir: Path, corpus_dir=None, output_dir=None):
        self.root = root
        self.project_dir = Path(project_dir).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir else self.project_dir / "output"
        self.index: Optional[SearchIndex] = None
        self.preprocessor: Optional[TextPreprocessor] = None
        self.selected_document: Optional[Document] = None
        self._result_documents = {}
        self._messages = queue.Queue()
        self._busy = False
        self._closed = False
        self._index_render_id = 0

        self.folder_var = tk.StringVar(value=str(Path(corpus_dir).resolve() if corpus_dir else self.project_dir / "corpus"))
        self.count_var = tk.StringVar(value="Dokumen valid: belum dihitung")
        self.query_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value="Masukkan artikel ke folder corpus, lalu klik Bangun Index."
        )
        self.preview_info_var = tk.StringVar(value="Pilih hasil untuk membaca teks dokumen.")
        self.index_info_var = tk.StringVar(value="Index belum dibuat.")
        self.index_mode_var = tk.StringVar(value="TF-IDF")

        self._configure_style()
        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll_messages)

    def _configure_style(self):
        self.root.title("Pencarian Artikel ASAG")
        self.root.geometry("1120x790")
        self.root.minsize(900, 650)
        self.root.configure(background="#f4f6f8")
        available = set(font.families(self.root))
        family = "Segoe UI" if "Segoe UI" in available else font.nametofont("TkDefaultFont").actual("family")
        self.root.option_add("*Font", (family, 10))
        self._font_family = family
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f8")
        style.configure("TLabel", background="#f4f6f8", foreground="#253342")
        style.configure("Title.TLabel", font=(family, 19, "bold"), foreground="#173c67")
        style.configure("Section.TLabel", font=(family, 11, "bold"))
        style.configure("TButton", padding=(12, 7))
        style.configure("Accent.TButton", background="#173c67", foreground="white")
        style.map("Accent.TButton", background=[("active", "#234f7e"), ("disabled", "#a4afbc")])
        style.configure("Treeview", rowheight=27, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=(family, 10, "bold"))
        style.map("Treeview", background=[("selected", "#173c67")], foreground=[("selected", "white")])
        style.configure("TNotebook", background="#f4f6f8")
        style.configure("TNotebook.Tab", padding=(16, 8))

    def _build_widgets(self):
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)
        ttk.Label(container, text="Pencarian Artikel ASAG", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 14)
        )

        folder_box = ttk.Frame(container)
        folder_box.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        folder_box.columnconfigure(1, weight=1)
        ttk.Label(folder_box, text="Folder korpus").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(folder_box, textvariable=self.folder_var, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=(0, 8)
        )
        self.folder_button = ttk.Button(folder_box, text="Pilih Folder", command=self._choose_folder)
        self.folder_button.grid(row=0, column=2, padx=(0, 8))
        self.build_button = ttk.Button(
            folder_box, text="Bangun Index", style="Accent.TButton", command=self._build_index
        )
        self.build_button.grid(row=0, column=3)
        ttk.Label(folder_box, textvariable=self.count_var).grid(
            row=1, column=1, columnspan=3, sticky="w", pady=(7, 0)
        )

        self.progress = ttk.Progressbar(container, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=3, column=0, sticky="nsew")
        search_tab = ttk.Frame(self.notebook, padding=12)
        index_tab = ttk.Frame(self.notebook, padding=12)
        notes_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(search_tab, text="Pencarian")
        self.notebook.add(index_tab, text="Lihat Index")
        self.notebook.add(notes_tab, text="Catatan Proses")
        self._build_search_tab(search_tab)
        self._build_index_tab(index_tab)
        self._build_notes_tab(notes_tab)
        ttk.Label(container, textvariable=self.status_var, wraplength=1060).grid(
            row=4, column=0, sticky="ew", pady=(12, 0)
        )

    def _build_search_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        query_box = ttk.Frame(parent)
        query_box.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        query_box.columnconfigure(1, weight=1)
        ttk.Label(query_box, text="Kueri (bahasa Inggris)").grid(row=0, column=0, padx=(0, 10))
        self.query_entry = ttk.Entry(query_box, textvariable=self.query_var)
        self.query_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.query_entry.bind("<Return>", lambda event: self._search())
        self.search_button = ttk.Button(query_box, text="Cari", style="Accent.TButton", command=self._search)
        self.search_button.grid(row=0, column=2)

        panes = ttk.Panedwindow(parent, orient="vertical")
        panes.grid(row=1, column=0, sticky="nsew")
        results_frame = ttk.Frame(panes)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        panes.add(results_frame, weight=1)
        self.results_tree = ttk.Treeview(
            results_frame, columns=("rank", "code", "title", "score"), show="headings", height=7,
            selectmode="browse"
        )
        for key, title, width, anchor in (
            ("rank", "Peringkat", 80, "center"),
            ("code", "Kode", 75, "center"),
            ("title", "Judul / nama dokumen", 570, "w"),
            ("score", "Skor cosine", 120, "e"),
        ):
            self.results_tree.heading(key, text=title)
            self.results_tree.column(key, width=width, minwidth=65, anchor=anchor, stretch=key == "title")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self._add_tree_scrollbars(results_frame, self.results_tree)
        self.results_tree.bind("<<TreeviewSelect>>", self._show_selected_document)

        preview_frame = ttk.Frame(panes, padding=(0, 12, 0, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(2, weight=1)
        panes.add(preview_frame, weight=1)
        ttk.Label(preview_frame, text="Pratinjau teks asli", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.open_button = ttk.Button(preview_frame, text="Buka Dokumen", command=self._open_document, state="disabled")
        self.open_button.grid(row=0, column=1, sticky="e")
        ttk.Label(preview_frame, textvariable=self.preview_info_var, wraplength=920).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 8)
        )
        self.preview_text = tk.Text(
            preview_frame, height=9, wrap="word", state="disabled", background="white",
            foreground="#253342", relief="solid", borderwidth=1, padx=10, pady=8,
            font=(self._font_family, 10)
        )
        self.preview_text.grid(row=2, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_text.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
        self.preview_text.configure(yscrollcommand=scrollbar.set)

    def _build_index_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        controls = ttk.Frame(parent)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Tabel bobot:").pack(side="left", padx=(0, 12))
        for mode in ("TF", "TF-IDF"):
            ttk.Radiobutton(
                controls, text=mode, variable=self.index_mode_var, value=mode,
                command=self._render_index
            ).pack(side="left", padx=(0, 16))
        ttk.Label(parent, textvariable=self.index_info_var, wraplength=1000).grid(
            row=1, column=0, sticky="ew", pady=(0, 10)
        )
        table_frame = ttk.Frame(parent)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.index_tree = ttk.Treeview(table_frame, show="headings", height=15)
        self.index_tree.grid(row=0, column=0, sticky="nsew")
        self._add_tree_scrollbars(table_frame, self.index_tree)

    def _build_notes_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text="Hasil pembacaan dokumen dan pesan pemasangan", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        self.notes_text = tk.Text(
            parent, wrap="word", state="disabled", background="white", foreground="#253342",
            relief="solid", borderwidth=1, padx=10, pady=10, font=(self._font_family, 10)
        )
        self.notes_text.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.notes_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.notes_text.configure(yscrollcommand=scrollbar.set)
        self._set_text(self.notes_text, "Index belum dibuat. Folder korpus harus berisi artikel asli berbahasa Inggris tentang ASAG.")

    @staticmethod
    def _add_tree_scrollbars(parent, tree):
        vertical = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _choose_folder(self):
        chosen = filedialog.askdirectory(title="Pilih folder artikel ASAG", initialdir=self.folder_var.get())
        if chosen:
            self.folder_var.set(str(Path(chosen).resolve()))
            self._reset_index()
            self.status_var.set("Folder diubah. Klik Bangun Index untuk membaca seluruh dokumen.")

    def _reset_results(self):
        self.results_tree.delete(*self.results_tree.get_children())
        self._result_documents.clear()
        self.selected_document = None
        self.open_button.configure(state="disabled")
        self.preview_info_var.set("Pilih hasil untuk membaca teks dokumen.")
        self._set_text(self.preview_text, "")

    def _reset_index(self):
        self.index = None
        self.preprocessor = None
        self._reset_results()
        self._index_render_id += 1
        self.index_tree.delete(*self.index_tree.get_children())
        self.index_tree.configure(columns=())
        self.index_info_var.set("Index belum dibuat.")
        self.count_var.set("Dokumen valid: belum dihitung")

    def _set_busy(self, busy):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for widget in (self.folder_button, self.build_button, self.search_button, self.query_entry):
            widget.configure(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _build_index(self):
        if self._busy:
            return
        folder = Path(self.folder_var.get())
        output_dir = self.output_dir
        self._reset_index()
        self._set_busy(True)
        self.status_var.set("Membaca dokumen, melakukan preprocessing, dan menghitung index...")
        self.count_var.set("Dokumen valid: sedang dihitung")
        self._set_text(self.notes_text, "Sedang memproses folder: " + str(folder))
        threading.Thread(target=self._build_worker, args=(folder, output_dir), daemon=True).start()

    def _build_worker(self, folder, output_dir):
        """Worker hanya memakai data biasa dan queue, tanpa mengakses widget Tk."""
        try:
            preprocessor = TextPreprocessor()
            loaded = load_corpus(folder, preprocessor)
            index = SearchIndex(loaded.documents) if loaded.documents else None
            warnings = list(loaded.warnings)
            exported = []
            if index is not None:
                try:
                    exported = index.export(output_dir)
                except Exception as exc:
                    warnings.append("Index dapat dipakai, tetapi ekspor ke " + str(output_dir) + " gagal: " + str(exc))
            self._messages.put(("complete", index, preprocessor, loaded.candidate_count, warnings, exported))
        except Exception as exc:
            self._messages.put(("error", str(exc)))

    def _poll_messages(self):
        if self._closed:
            return
        try:
            while True:
                message = self._messages.get_nowait()
                self._set_busy(False)
                if message[0] == "error":
                    self.count_var.set("Dokumen valid: belum dihitung")
                    self.status_var.set("Index gagal dibuat. Baca penjelasan pada Catatan Proses.")
                    self._set_text(self.notes_text, message[1])
                    self.notebook.select(2)
                    continue
                _, self.index, self.preprocessor, candidates, warnings, exported = message
                valid_count = len(self.index.documents) if self.index is not None else 0
                self.count_var.set("Dokumen valid: {} dari {} file artikel".format(valid_count, candidates))
                lines = ["Folder: " + self.folder_var.get(), "File artikel: {} | Dokumen valid: {}".format(candidates, valid_count)]
                if exported:
                    lines.append("Hasil index tersimpan di: " + str(self.output_dir))
                    lines.extend("  " + Path(path).name for path in exported)
                if warnings:
                    lines.append("\nCatatan:")
                    lines.extend("- " + warning for warning in warnings)
                self._set_text(self.notes_text, "\n".join(lines))
                if self.index is None:
                    self.status_var.set("Belum ada dokumen valid. Periksa folder dan Catatan Proses, lalu bangun index kembali.")
                    self.notebook.select(2)
                else:
                    message = "Index selesai: {} dokumen, {} term.".format(valid_count, len(self.index.vocabulary))
                    if warnings:
                        message += " Ada {} catatan; lihat Catatan Proses.".format(len(warnings))
                    self.status_var.set(message)
                    self._render_index()
                    self.query_entry.focus_set()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _search(self):
        if self._busy:
            return
        if self.index is None or self.preprocessor is None:
            self.status_var.set("Index belum dibuat. Pilih folder dan klik Bangun Index terlebih dahulu.")
            return
        self._reset_results()
        try:
            response = self.index.search(self.query_var.get(), self.preprocessor)
        except Exception as exc:
            self.status_var.set("Pencarian gagal: " + str(exc))
            return
        for result in response.results:
            document = result.document
            self._result_documents[document.code] = document
            self.results_tree.insert(
                "", "end", iid=document.code,
                values=(result.rank, document.code, document.title, "{:.4f}".format(result.score))
            )
        self.status_var.set(response.message)
        self.notebook.select(0)
        if response.results:
            first = response.results[0].document.code
            self.results_tree.selection_set(first)
            self.results_tree.focus(first)
            self._show_selected_document()

    def _show_selected_document(self, event=None):
        selection = self.results_tree.selection()
        if not selection:
            return
        document = self._result_documents.get(selection[0])
        if document is None:
            return
        self.selected_document = document
        source = document.source_url or "[URL sumber belum diisi pada metadata.csv]"
        self.preview_info_var.set("{} | {}\nSumber: {}".format(document.code, document.filename, source))
        self._set_text(self.preview_text, document.text)
        self.open_button.configure(state="normal")

    def _open_document(self):
        document = self.selected_document
        if document is None:
            self.status_var.set("Pilih dokumen pada tabel hasil terlebih dahulu.")
            return
        path = Path(document.path).resolve()
        if not path.is_file():
            self.status_var.set("Dokumen tidak ditemukan: " + document.filename)
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Dokumen gagal dibuka", str(exc), parent=self.root)

    def _render_index(self):
        self._index_render_id += 1
        render_id = self._index_render_id
        self.index_tree.delete(*self.index_tree.get_children())
        if self.index is None:
            self.index_info_var.set("Index belum dibuat.")
            return
        index = self.index
        mode = self.index_mode_var.get()
        document_columns = ["doc_" + document.code for document in index.documents]
        columns = ["term", "df", "idf"] + document_columns
        self.index_tree.configure(columns=columns)
        for column, title, width, anchor in (
            ("term", "Term", 170, "w"), ("df", "DF", 75, "e"), ("idf", "IDF (log10)", 125, "e")
        ):
            self.index_tree.heading(column, text=title)
            self.index_tree.column(column, width=width, minwidth=width, stretch=False, anchor=anchor)
        for column, document in zip(document_columns, index.documents):
            self.index_tree.heading(column, text=document.code)
            self.index_tree.column(column, width=105, minwidth=90, stretch=False, anchor="e")
        # Engine menyimpan dokumen × term; tampilan harus term × dokumen.
        matrix = (index.tf if mode == "TF" else index.tfidf).T
        total = len(index.vocabulary)
        self.index_info_var.set("{} | Baris = term; kolom = kode dokumen. Memuat {} term...".format(mode, total))

        def insert_chunk(start=0):
            if self._closed or render_id != self._index_render_id:
                return
            stop = min(start + 150, total)
            for term_number in range(start, stop):
                weights = matrix[term_number]
                values = [
                    index.vocabulary[term_number], int(index.df[term_number]),
                    "{:.8f}".format(float(index.idf[term_number]))
                ]
                if mode == "TF":
                    values.extend(int(value) for value in weights)
                else:
                    values.extend("{:.8f}".format(float(value)) for value in weights)
                self.index_tree.insert("", "end", values=values)
            if stop < total:
                self.root.after(1, lambda: insert_chunk(stop))
            else:
                self.index_info_var.set(
                    "{} | {} term × {} dokumen. DF dan IDF berasal dari korpus; kueri tidak mengubah index.".format(
                        mode, total, len(index.documents)
                    )
                )

        insert_chunk()

    def _close(self):
        self._closed = True
        self._index_render_id += 1
        self.root.destroy()
