Nama Kelompok :
1. 71230966 - Jevon Christian Putra Funay
2. 71230987 - Prastha Pradipta Purusa
3. 71231033	- Imel Rantetandung


Sumber Artikel 
1. Fast and Easy Short Answer Grading with High Accuracy
https://aclanthology.org/N16-1123/

2. Pre-Training BERT on Domain Resources for Short Answer Grading
https://aclanthology.org/D19-1628/

3. Distributed Vector Representations for Unsupervised Automatic Short Answer Grading
https://aclanthology.org/W16-4904/

4. Wisdom of Students: A Consistent Automatic Short Answer Grading Technique
https://aclanthology.org/W16-6324/

5. Text-to-Text Semantic Similarity for Automatic Short Answer Grading
https://aclanthology.org/E09-1065/

6. ACTA: Short-Answer Grading in High-Stakes Medical Exams
https://aclanthology.org/2023.bea-1.36/

7. ExASAG: Explainable Framework for Automatic Short Answer Grading
https://aclanthology.org/2023.bea-1.29/

8. Get Semantic With Me! The Usefulness of Different Feature Types for Short-Answer Grading
https://aclanthology.org/C16-1206/

9. A Fluctuation Smoothing Approach for Unsupervised Automatic Short Answer Grading
https://aclanthology.org/W16-4911/

10. Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments
https://aclanthology.org/P11-1076/

11. Identifying Patterns For Short Answer Scoring Using Graph-based Lexico-Semantic Text Matching
https://aclanthology.org/W15-0612/ 

12. Universal Automatic Short Answer Grading (ASAG) Model: A Comprehensive Approach
https://www.jisem-journal.com/index.php/journal/article/view/11686/5438

13. Inject Rubrics into Short Answer Grading System
https://aclanthology.org/D19-6119/

14. Automatic Scoring of Short Answers Using Justification Cues Estimated by BERT
https://aclanthology.org/2022.bea-1.2/ 

15. Automatic Short Answer Grading on High School’s E-Learning Using Semantic Similarity Methods
https://www.temjournal.com/content/121/TEMJournalFebruary2023_297_302.pdf 

DESKRIPSI PROGRAM
Aplikasi ini mencari artikel tentang Automated Short Answer Grading
(ASAG) menggunakan TF-IDF dan cosine similarity. Hasil pencarian
diurutkan dari skor kemiripan tertinggi.

FUNGSI FILE DAN FOLDER
- main.py: menjalankan aplikasi.
- corpus.py: membaca teks dari dokumen PDF atau TXT.
- preprocessing.py: mengubah teks menjadi huruf kecil, memisahkan
  kata, menghapus stopword, dan melakukan stemming.
- search_engine.py: menghitung TF, DF, IDF, TF-IDF, serta menentukan
  peringkat dokumen menggunakan cosine similarity.
- ui.py: mengatur tampilan, tombol, tabel, dan pratinjau dokumen.
- requirements.txt: daftar pustaka yang perlu dipasang.
- build_zip.py: mengemas proyek menjadi ZIP untuk pengumpulan.
- corpus/: tempat menyimpan artikel.
- corpus/metadata.csv: informasi judul dan URL sumber (opsional).
- output/: tempat hasil indeks yang dibuat otomatis.
- tests/: berisi pengujian pembacaan, preprocessing, dan pencarian.
- README.txt: informasi kelompok, sumber artikel, dan petunjuk program.

TAHAP AWAL PENGERJAAN
1. Mengumpulkan 15 artikel berbahasa Inggris tentang ASAG.
2. Menyimpan artikel dalam folder corpus dengan nama D01.pdf
   sampai D15.pdf. Satu file berisi satu artikel.
3. Mencatat judul dan URL sumber artikel.
4. Menyiapkan Python 3.9 atau lebih baru yang memiliki Tkinter.
5. Membuka terminal di folder proyek dan memasang kebutuhan:
   python -m pip install -r requirements.txt
   python -m nltk.downloader stopwords
6. Menjalankan aplikasi:
   python main.py
7. Memilih folder corpus, lalu menekan Bangun Index.
8. Memastikan 15 dokumen berhasil dibaca, lalu mencoba pencarian.

ALUR PENGOLAHAN PROGRAM
1. Membaca teks seluruh halaman setiap dokumen.
2. Melakukan preprocessing pada teks.
3. Menghitung TF (jumlah kemunculan kata) dan DF (jumlah dokumen
   yang mengandung kata).
4. Menghitung IDF dan bobot TF-IDF.
5. Menyimpan hasil indeks ke folder output.
6. Mengolah kata pencarian dengan preprocessing yang sama.
7. Menghitung cosine similarity antara pencarian dan setiap dokumen.
8. Menampilkan dokumen dari skor kemiripan tertinggi.

Program dijalankan langsung tanpa kompilasi manual.
Jika dokumen ditambah atau diubah, lakukan Bangun Index kembali.