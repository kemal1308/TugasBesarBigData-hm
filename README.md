# 🛍️ H&M Personalized Fashion Recommendation System using PySpark MLlib

![PySpark](https://img.shields.io/badge/PySpark-MLlib-E25A1C?logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)
![Algorithm](https://img.shields.io/badge/Algorithm-ALS-blueviolet)

Repositori ini berisi implementasi **sistem rekomendasi produk mode (fashion recommendation) terpersonalisasi** untuk data transaksi H&M menggunakan **Apache Spark (PySpark MLlib)** dengan algoritma **Collaborative Filtering – Alternating Least Squares (ALS)** berbasis *implicit feedback*.

---

## 📑 Daftar Isi

- [Gambaran Proyek](#-gambaran-proyek)
- [Alur Kerja & Pembagian Tugas](#-alur-kerja--pembagian-tugas)
- [Tautan Eksekusi Cepat](#-tautan-eksekusi-cepat)
- [Struktur Direktori](#-struktur-direktori)
- [Data & Model Checkpoints](#-data--model-checkpoints)
- [Hasil Evaluasi Model Baseline](#-hasil-evaluasi-model-baseline)
- [Panduan Instalasi & Eksekusi Lokal](#%EF%B8%8F-panduan-instalasi--eksekusi-lokal)

---

## 📌 Gambaran Proyek

Proyek ini bertujuan mengolah data transaksi berskala besar (**~31,7 juta baris**) untuk memprediksi preferensi pembelian pelanggan dan menghasilkan **Top-N rekomendasi** produk pakaian yang relevan.

| Aspek | Keterangan |
|---|---|
| **Metode** | Collaborative Filtering (Matrix Factorization) menggunakan PySpark MLlib ALS |
| **Target Rating** | Implicit feedback berdasarkan frekuensi pembelian produk (`purchase_count`) |
| **Lingkungan Komputasi** | Google Colab (RAM ~12 GB, Cloud Runtime) terintegrasi dengan Google Drive dan GitHub |

---

## 👥 Alur Kerja & Pembagian Tugas

Proyek ini dikerjakan secara kolaboratif dengan pembagian peran sebagai berikut:

### 01: Persiapan Data Skala Besar

- Menangani dataset mentah dari file CSV yang berisi ~31,7 juta baris transaksi.
- Membersihkan data dari nilai *null* dan menggabungkan (*join*) tabel transaksi dengan metadata produk (tabel `articles`).
- Mengekspor seluruh 31,7 juta baris data gabungan yang sudah bersih ke format terkompresi `cleaned_data.parquet` agar pembacaan oleh PySpark jauh lebih cepat dan efisien.

📓 **Dokumentasi:** `notebooks/01_data_cleaning_eda.ipynb`

### 02: Pemodelan & Preprocessing Lanjutan

- **Filtering & penanganan memori:** Membaca file `.parquet` (31,7 juta baris), lalu menyaring data berdasarkan waktu (`t_dat >= 2020-08-01`) di awal pipeline untuk mencegah Google Colab mengalami *Out of Memory* (OOM). Data menyusut menjadi **2.035.461 baris aktif**.
- **Transformasi fitur:** Mengonversi ID pelanggan dan artikel (string) menjadi angka indeks menggunakan `StringIndexer`.
- **Pemodelan:** Membangun fitur implicit feedback berdasarkan frekuensi pembelian, lalu melatih dan menguji algoritma ALS dengan metrik RMSE.

📓 **Dokumentasi:** `notebooks/02_run_ml_pipeline.ipynb`, `src/preprocessing.py`, `src/model.py`

### 03: Tuning & Laporan

- Melakukan *hyperparameter tuning* (kombinasi `rank`, `maxIter`, `regParam`) untuk mendapatkan performa model terbaik.
- Menyusun laporan komprehensif dan visualisasi hasil rekomendasi.

---

## 🚀 Tautan Eksekusi Cepat

Jalankan eksperimen interaktif end-to-end langsung melalui Google Colab:

<!-- GANTI LINK_COLAB_ANDA dengan URL notebook Colab kamu -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_COLAB_ANDA)

> Notebook statis versi repositori tersedia pada direktori [`notebooks/02_run_ml_pipeline.ipynb`](notebooks/02_run_ml_pipeline.ipynb).

---

## 📁 Struktur Direktori

```text
├── data/                       # Direktori dataset lokal (diabaikan oleh git)
│   └── .gitkeep
├── models/                     # Direktori checkpoint model lokal (diabaikan oleh git)
│   └── .gitkeep
├── notebooks/                  # Jupyter / Colab Notebooks untuk eksperimen & evaluasi
│   ├── 01_data_cleaning_eda.ipynb
│   └── 02_run_ml_pipeline.ipynb
├── src/                        # Modul Python utama (modular pipeline)
│   ├── config.py               # Parameter konfigurasi Spark & ALS
│   ├── model.py                # Fungsi pelatihan, evaluasi (RMSE), dan inferensi
│   └── preprocessing.py        # Agregasi implicit feedback & StringIndexer pipeline
├── main_ml.py                  # Skrip eksekusi pipeline ML mandiri
├── .gitignore
└── README.md
```

---

## 💾 Data & Model Checkpoints

Karena batasan ukuran file di GitHub (< 100 MB), dataset utama (`.parquet`) dan artefak biner model PySpark (`models/`) disimpan secara eksternal.

- **☁️ Penyimpanan Cloud (Google Drive):** [Akses Folder Data & Model di Google Drive](LINK_GOOGLE_DRIVE_ANDA)
- **📦 Dataset:** Unduh dan simpan hasil unduhan ke direktori `data/cleaned_data.parquet`.
- **🧠 Model Artefak:** Hasil serialisasi model tersimpan di direktori:
  - `models/string_indexer_pipeline` — pipeline transformasi indeks user dan item
  - `models/als_recommender_model` — bobot matriks faktor laten ALS

> **Catatan:** Direktori `data/` dan `models/` di repositori ini hanya berisi file `.gitkeep` sebagai placeholder struktur direktori dan otomatis diabaikan oleh `.gitignore`.

---

## 📊 Hasil Evaluasi Model Baseline

Eksperimen baseline melatih data transaksi pada periode 2 bulan terakhir (`t_dat >= 2020-08-01`) guna memastikan efisiensi memori komputasi serta relevansi tren mode terkini.

## 🎛️ Hasil Hyperparameter Tuning & Evaluasi

### Hasil Grid Search (RMSE validasi, subset 20% data latih)

| rank | maxIter | regParam | RMSE Validasi |
|---|---|---|---|
| 20 | 10 | 0.05 | **1.21843** (terbaik) |
| 20 | 10 | 0.10 | 1.21856 (konfigurasi baseline) |
| 10 | 10 | 0.05 | 1.21926 |
| 10 | 10 | 0.10 | 1.21936 |

**Parameter terbaik:** rank = 20, maxIter = 10, regParam = 0.05

### Baseline vs Tuned (seed = 42)

| Metrik | Popularity | ALS Baseline | ALS Tuned |
|---|---|---|---|
| RMSE Test | - | 1.0794 | 1.0788 |
| Precision@10 | 0.00236 | 0.01314 | 0.01314 |
| Recall@10 | 0.01420 | 0.08163 | 0.08157 |
| MAP@10 | 0.00483 | 0.03084 | 0.03073 |
| NDCG@10 | 0.00791 | 0.04737 | 0.04725 |

Metrik ranking dihitung pada 9.827 pelanggan sampel dari data uji.

### Kesimpulan
- ALS mengungguli rekomendasi berbasis popularitas pada semua metrik ranking (sekitar 5-6 kali lipat).
- Selisih baseline vs tuned sangat kecil (RMSE 1.0794 → 1.0788), artinya konfigurasi baseline sudah mendekati optimal pada ruang pencarian yang diuji (4 kombinasi, subset 20% data).
- RMSE kurang ideal untuk implicit feedback, sehingga kualitas model juga dinilai dengan metrik ranking.
- Nilai absolut metrik ranking kecil karena katalog sangat besar dan matriks interaksi sangat renggang.

📊 Grafik hasil tuning dan evaluasi dapat dilihat pada `notebooks/03_model_tuning_and_evaluation.ipynb`.

---

## 🛠️ Panduan Instalasi & Eksekusi Lokal

### 1. Clone Repositori

```bash
git clone https://github.com/kemal1308/TugasBesarBigData-hm.git
cd TugasBesarBigData-hm
```

### 2. Pasang Dependensi

Pastikan **Java 8/11** dan **Python 3.9+** sudah terpasang, lalu jalankan:

```bash
pip install pyspark pandas matplotlib seaborn
```

### 3. Jalankan Pipeline

Letakkan file `cleaned_data.parquet` pada folder `data/`, lalu jalankan:

```bash
python main_ml.py
```
