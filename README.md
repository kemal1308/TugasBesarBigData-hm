# H&M Personalized Fashion Recommendation System using PySpark MLlib

Repositori ini berisi implementasi sistem rekomendasi produk mode (*fashion recommendation*) terpersonalisasi untuk data transaksi H&M menggunakan **Apache Spark (PySpark MLlib)** dengan algoritma **Collaborative Filtering - Alternating Least Squares (ALS)** berbasis *implicit feedback*.

---

## 📌 Gambaran Proyek

Proyek ini bertujuan mengolah data transaksi berskala besar (~31,7 juta baris transaksi) untuk memprediksi preferensi pembelian pelanggan dan menghasilkan Top-N rekomendasi produk pakaian yang relevan.

* **Metode:** Collaborative Filtering (Matrix Factorization) menggunakan PySpark MLlib ALS.
* **Target Rating:** *Implicit Feedback* berdasarkan frekuensi pembelian produk (`purchase_count`).
* **Lingkungan Komputasi:** Google Colab (RAM ~12 GB Cloud Runtime) terintegrasi dengan Google Drive dan GitHub.

---

## 🚀 Tautan Eksekusi Cepat

Jalankan eksperimen interaktif *end-to-end* langsung melalui Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15zm9vMmF2hRMCZDrS9ibv7SqMBLGJ7Mq?usp=sharing)

Notebook statis versi repositori tersedia pada direktori [`notebooks/02_run_ml_pipeline.ipynb`](notebooks/02_run_ml_pipeline.ipynb).

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

Karena batasan ukuran file di GitHub (< 100 MB), dataset utama (`.parquet`) dan artefak biner model PySpark (`models/`) disimpan secara eksternal:

* **Penyimpanan Cloud (Google Drive):** [Akses Folder Data & Model di Google Drive](https://drive.google.com/drive/folders/1MOEoXRvFr4jEGYtro00kPjD9-6ph_zdS?usp=drive_link)
* **Dataset:** Unduh dan simpan folder hasil unduhan ke dalam direktori `data/cleaned_data.parquet`.
* **Model Artefak:** Hasil serialisasi model tersimpan dalam direktori:
  * `models/string_indexer_pipeline` (Pipeline transformasi indeks user dan item)
  * `models/als_recommender_model` (Bobot matriks faktor laten ALS)

> **Catatan:** Direktori `data/` dan `models/` di repositori GitHub ini hanya berisi file `.gitkeep` sebagai *placeholder* struktur direktori dan secara otomatis diabaikan oleh `.gitignore`.

---

## 📊 Hasil Evaluasi Model Baseline

Eksperimen baseline melatih data transaksi pada periode 2 bulan terakhir (`t_dat >= 2020-08-01`) guna memastikan efisiensi memori komputasi serta relevansi tren mode terkini.

| Metrik / Parameter | Nilai / Konfigurasi |
| :--- | :--- |
| **Total Baris Data Aktif** | 2.035.461 baris |
| **Data Latih (Train Split 80%)** | 1.434.653 baris |
| **Data Uji (Test Split 20%)** | 358.961 baris |
| **Evaluasi Test RMSE** | **`1.0794`** |
| **Latent Factors (`rank`)** | 20 |
| **Max Iterations (`maxIter`)** | 10 |
| **Regularization (`regParam`)** | 0.1 |
| **Confidence Multiplier (`alpha`)** | 40.0 |
| **Feedback Type** | Implicit (`implicitPrefs=True`) |
| **Cold Start Strategy** | Drop (`coldStartStrategy='drop'`) |

---

## 🛠️ Panduan Instalasi & Eksekusi Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/kemal1308/TugasBesarBigData-hm.git
cd TugasBesarBigData-hm
```

### 2. Pasang Dependensi
Pastikan telah terpasang Java 8/11 dan Python 3.9+:
```bash
pip install pyspark pandas matplotlib seaborn
```

### 3. Jalankan Pipeline
Letakkan file `cleaned_data.parquet` pada folder `data/`, lalu jalankan:
```bash
python main_ml.py
```
