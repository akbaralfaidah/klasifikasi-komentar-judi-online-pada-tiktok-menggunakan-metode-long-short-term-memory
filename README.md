# Klasifikasi Komentar Judi Online pada TikTok Menggunakan Metode LSTM

Aplikasi ini merupakan perangkat lunak berbasis web  yang dibangun menggunakan **Streamlit** untuk mendeteksi dan mengklasifikasikan komentar TikTok menjadi dua kategori utama: **Judi Online** dan **Non-Judi Online**. Sistem ini menggunakan model *Deep Learning* **Long Short-Term Memory (LSTM)** dan dikembangkan sebagai bagian dari Tugas Akhir/Skripsi Program Studi Teknik Informatika.

---

## 📋 Fitur Utama

1. **Multi-Skenario Model (12 Skenario LSTM)**
   Mendukung pengujian 12 variasi model berdasarkan kombinasi *Learning Rate*, *Batch Size*, dan *Epoch*.

2. **Otomatisasi Preprocessing**

   * Auto-load data validasi saat aplikasi dijalankan.
   * Pembersihan teks real-time (regex, normalisasi slang, stopword removal, stemming Sastrawi).

3. **Analisis Hasil Validasi**

   * Tabel hasil prediksi lengkap.
   * Filter Kategori: **True Positive (TP)**, **True Negative (TN)**, **False Positive (FP)**, **False Negative (FN)**.
   * Fitur paginasi dengan kontrol *Next/Prev* dan jumlah item.

4. **Klasifikasi Teks Baru**

   * **Input Tunggal** untuk klasifikasi cepat.
   * **Batch Upload** melalui file `.csv` atau `.txt`.

---

## 🛠️ Struktur Folder

```text
📂 folder_proyek_anda/
│
├── 📄 app.py                   # Streamlit utama
├── 📄 preprocessing.py         # Pembersihan teks (Sastrawi & Regex)
├── 📄 word_embedding.py        # Tokenizer & Sequence
├── 📄 model_builder.py         # Load & Predict Model LSTM
├── 📄 requirements.txt         # Daftar library Python
│
├── 📂 .streamlit/              # Konfigurasi Tema
│   └── 📄 config.toml
│
├── 📄 kamus_slang.json         # Normalisasi slang
├── 📄 tokenizer.json           # Tokenizer Keras (wajib)
├── 📄 data_validasi_mentah.csv # Data validasi (wajib)
│
├── 📄 model_1.h5            # Model LSTM Skenario 1
├── 📄 model_2.h5            # Model LSTM Skenario 2
│   ...
└── 📄 model_12.h5           # Model LSTM Skenario 12
```

---

## 🚀 Instalasi & Cara Menjalankan

### 1. Prasyarat

Pastikan Python 3.8 atau lebih baru telah terpasang.

### 2. Buat Virtual Environment (Disarankan)

**Windows:**

```bash
python -m venv venv
.\/venv\/Scripts\/activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Library

```bash
pip install -r requirements.txt
```

### 4. Download Data NLTK

```bash
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi otomatis terbuka di browser ([http://localhost:8501](http://localhost:8501)).

---

## 📊 Panduan Penggunaan

### **Tab 1: Pilih Model**

* Tunggu proses *Inisialisasi Aplikasi* selesai.
* Pilih salah satu dari 12 model LSTM.
* Aktifkan model → tampilan indikator hijau.
* Klik **"Check Komentar Hasil Klasifikasi"** untuk melihat tabel validasi.
* Gunakan filter TP/TN/FP/FN untuk analisis khusus.

### **Tab 2: Klasifikasi Teks Baru**

* Pastikan model telah diaktifkan.
* **Input Tunggal:** ketik komentar, klik *Check Komentar*.
* **File Upload:** unggah `.csv` (kolom: `text`) atau `.txt`, lalu klik *Check File*.

---

## 📦 Dependensi Utama

* **Streamlit** — Antar muka interaktif.
* **TensorFlow / Keras** — LSTM dan prediksi.
* **Sastrawi** — Stemming Bahasa Indonesia.
* **Pandas & NumPy** — Manipulasi data.
* **NLTK** — Tokenizing dan stopword removal.

---

## ✍️ Pengembang

**Akbar Alfaidah**
Program Studi Teknik Informatika – Universitas Sriwijaya
Tahun: **2025**
