# FINAL PROJECT - BIG DATA DAN DATA MINING

## Analisis dan Prediksi Data dengan Pendekatan Big Data Mining

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rehanalfarizu/big-data-mining-/blob/main/Final_Project_BigData_Mining.ipynb)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

### 🚀 Quick Start - Jalankan di Google Colab

Klik tombol di bawah untuk langsung menjalankan notebook di Google Colab:

| Notebook | Link |
|----------|------|
| 📊 **Final Project Big Data Mining** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rehanalfarizu/big-data-mining-/blob/main/Final_Project_BigData_Mining.ipynb) |

> ⚠️ **Note**: Ganti `YOUR_USERNAME` dengan username GitHub Anda setelah repository di-push ke GitHub.

---

### Informasi Proyek
- **Mata Kuliah**: Big Data dan Data Mining
- **Metode**: Predictive Analytics & Descriptive Analytics
- **Algoritma**: Klasifikasi, Clustering, Association Rule Mining
- **Visualisasi Interaktif**: PyGWalker

---

## 📑 DAFTAR ISI

1. [📝 Deskripsi Proyek](#-deskripsi-proyek)
2. [📊 Dataset Publik](#-dataset-publik)
3. [🔬 Metode yang Digunakan](#-metode-yang-digunakan)
4. [📁 Struktur Program](#-struktur-program)
5. [⚙️ Cara Menjalankan](#️-cara-menjalankan)
6. [📈 Hasil Analisis](#-hasil-analisis)
7. [💡 Kesimpulan dan Rekomendasi](#-kesimpulan-dan-rekomendasi)
8. [🛠️ Teknologi](#️-teknologi)

---

## 📝 DESKRIPSI PROYEK

Proyek ini mengimplementasikan berbagai teknik Big Data Mining untuk menganalisis data customer dengan tujuan:
1. **Prediksi High Value Customer** - Menggunakan metode klasifikasi untuk memprediksi customer bernilai tinggi
2. **Customer Segmentation** - Menggunakan clustering untuk mengelompokkan customer berdasarkan RFM
3. **Market Basket Analysis** - Menggunakan Association Rule Mining untuk menemukan pola pembelian

**Catatan**: Program menggunakan **1 Dataset Publik** untuk semua metode analisis.

---

## 📊 DATASET PUBLIK

### Groceries Dataset (Machine Learning with R)
- **Sumber**: Machine Learning with R Datasets
- **URL**: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv
- **Jumlah Transaksi**: 9,835 transaksi
- **Jumlah Produk Unik**: 169 produk
- **Format**: CSV (diload langsung dari URL)

**Penggunaan Dataset untuk 3 Metode:**
1. **Association Rules** - Langsung menggunakan data transaksi dari URL
2. **RFM Analysis** - Dikonversi dengan CustomerID untuk Klasifikasi & Clustering
3. **Klasifikasi** - Prediksi High Value Customer berdasarkan fitur RFM

---

## 🔬 METODE YANG DIGUNAKAN

### A. KLASIFIKASI (Predictive Analytics)
**Target**: High Value Customer (berdasarkan Monetary & Frequency)

| Algoritma | Akurasi | Keterangan |
|-----------|---------|------------|
| Random Forest | ~95%+ | Ensemble method (TERBAIK) |
| Decision Tree | ~94%+ | Tree-based |
| K-Nearest Neighbors | ~93%+ | Distance-based |

**Feature Importance (Random Forest):**
| Fitur | Deskripsi |
|-------|-----------|
| Monetary | Total spending customer |
| Frequency | Jumlah transaksi customer |
| Recency | Hari sejak transaksi terakhir |
| TotalQuantity | Total quantity dibeli |
| UniqueProducts | Jumlah produk unik |

### B. CLUSTERING (Descriptive Analytics)
**Metode**: K-Means dan Hierarchical Clustering

| Metode | Silhouette Score | Jumlah Cluster |
|--------|------------------|----------------|
| K-Means | ~0.45+ | Optimal berdasarkan silhouette |
| Hierarchical | ~0.40+ | Ward linkage |

**Karakteristik Segmen Customer (RFM):**
- **Champions**: Recency rendah, Frequency tinggi, Monetary tinggi
- **Loyal Customers**: Frequency tinggi
- **At Risk**: Recency tinggi, perlu perhatian

### C. ASSOCIATION RULE MINING
- **Algoritma**: Apriori
- **Dataset**: Groceries (9,835 transaksi dari URL publik)
- **Minimum Support**: 0.01 (1%)
- **Minimum Lift**: 1.0

**Contoh Association Rules:**
| Antecedent | Consequent | Interpretasi |
|------------|------------|--------------|
| whole milk, yogurt | curd | Customer yang beli susu dan yogurt cenderung juga beli curd |
| tropical fruit | other vegetables | Customer yang beli buah tropis cenderung juga beli sayuran |

---

## 📁 STRUKTUR PROGRAM

```
big-data-mining-/
├── Final_Project_BigData_Mining.ipynb   # Jupyter Notebook (UTAMA)
├── requirements.txt                     # Library yang dibutuhkan
├── README.md                            # Dokumentasi
├── LICENSE                              # Lisensi
└── .venv/                               # Virtual environment
```

---

## ⚙️ CARA MENJALANKAN

### Option 1: Google Colab (Recommended) ☁️

1. Klik badge **Open in Colab** di atas
2. Notebook akan terbuka di Google Colab
3. Klik **Runtime** → **Run all** untuk menjalankan semua cell
4. Semua library akan otomatis terinstall

### Option 2: Jupyter Notebook (VS Code) 💻

1. Buka file `Final_Project_BigData_Mining.ipynb`
2. Pilih kernel **".venv"** (Python 3.x)
3. Klik **Run All** atau tekan `Ctrl+Shift+Enter`

### Library yang Dibutuhkan
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- mlxtend
- scipy
- **pygwalker** (untuk visualisasi interaktif)

Instalasi:
```bash
pip install -r requirements.txt
```

---

## 📈 HASIL ANALISIS

### 1. KLASIFIKASI - High Value Customer Prediction
- **Dataset**: Groceries (dari URL Publik)
- **Target**: High Value Customer (Monetary & Frequency tinggi)
- **Model Terbaik**: Random Forest dengan akurasi ~95%+
- **Fitur Paling Berpengaruh**: Monetary, Frequency, Recency

### 2. CLUSTERING - Customer Segmentation (RFM)
- **Dataset**: Groceries (dari URL Publik)
- **Metode Terbaik**: K-Means
- **Fitur**: Recency, Frequency, Monetary
- **Segmentasi**:
  - **Champions**: Pelanggan terbaik
  - **Loyal Customers**: Frekuensi tinggi
  - **At Risk**: Perlu perhatian

### 3. ASSOCIATION RULE MINING - Market Basket Analysis
- **Dataset**: Groceries (9,835 transaksi dari URL)
- **Algoritma**: Apriori
- **Contoh Rule**: whole milk + yogurt → curd

---

## 💡 KESIMPULAN DAN REKOMENDASI

### Kesimpulan
1. **Klasifikasi**: Model Decision Tree memberikan akurasi terbaik (79.42%) untuk prediksi churn. Fitur TotalCharges, tenure, dan MonthlyCharges paling berpengaruh.

2. **Clustering**: Customer dapat disegmentasi menjadi 2 kelompok:
   - Pelanggan dengan spending rendah
   - Pelanggan loyal dengan spending tinggi

3. **Association Rules**: Ditemukan rules dengan pola pembelian yang kuat, terutama produk dairy (whole milk, yogurt, curd).

### Rekomendasi Bisnis

#### 1. HIGH VALUE CUSTOMER RETENTION (Berdasarkan Klasifikasi)
- Identifikasi customer dengan potensi high-value menggunakan model
- Berikan program VIP untuk customer high-value
- Personalisasi penawaran berdasarkan pola pembelian
- Monitor customer dengan Monetary dan Frequency tinggi

#### 2. CUSTOMER SEGMENTATION (Berdasarkan Clustering RFM)
- Segmentasi customer berdasarkan Recency, Frequency, Monetary
- Strategi marketing berbeda untuk setiap segmen
- Champions: Berikan reward dan program referral
- At Risk: Kampanye win-back dan diskon khusus

#### 3. PRODUCT BUNDLING (Berdasarkan Association Rules)
- Buat bundle produk berdasarkan pola pembelian
- Implementasi cross-selling recommendation
- Optimasi product placement di toko
- Promo bundle untuk produk dengan lift tinggi

---

## 🛠️ TEKNOLOGI

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python 3.x |
| Data Processing | pandas, numpy |
| Visualisasi | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Association Rules | mlxtend |
| Environment | Virtual Environment (.venv) |
| IDE | VS Code, Jupyter Lab |

---

## REFERENSI

1. **Groceries Dataset**: 
   - https://github.com/stedy/Machine-Learning-with-R-datasets

2. **Scikit-learn Documentation**: 
   - https://scikit-learn.org/stable/

3. **MLxtend Documentation**: 
   - https://rasbt.github.io/mlxtend/

---

## 📧 KONTRIBUTOR

- **Nama**: [muhammad rehan alfarizi]
- **NIM**: [23115548]
- **Mata Kuliah**: Big Data dan Data Mining

---

<p align="center">
  <b>🎓 Final Project Big Data Mining</b><br>
  <i>1 Dataset Publik untuk semua metode analisis!</i>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/rehanalfarizu/big-data-mining-/blob/main/Final_Project_BigData_Mining.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>
