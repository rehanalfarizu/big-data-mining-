# FINAL PROJECT - BIG DATA DAN DATA MINING

## Analisis dan Prediksi Data dengan Pendekatan Big Data Mining

### Informasi Proyek
- **Mata Kuliah**: Big Data dan Data Mining
- **Metode**: Predictive Analytics & Descriptive Analytics
- **Algoritma**: Klasifikasi, Clustering, Association Rule Mining

---

## DAFTAR ISI

1. [Deskripsi Proyek](#deskripsi-proyek)
2. [Dataset Publik](#dataset-publik)
3. [Metode yang Digunakan](#metode-yang-digunakan)
4. [Struktur Program](#struktur-program)
5. [Cara Menjalankan](#cara-menjalankan)
6. [Hasil Analisis](#hasil-analisis)
7. [Kesimpulan dan Rekomendasi](#kesimpulan-dan-rekomendasi)

---

## DESKRIPSI PROYEK

Proyek ini mengimplementasikan berbagai teknik Big Data Mining untuk menganalisis data customer dengan tujuan:
1. **Prediksi Customer Churn** - Menggunakan metode klasifikasi untuk memprediksi customer yang akan churn
2. **Customer Segmentation** - Menggunakan clustering untuk mengelompokkan customer berdasarkan perilaku
3. **Market Basket Analysis** - Menggunakan Association Rule Mining untuk menemukan pola pembelian

---

## DATASET PUBLIK

### 1. Telco Customer Churn (IBM)
- **Sumber**: IBM GitHub Repository
- **URL**: https://github.com/IBM/telco-customer-churn-on-icp4d
- **Jumlah Data**: 7,043 customers
- **Fitur** (21 kolom):
  - customerID, gender, SeniorCitizen, Partner, Dependents
  - tenure, PhoneService, MultipleLines, InternetService
  - OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
  - StreamingTV, StreamingMovies, Contract, PaperlessBilling
  - PaymentMethod, MonthlyCharges, TotalCharges
- **Target**: Churn (Yes/No)
- **Kegunaan**: Klasifikasi & Clustering

### 2. Groceries Dataset (Machine Learning with R)
- **Sumber**: Machine Learning with R Datasets
- **URL**: https://github.com/stedy/Machine-Learning-with-R-datasets
- **Jumlah Transaksi**: 9,835 transaksi
- **Jumlah Produk Unik**: 169 produk
- **Kegunaan**: Association Rule Mining

---

## METODE YANG DIGUNAKAN

### A. KLASIFIKASI (Predictive Analytics)
| Algoritma | Akurasi | Keterangan |
|-----------|---------|------------|
| Decision Tree | 79.42% | Tree-based (TERBAIK) |
| Random Forest | 78.78% | Ensemble method |
| K-Nearest Neighbors | 74.66% | Distance-based |

**Top 10 Feature Importance (Random Forest):**
| Fitur | Importance |
|-------|------------|
| TotalCharges | 19.21% |
| tenure | 17.47% |
| MonthlyCharges | 16.84% |
| PaymentMethod_Electronic check | 3.88% |
| InternetService_Fiber optic | 3.86% |

### B. CLUSTERING (Descriptive Analytics)
| Metode | Silhouette Score | Jumlah Cluster |
|--------|------------------|----------------|
| K-Means | 0.4797 | 2 cluster (TERBAIK) |
| Hierarchical | 0.4300 | 2 cluster |

**Karakteristik Cluster (K-Means):**
| Cluster | Tenure (mean) | Monthly Charges (mean) | Total Charges (mean) | Jumlah |
|---------|---------------|------------------------|----------------------|--------|
| 0 | 20.03 bulan | $52.17 | $869.31 | 4,683 (66.5%) |
| 1 | 56.87 bulan | $89.76 | $5,084.99 | 2,360 (33.5%) |

### C. ASSOCIATION RULE MINING
- **Algoritma**: Apriori
- **Minimum Support**: 0.01 (1%)
- **Minimum Lift**: 1.0
- **Frequent Itemsets**: 333
- **Association Rules**: 598

**Top 5 Association Rules (berdasarkan Lift):**
| Antecedent | Consequent | Support | Confidence | Lift |
|------------|------------|---------|------------|------|
| whole milk, yogurt | curd | 0.0101 | 17.97% | 3.37 |
| curd | whole milk, yogurt | 0.0101 | 18.89% | 3.37 |
| other vegetables, citrus fruit | root vegetables | 0.0104 | 35.92% | 3.30 |
| yogurt, other vegetables | whipped/sour cream | 0.0102 | 23.42% | 3.27 |
| tropical fruit, other vegetables | root vegetables | 0.0123 | 34.28% | 3.14 |

---

## STRUKTUR PROGRAM

```
big-data-mining-/
├── main.py                              # Program Python (untuk terminal)
├── Final_Project_BigData_Mining.ipynb   # Jupyter Notebook
├── requirements.txt                     # Library yang dibutuhkan
├── README.md                            # Dokumentasi
├── LICENSE                              # Lisensi
└── .venv/                               # Virtual environment
```

---

## CARA MENJALANKAN

### Opsi 1: Menggunakan Python Script (Terminal)

```bash
# 1. Aktifkan virtual environment
source .venv/bin/activate

# 2. Install dependencies (jika belum)
pip install -r requirements.txt

# 3. Jalankan program
python main.py
```

### Opsi 2: Menggunakan Jupyter Notebook

```bash
# 1. Aktifkan virtual environment
source .venv/bin/activate

# 2. Buka Jupyter Lab
jupyter lab

# 3. Buka file Final_Project_BigData_Mining.ipynb
# 4. Run All Cells
```

### Opsi 3: Menggunakan VS Code

1. Buka file `Final_Project_BigData_Mining.ipynb`
2. Pilih kernel **"Python (Big Data Mining)"** atau **.venv**
3. Klik **Run All** atau tekan `Ctrl+Shift+Enter`

### Library yang Dibutuhkan
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- mlxtend
- scipy

---

## HASIL ANALISIS

### 1. KLASIFIKASI - Customer Churn Prediction
- **Dataset**: Telco Customer Churn (IBM)
- **Jumlah Data**: 7,043 customers
- **Model Terbaik**: Decision Tree dengan akurasi **79.42%**
- **Fitur Paling Berpengaruh**: TotalCharges (19.21%)
- **Distribusi Target**:
  - Tidak Churn: 5,174 (73.46%)
  - Churn: 1,869 (26.54%)

### 2. CLUSTERING - Customer Segmentation
- **Dataset**: Telco Customer Churn (IBM)
- **Jumlah Cluster Optimal**: 2 cluster
- **Metode Terbaik**: K-Means (Silhouette: 0.4797)
- **Segmentasi**:
  - **Cluster 0**: Pelanggan baru dengan charges rendah (66.5%)
  - **Cluster 1**: Pelanggan loyal dengan charges tinggi (33.5%)

### 3. ASSOCIATION RULE MINING - Market Basket Analysis
- **Dataset**: Groceries
- **Jumlah Transaksi**: 9,835
- **Frequent Itemsets**: 333
- **Association Rules**: 598
- **Rule Terbaik**: whole milk + yogurt → curd (Lift: 3.37)

---

## KESIMPULAN DAN REKOMENDASI

### Kesimpulan
1. **Klasifikasi**: Model Decision Tree memberikan akurasi terbaik (79.42%) untuk prediksi churn. Fitur TotalCharges, tenure, dan MonthlyCharges paling berpengaruh.

2. **Clustering**: Customer dapat disegmentasi menjadi 2 kelompok:
   - Pelanggan baru dengan spending rendah
   - Pelanggan loyal dengan spending tinggi

3. **Association Rules**: Ditemukan 598 rules dengan pola pembelian yang kuat, terutama produk dairy (whole milk, yogurt, curd).

### Rekomendasi Bisnis

#### 1. CHURN PREVENTION (Berdasarkan Klasifikasi)
- Fokus pada customer dengan tenure rendah (pelanggan baru)
- Monitor customer dengan kontrak month-to-month (rentan churn)
- Berikan program loyalitas untuk meningkatkan retention
- Perhatikan customer dengan monthly charges tinggi

#### 2. CUSTOMER SEGMENTATION (Berdasarkan Clustering)
- Segmentasi customer berdasarkan tenure dan monthly charges
- Strategi marketing berbeda untuk setiap segmen
- Identifikasi high-value customers untuk program VIP
- Tawarkan upgrade layanan ke segmen yang sesuai

#### 3. PRODUCT BUNDLING (Berdasarkan Association Rules)
- Buat bundle produk berdasarkan pola pembelian
- Implementasi cross-selling recommendation
- Optimasi product placement di toko
- Promo bundle untuk produk dengan lift tinggi (whole milk + yogurt + curd)

---

## TEKNOLOGI

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python 3.14 |
| Data Processing | pandas, numpy |
| Visualisasi | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Association Rules | mlxtend |
| Environment | Virtual Environment (.venv) |
| IDE | VS Code, Jupyter Lab |

---

## REFERENSI

1. **Telco Customer Churn Dataset**: 
   - https://github.com/IBM/telco-customer-churn-on-icp4d

2. **Groceries Dataset**: 
   - https://github.com/stedy/Machine-Learning-with-R-datasets

3. **Scikit-learn Documentation**: 
   - https://scikit-learn.org/stable/

4. **MLxtend Documentation**: 
   - https://rasbt.github.io/mlxtend/

---

**Program Final Project Big Data Mining berhasil dijalankan!**
