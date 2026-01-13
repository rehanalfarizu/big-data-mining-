# FINAL PROJECT - BIG DATA DAN DATA MINING

## Analisis dan Prediksi Data dengan Pendekatan Big Data Mining

### Informasi Proyek
- **Mata Kuliah**: Big Data dan Data Mining
- **Metode**: Predictive Analytics & Descriptive Analytics
- **Algoritma**: Klasifikasi, Clustering, Association Rule Mining

---

## DAFTAR ISI

1. [Deskripsi Proyek](#deskripsi-proyek)
2. [Dataset](#dataset)
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

## DATASET

### 1. Dataset Klasifikasi (Customer Churn)
- **Jumlah data**: 500 customer
- **Fitur**:
  - Usia
  - Pendapatan
  - Lama Berlangganan
  - Jumlah Transaksi
  - Total Spending
  - Frekuensi Komplain
  - Skor Kepuasan
- **Target**: Churn (0/1)

### 2. Dataset Clustering (RFM Analysis)
- **Jumlah data**: 500 customer
- **Fitur**:
  - Recency (hari sejak transaksi terakhir)
  - Frequency (frekuensi transaksi)
  - Monetary (total spending)
  - Tenure (lama menjadi customer)

### 3. Dataset Association Rule Mining
- **Jumlah transaksi**: 300 transaksi
- **Produk**: 12 jenis produk

---

## METODE YANG DIGUNAKAN

### A. KLASIFIKASI (Predictive Analytics)
| Algoritma | Akurasi | Keterangan |
|-----------|---------|------------|
| Random Forest | 76.00% | Ensemble method |
| Decision Tree | 76.00% | Tree-based |
| K-Nearest Neighbors | 78.00% | Distance-based (TERBAIK) |

### B. CLUSTERING (Descriptive Analytics)
| Metode | Silhouette Score | Jumlah Cluster |
|--------|------------------|----------------|
| K-Means | 0.2315 | 9 cluster (TERBAIK) |
| Hierarchical | 0.1691 | 9 cluster |

### C. ASSOCIATION RULE MINING
- **Algoritma**: Apriori
- **Minimum Support**: 0.1
- **Minimum Confidence**: 0.3
- **Frequent Itemsets**: 78
- **Association Rules**: 129

---

## STRUKTUR PROGRAM

```
big-data-mining-/
|-- main.py                      # Program utama
|-- requirements.txt             # Library yang dibutuhkan
|-- README.md                    # Dokumentasi
|
|-- OUTPUT FILES:
|   |-- hasil_klasifikasi.csv    # Hasil prediksi churn
|   |-- hasil_clustering.csv     # Hasil segmentasi customer
|   |-- hasil_association_rules.csv  # Association rules
|
|-- VISUALISASI:
    |-- output_visualisasi_1.png # Distribusi Churn, Perbandingan Model
    |-- output_visualisasi_2.png # Heatmap, Feature Importance, Cluster
    |-- output_confusion_matrix.png # Confusion Matrix
```

---

## CARA MENJALANKAN

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Program
```bash
python3 main.py
```

### 3. Library yang Dibutuhkan
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- mlxtend
- scipy

---

## HASIL ANALISIS

### 1. KLASIFIKASI
- Model terbaik: **KNN** dengan akurasi **78.00%**
- Fitur paling berpengaruh: **Frekuensi Komplain** (22.63%)
- Cross-validation menunjukkan model stabil

### 2. CLUSTERING
- Jumlah cluster optimal: **9 cluster**
- Metode terbaik: **K-Means**
- Customer tersegmentasi berdasarkan RFM

### 3. ASSOCIATION RULE MINING
- Rule terbaik: Snack -> Kopi (Lift: 1.23)
- 129 rules ditemukan untuk rekomendasi produk

---

## KESIMPULAN DAN REKOMENDASI

### Kesimpulan
1. Model KNN memberikan akurasi terbaik untuk prediksi churn
2. Customer dapat disegmentasi menjadi 9 kelompok berbeda
3. Terdapat pola pembelian yang dapat dimanfaatkan untuk bundling

### Rekomendasi Bisnis
1. **Churn Prevention**: Fokus pada customer dengan komplain tinggi
2. **Customer Segmentation**: Strategi marketing berbeda per cluster
3. **Product Bundling**: Bundling Snack + Kopi berdasarkan association rules

---

## OUTPUT FILES

| File | Deskripsi |
|------|-----------|
| hasil_klasifikasi.csv | Dataset dengan prediksi churn |
| hasil_clustering.csv | Dataset dengan label cluster |
| hasil_association_rules.csv | Association rules yang ditemukan |
| output_visualisasi_1.png | Visualisasi utama |
| output_visualisasi_2.png | Visualisasi tambahan |
| output_confusion_matrix.png | Confusion matrix per model |

---

**Program berhasil dijalankan dan menghasilkan semua output yang diperlukan.**
