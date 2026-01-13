#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL PROJECT - BIG DATA DAN DATA MINING
=========================================
Analisis dan Prediksi Data dengan Pendekatan Big Data Mining

Metode:
1. Klasifikasi (Random Forest, Decision Tree, KNN)
2. Clustering (K-Means, Hierarchical)
3. Association Rule Mining (Apriori)

Dataset Publik:
- Telco Customer Churn (IBM) - untuk Klasifikasi & Clustering
- Groceries Dataset - untuk Association Rule Mining
"""

# ============================================================================
# BAGIAN 1: IMPORT LIBRARY
# ============================================================================
print("="*70)
print("FINAL PROJECT - BIG DATA DAN DATA MINING")
print("="*70)
print("\n[1] Importing libraries...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import urllib.request
import ssl
warnings.filterwarnings('ignore')

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

# Library untuk Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, silhouette_score)

# Library untuk Klasifikasi
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Library untuk Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering

# Library untuk Association Rule Mining
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

print("Semua library berhasil diimport!")

# ============================================================================
# BAGIAN 2: LOAD DATASET DARI URL
# ============================================================================
print("\n" + "="*70)
print("BAGIAN 2: LOAD DATASET PUBLIK")
print("="*70)

# 2.1 Load Dataset Telco Customer Churn
print("\n[2.1] Loading Telco Customer Churn Dataset...")
print("-"*50)

URL_TELCO = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    df_churn = pd.read_csv(URL_TELCO)
    print(f"Dataset berhasil dimuat!")
    print(f"Ukuran: {df_churn.shape[0]} baris x {df_churn.shape[1]} kolom")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 2.2 Load Dataset Groceries untuk Association Rule Mining
print("\n[2.2] Loading Groceries Dataset...")
print("-"*50)

URL_GROCERIES = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"

transactions = []
try:
    with urllib.request.urlopen(URL_GROCERIES) as response:
        content = response.read().decode('utf-8')
        for line in content.strip().split('\n'):
            items = [item.strip() for item in line.split(',') if item.strip()]
            if items:
                transactions.append(items)
    
    all_items = set()
    for trans in transactions:
        all_items.update(trans)
    
    print(f"Dataset berhasil dimuat!")
    print(f"Jumlah transaksi: {len(transactions)}")
    print(f"Jumlah produk unik: {len(all_items)}")
except Exception as e:
    print(f"Error loading groceries: {e}")
    exit(1)

# 2.3 Preview Dataset
print("\n[2.3] Preview Dataset Telco Customer Churn")
print("-"*50)
print(df_churn.head())

# ============================================================================
# BAGIAN 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*70)
print("BAGIAN 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

# 3.1 Info Dataset
print("\n[3.1] Informasi Dataset")
print("-"*50)
print(f"Shape: {df_churn.shape}")
print(f"Columns: {list(df_churn.columns)}")

# 3.2 Distribusi Target
print("\n[3.2] Distribusi Target Variable (Churn)")
print("-"*50)

churn_counts = df_churn['Churn'].value_counts()
print(f"Tidak Churn (No) : {churn_counts['No']} ({churn_counts['No']/len(df_churn)*100:.2f}%)")
print(f"Churn (Yes)      : {churn_counts['Yes']} ({churn_counts['Yes']/len(df_churn)*100:.2f}%)")

# 3.3 Missing Values
print("\n[3.3] Pengecekan Missing Values")
print("-"*50)
missing = df_churn.isnull().sum()
if missing.sum() > 0:
    print("Kolom dengan missing values:")
    print(missing[missing > 0])
else:
    print("Tidak ada missing values (NULL)!")

empty_total = (df_churn['TotalCharges'] == ' ').sum()
print(f"Nilai kosong (string) di TotalCharges: {empty_total}")

# ============================================================================
# BAGIAN 4: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("BAGIAN 4: DATA PREPROCESSING")
print("="*70)

# 4.1 Preprocessing
print("\n[4.1] Preprocessing Dataset")
print("-"*50)

df = df_churn.copy()

# Hapus customerID
df = df.drop('customerID', axis=1)
print("- Kolom 'customerID' dihapus")

# Konversi TotalCharges ke numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("- Kolom 'TotalCharges' dikonversi ke numeric")

# Isi missing dengan median
median_total = df['TotalCharges'].median()
df['TotalCharges'].fillna(median_total, inplace=True)
print(f"- Missing values diisi dengan median ({median_total:.2f})")

# 4.2 Encoding
print("\n[4.2] Encoding Categorical Variables")
print("-"*50)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Kolom kategorikal ({len(cat_cols)}): {cat_cols}")

# Label Encoding untuk target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])
print("- Target 'Churn' di-encode: No=0, Yes=1")

# One-Hot Encoding
cat_cols.remove('Churn')
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"- One-Hot Encoding diterapkan")
print(f"Ukuran setelah encoding: {df_encoded.shape}")

# 4.3 Split dan Normalisasi
print("\n[4.3] Split dan Normalisasi Data")
print("-"*50)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Jumlah fitur: {X.shape[1]}")
print(f"Data Training: {len(X_train)} sampel (80%)")
print(f"Data Testing : {len(X_test)} sampel (20%)")

# ============================================================================
# BAGIAN 5: IMPLEMENTASI KLASIFIKASI
# ============================================================================
print("\n" + "="*70)
print("BAGIAN 5: IMPLEMENTASI KLASIFIKASI")
print("="*70)

hasil_klasifikasi = {}

# 5.1 Random Forest
print("\n[5.1] Random Forest Classifier")
print("-"*50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
hasil_klasifikasi['Random Forest'] = rf_acc

print(f"Akurasi: {rf_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['Tidak Churn', 'Churn']))

# 5.2 Decision Tree
print("\n[5.2] Decision Tree Classifier")
print("-"*50)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
hasil_klasifikasi['Decision Tree'] = dt_acc

print(f"Akurasi: {dt_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, dt_pred, target_names=['Tidak Churn', 'Churn']))

# 5.3 KNN
print("\n[5.3] K-Nearest Neighbors (KNN)")
print("-"*50)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
hasil_klasifikasi['KNN'] = knn_acc

print(f"Akurasi: {knn_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, knn_pred, target_names=['Tidak Churn', 'Churn']))

# 5.4 Perbandingan Model
print("\n[5.4] Perbandingan Akurasi Model")
print("-"*50)
print(f"{'Model':<20} {'Akurasi':>10}")
print("-"*32)
for model, acc in sorted(hasil_klasifikasi.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:<20} {acc*100:>9.2f}%")

best_model = max(hasil_klasifikasi, key=hasil_klasifikasi.get)
print(f"\nModel Terbaik: {best_model} ({hasil_klasifikasi[best_model]*100:.2f}%)")

# 5.5 Feature Importance
print("\n[5.5] Feature Importance (Random Forest)")
print("-"*50)

feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 Fitur Paling Berpengaruh:")
print(feature_imp.head(10).to_string(index=False))

# ============================================================================
# BAGIAN 6: IMPLEMENTASI CLUSTERING
# ============================================================================
print("\n" + "="*70)
print("BAGIAN 6: IMPLEMENTASI CLUSTERING")
print("="*70)

# 6.1 Persiapan Data
print("\n[6.1] Persiapan Data untuk Clustering")
print("-"*50)

cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_cluster = df[cluster_features].copy()

scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df_cluster)

print(f"Fitur untuk clustering: {cluster_features}")
print(f"Jumlah sampel: {len(X_cluster)}")

# 6.2 Elbow Method
print("\n[6.2] Menentukan Jumlah Cluster Optimal")
print("-"*50)

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_cluster, kmeans.labels_))

print(f"{'K':>3} {'Inertia':>12} {'Silhouette':>12}")
print("-"*30)
for k, iner, sil in zip(K_range, inertias, silhouettes):
    print(f"{k:>3} {iner:>12.2f} {sil:>12.4f}")

optimal_k = list(K_range)[np.argmax(silhouettes)]
print(f"\nJumlah cluster optimal: K = {optimal_k}")

# 6.3 K-Means Clustering
print(f"\n[6.3] K-Means Clustering (K={optimal_k})")
print("-"*50)

kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_cluster['Cluster_KMeans'] = kmeans_model.fit_predict(X_cluster)

print("Distribusi Cluster:")
km_counts = df_cluster['Cluster_KMeans'].value_counts().sort_index()
for cluster, count in km_counts.items():
    print(f"  Cluster {cluster}: {count} customers ({count/len(df_cluster)*100:.1f}%)")

sil_kmeans = silhouette_score(X_cluster, df_cluster['Cluster_KMeans'])
print(f"\nSilhouette Score: {sil_kmeans:.4f}")

# 6.4 Hierarchical Clustering
print(f"\n[6.4] Hierarchical Clustering (K={optimal_k})")
print("-"*50)

hc_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
df_cluster['Cluster_HC'] = hc_model.fit_predict(X_cluster)

print("Distribusi Cluster:")
hc_counts = df_cluster['Cluster_HC'].value_counts().sort_index()
for cluster, count in hc_counts.items():
    print(f"  Cluster {cluster}: {count} customers ({count/len(df_cluster)*100:.1f}%)")

sil_hc = silhouette_score(X_cluster, df_cluster['Cluster_HC'])
print(f"\nSilhouette Score: {sil_hc:.4f}")

# 6.5 Perbandingan
print("\n[6.5] Perbandingan Metode Clustering")
print("-"*50)
print(f"K-Means Silhouette Score     : {sil_kmeans:.4f}")
print(f"Hierarchical Silhouette Score: {sil_hc:.4f}")

best_clustering = 'K-Means' if sil_kmeans >= sil_hc else 'Hierarchical'
print(f"\nMetode Terbaik: {best_clustering}")

# 6.6 Karakteristik Cluster
print("\n[6.6] Karakteristik Cluster (K-Means)")
print("-"*50)
cluster_summary = df_cluster.groupby('Cluster_KMeans')[cluster_features].agg(['mean', 'std']).round(2)
print(cluster_summary)

# ============================================================================
# BAGIAN 7: ASSOCIATION RULE MINING
# ============================================================================
print("\n" + "="*70)
print("BAGIAN 7: ASSOCIATION RULE MINING")
print("="*70)

# 7.1 Encoding Transaksi
print("\n[7.1] Encoding Transaksi")
print("-"*50)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_array, columns=te.columns_)
df_trans.columns = [str(col) for col in df_trans.columns]

print(f"Jumlah transaksi: {len(transactions)}")
print(f"Jumlah produk unik: {len(te.columns_)}")

# 7.2 Frequent Itemsets
print("\n[7.2] Frequent Itemsets (Apriori)")
print("-"*50)

frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print(f"Jumlah frequent itemsets: {len(frequent_itemsets)}")
print(f"\nTop 15 Frequent Itemsets:")
print(f"{'Itemset':<45} {'Support':>10}")
print("-"*57)

top_items = frequent_itemsets.nlargest(15, 'support')
for _, row in top_items.iterrows():
    items_str = ', '.join([str(i) for i in row['itemsets']])
    print(f"{items_str:<45} {row['support']:>10.4f}")

# 7.3 Association Rules
print("\n[7.3] Association Rules")
print("-"*50)

# Konversi itemsets ke string untuk kompatibilitas
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(
    lambda x: frozenset([str(item) for item in x])
)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0,
                          num_itemsets=len(frequent_itemsets))

if len(rules) > 0:
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join([str(i) for i in x]))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join([str(i) for i in x]))
    
    print(f"Jumlah rules ditemukan: {len(rules)}")
    print(f"\nTop 15 Rules (berdasarkan Lift):")
    print(f"{'Antecedent':<25} {'Consequent':<20} {'Supp':>7} {'Conf':>7} {'Lift':>7}")
    print("-"*72)
    
    top_rules = rules.nlargest(15, 'lift')
    for _, row in top_rules.iterrows():
        ant = row['antecedents_str'][:23]
        cons = row['consequents_str'][:18]
        print(f"{ant:<25} {cons:<20} {row['support']:>7.4f} {row['confidence']:>7.4f} {row['lift']:>7.4f}")
    
    # 7.4 Interpretasi
    print("\n[7.4] Interpretasi Association Rules")
    print("-"*50)
    
    best_rule = rules.loc[rules['lift'].idxmax()]
    print("Rule dengan Lift Tertinggi:")
    print(f"\n  IF customer membeli: [{best_rule['antecedents_str']}]")
    print(f"  THEN kemungkinan juga membeli: [{best_rule['consequents_str']}]")
    print(f"\n  Metrics:")
    print(f"    Support   : {best_rule['support']:.4f} ({best_rule['support']*100:.2f}% transaksi)")
    print(f"    Confidence: {best_rule['confidence']:.4f} ({best_rule['confidence']*100:.2f}%)")
    print(f"    Lift      : {best_rule['lift']:.4f}")
else:
    print("Tidak ada rules yang memenuhi kriteria.")

# ============================================================================
# BAGIAN 8: KESIMPULAN DAN REKOMENDASI
# ============================================================================
print("\n" + "="*70)
print("BAGIAN 8: KESIMPULAN DAN REKOMENDASI")
print("="*70)

print("\n" + "="*70)
print("RINGKASAN HASIL ANALISIS")
print("="*70)

print("\n[1] KLASIFIKASI - Customer Churn Prediction")
print("-"*50)
print(f"    Dataset      : Telco Customer Churn (IBM)")
print(f"    Jumlah Data  : {len(df_churn)} customers")
print(f"    Model Terbaik: {best_model}")
print(f"    Akurasi      : {hasil_klasifikasi[best_model]*100:.2f}%")
print(f"    Top Feature  : {feature_imp.iloc[0]['Feature']}")

print("\n[2] CLUSTERING - Customer Segmentation")
print("-"*50)
print(f"    Dataset       : Telco Customer Churn (IBM)")
print(f"    Jumlah Cluster: {optimal_k}")
print(f"    Metode Terbaik: {best_clustering}")
print(f"    Silhouette    : {max(sil_kmeans, sil_hc):.4f}")

print("\n[3] ASSOCIATION RULE MINING - Market Basket Analysis")
print("-"*50)
print(f"    Dataset          : Groceries")
print(f"    Jumlah Transaksi : {len(transactions)}")
print(f"    Frequent Itemsets: {len(frequent_itemsets)}")
print(f"    Association Rules: {len(rules)}")

print("\n" + "="*70)
print("REKOMENDASI BISNIS")
print("="*70)

print("""
1. CHURN PREVENTION (Berdasarkan Klasifikasi)
   - Fokus pada customer dengan tenure rendah (pelanggan baru)
   - Monitor customer dengan kontrak month-to-month (rentan churn)
   - Berikan program loyalitas untuk meningkatkan retention
   - Perhatikan customer dengan monthly charges tinggi

2. CUSTOMER SEGMENTATION (Berdasarkan Clustering)
   - Segmentasi customer berdasarkan tenure dan monthly charges
   - Strategi marketing berbeda untuk setiap segmen
   - Identifikasi high-value customers untuk program VIP
   - Tawarkan upgrade layanan ke segmen yang sesuai

3. PRODUCT BUNDLING (Berdasarkan Association Rules)
   - Buat bundle produk berdasarkan pola pembelian
   - Implementasi cross-selling recommendation
   - Optimasi product placement di toko
   - Promo bundle untuk produk dengan lift tinggi
""")

print("="*70)
print("PROGRAM FINAL PROJECT SELESAI")
print("="*70)
