# TubesBesar BigData - Analisis Prediktif IGD 🏥

Platform untuk analisis performa **Emergency Department (IGD)** menggunakan machine learning dan time-series forecasting untuk optimasi operasional rumah sakit skala besar.

## 📋 Deskripsi

**TubesBesar BigData** adalah project penelitian yang menganalisis data pasien di Instalasi Gawat Darurat (IGD) untuk membangun model prediktif yang dapat membantu dalam:
- Memprediksi disposisi pasien (admission vs discharge)
- Memproyeksikan occupancy (kepadatan) IGD per departemen
- Mengukur performa infrastruktur dan pipeline data

Project ini dikembangkan dengan fokus pada penanganan **data skala besar** dari IGD dengan mengimplementasikan berbagai machine learning algorithms dan time-series forecasting techniques.

## 🎯 Tujuan Project

- **M1**: Membangun model prediktif untuk disposisi pasien IGD dengan akurasi tinggi
- **M2**: Memprediksi occupancy IGD per jam untuk 4-6 jam ke depan per departemen
- **Infrastructure**: Mengukur latensi dan throughput dalam pipeline data Lambda
- Mengoptimalkan resource allocation dan patient flow management
- Memberikan insights berbasis data untuk pengambilan keputusan klinis dan administratif

## 📊 Dataset

Project menggunakan **IGD Patient Dataset** dengan fitur-fitur klinis:

- **5v_cleandf_relevant.csv** - Data pasien IGD yang telah di-preprocess (5 variables utama)
  - Variabel klinis: vital signs, triage level
  - Variabel demografis: umur, jenis kelamin
  - Variabel temporal: arrival hour, length of stay
  - Target: disposition (Admit/Discharge)

### Karakteristik Dataset
- **Total Records**: [✓ Sesuai analisis M1/M2]
- **Time Range**: Historical data dengan hourly resolution
- **Departemen**: Multiple departments dalam IGD
- **Format**: CSV dengan preprocessing lengkap

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- pandas, numpy, scikit-learn
- XGBoost
- statsmodels (untuk time-series)
- matplotlib, seaborn (visualisasi)
- Git

### Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/giovaando/TubesBigData.git
   cd TubesBigData
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn
   ```

3. **Jalankan Notebook Eksperimen**
   ```bash
   jupyter notebook
   # Buka EKSPERIMEN_M1_—_Prediksi_Disposisi_Pasien_IGD_(Admit_vs_Discharge).ipynb
   # atau
   # Buka EKSPERIMEN_M2_—_Prediksi_Occupancy_IGD.ipynb
   ```

## 📈 Eksperimen & Skenario Testing

### 1️⃣ **EKSPERIMEN M1**: Prediksi Disposisi Pasien IGD (Admit vs Discharge)

**File**: `EKSPERIMEN_M1_—_Prediksi_Disposisi_Pasien_IGD_(Admit_vs_Discharge).ipynb`

**Tujuan**: Membangun model klasifikasi untuk memprediksi apakah pasien akan di-admit atau di-discharge dari IGD

**Alur Eksperimen**:
1. **Load Data & Feature Selection** - Memilih fitur relevan dari dataset
2. **Preprocessing** - Imputasi missing values, encoding kategori, feature engineering
3. **Split Walk-Forward** - 70% training | 15% validation | 15% testing
4. **Medical Rule Engine** - Implementasi hierarki prioritas ESI (Emergency Severity Index)
5. **Model Training**:
   - Logistic Regression (baseline)
   - Decision Tree
   - Random Forest
   - XGBoost (gradient boosting)
6. **Evaluasi Lengkap** - ROC-AUC, F1-Score, Recall, Precision, Confusion Matrix

**Output Utama**:
- Model pickle files untuk production deployment
- Confusion matrices per model
- ROC curves comparison
- Feature importance analysis
- Performance metrics dashboard

---

### 2️⃣ **EKSPERIMEN M2**: Prediksi Occupancy IGD

**File**: `EKSPERIMEN_M2_—_Prediksi_Occupancy_IGD.ipynb`

**Tujuan**: Memproyeksikan kepadatan IGD per jam menggunakan time-series forecasting

**Alur Eksperimen**:
1. **Data Aggregation** - Konversi data arrival menjadi time-series hourly occupancy
2. **Time-Series Analysis**:
   - Stationarity testing (Augmented Dickey-Fuller)
   - ACF/PACF plots untuk identifikasi autoregressive order
3. **Model Comparison**:
   - **SARIMA** (Seasonal ARIMA) - Traditional time-series
   - **Hybrid SARIMA + XGBoost** - Ensemble approach
4. **Forecasting** - Prediksi 4-6 jam ke depan per departemen (A, B, C)
5. **Evaluasi** - MAPE (Mean Absolute Percentage Error) per departemen

**Output Utama**:
- Forecast visualizations per departemen
- MAPE comparison: SARIMA vs Hybrid model
- Time-series decomposition plots
- Seasonal patterns analysis

---

### 3️⃣ **EKSPERIMEN BONUS**: Evaluasi Latensi dan Throughput Pipeline Lambda

**File**: `Evaluasi_Latensi_dan_Throughput_Pipeline_Lambda.ipynb`

**Tujuan**: Mengukur performa infrastruktur data processing pipeline

**Metrik Utama**:
- End-to-end latency data pipeline
- Throughput (records/detik) vs load
- Bottleneck identification
- Resource utilization analysis

## 🏗️ Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│              (IGD Patient Events)                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Raw Admission Data → ETL Pipeline → 5v_cleandf       │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│              APPLICATION / ANALYSIS LAYER                    │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │   EKSPERIMEN M1          │  │  EKSPERIMEN M2          │ │
│  │ ─────────────────────    │  │ ─────────────────────   │ │
│  │ Klasifikasi:             │  │ Time-Series Forecast:   │ │
│  │ • Logistic Regression    │  │ • SARIMA                │ │
│  │ • Decision Tree          │  │ • Hybrid Ensemble       │ │
│  │ • Random Forest          │  │ • Per-Department        │ │
│  │ • XGBoost                │  │   Predictions           │ │
│  │                          │  │                         │ │
│  │ Output: Admit/Discharge  │  │ Output: Occupancy       │ │
│  │ Predictions              │  │ Forecasts (hourly)      │ │
│  └──────────────────────────┘  └──────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │    Infrastructure Evaluation (Latensi & Throughput) │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│              OUTPUT & VISUALIZATION LAYER                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Dashboards | Models | Insights | Recommendations   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## 📁 Struktur Project

```
TubesBigData/
├── README.md                              # Dokumentasi ini
├── 5v_cleandf_relevant.csv               # Dataset utama (cleaned & preprocessed)
│
├── EKSPERIMEN_M1_—_Prediksi_Disposisi_Pasien_IGD_(Admit_vs_Discharge).ipynb
│   └── Model klasifikasi dengan 4 algoritma
│
├── EKSPERIMEN_M2_—_Prediksi_Occupancy_IGD.ipynb
│   └── Time-series forecasting SARIMA vs Hybrid
│
└── Evaluasi_Latensi_dan_Throughput_Pipeline_Lambda.ipynb
    └── Infrastructure performance analysis
```

## 📊 Output & Results

Hasil analisis dari setiap eksperimen:

### M1 Output
- **Model Files**: Pickle files untuk setiap trained model
- **Performance Metrics**: 
  - ROC-AUC scores per model
  - F1-Score, Precision, Recall
  - Confusion matrices
- **Visualizations**: ROC curves, feature importance plots

### M2 Output
- **Forecasts**: Per-department occupancy projections (4-6 jam)
- **MAPE Comparison**: SARIMA vs Hybrid model performance
- **Time-Series Plots**: Actual vs predicted dengan seasonal decomposition
- **Statistical Tests**: ADF test results, ACF/PACF plots

### Infrastructure Output
- **Latency Metrics**: P50, P95, P99 latencies
- **Throughput Analysis**: Records/sec vs system load
- **Resource Utilization**: CPU, memory, I/O usage patterns

## 🚀 How to Run

### Option 1: Jupyter Notebook (Interactive)
```bash
jupyter notebook
# Pilih notebook sesuai eksperimen yang ingin dijalankan
```

### Option 2: Command Line dengan nbconvert
```bash
# M1 - Prediksi Disposisi
jupyter nbconvert --to notebook --execute "EKSPERIMEN_M1_—_Prediksi_Disposisi_Pasien_IGD_(Admit_vs_Discharge).ipynb"

# M2 - Prediksi Occupancy
jupyter nbconvert --to notebook --execute "EKSPERIMEN_M2_—_Prediksi_Occupancy_IGD.ipynb"

# Infrastructure Evaluation
jupyter nbconvert --to notebook --execute "Evaluasi_Latensi_dan_Throughput_Pipeline_Lambda.ipynb"
```

### Option 3: Python Script Extraction
```python
# Extract cells dari notebook menjadi Python script, lalu run dengan:
python extracted_script.py
```

## 📌 Catatan Penting

⚠️ **Dataset Size**: File `5v_cleandf_relevant.csv` adalah hasil dari preprocessing dan sudah dioptimasi untuk eksperimen. Untuk kembali ke data raw, hubungi data keeper.

⚠️ **Dependencies**: Pastikan semua library terinstall dengan benar, khususnya `statsmodels` untuk SARIMA (untuk M2) dan `xgboost` untuk gradient boosting.

ℹ️ **Reproducibility**: Set random seed di awal notebook untuk memastikan hasil yang reproducible across runs:
```python
import numpy as np
np.random.seed(42)
```

ℹ️ **Google Colab Support**: Semua notebook kompatibel dengan Google Colab. Upload dataset ke Google Drive dan adjust file path sesuai kebutuhan.

## 👥 Tim Kontributor

| No | Nama | Tugas dan Tanggung Jawab | Kontribusi (%) |
|:--:|:-----|:------------------------:|:---------------:|
| 1 | Muhammad Nurikhsan | Bab II - Identifikasi Research Gap dan Kontribusi Jurnal | 12.5% |
| 2 | Miftahul Khoiriyah | Bab II - Review Jurnal dan Analisis State-of-the-Art | 12.5% |
| 3 | Andini Rahma Kemala | Bab V - Hasil & Analisis, Bab VI - Kesimpulan | 12.5% |
| 4 | Giovan Lado | Bab I - Pendahuluan dan Latar Belakang Masalah | 12.5% |
| 5 | Zahwa Natasya Hamzah | Abstrak, Daftar Pustaka dan Daftar Isi | 12.5% |
| 6 | Aditya Hot Martua Sihite | Bab II - Tabel Perbandingan Research dan Domain Analysis | 12.5% |
| 7 | Aryasatya Widyatna Akbar | Bab IV - Desain Eksperimen dan Metodologi | 12.5% |
| 8 | Rian Rafael Sangap Tamba | Bab III - Metodologi dan Pendekatan Teknis | 12.5% |
| | **Total** | | **100%** |

## 📋 Rencana Pengerjaan (Roadmap)

### 📌 Phase 1: Foundation (Bab I-II)
| Bab | Deskripsi | Target | Status |
|:---|:-------|:------:|:------:|
| **Bab I** | Pendahuluan & Latar Belakang Masalah | Minggu 1 | ✅ |
| **Bab II** | Review Literatur & Research Gap | Minggu 2-3 | ✅ |

**Key Activities**:
- Identifikasi gap dalam literatur terkait IGD occupancy prediction
- Review state-of-the-art machine learning applications di emergency care
- Tabel perbandingan penelitian sejenis

---

### 📌 Phase 2: Methodology (Bab III-IV)
| Bab | Deskripsi | Target | Status |
|:---|:-------|:------:|:------:|
| **Bab III** | Metodologi & Approach Teknis | Minggu 4 | ✅ |
| **Bab IV** | Desain Eksperimen | Minggu 5 | ✅ |

**Key Activities**:
- Definisi metodologi penelitian (data collection, preprocessing)
- Desain eksperimen M1 dan M2
- Pemilihan algorithms dan hyperparameter tuning strategy
- Metrics dan evaluation methodology

---

### 📌 Phase 3: Implementation (Bab III Extended)
| Task | Deskripsi | Target | Status |
|:---|:-------|:------:|:------:|
| **Data Preparation** | Cleaning, feature engineering, split strategy | Minggu 4-5 | ✅ |
| **M1 Implementation** | Model training & evaluation untuk Admit/Discharge prediction | Minggu 6 | ✅ |
| **M2 Implementation** | SARIMA dan Hybrid model untuk occupancy forecasting | Minggu 7 | ✅ |
| **Infrastructure Eval** | Latensi dan throughput testing | Minggu 8 | ✅ |

**Key Activities**:
- Implementasi klasifikasi (LR, DT, RF, XGBoost) di M1
- Implementasi time-series (SARIMA + Hybrid) di M2
- Evaluation dan hyperparameter optimization
- Result collection dan result analysis

---

### 📌 Phase 4: Analysis & Results (Bab V)
| Task | Deskripsi | Target | Status |
|:---|:-------|:------:|:------:|
| **M1 Analysis** | Result interpretation, feature importance, model comparison | Minggu 9 | 🔄 |
| **M2 Analysis** | Forecast accuracy per departemen, seasonal patterns | Minggu 9 | 🔄 |
| **Infrastructure Analysis** | Bottleneck identification, optimization recommendations | Minggu 9 | 🔄 |

**Key Deliverables**:
- Comprehensive performance comparison: LR vs DT vs RF vs XGBoost
- MAPE breakdown per departemen (A, B, C)
- Latency and throughput insights
- Business recommendations based on findings

---

### 📌 Phase 5: Conclusion & Documentation (Bab VI + Appendix)
| Task | Deskripsi | Target | Status |
|:---|:-------|:------:|:------:|
| **Kesimpulan** | Summary findings, limitations, future work | Minggu 10 | 🔄 |
| **Appendix** | Daftar Pustaka, kode tambahan, supplementary figures | Minggu 10 | 🔄 |
| **Final Review** | Peer review dan revisi | Minggu 11 | ⏳ |
| **Submission** | Final laporan dan repository cleanup | Minggu 12 | ⏳ |

**Deliverables**:
- Final report PDF
- Clean, documented GitHub repository
- Reproducible analysis notebooks
- Presentation materials

---

## 🔑 Key Insights & Findings (To be updated)

### M1 - Prediksi Disposisi Pasien IGD
- **Best Model**: [TBD after M1 completion]
- **Key Features**: [Feature importance akan ditampilkan setelah training]
- **Business Impact**: Membantu triage dan resource planning

### M2 - Prediksi Occupancy IGD
- **MAPE by Department**: [TBD after M2 completion]
- **Best Forecasting Approach**: [SARIMA vs Hybrid comparison TBD]
- **Operational Value**: Real-time occupancy forecasting untuk staff scheduling

### Infrastructure
- **Bottlenecks**: [TBD after infrastructure evaluation]
- **Recommendations**: [Optimization suggestions TBD]

## 📚 Referensi & Resources

### Academic References
- [1] Donita, S., et al. (2024) "Emergency Department Occupancy Prediction using Machine Learning"
- [2] Ahmad, R., et al. (2023) "Time-Series Forecasting for Healthcare Systems"
- [3] Chen, T., & Guestrin, C. (2016) "XGBoost: A Scalable Tree Boosting System"

### Tools & Libraries
- **scikit-learn** - Machine Learning: https://scikit-learn.org
- **XGBoost** - Gradient Boosting: https://xgboost.readthedocs.io
- **statsmodels** - Time-Series: https://www.statsmodels.org
- **pandas** - Data Manipulation: https://pandas.pydata.org
- **matplotlib/seaborn** - Visualization: https://matplotlib.org

## 💡 Tips & Best Practices

1. **Data Lineage**: Selalu dokumentasikan transformasi data dari raw ke processed
2. **Model Versioning**: Simpan metadata model (training date, hyperparams, performance) 
3. **Cross-Validation**: Gunakan stratified k-fold untuk imbalanced data (jika ada)
4. **Hyperparameter Tuning**: GridSearchCV atau RandomizedSearchCV dengan appropriate CV strategy
5. **Reproducibility**: Set random seeds dan simpan data splits untuk future reference

## 🔗 Repository Links

- **GitHub Repository**: https://github.com/giovaando/TubesBigData
- **Project Issues**: Report issues di GitHub Issues tab
- **Contributions**: Silakan fork dan submit pull requests

## 📝 License

Project ini tersedia untuk keperluan akademik dan research di bawah [Specify Your License - e.g., MIT, Apache 2.0, CC-BY-4.0]

---

## 📞 Contact & Support

**Project Lead**: [Project Coordinator Name]  
**Questions/Issues**: Buka GitHub issue atau hubungi tim

**Last Updated**: May 29, 2026  
**Status**: 🟡 Active Development (Phase 3-4 in progress)

---

### 🎓 Institut Teknologi Sumatera

Penelitian ini adalah bagian dari Big Data course project di ITERA Lampung.

**Happy Analyzing! 🚀📊**
