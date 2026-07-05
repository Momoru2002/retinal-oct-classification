<h1 align="center">🩺 Retinal OCT Disease Classification</h1>

<p align="center">
  <b>ResNet50 vs EfficientNetV2 vs DenseNet121 — with Explainable AI (Grad-CAM)</b>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=18&duration=3000&pause=900&color=3B82F6&center=true&vCenter=true&width=800&lines=Classifying+8+retinal+diseases+from+OCT+scans.;Transfer+Learning+%2B+Fine-Tuning+%2B+Grad-CAM.;ResNet50+%7C+EfficientNetV2+%7C+DenseNet121.;Final+Project+%E2%80%94+Universitas+Muhammadiyah+Riau."/>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/Momoru2002/retinal-oct-classification/blob/main/Perbandingan_Kinerja_ResNet50%2C_EfficientNetV2%2C_dan_DenseNet121_untuk_Klasifikasi_Penyakit_Retina_pada_Citra_Optical_Coherence_Tomography_Menggunakan_Explainable_AI.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-3B82F6?style=flat-square">
</p>

<br>

## 📌 Tentang Proyek

Proyek ini membandingkan performa tiga arsitektur *Convolutional Neural Network* — **ResNet50**, **EfficientNetV2S**, dan **DenseNet121** — untuk mengklasifikasikan **8 jenis kondisi retina** dari citra **Optical Coherence Tomography (OCT)**. Setiap model dilatih menggunakan pendekatan **Transfer Learning** (ImageNet) yang kemudian di-*fine-tune*, dan hasil prediksinya diinterpretasikan menggunakan **Explainable AI (Grad-CAM)** agar keputusan model dapat divisualisasikan secara medis, bukan sekadar *black-box*.

> Dikerjakan sebagai bagian dari riset terapan AI di bidang **Medical Image Classification**, oleh **Muhammad Almas Albirra Hamid (NIM 230401193)** — Teknik Informatika, Universitas Muhammadiyah Riau.

<br>

## 🧬 Dataset

| Detail | Keterangan |
|---|---|
| Sumber | [Kaggle — `obulisainaren/retinal-oct-c8`](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8) |
| Nama | RetinalOCT-C8 |
| Jumlah Kelas | 8 |
| Kelas | `AMD` `CNV` `CSR` `DME` `DR` `DRUSEN` `MH` `NORMAL` |
| Ukuran Input | 224 × 224 × 3 |
| Split | Train / Validation / Test |

<br>

## 🧠 Metodologi

```text
1. Load & preprocessing dataset OCT (image_dataset_from_directory)
2. Transfer Learning — base model frozen (ImageNet weights)
   ├── ResNet50
   ├── EfficientNetV2S
   └── DenseNet121
3. Fine-Tuning — unfreeze sebagian layer, learning rate rendah (1e-5)
4. Evaluasi — accuracy, precision, recall, f1-score, confusion matrix
5. Explainable AI — Grad-CAM heatmap untuk interpretasi prediksi
```

**Head classifier** yang digunakan sama di ketiga model, untuk menjaga perbandingan tetap *fair*:

```
Input(224,224,3) → preprocess_input → base_model (frozen) 
→ GlobalAveragePooling2D → Dropout(0.3) → Dense(256, relu) → Dense(8, softmax)
```

Callback yang dipakai selama training: `EarlyStopping` (monitor `val_accuracy`) dan `ReduceLROnPlateau` (monitor `val_loss`).

<br>

## 📊 Hasil Perbandingan Model

| Model | Akurasi Baseline (Frozen) | Akurasi Setelah Fine-Tuning | Δ Peningkatan |
|---|:---:|:---:|:---:|
| **ResNet50** | 89.39% | 🏆 **94.32%** | +4.93% |
| EfficientNetV2S | 89.79% | 93.46% | +3.67% |
| DenseNet121 | 87.82% | 91.68% | +3.86% |

> **ResNet50** menghasilkan performa terbaik setelah fine-tuning, dengan akurasi uji **94.32%** pada 2.800 citra test (350 citra/kelas).

<details>
<summary><b>📄 Classification Report — ResNet50 (Fine-Tuned)</b></summary>

```
              precision    recall  f1-score   support

         AMD       1.00      1.00      1.00       350
         CNV       0.88      0.90      0.89       350
         CSR       1.00      1.00      1.00       350
         DME       0.90      0.93      0.91       350
          DR       1.00      1.00      1.00       350
      DRUSEN       0.90      0.77      0.83       350
          MH       1.00      1.00      1.00       350
      NORMAL       0.88      0.95      0.91       350

    accuracy                           0.94      2800
   macro avg       0.94      0.94      0.94      2800
weighted avg       0.94      0.94      0.94      2800
```

</details>

Kelas dengan tantangan klasifikasi tertinggi secara konsisten di ketiga model adalah **`DRUSEN`**, **`CNV`**, dan **`NORMAL`** — hal ini wajar mengingat kemiripan visual antar kondisi tersebut pada citra OCT.

<br>

## 🔍 Explainable AI — Grad-CAM

Untuk memastikan model tidak hanya "menghafal" pola, melainkan benar-benar fokus pada area retina yang relevan secara klinis, setiap prediksi divisualisasikan menggunakan **Grad-CAM** terhadap layer konvolusi terakhir dari base model:

```
Original Image  →  Grad-CAM Heatmap  →  Overlay (Superimposed)
```

Heatmap ini menyoroti region yang paling berkontribusi terhadap keputusan klasifikasi model — langkah penting untuk membangun kepercayaan pada model AI di domain medis.

<br>

## ⚙️ Tech Stack

<img src="https://skillicons.dev/icons?i=python,tensorflow,opencv,sklearn"/>

`TensorFlow` `Keras` `OpenCV` `Scikit-Learn` `NumPy` `Matplotlib` `Seaborn` `Google Colab`

<br>

## 🚀 Cara Menjalankan

1. Buka notebook langsung di Google Colab lewat badge **"Open In Colab"** di atas.
2. Siapkan **Kaggle API Token** (`kaggle.json`) untuk mengunduh dataset:
   ```bash
   pip install kaggle -q
   kaggle datasets download -d obulisainaren/retinal-oct-c8
   unzip -o retinal-oct-c8.zip
   ```
3. Jalankan notebook secara berurutan dari atas ke bawah — mulai dari load dataset, training ketiga model, evaluasi, hingga visualisasi Grad-CAM.
4. Model hasil training akan tersimpan dalam format `.keras` (`resnet50_retina.keras`, `efficientnetv2_finetune.keras`, `densenet121_finetune.keras`).

<br>

## 📁 Struktur Repo

```
retinal-oct-classification/
├── Perbandingan_Kinerja_ResNet50,_EfficientNetV2,_dan_DenseNet121_...ipynb
└── README.md
```

<br>

## 👤 Author

<p align="center">
  <b>Muhammad Almas Albirra Hamid</b><br>
  NIM 230401193 · Teknik Informatika · Universitas Muhammadiyah Riau
</p>

<p align="center">
  <a href="mailto:seriesmomoru@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white"></a>
  <a href="https://www.linkedin.com/in/muhammad-almas-albirra-hamid-3812202b8"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white"></a>
  <a href="https://github.com/Momoru2002"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"></a>
</p>

<br>

<p align="center">
  <i>"A model isn't explainable until it earns clinical trust — that's what Grad-CAM is for."</i>
</p>
