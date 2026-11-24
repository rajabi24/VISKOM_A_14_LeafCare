# 🌿 LeafCare — Klasifikasi Kondisi Daun Cabai

LeafCare merupakan aplikasi berbasis web untuk membantu mengklasifikasikan kondisi daun cabai (*Capsicum annuum L.*) berdasarkan citra. Aplikasi ini menggunakan pendekatan *Deep Learning* dengan arsitektur **MobileNetV2** dan teknik *Transfer Learning*.

Selain menampilkan hasil klasifikasi, LeafCare menyediakan visualisasi **Grad-CAM** untuk membantu pengguna memahami bagian citra yang menjadi perhatian model. Aplikasi juga dilengkapi fitur validasi awal citra untuk menyaring masukan yang tidak sesuai dengan karakteristik citra daun.

LeafCare dikembangkan sebagai proyek mata kuliah Visi Komputer dengan menggabungkan klasifikasi citra, interpretasi hasil model, dan penyediaan informasi mengenai kondisi daun cabai.

---

## 🌱 Fitur Utama

### 1. Klasifikasi Kondisi Daun Cabai
- Menggunakan model MobileNetV2 yang telah dilatih dengan dataset citra daun cabai.
- Mengklasifikasikan citra ke dalam enam kategori kondisi daun.
- Menampilkan kelas prediksi dan tingkat kepercayaan (*confidence score*).
- Menampilkan probabilitas prediksi untuk setiap kelas.

### 2. Validasi Awal Citra
- Memeriksa karakteristik citra sebelum diteruskan ke proses klasifikasi.
- Menggunakan pemeriksaan berbasis karakteristik warna, tekstur, dan area citra.
- Memberikan peringatan apabila citra tidak memenuhi kriteria masukan yang ditentukan.

**Catatan:** Validasi ini menggunakan pendekatan heuristik, bukan model deteksi objek atau model OOD yang dilatih secara khusus. Oleh karena itu, fitur ini merupakan penyaringan awal dan tidak menjamin seluruh citra non-daun dapat dikenali.

### 3. Visualisasi Grad-CAM
- Menghasilkan *heatmap* berdasarkan aktivasi fitur dan gradien pada model MobileNetV2.
- Menggunakan lapisan fitur terakhir model sebagai target visualisasi.
- Menampilkan gambar asli, *heatmap*, dan hasil *overlay*.
- Menyediakan pilihan skema warna JET, TURBO, VIRIDIS, dan INFERNO.
- Memungkinkan pengguna mengatur transparansi visualisasi.

Grad-CAM digunakan sebagai alat bantu interpretasi untuk melihat area citra yang menjadi perhatian model, bukan sebagai bukti pasti penyebab suatu penyakit.

### 4. Beragam Sumber Masukan Citra
- **Unggah gambar:** Mendukung format JPG, JPEG, PNG, dan WEBP.
- **Kamera:** Mengambil gambar melalui kamera yang tersedia pada perangkat.
- **Gambar sampel:** Menggunakan gambar contoh yang disediakan aplikasi.

### 5. Informasi Kondisi Daun
Aplikasi menyediakan informasi tambahan berdasarkan kelas hasil klasifikasi, meliputi:
- Gejala yang dapat diamati.
- Informasi penyebab atau faktor terkait.
- Rekomendasi penanganan.
- Langkah pencegahan.

Informasi tersebut disediakan sebagai referensi awal dan bukan pengganti pemeriksaan langsung oleh tenaga ahli pertanian.

### 6. Fitur Pendukung
- Tampilan peringkat tiga prediksi teratas (*Top-3 predictions*).
- Informasi waktu inferensi.
- Ensiklopedia enam kategori kondisi daun.
- Ekspor hasil klasifikasi dalam format TXT.

---

## 🏷️ Kategori Klasifikasi

Model LeafCare memiliki enam kelas keluaran sebagai berikut:

| Indeks | Nama Kelas Model | Keterangan |
|:---:|---|---|
| 0 | `Bacterial Spot` | Bercak bakteri |
| 1 | `Cercospora Leaf Spot` | Bercak daun Cercospora |
| 2 | `Curl Virus` | Gejala keriting daun |
| 3 | `Healthy Leaf` | Daun sehat |
| 4 | `Nutrition Deficiency` | Kekurangan nutrisi |
| 5 | `White spot` | Bercak putih |

---

## 🧠 Teknologi dan Metode

LeafCare dikembangkan menggunakan beberapa teknologi dan metode berikut:

| Komponen | Teknologi atau Metode |
|---|---|
| Bahasa pemrograman | Python |
| Antarmuka aplikasi | Streamlit |
| Pengolahan citra | PIL dan OpenCV |
| Framework Deep Learning | PyTorch |
| Arsitektur model | MobileNetV2 |
| Metode pengembangan model | Transfer Learning dan Fine-Tuning |
| Visualisasi interpretasi | Grad-CAM |
| Evaluasi model | Accuracy, Precision, Recall, F1-score, dan Confusion Matrix |

### Konfigurasi Model
- Arsitektur: MobileNetV2.
- Bobot awal: ImageNet V1.
- Ukuran masukan: 224 × 224 piksel.
- Jumlah kelas keluaran: 6 kelas.
- *Dropout*: 0,3.
- Normalisasi: menggunakan nilai *mean* dan *standard deviation* ImageNet.

### Hasil Evaluasi Model

Berdasarkan pengujian pada *testing set* yang terdiri dari 267 citra, model memperoleh hasil:

| Metrik | Hasil |
|---|---:|
| Accuracy | 99,25% |
| Macro Precision | 98,87% |
| Macro Recall | 98,57% |
| Macro F1-score | 98,71% |
| Balanced Accuracy | 98,57% |

Hasil tersebut merupakan evaluasi model pada dataset pengujian yang digunakan dalam eksperimen, bukan jaminan tingkat akurasi pada seluruh kondisi citra di lingkungan nyata.

---

## 📁 Struktur Proyek

Struktur direktori utama LeafCare terdiri atas aplikasi, modul pendukung, file model, dan konfigurasi kelas. Sesuaikan susunan direktori berikut dengan struktur aktual proyek:

```text
leaf_deases_detection/
├── app.py
├── model_loader.py
├── preprocess.py
├── predict.py
├── gradcam.py
├── leaf_validator.py
├── disease_info.py
├── models/
│   └── mobilenetv2_chili.pth
├── utils/
│   └── class_names.json
├── assets/
│   └── samples/
├── requirements.txt
└── README.md
```

Keterangan:
- `app.py`: Menjalankan antarmuka aplikasi berbasis web.
- `model_loader.py`: Memuat model dan bobot hasil pelatihan.
- `preprocess.py`: Melakukan prapemrosesan citra.
- `predict.py`: Menjalankan proses prediksi kelas.
- `gradcam.py`: Menghasilkan visualisasi Grad-CAM.
- `leaf_validator.py`: Melakukan validasi awal citra.
- `disease_info.py`: Menyediakan informasi gejala dan rekomendasi penanganan.
- `class_names.json`: Menyimpan daftar nama kelas model.
- `mobilenetv2_chili.pth`: File bobot model hasil pelatihan.
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan.

---

## 💻 Instalasi dan Cara Menjalankan

### 1. Persiapan
Pastikan perangkat telah memiliki:
- Python.
- pip.
- Git (opsional).

### 2. Masuk ke Direktori Proyek

Buka terminal, kemudian masuk ke direktori proyek:

```bash
cd leaf_deases_detection
```

### 3. Membuat Virtual Environment (Opsional)

```bash
python -m venv venv
```

Aktifkan virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Instalasi Dependensi

Jalankan perintah berikut:

```bash
pip install -r requirements.txt
```

### 5. Menjalankan Aplikasi

Setelah dependensi terpasang, jalankan:

```bash
streamlit run app.py
```

Aplikasi kemudian dapat dibuka melalui browser pada alamat lokal:

```text
http://localhost:8501
```

---

## 📸 Cara Menggunakan Aplikasi

1. Buka aplikasi LeafCare melalui browser.
2. Pilih sumber gambar melalui fitur unggah gambar, kamera, atau sampel yang tersedia.
3. Masukkan citra daun cabai yang ingin diperiksa.
4. Tunggu proses validasi dan klasifikasi citra.
5. Amati kategori hasil prediksi beserta nilai kepercayaannya.
6. Periksa probabilitas kelas dan visualisasi Grad-CAM.
7. Baca informasi gejala serta rekomendasi penanganan yang tersedia.
8. Gunakan fitur ekspor apabila ingin menyimpan hasil klasifikasi dalam format TXT.

---

## ⚠️ Batasan Sistem

LeafCare memiliki beberapa batasan yang perlu diperhatikan:
- Model dikembangkan berdasarkan enam kategori citra daun cabai pada dataset yang digunakan.
- Hasil klasifikasi dipengaruhi oleh kualitas, pencahayaan, sudut pengambilan, dan karakteristik citra masukan.
- Validasi citra merupakan pemeriksaan awal berbasis heuristik, bukan deteksi non-daun yang sempurna.
- Grad-CAM hanya memberikan visualisasi area yang menjadi perhatian model dan tidak membuktikan penyebab penyakit.
- Informasi penanganan merupakan referensi awal, sehingga keputusan penanganan tetap perlu mempertimbangkan kondisi tanaman dan saran ahli pertanian.

---

## 📚 Sumber Dataset

Dataset yang digunakan dalam pengembangan model berasal dari:

**Image Dataset on Chili Leaf Diseases in the Krishna River Basin of the Deccan Plateau, India**

Publikasi: Malghan et al. (2026).

Mendeley Data, Version 3.

DOI: https://doi.org/10.17632/ymt8k9bjkn.3

---

## 🎓 Informasi Proyek

**Nama Proyek:** LeafCare  
**Bidang:** Visi Komputer  
**Metode:** Deep Learning – MobileNetV2 Transfer Learning  
**Interpretabilitas:** Grad-CAM  
**Platform:** Aplikasi Berbasis Web  
