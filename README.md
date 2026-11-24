# 🌿 AgriConnect — Deteksi & Solusi Penyakit Daun Cabai

Sistem pakar diagnostik penyakit daun tanaman cabai (*Capsicum annuum L.*) berbasis **Deep Learning MobileNetV2 Transfer Learning**, **Explainable AI (Grad-CAM)**, serta **Guardrail Validasi Non-Daun (Out-of-Distribution Filter)** dengan antarmuka modern terinspirasi desain pertanian presisi **AgriConnect**.

---

## 🎨 Desain Antarmuka Modern & Natural (AgriConnect Theme)

1. **Editorial Agricultural Aesthetics**:
   - Skema warna alami: *Warm Ivory Canvas* (`#FBFBFA`), *Deep Forest Evergreen* (`#173628`), *Leaf Green* (`#2D5F45`), dan aksen sage lembut (`#E9F1EB`).
   - Tipografi elegan kombinasi font serif editorial (*Playfair Display*) untuk judul utama dan sans-serif bersih (*Plus Jakarta Sans*) untuk kenyamanan membaca.
   - **Bebas dari ikon sci-fi / AI**: Seluruh visual menggunakan simbol botani dan pertanian alami (🌿, 🌱, 🍃, 🌾, 🔬, 🛡️, 📋).
2. **Komponen Visual Berstandar Tinggi**:
   - **Top Navigation Bar**: Brand logo botani, menu tautan navigasi, dan tombol pill aksi.
   - **Hero Section**: Tagline *"Empowering Farmers. Growing Tomorrow."* dengan kartu showcase diagnostik dan *floating glassmorphic crop health score*.
   - **4 Feature Cards**: Solusi manajemen penyakit, visualisasi Grad-CAM, guardrail anti-salah tebak, dan rekomendasi agronomi terpadu.
   - **Evergreen Statistics Strip**: Statistik performa (Akurasi Uji 94.22%, 6 Kelas Daun, MobileNetV2, Latensi < 50ms).
   - **Studio Deteksi Bersih**: Layout kartu modern dengan perbandingan berdampingan, warning card terracotta untuk citra non-daun, dan tab ensiklopedia.

---

## 🚀 Fitur Fungsional Unggulan

1. **Akurasi & Penyesuaian Model Presisi**:
   - Memuat file bobot terlatih `models/mobilenetv2_chili.pth` secara langsung dan akurat.
   - Konfigurasi checkpoint: normalisasi ImageNet, resolusi masukan $224 \times 224$, dropout $0.3$, dan 6 kelas daun cabai.
2. **🛡️ Filter Validasi Non-Daun (Anti-Tebakan Asal / OOD Detector)**:
   - Mencegah model menebak sembarangan ketika gambar yang dimasukkan adalah wajah manusia, hewan, kendaraan, benda mati, atau latar belakang kosong.
   - Menggunakan analisis spektrum klorofil/jaringan daun (HSV), filter morfologi spasial, deteksi warna non-daun dominan (biru/ungu), serta variasi tekstur helai daun.
   - Menampilkan kartu peringatan tegas dan edukatif ketika citra yang diunggah tidak valid.
3. **Visualisasi Explainable AI (Grad-CAM)**:
   - Menghasilkan heatmap atensi visual dari lapisan konvolusi terakhir (`features.18`).
   - Fitur interaktif pengaturan transparansi heatmap ($\alpha$) dan pilihan skema warna (JET, TURBO, VIRIDIS, INFERNO).
   - Tampilan perbandingan berdampingan: Gambar Asli, Heatmap Grad-CAM, dan Hasil Overlay.
4. **Multi-Input Gambar Fleksibel**:
   - **Unggah File**: Mendukung format JPG, JPEG, PNG, dan WEBP.
   - **Kamera Langsung**: Ambil foto daun secara *real-time* via webcam/kamera laptop.
   - **Sampel Uji Siap Pakai**: Termasuk sampel daun sehat, daun terinfeksi bercak/defisiensi, serta **sampel uji non-daun** untuk menguji ketahanan filter guardrail.
5. **Diagnosis Agronomi Komprehensif**:
   - Status penyakit & tingkat risiko (Aman, Sedang, Tinggi, Sangat Tinggi).
   - Indikasi gejala visual dan faktor patogen/penyebab.
   - Rekomendasi penanganan kimiawi, pengendalian organik/hayati, serta langkah pencegahan preventif.
   - Fitur ekspor/unduh laporan hasil diagnosis (.txt).
6. **Ensiklopedia Penyakit & Katalog Teknis**:
   - Panduan lengkap mengenai 6 kelas kondisi daun cabai.
   - Spesifikasi teknis arsitektur model dan pemetaan indeks kelas.

---

## 🏷️ 6 Kelas Daun Cabai yang Dideteksi

| Indeks | Nama Kelas (Model) | Nama Bahasa Indonesia | Kategori Patogen / Kondisi |
| :---: | :--- | :--- | :--- |
| **0** | `Bacterial Spot` | Bercak Bakteri | Bakteri (*Xanthomonas campestris pv. vesicatoria*) |
| **1** | `Cercospora Leaf Spot` | Bercak Daun Cercospora (Mata Katak) | Jamur / Fungi (*Cercospora capsici*) |
| **2** | `Curl Virus` | Virus Keriting (Gemini Virus / Bulai) | Virus (*Begomovirus* / ChiLCV) & Kutu Kebul |
| **3** | `Healthy Leaf` | Daun Sehat | Tanaman Normal & Prima |
| **4** | `Nutrition Deficiency` | Defisiensi Nutrisi | Gangguan Fisiologis / Kekurangan Hara (N/P/K/Mg/Ca) |
| **5** | `White spot` | Bercak Putih | Jamur / Nekrosis Jaringan Daun |

---

## 📁 Struktur Direktori

```text
leaf_deases_detection/
├── assets/
│   └── samples/                  # Gambar sampel daun & non-daun untuk uji instan
│       ├── sampel_daun_sehat.jpg
│       ├── sampel_bercak_daun.jpg
│       ├── sampel_defisiensi_hara.jpg
│       ├── sampel_bercak_bakteri.jpg
│       ├── sampel_non_daun_benda.jpg
│       └── sampel_non_daun_wajah.jpg
├── models/
│   └── mobilenetv2_chili.pth     # Model bobot PyTorch terlatih (MobileNetV2)
├── utils/
│   ├── class_names.json          # Daftar 6 nama kelas terstandarisasi
│   ├── disease_info.py           # Basis pengetahuan agronomi, gejala & solusi
│   ├── gradcam.py                # Implementasi Grad-CAM & Heatmap Overlay
│   ├── leaf_validator.py         # Guardrail validasi citra daun vs non-daun (OOD)
│   ├── model_loader.py           # Loader checkpoint model & ekstraksi metadata
│   ├── predict.py                # Pipeline inferensi & ranking probabilitas
│   └── preprocess.py             # Normalisasi & transformasi tensor PyTorch
├── app.py                        # Aplikasi antarmuka modern Streamlit (AgriConnect Theme)
├── requirements.txt              # Daftar dependensi library Python
└── README.md                     # Dokumentasi proyek
```

---

## 💻 Cara Menjalankan Aplikasi

Pastikan terminal berada di direktori proyek, lalu jalankan:

```bash
streamlit run app.py
```

Aplikasi dapat langsung diakses melalui browser di alamat:
👉 **`http://localhost:8501`**
