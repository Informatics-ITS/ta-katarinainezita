# 🏁 Tugas Akhir (TA) - Final Project

**Nama Mahasiswa**: Katarina Inezita Prambudi  
**NRP**: 5025211148  
**Judul TA**: Segmentasi Citra Tulang Alveolar dan Kanal Mandibula pada Citra CBCT Menggunakan Metode YOLOv9   
**Dosen Pembimbing**: Prof. Dr. Eng. Chastine Fatichah, S.Kom., M.Kom.  
**Dosen Ko-pembimbing**: Dini Adni Navastara, S.Kom., M.Sc.


## 📺 Demo Aplikasi

[![Demo Aplikasi](https://github.com/user-attachments/assets/a42668a3-00ba-4d59-a177-c63a6c69ce02)](https://www.youtube.com/watch?v=-g9b4dRYBMc)
*Klik gambar di atas untuk menonton demo*



---

## 🛠 Panduan Instalasi & Menjalankan Software

### Prasyarat

* Python 3.10 atau lebih baru
* pip (Python package manager)
* (Opsional) virtual environment

### Langkah-langkah

1. **Clone Repository**

   ```bash
   git clone https://github.com/katarinainezita/TA-Segmentasi-YOLOv9.git
   cd TA-Segmentasi-YOLOv9
   ```

2. **Instalasi Dependensi**

   ```bash
   pip install -r requirements.txt
   ```

3. **Menjalankan Notebook Eksperimen**
   Buka Jupyter Notebook lalu pilih salah satu dari folder `notebooks/`:

   * YOLOv8 Medium
   * YOLOv9 Compact
   * YOLOv9 Extensive
   * YOLOv9 Extensive with HE
   * YOLOv9 Extensive with KFOLD

4. **Menjalankan Aplikasi**
   Jika aplikasi menggunakan Streamlit:

   ```bash
   streamlit run app.py
   ```

---

## 🧪 Contoh Dataset untuk Uji Coba Aplikasi

Untuk keperluan pengujian aplikasi, tersedia folder `dataset_sample/` yang berisi contoh citra CBCT yang dapat digunakan untuk segmentasi menggunakan model YOLOv9.

---

## 📊 Hasil Evaluasi

Hasil evaluasi dari semua model dan skenario disimpan dalam:

📄 `outputs/model_evaluation_results.xlsx`

---

## 🔍 Model Terlatih

Model YOLOv9 terbaik disimpan di: 

🔗 [Download model best.pt (Google Drive)](https://drive.google.com/drive/folders/1vpLnA6EA_z33_rAwR4fb-S8ptaLLqzNC?usp=sharing)

---


## 📌 Informasi Dataset

Dataset CBCT digunakan dalam tugas akhir ini bersumber dari Roboflow dan telah dianotasi oleh dokter gigi ahli. Seluruh data digunakan hanya untuk tujuan akademik dan penelitian.

---

## ⁉️ Pertanyaan?

Silakan hubungi:

* Penulis: [5025211148@student.its.ac.id](mailto:5025211148@student.its.ac.id)
* Pembimbing: [chastine@if.its.ac.id](mailto:chastine@if.its.ac.id)

---

## 📄 Lisensi

MIT License — silakan lihat file `LICENSE` untuk informasi selengkapnya.
