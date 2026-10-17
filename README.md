# Project-1 YOLO

**Nur Hafidzatun Nabila** | NIM 4222401012 | RET503 Computer Vision dan Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hafizatunabila/Project-1-Computer-Vision-and-Deep-Learning/blob/main/Project1_503_4222401012_Hafiza.ipynb)

## Tentang project

Project ini melatih model YOLO26n untuk mendeteksi **mobil (car)** dan **motor (motorcycle)** pada foto, sebagai bagian dari kebutuhan Security Patrol Robot. Model diuji dengan tiga eksperimen untuk melihat pengaruh transfer learning dan pengaturan ukuran gambar.

| Eksperimen | Pengaturan |
|---|---|
| E1 (pretrained) | Transfer learning dari bobot COCO, `imgsz=416` |
| E2 (scratch) | Tanpa transfer learning, bobot dimulai dari awal, `imgsz=416` |
| E3 (tuning) | Seperti E1, dengan satu perubahan: `imgsz=640` |

Pengaturan yang sama untuk ketiganya: 30 epoch, batch 16, seed 0.

## Dataset

- 114 foto dari internet (mobil 54, motor 60). Sebanyak 103 foto berlabel dan dipakai, 11 foto dilewati karena tidak memiliki kotak objek.
- Label awal dibuat secara otomatis dengan model YOLO COCO, lalu dicek dengan `01_cek_anotasi.py` (tidak ditemukan masalah teknis). Hasil perlu dibaca dengan catatan ini.
- Pembagian data: train 72, val 20, test 11 foto.

## Hasil

Nilai pada data validasi:

| Eksperimen | mAP50 | mAP50-95 | Precision | Recall | Waktu latih |
|---|---|---|---|---|---|
| E1 pretrained | 0,978 | 0,900 | 0,964 | 0,952 | 8,6 menit |
| E2 scratch | 0,144 | 0,041 | 0,165 | 0,150 | 7,9 menit |
| E3 tuning (`imgsz=640`) | 0,995 | 0,932 | 0,991 | 1,000 | 19,6 menit |

Model terpilih adalah **E3** (mAP50-95 validasi tertinggi). Hasil pada data test (diuji sekali): **mAP50 0,900 dan mAP50-95 0,745**.

Ringkasan:

- **Transfer learning sangat membantu.** E1 jauh lebih baik daripada E2. Dengan data yang sedikit (72 foto latih), model yang dilatih dari awal belum sempat belajar dengan baik.
- **Ukuran gambar 640 memberi kenaikan kecil.** E3 sedikit lebih tinggi daripada E1 (0,932 dibanding 0,900), tetapi waktu latihnya sekitar 2,3 kali lebih lama.
- **Mohon dibaca dengan hati-hati.** Data validasi hanya 20 foto dan data test hanya 11 foto, sehingga selisih kecil antar eksperimen belum tentu bermakna. Label awal yang dibuat otomatis juga dapat membuat angka terlihat lebih tinggi.

## Letak hasil dan file di repo ini

| Isi | Lokasi |
|---|---|
| Notebook Colab (`.ipynb`) | [`Project1_503_4222401012_Hafiza.ipynb`](Project1_503_4222401012_Hafiza.ipynb) |
| Tabel hasil (`.txt`) | [`hasil_praktik_4222401012.txt`](hasil_praktik_4222401012.txt) |
| Tabel hasil (`.csv`) | [`hasil_praktik_4222401012.csv`](hasil_praktik_4222401012.csv) |
| Grafik pelatihan E1 (`results.png`) | [`hasil_praktik/E1_pretrained/results.png`](hasil_praktik/E1_pretrained/results.png) |
| Grafik pelatihan E3 (`results.png`) | [`hasil_praktik/E3_tuning/results.png`](hasil_praktik/E3_tuning/results.png) |
| Hasil lengkap E1 (confusion matrix, `results.csv`, `weights/best.pt`) | [`hasil_praktik/E1_pretrained/`](hasil_praktik/E1_pretrained/) |
| Hasil lengkap E3 | [`hasil_praktik/E3_tuning/`](hasil_praktik/E3_tuning/) |
| Hasil E2 (pembanding) | [`hasil_praktik/E2_scratch/`](hasil_praktik/E2_scratch/) |
| Dataset siap latih (train/val/test dan `data.yaml`) | [`dataset_v1/`](dataset_v1/) |
| Semua foto dan label sebelum dibagi | [`dataset_yolo/`](dataset_yolo/) |
| Foto contoh dengan kotak label | [`dataset_yolo/cek_visual/`](dataset_yolo/cek_visual/) |
| Foto asli | [`dataset_patrol/`](dataset_patrol/) |

## Program

| Program | Fungsi |
|---|---|
| `00_siapkan_dataset_yolo.py` | Mengubah foto menjadi `dataset_yolo/` (dengan label otomatis) |
| `01_cek_anotasi.py` | Memeriksa label dan membuat gambar kotak di `cek_visual/` |
| `02_split_dataset_yolo.py` | Membagi data train 70%, val 20%, test 10% menjadi `dataset_v1/` |
| `03_latih_bandingkan.py` | Melatih E1, E2, E3, memilih model terbaik, lalu menguji sekali di data test |

Program 01 sampai 03 berasal dari program praktik dosen. Hanya nama filenya yang disesuaikan.

## Cara menjalankan

**Google Colab:** klik tombol *Open In Colab* di atas, aktifkan GPU (Runtime > Change runtime type > T4 GPU), lalu jalankan semua sel dari atas ke bawah.

**Di komputer sendiri** (dari folder repo):

```
pip install ultralytics
python 00_siapkan_dataset_yolo.py
python 01_cek_anotasi.py dataset_yolo
python 02_split_dataset_yolo.py
python 03_latih_bandingkan.py --nim 4222401012 --nama "Nur Hafidzatun Nabila"
```
