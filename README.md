# Project 1 - Deteksi Mobil dan Motor (dataset_patrol)

Nama / NIM : ...
Kelas      : car, motorcycle (2 kelas)
Model      : YOLO26n (Ultralytics)

## Urutan program
| Program | Fungsi |
|---|---|
| `00_siapkan_dataset_yolo.py` | dataset_patrol.zip -> `dataset_yolo/` (gambar, label, classes.txt) |
| `01_cek_anotasi.py` | memeriksa label dan menggambar kotaknya di `cek_visual/` |
| `02_split_dataset_yolo.py` | membagi train 70% / val 20% / test 10% -> `dataset_v1/` |
| `03_latih_bandingkan.py` | melatih E1, E2, E3, memilih yang terbaik (val), menguji sekali (test) |

Program 01-03 berasal dari program praktik dosen (hanya nama file yang disesuaikan).

## Cara menjalankan (dari folder Project1)
```
pip install ultralytics
python 00_siapkan_dataset_yolo.py
python 01_cek_anotasi.py dataset_yolo
python 02_split_dataset_yolo.py
python 03_latih_bandingkan.py --nim <NIM> --nama "<Nama>"
```

## Catatan label
Dataset awal berisi gambar tanpa label. Label dibuat otomatis oleh model YOLO COCO,
kemudian diperiksa dan diperbaiki: (isi: berapa gambar dikoreksi, alat yang dipakai).

## Eksperimen
| Eksperimen | Pengaturan |
|---|---|
| E1_pretrained | transfer learning (bobot COCO), pengaturan dasar |
| E2_scratch | arsitektur sama, bobot acak |
| E3_tuning | E1 + satu perubahan: imgsz=640 |

## Hasil
(salin dari hasil_praktik_<NIM>.txt)

## Analisis
- E1 vs E2 (efek transfer learning): ...
- E1 vs E3 (efek tuning): ...
- Model terpilih dan hasil di data test: ...
- Kelemahan / contoh gagal: ...
