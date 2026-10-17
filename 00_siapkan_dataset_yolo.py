"""Mengubah dataset_patrol.zip (folder car/ dan motorcycle/) menjadi dataset_yolo/ untuk program 01-03.

Hasil (struktur sama dengan keluaran 00_ambil_subset_coco.py milik dosen):
    dataset_yolo/
    ├── classes.txt        # car, motorcycle
    ├── images/
    └── labels/

Label:
  - Jika di samping gambar sudah ada file .txt (label YOLO), label itu yang dipakai.
  - Jika belum ada, label dibuat OTOMATIS oleh model YOLO COCO (yolo26n.pt). Label otomatis ini
    HARUS diperiksa dengan 01_cek_anotasi.py dan diperbaiki sebelum hasil dipakai untuk laporan.
  - Gambar yang tidak terdeteksi apa pun tidak dimasukkan; salinannya ada di perlu_label_manual/.

Pemakaian (jalankan dari folder Project1, di samping dataset_patrol.zip):
    python 00_siapkan_dataset_yolo.py            # --timpa untuk membuat ulang dataset_yolo/
Lanjutan  : python 01_cek_anotasi.py dataset_yolo
"""
import argparse
import shutil
import zipfile
from pathlib import Path

ZIP = Path("dataset_patrol.zip")
RAW = Path("raw")
OUT = Path("dataset_yolo")
MANUAL = Path("perlu_label_manual")
KELAS = ["car", "motorcycle"]              # urutan = class_id 0, 1
COCO_ID = {"car": 2, "motorcycle": 3}      # id kelas yang sama di COCO
CONF = 0.25
EXT = {".jpg", ".jpeg", ".png"}

p = argparse.ArgumentParser()
p.add_argument("--timpa", action="store_true", help="hapus dataset_yolo/ lama dan buat ulang")
args = p.parse_args()

if OUT.exists() and not args.timpa:
    raise SystemExit(f"{OUT}/ sudah ada (mungkin sudah kamu perbaiki labelnya), jadi tidak ditimpa.\n"
                     "Hapus foldernya atau jalankan dengan --timpa jika memang ingin membuat ulang.")
if not ZIP.exists() and not RAW.exists():
    raise SystemExit(f"{ZIP} tidak ditemukan. Taruh di folder yang sama dengan skrip ini.")
if not RAW.exists():
    print(f"Mengekstrak {ZIP} ...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW)


def kelas_dari_path(path: Path):
    for bagian in path.relative_to(RAW).parts:
        if bagian.lower() in KELAS:
            return bagian.lower()
    return None


gambar = sorted(f for f in RAW.rglob("*") if f.suffix.lower() in EXT and kelas_dari_path(f))
ada_label_asli = any(f.with_suffix(".txt").exists() for f in gambar)
model = None
if not ada_label_asli:
    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")             # hanya untuk membuat label otomatis

if OUT.exists():
    shutil.rmtree(OUT)
if MANUAL.exists():
    shutil.rmtree(MANUAL)
(OUT / "images").mkdir(parents=True)
(OUT / "labels").mkdir(parents=True)
(OUT / "classes.txt").write_text("\n".join(KELAS) + "\n", encoding="utf-8")

hit_gambar = {k: 0 for k in KELAS}
hit_kotak = {k: 0 for k in KELAS}
dari_asli = otomatis = 0
tanpa_kotak = []
for f in gambar:
    k = kelas_dari_path(f)
    sumber = f.with_suffix(".txt")
    if sumber.exists():
        baris = [b for b in sumber.read_text().splitlines() if b.strip()]
        asli = True
    else:
        r = model.predict(str(f), conf=CONF, verbose=False)[0]
        baris = [f"{KELAS.index(k)} " + " ".join(f"{v:.6f}" for v in box)
                 for box, c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist())
                 if int(c) == COCO_ID[k]]
        asli = False
    nama = f"{k}_{f.stem}"
    if not baris:
        MANUAL.mkdir(exist_ok=True)
        shutil.copy(f, MANUAL / (nama + f.suffix.lower()))
        tanpa_kotak.append(nama + f.suffix.lower())
        continue
    shutil.copy(f, OUT / "images" / (nama + f.suffix.lower()))
    (OUT / "labels" / (nama + ".txt")).write_text("\n".join(baris) + "\n")
    hit_gambar[k] += 1
    hit_kotak[k] += len(baris)
    dari_asli, otomatis = dari_asli + asli, otomatis + (not asli)

(OUT / "CATATAN_LABEL.txt").write_text(
    f"Gambar berlabel asli      : {dari_asli}\n"
    f"Gambar berlabel otomatis  : {otomatis} (model yolo26n.pt/COCO, conf {CONF})\n"
    f"Gambar tanpa kotak        : {len(tanpa_kotak)} (tidak dipakai)\n"
    "Label otomatis hanya memuat objek yang sesuai nama folder gambar (mis. folder motorcycle -> motor saja),\n"
    "sehingga kendaraan lain di latar bisa tidak berlabel. Periksa dengan 01_cek_anotasi.py.\n", encoding="utf-8")

print(f"Kelas           : {', '.join(f'{i} = {k}' for i, k in enumerate(KELAS))}")
print(f"Gambar dipakai  : " + ", ".join(f"{k} {n}" for k, n in hit_gambar.items()))
print(f"Jumlah kotak    : " + ", ".join(f"{k} {n}" for k, n in hit_kotak.items()))
print(f"Label asli / otomatis : {dari_asli} / {otomatis}")
if tanpa_kotak:
    print(f"{len(tanpa_kotak)} gambar tanpa kotak -> salinan di {MANUAL}/ (beri label manual jika mau dipakai)")
for sisa in ("dataset_v1", "hasil_praktik"):
    if Path(sisa).exists():
        print(f"[!] Folder lama '{sisa}/' masih ada. Hapus dulu sebelum menjalankan 02 dan 03 agar tidak bercampur.")
if otomatis:
    print("\nPERHATIAN: sebagian label dibuat otomatis. Periksa dan perbaiki dulu sebelum dipakai untuk laporan.")
print(f"Siap di {OUT}/ -> lanjutkan dengan: python 01_cek_anotasi.py {OUT}")
