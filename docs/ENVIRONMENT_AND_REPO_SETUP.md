# Panduan Setup Environment, Struktur Repositori, dan Verifikasi Awal

Dokumen ini menjadi panduan praktis langkah-demi-langkah bagi seluruh anggota tim untuk menyiapkan mesin pengembangan secara seragam, menjaga kebersihan repositori publik, serta memverifikasi aset pertandingan sebelum mulai menulis modul produksi.

---

## 1. Prasyarat Hardware dan Software

1. **Python:** Disarankan **Python 3.10 atau 3.11** (64-bit). Catatan dari [README.md](../README.md): evaluator resmi teruji identik pada Python 3.8 dan 3.11, sementara notebook starter (`requirements-starter.txt`) bermasalah pada Python 3.8 di Windows karena error build `pywinpty` sehingga Python 3.9+ / 3.10 / 3.11 lebih aman.
2. **Paket Sistem:**
   - Git
   - FFmpeg (opsional untuk rendering animasi/inspeksi cepat video, namun sangat membantu bersama OpenCV).
   - GPU NVIDIA (CUDA) sangat direkomendasikan untuk mempercepat inferensi deteksi bola CV pada ~158.857 frame gameplay video taktis 1920x1080 (CPU tetap bisa dijalankan secara chunked namun jauh lebih lambat).
3. **Kapasitas Disk:** Siapkan ruang kosong memadai untuk bundle video & tracking `data/` (di luar git) serta folder `cache/` lokal untuk menyimpan file `.parquet` (jangan pernah menyimpan ekstrak gambar frame ke disk).

---

## 2. Pengamanan `.gitignore` Sebelum Mulai Bekerja (Gate P0)

Aturan [RULES.md](../RULES.md) dan [SUBMISSION.md](../SUBMISSION.md) melarang keras redistribusi aset pertandingan dan data turunan yang dapat merekonstruksi data pertandingan ke repositori GitHub publik.

Saat ini `.gitignore` bawaan sudah mengecualikan `data/*`, `evaluation/results/*.csv` (kecuali `baseline_*.csv`), `evaluation/reports/`, `evaluation/scoreboard.csv`, `*.mp4`, dan `*.pdf`. Sebelum anggota tim menghasilkan file cache atau artefak eksperimen baru, **tambahkan aturan berikut ke `.gitignore`**:

```gitignore
# -- derived local caches and pipeline outputs (MUST NOT be pushed) --
cache/
outputs/
*.parquet
*.feather
*.npy
*.npz
*.pkl
*.pt
*.pth
*.onnx
```

*(Catatan: jika tim ingin melampirkan dokumen teknis `<team_name>.md` atau `.pdf`, perhatikan bahwa `*.pdf` saat ini di-ignore oleh `.gitignore` bawaan. Upload file `<team_name>.pdf` dilakukan langsung lewat Google Form submission; sementara di repositori publik, simpan ringkasan dokumen teknis dalam format `.md` atau dokumentasi di `README.md` / `docs/`.)*

---

## 3. Setup Virtual Environment dan Pemisahan Dependency

Sesuai strategi di [requirements.txt](../requirements.txt) dan bagian 20 pada [PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md](../PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md), pertahankan file asli [requirements.txt](../requirements.txt) (`numpy`, `pandas`) atau pisahkan dengan jelas agar evaluator tetap ringan dan bebas konflik.

### Opsi A: Virtual Environment Lokal (`venv` atau `uv`)

Pada Windows PowerShell:

```powershell
# 1. Buat dan aktifkan virtual environment (misal Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Pasang dependency resmi evaluator & starter notebooks
pip install -r requirements-starter.txt

# 3. Verifikasi evaluator resmi berjalan sebelum mengunduh data
python evaluation/run_evaluation.py
```

### Pemisahan Dependency Pipeline CV/ML (`requirements-pipeline.txt`)

Untuk tahap pengembangan pipeline tim, buat file `requirements-pipeline.txt` terpisah dengan **versi terkunci (pinned)** agar saat panitia melakukan reproduksi, seluruh versi library tercatat persis. Contoh komponen yang akan di-pin saat modul diimplementasikan:

```text
-r requirements.txt
opencv-python-headless==<pinned_version>
scipy==<pinned_version>
pyarrow==<pinned_version>
pyyaml==<pinned_version>
pytest==<pinned_version>
# Tambahkan library CV/ML yang benar-benar dipakai di pipeline final, misal:
# torch==<pinned_version>
# ultralytics==<pinned_version>
# filterpy==<pinned_version>
# scikit-learn==<pinned_version>
# lightgbm==<pinned_version>
# hmmlearn==<pinned_version>
```

**Penting:** Hanya masukkan paket yang benar-benar dipanggil oleh `run.py` saat final submission agar waktu instalasi reproduksi panitia tetap singkat dan bersih. Instruksi instalasi di `README.md` nanti wajib menyebutkan perintah tepat (misal `pip install -r requirements-pipeline.txt`).

---

## 4. Penempatan Data Pertandingan dan Verifikasi Awal (Gate P0 & P1)

### 4.1 Lokasi Bundle Data

Ekstrak bundle data resmi dari link registrasi ke folder `data/` (atau simpan di drive terpisah dan set environment variable `MTAH_DATA_DIR`):

```powershell
# Jika data disimpan di luar folder repo (misal D:\datasets\mtah2027):
$env:MTAH_DATA_DIR = "D:\datasets\mtah2027"
```

Struktur folder yang diharapkan oleh [games.py](../games.py):

```text
data/
├── tactical/
│   ├── BAR-ATH_tactical.mp4
│   ├── BAR-ATH_tactical_atd.xml
│   ├── BAR-ATH_tactical_atd.txt
│   ├── BAR-ATH_tactical_homography.csv
│   └── BAR-ATH_tactical_smart_tagging.xml
└── broadcast/
    ├── BAR-ATH_broadcast.mp4
    ├── BAR-ATH_broadcast_atd.xml
    ├── BAR-ATH_broadcast_atd.txt
    └── BAR-ATH_broadcast_homography.csv
```

### 4.2 Perintah Verifikasi Aset dan Alignment (Wajib Lulus di Hari ke-1)

Setelah data tersedia, jalankan seluruh pemeriksaan bawaan ini secara berurutan:

```powershell
# 1. Cek seluruh file tactical tersedia dan dapat ditemukan games.py
python -c "from games import require; print(require(['video','atd_meta','atd_raw','homography','smart_tagging']))"

# 2. Cek alignment ATD 1..199319 (streaming cepat tanpa kloppy penuh)
python -c "import sys; sys.path.insert(0, 'starter'); import load_atd; print(load_atd.check_alignment_raw())"

# 3. Cek alignment dan statistik homography (tactical & broadcast)
python starter/load_homography.py

# 4. Cek Smart Tagging time-base terhadap kedua kick-off
python starter/load_smart_tagging.py

# 5. Cek jumlah frame video taktis terbaca OpenCV tepat 199,319
python -c "import cv2; from games import video_path; cap = cv2.VideoCapture(video_path()); print('frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 'fps:', cap.get(cv2.CAP_PROP_FPS))"

# 6. Regenerasi 4 baseline resmi dan jalankan evaluator publik
python starter/baseline.py
python evaluation/run_evaluation.py
python evaluation/validate_submission.py evaluation/results/baseline_smooth50.csv
```

---

## 5. Struktur Target Repositori Proyek

Berikut struktur target repositori untuk fase implementasi (dikembangkan dari Bagian 11 [PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md](../PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md), dengan penyesuaian agar folder resmi `evaluation/`, `starter/`, `games.py`, dan `data/` tetap utuh):

```text
more-than-a-hack-2027/
├── data/                         # GITIGNORED (kecuali data/README.md)
├── cache/                        # GITIGNORED (derived parquet & metadata)
├── outputs/                      # GITIGNORED (working output CSVs)
├── docs/                         # Dokumentasi PRD, arsitektur, rencana, kepatuhan
├── configs/
│   ├── cv.yaml
│   ├── fusion.yaml
│   ├── possession.yaml
│   ├── temporal.yaml
│   └── final.yaml
├── evaluation/                   # ASLI RESMI (JANGAN DIMODIFIKASI)
│   ├── gt/
│   ├── results/                  # GITIGNORED kecuali baseline_*.csv
│   ├── possession_metrics.py
│   ├── run_evaluation.py
│   └── validate_submission.py
├── starter/                      # ASLI RESMI (Referensi & loader helpers)
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── config.py             # Loader & validator YAML
│   │   └── determinism.py        # Seed locking & SHA-256 manifest
│   ├── tracking/
│   │   ├── atd_loader.py         # Parser raw ATD (frame_id - 1) & XML teams
│   │   └── smart_tagging.py      # Parser XML, normalisasi 'Atletic Club', polaritas
│   ├── video/
│   │   └── decoder.py            # Streaming chunked video reader
│   ├── cv/
│   │   ├── ball_detector.py      # Pretrained YOLO/RT-DETR + ROI/tiling + filter
│   │   └── player_tracker.py     # Opsional (ADV-3): ByteTrack + jersey cluster
│   ├── geometry/
│   │   └── homography.py         # Wrapper load_homography + unproject -> meter
│   ├── fusion/
│   │   └── kalman_fusion.py      # 2D Kalman + RTS smoother + ATD/CV confidence fusion
│   ├── features/
│   │   └── feature_builder.py    # Pembentukan features.parquet per video_frame
│   ├── possession/
│   │   ├── dead_ball.py          # Dead-ball engine (boundary, stillness, set-piece)
│   │   ├── control_model.py      # Estimator kontrol H/A -> P(H), P(A), P(D)
│   │   └── abstention.py         # Opsional kebijakan label X yang aman
│   ├── temporal/
│   │   ├── state_machine.py      # Baseline transisi berhisteresis
│   │   └── hmm_viterbi.py        # 3-State HMM / Viterbi decoding per babak
│   ├── diagnostics/
│   │   ├── audit_data.py         # Laporan missingness ATD & Smart Tagging
│   │   └── non_gt_eval.py        # Benchmark synthetic ball-masking & spell stats
│   └── pipeline.py               # Orkestrator end-to-end
├── scripts/
│   ├── run_ablation.py           # Menjalankan eksperimen E0..E8 terstruktur
│   └── check_reproducibility.py  # Uji dua run identik + validate_submission
├── tests/
│   ├── test_alignment.py
│   ├── test_homography.py
│   ├── test_smart_tagging.py
│   ├── test_kalman_fusion.py
│   └── test_temporal_decoder.py
├── games.py                      # ASLI RESMI
├── requirements.txt              # ASLI RESMI (numpy, pandas)
├── requirements-starter.txt      # ASLI RESMI
├── requirements-pipeline.txt     # Dependency pipeline tim (versi ter-pin)
├── run.py                        # Entrypoint CLI tunggal untuk reproduksi submission
└── README.md                     # Diperbarui di akhir dengan instruksi reproduksi tim
```

**Catatan Penting:** Jangan mengubah isi `evaluation/*.py` atau `games.py`. Seluruh logika tim berada di `src/`, `configs/`, `scripts/`, `tests/`, dan `run.py`.

