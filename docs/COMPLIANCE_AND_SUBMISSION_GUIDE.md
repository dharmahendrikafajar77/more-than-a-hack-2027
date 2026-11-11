# Panduan Kepatuhan Aturan, Keamanan Repositori, dan Final Submission

Dokumen ini memastikan tim terhindar dari seluruh penyebab diskualifikasi atau hasil gagal-nilai (skipped submission) berdasarkan [RULES.md](../RULES.md), [SUBMISSION.md](../SUBMISSION.md), [OVERVIEW.md](../OVERVIEW.md), dan [FAQ.md](../FAQ.md).

---

## 1. Matriks Kepatuhan Aturan Kompetisi (`RULES.md` & `OVERVIEW.md`)

| Aturan Resmi | Diperbolehkan | Dilarang Keras (Risiko Diskualifikasi / Nilai 0) | Mekanisme Pengendalian di Proyek |
| --- | --- | --- | --- |
| **1. Sumber Label Prediksi** | Seluruh 199.319 label dihasilkan secara otomatis oleh kode (`CV`, `tracking`, `geometry`, `ML`, `HMM`). Pseudo-label algoritmik tanpa campur tangan keputusan manusia di frame tertentu diperbolehkan. | Menonton video lalu mengoreksi label per frame, per segmen, atau di titik transisi; memperbaiki posisi bola atau ID switch ATD secara manual; menulis rule/konstanta berbasis nomor frame spesifik hasil pengamatan (misal `if frame == 113742`). | Seluruh konstanta berada di `configs/*.yaml` dan berlaku umum secara fisika/taktis. Tidak ada edit manual atas output CSV. |
| **2. Penggunaan Public 5-Minute GT** | Mengecek keselarasan frame, makna label `H`/`A`/`D`/`X`, validitas format CSV, serta mendiagnosis bug konseptual pada pipeline end-to-end. | Melatih model, memilih smoothing window, menyetel threshold, atau memilih versi model berdasarkan nilai `spellF1` / `macroF1` di public GT (`11185..18684`). | Evaluasi utama memakai uji synthetic ball-masking dan diagnostik non-GT di full match (`EXECUTION_PLAN_AND_EXPERIMENTS.md`). |
| **3. Data Eksternal dan Game Lain** | Menggunakan dataset publik atau data berlabel manual dari **pertandingan lain** selama sumber, jumlah, siapa pelabelnya, dan cara aksesnya dijelaskan di dokumen teknis. | Menggunakan private GT `BAR-ATH`, atau data buatan manusia tentang laga FC Barcelona vs Athletic Club ini (seperti log event Opta, StatsBomb, Wyscout, LaLiga, atau anotasi pihak ketiga). | Jika memakai model ML/detector fine-tuned, catat seluruh dataset eksternal yang berasal dari game lain pada tabel transparansi aset. |
| **4. Penggunaan LLM** | Memakai LLM sebagai asisten penulisan kode/dokumentasi (tanpa mengirim data lomba), atau memakai model open-weights yang dijalankan lokal bila relevan. | **Mengunggah, membocorkan, atau mengirimkan video, dataset ATD/homography/Smart Tagging, maupun cuplikannya ke API LLM cloud pihak ketiga** (ChatGPT, Claude API, Gemini API hosted, dll.). | Jalur inferensi tidak menggunakan LLM sama sekali. Tim dilarang keras mengunggah file dari `data/` ke layanan AI cloud mana pun. |
| **5. Model & Library Publik** | Menggunakan model pretrained publik (YOLO, RT-DETR, dll.) atau library open-source apa pun selama panitia dapat mengakses dan menjalankannya. | Bergantung pada model internal privat/berbayar tertutup atau dataset tertutup yang tidak bisa diakses penyelenggara. | Dokumentasikan nama model, versi/checkpoint exact, dan cara unduh otomatis/resmi di `README.md` serta dokumen teknis. |
| **6. Keamanan Data & Repo Publik** | Mempublikasikan kode sumber (`src/`, `configs/`, `scripts/`, `README.md`, `requirements*.txt`) di GitHub publik. | Meng-commit folder `data/`, video `.mp4`, file `cache/`, hasil CSV non-baseline, atau data frame-level turunan yang dapat merekonstruksi data pertandingan ke repo publik. | Perkuat `.gitignore` sebelum mulai bekerja dan lakukan audit `git ls-files` sebelum submission. |
| **7. Reproducibility** | Menjalankan satu perintah CLI menghasilkan ulang file CSV yang sama persis dengan file yang diunggah ke Google Form. | Mengirim CSV yang tidak bisa diregenerasi persis oleh kode di repositori publik tim. | Uji regenerasi 2 kali dengan verifikasi hash SHA-256 identik dan uji instalasi di virtual environment bersih. |

---

## 2. Protokol Audit Kebersihan Repositori Publik (Sebelum Push & Submit)

Jalankan perintah pemeriksaan berikut di PowerShell sebelum melakukan `git push` final dan mengisi formulir submission:

```powershell
# 1. Pastikan tidak ada file di folder data/ kecuali README.md yang terlacak git
git ls-files data/

# 2. Pastikan tidak ada file cache, outputs, parquet, video, atau pdf privat terlacak git
git ls-files cache/ outputs/
git ls-files *.mp4 *.parquet *.pkl *.csv | Where-Object { $_ -notmatch "evaluation/gt/BAR-ATH_possession_gt_public.csv" -and $_ -notmatch "evaluation/results/baseline_.*\.csv" }

# 3. Pastikan skrip evaluator resmi tidak termodifikasi secara tidak sengaja
git diff evaluation/possession_metrics.py evaluation/run_evaluation.py evaluation/validate_submission.py games.py
```

**Kriteria Lulus:**
- Perintah #1 hanya menampilkan `data/README.md`.
- Perintah #2 tidak mengembalikan satu file pun (kosong).
- Perintah #3 tidak menunjukkan perubahan pada file evaluator resmi dan `games.py`.

---

## 3. Verifikasi Kontrak CSV dan Reproducibility Final

Sebelum mengunggah `<team_name>.csv` ke Google Form, jalankan urutan verifikasi ini pada **file eksak yang akan diunggah**:

```powershell
# 1. Jalankan ulang pipeline dari konfigurasi final untuk menghasilkan <team_name>.csv
python run.py --config configs/final.yaml --output outputs/<team_name>.csv

# 2. Jalankan run kedua ke file pembanding untuk memastikan determinisme 100%
python run.py --config configs/final.yaml --output outputs/<team_name>_verify.csv

# 3. Bandingkan SHA-256 kedua file (harus identik persis)
Get-FileHash outputs/<team_name>.csv, outputs/<team_name>_verify.csv -Algorithm SHA256

# 4. Jalankan validator resmi kompetisi pada file yang akan diunggah
python evaluation/validate_submission.py outputs/<team_name>.csv
```

**Checklist Output `validate_submission.py` yang Wajib Diperiksa:**
- Status dimulai dengan `OK`.
- Tertulis tepat `199319 rows, frame 0..199318`.
- Tidak ada peringatan error format, missing column, atau blank cells.
- Jumlah prediksi possession (`reports X possessions of mean Y.Y s`) dan distribusi persentase `H` / `A` / `D` / `X` terlihat wajar (tidak ratusan/ribuan fragmen pendek dan tidak konstan 100% satu label).

---

## 4. Template Resmi Dokumen Teknis (`<team_name>.md` / `<team_name>.pdf`)

Berdasarkan [SUBMISSION.md](../SUBMISSION.md), deliverable ke-3 adalah dokumen teknis singkat (beberapa halaman, format PDF atau Markdown, dinamai **`<team_name>.pdf`** atau **`<team_name>.md`**) yang wajib menjawab enam komponen inti. Gunakan kerangka di bawah ini untuk menyusun dokumen final tim:

```markdown
# Technical Report: <Team Name>
## More Than a Hack 2027 - Phase 1: Frame-Level Ball Possession

**Team Name:** <Nama Tim Sesuai Registrasi>  
**Members:** <Daftar 3-5 Anggota Terdaftar>  
**Repository URL:** <URL GitHub Publik>  

---

### 1. Approach Overview and Rationale
- Ringkasan arsitektur multimodal: Tactical Video (CV Ball Recovery) + ATD Tracking + Inverse Homography + Smart Tagging Priors + Kalman/RTS Fusion + Dead-Ball/Control Engine + Per-Half HMM/Viterbi Decoding.
- Alasan pemilihan metode dibandingkan smoothing frame sederhana (mengatasi missing ball 41%, memisahkan dead-ball dari occlusion, serta mengoptimalkan kontinuitas interval `spellF1` bukan hanya akurasi per frame).

### 2. Assets Used, Modified, or Dropped
| Asset | Status | Cara Penggunaan / Alasan Drop |
| --- | --- | --- |
| `BAR-ATH_tactical.mp4` | Digunakan | Ekstraksi kandidat bola visual via streaming chunked decoder. |
| `BAR-ATH_tactical_atd.txt/.xml` | Digunakan | Posisi 46 tracklet tim dan observasi awal bola (`video_frame = frame_id - 1`, skala 105.3 x 68.0 m). |
| `BAR-ATH_tactical_homography.csv` | Digunakan | Proyeksi balik piksel CV ke koordinat meter (`unproject`) dan kontrol batas lapangan (`valid`, `reprojection_error`). |
| `BAR-ATH_tactical_smart_tagging.xml` | Digunakan | Prior taktis `SET PIECES`, `GOALS`, dan fase dengan pemetaan polaritas ofensif/defensif serta penanganan `Team = N/A` pada throw-in. |
| `BAR-ATH_broadcast.*` | [Dipakai / Di-drop] | [Jelaskan alasan objektif, misal di-drop karena tidak tersinkronisasi dan 34.2% homography invalid sehingga berisiko menambah error alignment dalam durasi sprint.] |

### 3. Pretrained Models and External Training Data
*(Wajib diisi secara jujur dan dapat diverifikasi oleh penyelenggara)*
- **Pretrained Models:** [Nama model, versi/checkpoint exact, misal `yolo...`, sumber resmi/URL unduhan publik, serta apakah dijalankan off-the-shelf atau fine-tuned.]
- **External Data (Other Matches Only):** [Jika tidak ada training eksternal, nyatakan eksplisit: "No external labelled match data was used for training." Jika ada, sebutkan nama dataset game lain, lisensi/akses publik, siapa yang melabeli, dan jumlah frame/klipnya. Tegaskan 0 label manusia berasal dari laga `BAR-ATH`.]

### 4. Possession and Dead-Ball Rules Implemented
- **Dead-Ball (`D`) Conditions:** Bola di luar dimensi lapangan (105.3 x 68.0 m + margin), penurunan kecepatan kolektif pemain saat stoppage, window set-piece Smart Tagging, dan deteksi kondisi pra-kickoff algoritmik (termasuk awal babak kedua tanpa hardcoding frame).
- **Ball Imputation & Occlusion Handling:** Fusi Kalman/RTS dengan gating fisika antara ATD dan CV; kebijakan mempertahankan kepemilikan aktif saat bola hilang sesaat di dalam lapangan tanpa langsung memvonis `D`.
- **Controlled Possession & Turnover:** Radius kontrol terhadap bola, kecepatan relatif, dwell consistency untuk mencegah turnover palsu akibat defleksi/sentuhan sesaat, serta Viterbi transition penalty per babak.
- **Abstention (`X`) Policy:** Penggunaan `X` pada frame di luar rentang gameplay (`0..11184`, `90652..110807`, `190198..199318`) dan kebijakan pada frame dinilai.

### 5. Validation Strategy, Ablations, and Limitations
- **Validation Without Overfitting Public GT:** Penjelasan uji synthetic ball-masking dan diagnostik non-GT full match; penegasan bahwa 5-minute public GT hanya dipakai untuk verifikasi alignment & format.
- **Ablation Summary:** Ringkasan hasil E0 hingga E6+ pada metrik proksi dan diagnostik.
- **Known Limitations & Future Work:** Kasus tersisa di mana sistem masih lemah (misal oklusi berkepanjangan di kotak penalti, keterbatasan tracklet ATD tanpa re-ID pemain tetap, potensi integrasi multi-kamera broadcast dan HSMM duration model).

### 6. Reproduction Instructions
1. Clone repository publik.
2. Unpack data resmi ke `./data` atau set `MTAH_DATA_DIR`.
3. Install dependencies: `pip install -r requirements-pipeline.txt`.
4. Jalankan satu perintah deterministik:
   `python run.py --config configs/final.yaml --output evaluation/results/<team_name>.csv`
5. Verifikasi format:
   `python evaluation/validate_submission.py evaluation/results/<team_name>.csv`
```

---

## 5. Checklist Final Sebelum Submit Google Form (Deadline: 17 November 2026)

- [ ] **Eligibilitas Tim:** 3 hingga 5 anggota terdaftar lengkap; `<team_name>` pada form sama persis dengan registrasi.
- [ ] **Akun Google:** Anggota pengirim sudah login akun Google yang aktif untuk unggahan Google Form (karena perlu upload file `.csv` ~1.9 MB dan `.pdf`/`.md`).
- [ ] **Deliverable 1 (`<team_name>.csv`):** Lulus `python evaluation/validate_submission.py`, tepat 199.319 baris (`0..199318`), diregenerasi dari kode terbaru yang ada di repo publik.
- [ ] **Deliverable 2 (Public Repo URL):** URL repositori publik dapat diakses secara anonim (cek mode incognito browser!), berisi kode lengkap + pinned `requirements-pipeline.txt` + `README.md` instruksi reproduksi, dan **100% bersih** dari `data/`, `cache/`, `outputs/`, dan file non-baseline lainnya.
- [ ] **Deliverable 3 (`<team_name>.pdf` atau `<team_name>.md`):** Menjawab seluruh 6 bagian wajib dari [SUBMISSION.md](../SUBMISSION.md) tanpa klaim yang tidak dapat diverifikasi.
- [ ] **Buffer Waktu:** Submit lebih awal sebelum batas akhir **17 November 2026** (ingat Google Form hanya mengizinkan satu respons yang dapat diedit; respons terakhir sebelum deadline yang akan dinilai).

