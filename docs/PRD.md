# Product Requirements Document (PRD): Sistem Inferensi Kepemilikan Bola Algoritmik dan Computer Vision

**Proyek:** More Than a Hack 2027 (Phase 1: Ball Possession)  
**Penyelenggara:** Metrica Sports, FC Barcelona (Barça Innovation Hub), Mobile World Capital Barcelona  
**Pertandingan Target:** FC Barcelona vs Athletic Club (`BAR-ATH`), LaLiga EA Sports 2024-25 Matchday 2  
**Tanggal Dokumen:** 10 November 2026  
**Tenggat Submission:** 17 November 2026 (Pengumuman hasil sebelum 26 November 2026)  
**Status:** Siap Eksekusi (Sprint 7 Hari)

---

## 1. Ringkasan Eksekutif

Kompetisi More Than a Hack 2027 menuntut tim menghasilkan prediksi status kepemilikan bola frame-demi-frame (`H`, `A`, `D`, atau `X`) untuk seluruh **199.319 frame** video taktis (25 FPS) pada laga FC Barcelona vs Athletic Club. Berbeda dari edisi 2026 yang dinilai juri secara kualitatif, edisi 2027 dinilai sepenuhnya secara otomatis menggunakan skrip [evaluation/run_evaluation.py](../evaluation/run_evaluation.py) terhadap ground truth privat sepanjang laga (158.857 frame dinilai). Metrik penentu peringkat utama adalah **`spellF1`** (rata-rata F1 pencocokan interval penguasaan bola 1-to-1 pada ambang IoU 0.3 hingga 0.7), dengan **`macroF1`** sebagai tie-break pertama.

Sistem yang dibangun mengintegrasikan **Computer Vision (CV) untuk pemulihan observasi bola dari video taktis**, **Automatic Tracking Data (ATD)** yang tersedia, **proyeksi geometri lapangan berbasis homografi**, **prior taktis dari Smart Tagging**, dan **dekoding temporal berbasis probabilistik/state model (HMM/Viterbi)** guna menghasilkan sequence possession yang utuh, minim fragmentasi, dan **100% reproducible melalui satu perintah CLI**. Seluruh jalur inferensi dirancang murni algoritmik tanpa koreksi manual dan tanpa mengirim aset pertandingan ke LLM cloud pihak ketiga.

---

## 2. Latar Belakang Masalah dan Tantangan Teknis

### 2.1 Mengapa Baseline Resmi Gagal Optimal

Repositori resmi menyediakan [starter/baseline.py](../starter/baseline.py) yang memetakan tim pemain terdekat ke bola sebagai pemilik bola, memberi label `D` setiap kali bola tidak terlihat di ATD, lalu melakukan majority vote smoothing. Hasil evaluasi resmi pada full match menunjukkan:

| Varian Baseline | macroF1 (Full Match) | spellF1 (Full Match) | Karakteristik |
| --- | --- | --- | --- |
| `baseline_raw` | 0.6532 | 0.0486 | Flicker parah; memecah penguasaan bola menjadi sangat banyak segmen pendek. |
| `baseline_smooth25` | **0.6752** | 0.1996 | macroF1 tertinggi di antara baseline, namun masih banyak transisi palsu. |
| **`baseline_smooth50`** | 0.6724 | **0.2433** | **Floor resmi yang wajib dikalahkan (`spellF1` 0.2433).** |
| `baseline_smooth125` | 0.6356 | 0.2242 | Smoothing terlalu lebar menghilangkan penguasaan pendek yang sah. |

Tiga akar masalah utama dari pendekatan baseline dan karakteristik dataset adalah:

1. **Bola hilang pada 40.7% frame dinilai (43.1% total video):** Menurut [DATA.md](../DATA.md), bola hanya terlacak pada 56.9% total frame (113.488 dari 199.319) dan 59.3% frame dinilai (94.214 dari 158.857). Bola sering hilang pada momen krusial: kotak penalti padat, duel udara, dan oklusi pemain. Menganggap bola hilang otomatis sebagai `D` menghancurkan kontinuitas penguasaan bola.
2. **Ketidaksinkronan metrik frame vs interval:** Sekitar 80%+ frame berada di tengah segmen stabil. Flickering 2 hingga 3 frame menurunkan `spellF1` secara drastis walau `macroF1` terlihat tinggi.
3. **Keterbatasan ATD dan Smart Tagging:** ATD hanya memiliki 46 slot tracklet tanpa identitas pemain tetap (`ShirtNumber` adalah indeks slot, rata-rata 20.5 objek terdeteksi, kadang melebihi 22 karena duplikasi) dan sering mengalami ID switch. Sementara itu, Smart Tagging (638 fase) bersifat otomatis dengan noise batas waktu sekitar 1 hingga 5 detik, dan seluruh 26 kejadian `Throw-in` bertag `Team = N/A`.

### 2.2 Rekonsiliasi Fakta Teknis dari Rancangan Awal terhadap Repositori Resmi

Audit menyeluruh terhadap seluruh file dokumentasi dan kode sumber di repositori ini menemukan **lima koreksi teknis kritis** yang wajib dipegang sebelum tim menulis kode, agar tidak terjadi bug fatal atau pelanggaran aturan:

| No | Asumsi di Draf Rancangan Awal | Fakta Resmi di Repositori | Keputusan Desain Wajib di PRD |
| --- | --- | --- | --- |
| 1 | Join ATD ke `video_frame` memakai offset per babak (`1702057` dan `1702058`) dari `games.video_to_id()`. | Di [DATA.md](../DATA.md) dan [starter/load_atd.py](../starter/load_atd.py), ATD tactical memiliki tepat `199.319` baris (`frame_id` `1..199319`) yang **sudah sejajar 1:1** dengan video taktis: **`video_frame = frame_id - 1`**. Offset di [games.py](../games.py) hanya memetakan ID internal pembuat ground truth ke video untuk mengetahui rentang frame dinilai via `half_video_blocks("BAR-ATH")` (`11185..90651` dan `110808..190197`). | **Wajib gunakan `video_frame = frame_id - 1` untuk membaca ATD tactical.** Gunakan `games.half_video_blocks("BAR-ATH")` hanya untuk membatasi pemrosesan temporal per babak dan menandai segmen dinilai vs tidak dinilai. |
| 2 | Proyeksi homografi digambarkan langsung membawa piksel `(u, v)` melalui `H(frame)` ke titik lapangan `(x_m, y_m)`. | Di [starter/load_homography.py](../starter/load_homography.py), matriks `H[frame]` memetakan **koordinat lapangan ternormalisasi `(0..1, 0..1)` ke piksel gambar `(u, v)`**. | Untuk mengubah deteksi CV piksel `(u, v)` menjadi koordinat lapangan, **wajib gunakan matriks invers** (`unproject(H[frame], u, v)` dengan `np.linalg.inv`), lalu kalikan dengan dimensi resmi `(105.3, 68.0)` meter dari `games.GAMES["BAR-ATH"]["pitch"]`. |
| 3 | Awal babak kedua diasumsikan dimulai langsung saat kick-off seperti babak pertama. | [FAQ.md](../FAQ.md) menegaskan babak kedua dinilai sejak frame `110.808`, padahal kick-off babak kedua baru terjadi di frame **`113.742`**. Sebanyak **2.934 frame** (`110808..113741`) masuk penilaian dan seluruhnya bernilai `D`. | Dead-ball engine **harus** mampu mendeteksi periode pra-kickoff babak kedua secara **algoritmik** (bola diam/hilang di sekitar titik tengah atau belum ada pergerakan permainan aktif, kecepatan kolektif pemain rendah/statis di posisi kickoff, ditambah tag `Kick Off (Start)`), **tanpa hardcoding nomor frame tertentu** (karena [RULES.md](../RULES.md) melarang konstanta manual yang terikat pada frame spesifik hasil pengamatan video). |
| 4 | Penyebutan `Athletic Club` pada Smart Tagging diasumsikan konsisten dengan metadata ATD. | [DATA.md](../DATA.md) mencatat nilai tag `Team` di XML Smart Tagging tertulis **`Atletic Club`** (tanpa huruf `h`), `Barcelona`, dan `N/A`. Kolom `Max Players in the box` bertipe string termasuk `"7+"`. | Parser Smart Tagging wajib menormalisasi string secara case-insensitive serta memetakan `"atletic club"` dan `"athletic club"` ke kelas `A`. |
| 5 | Penggunaan public 5-minute GT (`11185..18684`) berisiko dipakai memilih parameter smoothing. | [EVALUATION.md](../EVALUATION.md) membuktikan `baseline_smooth125` unggul di public 5 menit (`0.4242` vs `0.3256`), tetapi kalah di full match (`0.2242` vs `0.2433`). | **Dilarang keras** melakukan training, tuning threshold, atau pemilihan model berdasarkan skor `spellF1` di public 5-minute GT. Public GT hanya dipakai untuk verifikasi alignment, format, dan sanity check regresi. |

---

## 3. Tujuan Produk, Target Metrik, dan Non-Goals

### 3.1 Tujuan Produk (Goals)

1. **Mengalahkan floor resmi (`spellF1 = 0.2433`) dengan margin yang bermakna dan stabil** melalui pemulihan posisi bola berbasis Computer Vision, fusi Kalman, logika bola mati berbasis geometri lapangan, aturan kontrol kepemilikan bola, dan dekoding sequence HMM/Viterbi.
2. **Menuntaskan seluruh 3 deliverable wajib kompetisi sebelum 17 November 2026:**
   - File CSV prediksi `<team_name>.csv` (199.319 baris, lolos `validate_submission.py`).
   - Repositori kode publik dengan dependency terkunci dan eksekusi reproduksi satu perintah yang menghasilkan CSV identik.
   - Dokumen teknis ringkas (`<team_name>.pdf` / `.md`) sesuai struktur wajib [SUBMISSION.md](../SUBMISSION.md).
3. **Kepatuhan Penuh dan Reproducibility 100%:** Zero hand-labeling pada `BAR-ATH`, zero kebocoran data/cache ke repositori publik, dan zero pemanggilan LLM cloud pada data pertandingan.

### 3.2 Target Keberhasilan (Success Criteria)

Karena ground truth full match bersifat privat, target keberhasilan dibagi menjadi indikator kepatuhan, indikator validasi proksi (non-GT), dan indikator sanity check:

| Kategori | Indikator | Target Minimal (MVP) | Target Lanjutan |
| --- | --- | --- | --- |
| Format & Validasi | `evaluation/validate_submission.py` | Status `OK`, 199.319 baris `0..199318`, 0 cell kosong. | Sama. |
| Reproducibility | Regenerasi CSV dari command tunggal (`run.py`) | Output identik (SHA-256 hash sama antar eksekusi berurutan). | Berjalan bersih di environment baru tanpa cache manual. |
| Cakupan Bola | Persentase frame berposisi bola valid selama gameplay dinilai (`158.857` frame) | Meningkat dari baseline ATD (`59.3%`) menjadi `>= 90%` melalui fusi CV + Kalman terkalibrasi ketidakpastiannya. | `>= 95%` dengan residual error Kalman terkendali pada uji synthetic masking. |
| Kualitas Bola CV | Akurasi detektor bola CV pada frame uji di mana ATD bola valid dan tidak stasioner | Precision dan recall terukur pada subset frame clean ATD (tanpa hand-label) serta menurunkan gap panjang `> 1 s`. | Penambahan verifikasi visual pemain/tim bila audit menunjukkan ATD pemain bermasalah. |
| Konsistensi Temporal | Jumlah possession spell dan durasi rata-rata full match | Eliminasi fragmentasi ekstrem (`baseline_raw` menghasilkan ratusan/ribuan spell); durasi rata-rata dan rasio transisi wajar untuk sepak bola 90 menit tanpa tuning ke public GT. | Pemodelan durasi eksplisit (HSMM) bila waktu dan validasi mendukung. |
| Kepatuhan Aturan | Audit `RULES.md` dan `SUBMISSION.md` | 100% bebas data `data/`, bebas file cache di git, bebas hand-correction, bebas tuning pada public GT. | Dokumentasi teknis final siap unggah. |

### 3.3 Non-Goals (Di Luar Ruang Lingkup)

- Tidak membangun ulang multi-object player tracking atau jersey re-identification lengkap pada fase MVP selama ATD tracking pemain sudah memadai.
- Tidak membangun sistem pengenalan aksi/event end-to-end berbasis video dari nol untuk menggantikan Smart Tagging, kecuali ada kebutuhan terukur di fase stretch.
- Tidak menggunakan `BAR-ATH_broadcast.mp4` sebagai dependensi MVP karena tidak sinkron, memiliki banyak cut/replay, dan 34.2% matriks homografinya tidak valid.
- Tidak menyematkan LLM dalam jalur inferensi scoring.
- Tidak membuat GUI/aplikasi web interaktif (output sistem adalah pipeline CLI batch yang deterministik dan artefak diagnostik lokal).

---

## 4. Pengguna, Pemangku Kepentingan, dan Persyaratan Tim

### 4.1 Pengguna Sistem dan Pemangku Kepentingan

1. **Tim Peserta (3 hingga 5 Anggota Terdaftar):** Engineer Video/CV, Engineer Tracking/Geometri, Engineer Possession/Temporal, dan Engineer Evaluasi/Integrasi yang mengoperasikan, menguji, dan mengaudit pipeline secara kolaboratif.
2. **Penyelenggara Kompetisi (Metrica Sports, Barça Innovation Hub, MWCB):** Akan mengunduh repositori publik tim, memasang dependency dari `requirements.txt`, menyambungkan bundle data `BAR-ATH`, dan menjalankan ulang perintah reproduksi untuk memastikan CSV akhir identik dengan file yang diunggah, serta membaca dokumen teknis tim.

### 4.2 Persyaratan Eligibilitas Tim (`RULES.md` & `SUBMISSION.md`)

- Tim beranggotakan **3 hingga 5 orang** (tim kurang dari 3 orang tidak memenuhi syarat untuk Innovation Challenge dan Final Pitch).
- Nama tim (`<team_name>`) pada formulir registrasi, penamaan CSV (`<team_name>.csv`), dan dokumen teknis (`<team_name>.pdf` / `.md`) harus identik.
- Diperlukan akun Google aktif untuk mengisi Google Form submission sebelum 17 November 2026.

---

## 5. Kebutuhan Fungsional (Functional Requirements)

### FR-01: Ingesti Aset, Penyelarasan Timeline, dan Validasi Input
- Sistem wajib membaca seluruh jalur file secara dinamis melalui [games.py](../games.py) (`asset_paths()`, `require()`, `MTAH_DATA_DIR`) tanpa hardcode path lokal.
- Sistem wajib memverifikasi jumlah baris/frame sumber tactical (`199.319` frame, `25` FPS):
  - ATD tactical: `video_frame = frame_id - 1` (verifikasi `1..199319` kontigu).
  - Homography tactical: indeks `frame` `0..199318` dan cek flag `valid` serta `reprojection_error`.
  - Smart Tagging: konversi `round(seconds * 25)` menghasilkan interval `[start_frame, end_frame)`, dengan normalisasi nama tim (`Barcelona` -> `H`, `Atletic Club` / `Athletic Club` -> `A`, `N/A` -> `None`).
- Sistem wajib membaca batas babak dinilai melalui `games.half_video_blocks("BAR-ATH")` (`[(1, 11185, 90651), (2, 110808, 190197)]`) untuk memisahkan pemrosesan babak 1 dan babak 2 serta menjaga frame di luar dua blok tersebut tetap aman (misalnya diberi label `X`).

### FR-02: Dekoder Video Taktis Berbasis Streaming / Chunk
- Sistem wajib mendekode `BAR-ATH_tactical.mp4` secara streaming/chunk (misalnya per chunk 1.000 hingga 2.500 frame) menggunakan OpenCV/FFmpeg/PyAV agar tidak menyimpan ratusan ribu file gambar PNG/JPG ke disk.
- Setiap frame yang dibaca wajib terikat ke primary key `video_frame` (`0..199318`), dan sistem wajib memverifikasi total frame video terbaca sama dengan `199.319` (sesuai peringatan [FAQ.md](../FAQ.md), jangan memotong atau menambah padding jika jumlah frame dekode berbeda).
- Untuk efisiensi di laptop/GPU tim, sistem boleh melewati dekoding visual pada frame non-graded (`0..11184`, `90652..110807`, `190198..199318`) dengan mekanisme indeks frame yang tetap terverifikasi tepat 1:1.

### FR-03: Deteksi Bola Visual (CV Ball Recovery)
- Sistem wajib menjalankan detektor objek publik/pretrained (seperti keluarga Ultralytics YOLO atau RT-DETR, atau model yang dilatih pada dataset game lain yang legal dan terdokumentasi) pada frame video taktis (terutama atau penuh di rentang graded) untuk menghasilkan kandidat bounding box bola per frame.
- Kandidat deteksi wajib difilter secara algoritmik berdasarkan ukuran piksel wajar, skor confidence, proyeksi posisi di area lapangan/sekitarnya via `unproject()`, dan konsistensi lintasan terhadap frame sebelumnya.
- Output modul `cv_ball` per frame wajib mencatat minimal: `video_frame`, `u_px`, `v_px`, `pitch_x_m`, `pitch_y_m`, `cv_confidence`, `bbox_w`, `bbox_h`, dan status deteksi (`detected` / `missing`).
- Dilarang melakukan koreksi bounding box atau pemilihan parameter per-frame secara manual pada video `BAR-ATH`.

### FR-04: Pelacakan Temporal Bola dan Fusi ATD + CV Berbasis Confidence
- Sistem wajib menggabungkan observasi bola dari ATD (`ball_x`, `ball_y` yang sudah diskalakan ke `105.3 x 68.0` m) dengan observasi bola CV menggunakan **Kalman Filter** (state posisi dan kecepatan 2D `[x, y, vx, vy]`) dalam koordinat meter lapangan.
- Mekanisme fusi wajib menangani empat kondisi secara deterministik:
  1. **ATD valid & CV confident:** fusi berbobot berdasarkan reliabilitas/jarak inovasi Kalman dan kualitas homografi (`reprojection_error`).
  2. **Hanya ATD valid:** perbarui Kalman menggunakan observasi ATD (dengan gating outlier agar lonjakan noise ATD tersaring).
  3. **ATD hilang & CV confident (target utama menutup gap 41%):** perbarui Kalman menggunakan koordinat `unproject` CV.
  4. **Keduanya hilang:** lakukan prediksi Kalman (coasting) untuk gap pendek dengan menaikkan kovarians ketidakpastian (`uncertainty`); jika gap panjang, turunkan `ball_confidence` dan siapkan sinyal konteks tim/centroid pemain tanpa otomatis memvonis `D`.
- Output fusi (`fused_tracking`) wajib menyimpan `ball_x_m`, `ball_y_m`, `ball_vx`, `ball_vy`, `ball_speed_mps`, `ball_confidence`, `ball_source` (`both`, `atd`, `cv`, `kalman_pred`, `missing`), dan `gap_duration_frames`.

### FR-05: Proyeksi Geometri dan Feature Engineering Spasial-Kinematika
- Sistem wajib menskalakan seluruh posisi pemain ATD dari `0..1` ke meter (`105.3 x 68.0` m) sebelum menghitung jarak Euclidean.
- Fitur minimal per `video_frame` wajib mencakup:
  - **Relasi bola-pemain:** `nearest_home_dist`, `nearest_away_dist`, `dist_margin` (`away - home`), jumlah pemain `H` dan `A` dalam radius kontrol (misal 1.5 m, 2.5 m, 4.0 m), serta kecepatan relatif bola terhadap pemain terdekat.
  - **Struktur tim dan konteks:** posisi centroid tim `H` dan `A`, kekompakan/kepadatan di sekitar bola atau sekitar centroid bola saat hilang lama, jumlah tracklet aktif.
  - **Geometri batas lapangan:** jarak bola ke garis tepi (`touchline`) dan garis gawang (`goal_line`), serta flag `outside_pitch` dengan margin toleransi fisik.
  - **Kinematika bola dan pemain:** kecepatan & percepatan bola, rata-rata/persentil kecepatan 22 pemain (mendeteksi pemain berjalan santai saat stoppage vs berlari aktif saat live play), serta arah lintasan bola (mendeteksi perubahan arah tajam akibat kontrol/operan vs lintasan balistik).
  - **Konteks Smart Tagging:** flag biner/skor kedekatan terhadap `SET PIECES` (dibedakan jenisnya: `Throw-in`, `Corner Kick`, `Goal Kick`, `Free Kick`, `Kick Off`), fase penguasaan tim `H`/`A` (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, dll.), dan `GOALS`, selalu dipetakan ke indeks frame tanpa memperlakukannya sebagai kebenaran mutlak.

### FR-06: Dead-Ball Engine (`D`) dan Estimator Kontrol Kepemilikan (`H` / `A`)
- **Dead-Ball Engine** wajib menghasilkan skor/probabilitas `P(D)` per frame berdasarkan kombinasi bukti algoritmik:
  - Bola berada di luar lapangan secara meyakinkan (`x < -margin` atau `x > 105.3 + margin`, `y < -margin` atau `y > 68.0 + margin`).
  - Penurunan kecepatan kolektif pemain secara serentak disertai bola diam atau hilang berkepanjangan di dekat batas lapangan / area set-piece.
  - Konteks `SET PIECES` / `GOALS` dari Smart Tagging serta deteksi kondisi pra-kickoff (seperti periode `110808..113741` di awal babak kedua di mana permainan belum dimulai secara kinematika).
  - **Aturan ketat:** low ball visibility atau bola hilang sesaat di tengah lapangan **tidak boleh** langsung menghasilkan `P(D)` tinggi.
- **Controlled Possession Estimator** wajib membedakan kontrol nyata dari sekadar sentuhan sekilas/defleksi/duel udara sesuai definisi [CHALLENGE.md](../CHALLENGE.md):
  - Menghitung probabilitas/skor emisi `P(H)`, `P(A)`, `P(D)` per frame (melalui model probabilistik terkalibrasi, rule-based terstruktur, atau classifier ML seperti LightGBM/XGBoost/Logistic Regression yang dilatih hanya pada data legal dari game lain atau pseudo-label algoritmik yang bersih).
  - Mempertahankan kepemilikan tim sebelumnya ketika bola dalam kondisi *loose* atau *contested* selama belum terpenuhi bukti kontrol berkelanjutan (*dwell frames*, konsistensi kedekatan tim, atau perubahan arah/kecepatan bola yang terkendali oleh lawan).

### FR-07: Model Temporal dan Dekoding Sequence (HMM / Viterbi)
- Sistem wajib menerapkan dekoder sequence temporal (minimal State Machine berhisteresis pada baseline lanjutan dan **3-State HMM / HSMM + Viterbi decoding** sebagai model utama) pada masing-masing blok babak (`half 1` dan `half 2` diproses terpisah agar tidak menyeberangi halftime).
- Matriks transisi dan/atau model durasi wajib mencerminkan dinamika durasi possession sepak bola yang realistis sehingga menghilangkan fragmentasi/flickering antar frame sambil tetap responsif pada titik turnover sah dan transisi `D -> H/A` saat restart.

### FR-08: Kebijakan Label Abstention (`X`)
- Frame di luar blok babak dinilai (`0..11184`, `90652..110807`, `190198..199318`) aman diisi `X` (atau label apapun karena tidak dinilai oleh `scored_confusion` maupun `valid_runs`).
- Pada frame dinilai, penggunaan `X` bersifat opsional dan hanya diizinkan secara algoritmik melalui kebijakan eksplisit berbasis konteks:
  - Diperbolehkan mempertimbangkan `X` hanya jika ketidakpastian sangat tinggi **di tengah** spell yang sudah stabil (karena `fill_abstentions` di `spellF1` akan meneruskan label sebelumnya tanpa memutus spell).
  - **Dilarang** mengeluarkan `X` di sekitar kandidat transisi/turnover atau awal babak karena akan membawa label lama melewati pergantian penguasaan dan menurunkan `macroF1` sekaligus `spellF1`.
  - Pada MVP awal, prediksi pada frame dinilai memprioritaskan `{H, A, D}` secara penuh agar tidak kehilangan poin `macroF1` (tie-break) secara sia-sia.

### FR-09: Orkestrasi Satu Perintah, Validasi, dan Reproducibility
- Sistem wajib menyediakan entrypoint tunggal (misalnya `python run.py --config configs/final.yaml --output evaluation/results/<team_name>.csv`) yang menjalankan seluruh tahapan end-to-end atau memakai cache turunan lokal secara deterministik dan dapat meregenerasi file CSV yang **sama persis**.
- Sistem wajib memanggil atau lolos [evaluation/validate_submission.py](../evaluation/validate_submission.py) sebelum deklarasi sukses.
- Sistem wajib menyediakan skrip eksperimen/ablasi (`E0`..`E8`) yang mencetak metrik diagnostik non-GT dan sanity check evaluator publik secara terstruktur.

---

## 6. Kebutuhan Non-Fungsional (Non-Functional Requirements)

1. **Determinisme dan Reproducibility (Wajib Keras):**
   - Seluruh random seed (`random`, `numpy`, `torch`, `PYTHONHASHSEED`) dan mode deterministik framework wajib dikunci di `configs/*.yaml`.
   - Menjalankan command utama dari repository bersih dengan data `data/` yang sama wajib menghasilkan file CSV identik.
2. **Keamanan Data, Privasi, dan Kebersihan Repositori Publik:**
   - File `.gitignore` wajib memblokir `data/*`, `cache/`, `outputs/`, file video (`*.mp4`), file frame-level cache (`*.parquet`, `*.pkl`, `*.npy`, `*.csv` selain baseline resmi), serta model checkpoint besar yang tidak boleh dipublikasi atau harus diunduh dari sumber publik terdokumentasi.
   - Tidak ada satu pun potongan data video, frame, tracking, atau XML `BAR-ATH` yang diunggah ke layanan LLM eksternal/cloud.
3. **Keterpisahan Environment Evaluator:**
   - Jangan pernah memodifikasi [evaluation/possession_metrics.py](../evaluation/possession_metrics.py), [evaluation/run_evaluation.py](../evaluation/run_evaluation.py), atau [evaluation/validate_submission.py](../evaluation/validate_submission.py).
   - Pisahkan dependency minimal evaluator (`requirements.txt`: `numpy`, `pandas`) dari dependency pipeline CV/ML lengkap (di dokumen `requirements-pipeline.txt` atau `requirements.txt` terkelola dengan versi ter-pin jelas untuk reproduksi submission).
4. **Efisiensi Penyimpanan dan Komputasi:**
   - Pemrosesan 199.319 frame video 1080p tidak boleh mengekstrak gambar ke disk secara permanen (estimasi penghematan ratusan GB I/O). Gunakan pemrosesan chunked in-memory dan simpan hanya feature/detection table terkompresi (`.parquet`) di `cache/`.
   - Pipeline full inference harus dapat diselesaikan dalam waktu yang masuk akal pada mesin tim maupun mesin verifikasi penyelenggara (target praktis: deteksi CV + fusi + temporal inference dapat berjalan ulang secara otomatis dan terdokumentasi hardware-nya).
5. **Kejelasan Kode dan Pengujian Unit:**
   - Kode terstruktur modular di `src/` dengan unit test `pytest` minimal untuk: alignment `frame_id - 1`, `unproject()` homografi, parser Smart Tagging (termasuk `"Atletic Club"` dan `N/A`), Kalman filter step/gap handling, HMM Viterbi per-half tanpa menyeberangi halftime, dan validasi format 199.319 baris.

---

## 7. Ruang Lingkup Bertahap: MVP vs Advanced

Untuk menjamin tim memiliki submission legal dan unggul di atas `0.2433` sebelum tenggat **17 November 2026**, pengerjaan dibagi menjadi jalur **MVP Wajib** dan **Ekspansi Advanced (Hanya Jika MVP Selesai & Teruji)**:

| Tingkat | Komponen | Target Hasil | Status |
| --- | --- | --- | --- |
| **MVP-0** | Setup env, `games.py`, audit data, regenerasi `starter/baseline.py` + `run_evaluation.py` | Baseline floor `0.2433` tereproduksi lokal; gap ATD & Smart Tagging terukur. | Wajib (Hari 1) |
| **MVP-1** | Streaming video decoder + CV Ball Detector (pretrained YOLO/RT-DETR) dengan filter geometri lapangan (`unproject`) | Tabel `cv_ball.parquet` + confidence untuk frame gameplay; terbukti menambal sebagian gap 41% bola ATD. | Wajib (Hari 2 hingga 3) |
| **MVP-2** | Kalman Filter + Confidence-Aware Fusion (ATD + CV) | `fused_tracking.parquet` menurunkan missing ball di frame graded secara signifikan tanpa lonjakan noise. | Wajib (Hari 3) |
| **MVP-3** | Spatial/Kinematic Feature Engineering + Dead-Ball Engine + Controlled Possession Score `P(H/A/D)` | Mengatasi kelemahan utama baseline (`D` saat bola hilang, turnover instan akibat defleksi, dan periode pra-kickoff). | Wajib (Hari 4) |
| **MVP-4** | 3-State HMM + Viterbi Decoding per babak + CLI `run.py` + `validate_submission.py` | File submission `<team_name>.csv` deterministik, bebas flicker, siap submit sebagai versi aman. | Wajib (Hari 5) |
| **ADV-1** | Fine-tuning detector bola pada dataset publik game lain (misal SoccerNet/public football) atau multi-scale/SAHI inference pada area penalti | Meningkatkan recall bola kecil/blur di video taktis jika precision/recall pretrained masih rendah. | Opsional (Hari 5 hingga 6) |
| **ADV-2** | Classifier ML (`LightGBM`/`XGBoost`) terlatih pada dataset game lain atau pseudo-label algoritmik bersih + HSMM (pemodelan durasi spell eksplisit) | Potensi gain tambahan pada batas transisi dan distribusi durasi spell. | Opsional (Hari 6) |
| **ADV-3** | Player CV + ByteTrack + Jersey Clustering (HSV/Lab) untuk koreksi/penambahan tracking pemain ATD yang hilang/duplikat | Hanya dikerjakan jika audit diagnostik menunjukkan tracking pemain ATD menjadi bottleneck utama. | Opsional |
| **ADV-4** | Broadcast Auxiliary Fusion (sinkronisasi algoritmik feed `broadcast` ke `tactical` berbasis kesamaan konfigurasi pemain/homografi) | Sumber bukti tambahan saat oklusi total di kamera taktis; risiko kompleksitas tinggi untuk sisa waktu 7 hari. | Stretch / Tidak diprioritaskan |

---

## 8. Aturan Validasi Tanpa Melanggar Ketentuan Kompetisi

Tim **wajib** mematuhi strategi validasi berikut agar tidak mengalami jebakan overfitting 5 menit (seperti kasus `baseline_smooth125`) dan tidak melanggar [RULES.md](../RULES.md):

1. **Jangan Tuning di Public 5-Minute GT:**
   - `BAR-ATH_possession_gt_public.csv` hanya digunakan pada cek alignment, cek arti label (`H`/`A`/`D`/`X`), dan regresi pipeline end-to-end.
   - Jangan membandingkan hyperparameter (seperti transition penalty, dwell frames, atau smoothing window) hanya karena skor public `spellF1` naik.
2. **Validasi Proksi Non-GT pada Full Match (Legal dan Berbasis Fisika/Data):**
   - **Kalman / CV Ball Recovery Benchmark:** Masking secara sintetis interval bola ATD yang bersih (durasi 10 hingga 75 frame), lalu ukur RMSE posisi (meter) dan recall detektor CV terhadap posisi ATD asli yang disembunyikan.
   - **Diagnostic Spatiotemporal Full Match:** Pantau jumlah possession spell per babak, distribusi durasi spell (detik), rasio fragmentasi, konsistensi antara event `SET PIECES` Smart Tagging dengan transisi menuju/keluar dari `D`, serta konsistensi antara `GOALS` dengan jeda `D` dan kick-off berikutnya.
   - **Validasi Supervised Eksternal (Jika Memakai ML/Fine-Tuning):** Gunakan data berlabel dari **pertandingan lain** yang terbuka dan dapat diakses panitia (misal subset dataset publik sepak bola), dan dokumentasikan sumber serta lisensinya.
3. **Inspeksi Error Tanpa Koreksi Manual:**
   - Tim boleh menonton video taktis untuk memahami fenomena kegagalan (misal dari `evaluation/reports/<name>_transitions.csv` atau kasus bola melambung), tetapi solusi perbaikan harus berupa **aturan algoritmik umum**, bukan `if frame == ...` atau koreksi label manual.

---

## 9. Risiko Utama dan Mitigasi

| Risiko | Dampak | Probabilitas | Mitigasi Spesifik |
| --- | --- | --- | --- |
| Salah alignment ATD karena mengikuti asumsi offset `1702057/1702058` | Seluruh pipeline ATD meleset atau menghasilkan `None` | Tinggi jika tidak dikoreksi | Gunakan `video_frame = frame_id - 1` sesuai [starter/load_atd.py](../starter/load_atd.py) dan pasang unit test verifikasi 199.319 baris. |
| Salah arah matriks homografi saat memproyeksikan deteksi bola CV | Koordinat bola CV meleset ratusan meter keluar lapangan | Tinggi jika memakai `project()` langsung | Gunakan `unproject()` (invers matriks `3x3`) dan buat unit test proyeksi balik piksel -> pitch -> piksel pada 4 sudut lapangan. |
| Waktu habis mengejar Player CV + ByteTrack + Broadcast sebelum MVP selesai | Tidak ada submission matang saat deadline 17 November 2026 | Tinggi mengingat sisa waktu 7 hari | Kunci jalur MVP-0 sampai MVP-4 selesai di Hari ke-5; Player CV/Broadcast dikunci di belakang gate prioritas. |
| Overfitting terhadap 5 menit public GT | Skor full match anjlok di leaderboard privat | Tinggi | Gunakan proksi fisik/kinematika full match dan cek statistik non-GT; jalankan `--bootstrap 1000` untuk melihat lebar noise interval. |
| Kebocoran data pertandingan atau file hasil ke repo GitHub publik | Pelanggaran ketentuan lisensi dan diskualifikasi | Sedang | Perbarui `.gitignore` di Hari ke-1 untuk `cache/`, `outputs/`, `*.parquet`, dan lakukan `git status` / audit repo sebelum push dan submit. |
| Ketidaksesuaian environment saat panitia menjalankan ulang repo | Submission dianggap tidak reproducible (tidak dinilai) | Sedang | Pin seluruh versi library di `requirements.txt`, kunci seluruh random seed, dokumentasikan pengunduhan bobot model publik, dan uji eksekusi bersih satu perintah. |

---

## 10. Kriteria Penerimaan Final (Release Gate)

Proyek dinyatakan tuntas dan siap dikumpulkan apabila seluruh poin berikut terpenuhi sebelum **17 November 2026**:

1. File `<team_name>.csv` berjumlah tepat 199.319 baris (`frame` `0..199318`), hanya berisi `H`/`A`/`D`/`X` tanpa cell kosong, dan lulus `python evaluation/validate_submission.py <team_name>.csv`.
2. Seluruh eksperimen ablasi (`E0`..`E6`) tercatat hasilnya dan membuktikan kontribusi terukur dari CV ball recovery, dead-ball engine, dan temporal decoding dibandingkan baseline `0.2433`.
3. Perintah reproduksi tunggal (`run.py`) menghasilkan ulang CSV yang **identik** dengan file yang diunggah, menggunakan kode di repositori publik yang bersih dari aset `data/` maupun derived frame-level data.
4. Dokumen teknis `<team_name>.pdf` / `.md` selesai disusun mencakup seluruh 6 bagian wajib di [SUBMISSION.md](../SUBMISSION.md).
5. Seluruh 3 hingga 5 anggota tim terdaftar resmi dan unggahan Google Form telah diverifikasi pada file final yang baru divalidasi.

