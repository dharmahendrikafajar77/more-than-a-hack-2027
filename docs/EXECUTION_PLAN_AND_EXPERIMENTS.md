# Rencana Eksekusi Sprint 7 Hari, Pembagian Tim, dan Protokol Eksperimen

**Rentang Eksekusi:** 10 November 2026 hingga 17 November 2026 (Tenggat Submit)  
**Target Utama:** Menyelesaikan jalur MVP penuh paling lambat Hari ke-5 (15 November 2026), lalu menyisakan 2 hari penuh untuk ablasi tanpa overfitting, verifikasi reproducibility, pengemasan repo publik, dan penyusunan dokumen teknis.

---

## 1. Pembagian Peran Tim (3 hingga 5 Orang)

Mengacu pada syarat eligibilitas di [RULES.md](../RULES.md) (3 hingga 5 anggota terdaftar) dan arsitektur tim di [PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md](../PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md):

| Peran | Fokus Modul Utama | Deliverable Teknis |
| --- | --- | --- |
| **Engineer A: Video & Computer Vision** | `src/video/decoder.py`, `src/cv/ball_detector.py`, filter ukuran/lapangan kandidat bola, optimasi kecepatan inferensi GPU/chunking. *(Opsional: `player_tracker.py` jika gate MVP lolos.)* | `cv_ball.parquet`, diagnostik recall/precision CV bola pada frame ATD bersih, dokumentasi model publik & checkpoint. |
| **Engineer B: Tracking, Geometri & Fusi** | `src/tracking/atd_loader.py`, `src/tracking/smart_tagging.py`, `src/geometry/homography.py`, `src/fusion/kalman_fusion.py`, `src/features/feature_builder.py`. | `fused_tracking.parquet`, `features.parquet`, unit test alignment `frame_id - 1` dan `unproject()`, benchmark synthetic ball-masking. |
| **Engineer C: Possession & Temporal Model** | `src/possession/dead_ball.py`, `src/possession/control_model.py`, `src/temporal/state_machine.py`, `src/temporal/hmm_viterbi.py`, `src/possession/abstention.py`. | `probabilities.parquet`, logika deteksi pra-kickoff & bola mati, dekoder Viterbi per babak bebas flicker. |
| **Engineer D: Evaluasi, Integrasi & Dokumentasi** | `src/pipeline.py`, `run.py`, `configs/*.yaml`, `scripts/run_ablation.py`, `src/diagnostics/non_gt_eval.py`, manajemen `.gitignore`, audit kepatuhan, dan penulisan dokumen teknis submission. | Command tunggal reproduksi deterministik, log ablasi E0..E6+, hasil `validate_submission.py`, repo publik bersih, `<team_name>.pdf`/`.md`. |

**Penyesuaian jika tim berjumlah 3 atau 5 orang:**
- **Jika 3 orang:** Gabungkan peran **Engineer B + sebagian Engineer D** (Tracking/Geometri/Fusi + Integrasi), sementara **Engineer C** menangani Possession/Temporal sekaligus memimpin penulisan Laporan Teknis, dan **Engineer A** fokus pada Video/CV serta membantu audit diagnostik.
- **Jika 5 orang:** Anggota ke-5 bertindak sebagai **Validation & ML/Data Engineer** yang secara khusus menyiapkan data training legal dari pertandingan lain (jika memakai model ML/fine-tuning bola CV) dan menjalankan inspeksi error transisi dari `evaluation/reports/*_transitions.csv` tanpa membuat aturan frame-specific.

---

## 2. Jadwal Eksekusi Harian dan Gerbang Prioritas (P0 hingga P11)

| Hari / Tanggal | Gate Prioritas | Target Pekerjaan Konkret | Kriteria Lulus (Exit Gate) |
| --- | --- | --- | --- |
| **Hari 1 (10-11 Nov)** | **P0, P1, P2, P3** | 1. Perkuat `.gitignore` (`cache/`, `outputs/`, `*.parquet`, dll.).<br>2. Setup `.venv`, `pip install -r requirements-starter.txt`.<br>3. Unduh & verifikasi `data/tactical/` via `games.require()`.<br>4. Jalankan `baseline.py` & `run_evaluation.py` (E0).<br>5. Tulis `atd_loader.py`, `smart_tagging.py`, `homography.py`, dan unit test alignment (`frame_id - 1`, `unproject()`).<br>6. Buat `audit_data.py` untuk memetakan panjang gap bola ATD (berapa gap <0.5 s, 0.5-2 s, >2 s) dan cakupan Smart Tagging. | - `validate_submission.py` & `run_evaluation.py` berjalan.<br>- Unit test alignment & homography lulus 100%.<br>- Laporan audit missingness ATD selesai. |
| **Hari 2 (11-12 Nov)** | **P4** | 1. Implementasi `video/decoder.py` (streaming chunk tanpa simpan PNG ke disk).<br>2. Prototype `cv/ball_detector.py` (pilih model publik pretrained + strategi tile/ROI-guided dari proyeksi `H` terhadap kluster pemain ATD).<br>3. Uji kualitas deteksi CV secara otomatis pada sampel frame di mana ATD bola valid dan bergerak. | - Total frame terbaca = 199.319.<br>- CV detector menghasilkan `cv_ball.parquet` + confidence dengan FPS yang cukup untuk memproses 158.857 frame gameplay dalam batas waktu tim. |
| **Hari 3 (12-13 Nov)** | **P4 (full) & P5** | 1. Jalankan ekstraksi `cv_ball.parquet` penuh untuk blok babak 1 & babak 2.<br>2. Implementasi `fusion/kalman_fusion.py` (Kalman 2D + gating fisik + RTS smoother per babak) dan gabungkan ATD + CV.<br>3. Ukur perbaikan cakupan bola (`59.3%` -> target `>= 90%`) dan benchmark synthetic ball-masking (E2 vs E3). | - `fused_tracking.parquet` selesai untuk 199.319 frame.<br>- Terbukti secara kuantitatif bahwa CV + Kalman menurunkan RMSE dan memperpendek gap bola hilang dibanding ATD-only. |
| **Hari 4 (13-14 Nov)** | **P6 & P7** | 1. Implementasi `feature_builder.py` menghasilkan `features.parquet` (skala meter 105.3 x 68.0, boundary, kecepatan kolektif, polaritas Smart Tagging).<br>2. Implementasi `dead_ball.py` (termasuk deteksi kondisi pra-kickoff babak 2 secara algoritmik) dan `control_model.py` (dwell consistency, radius kontrol, pertahankan state saat loose ball).<br>3. Jalankan eksperimen E4 dan E5 pada diagnostik non-GT. | - `probabilities.parquet` (`P(H), P(A), P(D)`) tersedia.<br>- Periode pra-kickoff dan set-piece terdeteksi sebagai `D` tanpa hardcoding frame.<br>- Turnover sesaat akibat defleksi berkurang. |
| **Hari 5 (14-15 Nov)** | **P8, P9 (MVP Freeze)** | 1. Implementasi `temporal/hmm_viterbi.py` (dekoding terpisah untuk `half 1` dan `half 2`, tidak melewati halftime).<br>2. Integrasi awal `run.py` end-to-end dengan `configs/final.yaml`.<br>3. Jalankan `validate_submission.py` dan cek statistik full-match (`n_pred_spells`, `mean_duration_s`, `fragmentation` relatif terhadap baseline) serta sanity check public 5-minute GT (`E6`).<br>4. **Simpan kandidat submission MVP aman.** | - **Milestone Kritis:** Tim sudah memiliki `<team_name>.csv` lengkap 199.319 baris yang valid, bebas flicker, dan siap submit jika terjadi kendala di Hari 6-7. |
| **Hari 6 (15-16 Nov)** | **P10 (Ablasi & Opsional ADV)** | 1. Jika MVP masih punya kelemahan terukur dari `non_gt_eval.py` atau error sistemik (misal bola sering hilang di penalti atau transisi `D -> H/A` terlalu lambat), lakukan perbaikan algoritmik terarah (misal kalibrasi parameter transisi berdasar dinamika fisik, atau eksplorasi ADV-1/ADV-2 hanya jika cukup waktu dan data legal).<br>2. Evaluasi apakah `abstention.py` (`X` di tengah spell berketidakpastian tinggi) membantu atau berisiko (default tetap nonaktifkan bila tidak terbukti aman).<br>3. Bekukan `configs/final.yaml` dan `requirements-pipeline.txt` (pinned versions). | - Tabel eksperimen `E0` hingga `E6` (+ opsional `E7`/`E8`) lengkap terisi.<br>- Keputusan arsitektur final dibekukan. |
| **Hari 7 (16-17 Nov)** | **P11 & Final Submit** | 1. Bersihkan cache dan jalankan `run.py` dua kali dari awal (atau di mesin/venv kedua) untuk membuktikan **hash SHA-256 CSV identik**.<br>2. Audit `.gitignore` dan seluruh riwayat file yang akan di-push ke GitHub publik (`git status`, cek tidak ada `data/`, `cache/`, non-baseline CSV).<br>3. Lengkapi `README.md` dan dokumen teknis `<team_name>.pdf` / `.md` sesuai 6 poin [SUBMISSION.md](../SUBMISSION.md).<br>4. Submit via Google Form (beri buffer minimal beberapa jam sebelum cutoff 17 November 2026). | - Dua kali regenerasi menghasilkan CSV byte-identical.<br>- `validate_submission.py` mencetak `OK`.<br>- Ketiga deliverable terunggah sukses ke Google Form. |

---

## 3. Protokol Validasi Tanpa Public GT Overfitting

Mengingat [EVALUATION.md](../EVALUATION.md) dan [RULES.md](../RULES.md) melarang pemilihan model/hyperparameter berdasarkan 5 menit public GT, tim menggunakan **tiga lapis evaluasi legal**:

### Lapis 1: Validasi Proksi Kuantitatif Berbasis Fisika dan Sintetik (Full Match, Tanpa GT)

1. **Synthetic Ball-Masking Benchmark (`non_gt_eval.py`):**
   - Ambil segmen panjang di mana bola ATD valid dan berkesinambungan (misal >= 100 frame).
   - Tutupi observasi ATD pada jendela berdurasi 10, 25, 50, dan 100 frame secara bergantian.
   - Jalankan detektor CV dan Kalman Fusion pada segmen tertutup tersebut, lalu ukur **RMSE jarak (meter)** dan **persentase recovery dalam radius < 2.0 m** terhadap koordinat ATD yang disembunyikan.
   - Ini memberikan angka evaluasi yang sah dan langsung membuktikan apakah CV ball detector + Kalman memang meningkatkan akurasi pemulihan bola.
2. **CV Agreement pada Frame ATD Bersih:**
   - Pada frame di mana bola ATD hadir dan bergerak konsisten, ukur tingkat deteksi (*detection rate*) serta median jarak (meter) antara `unproject(CV)` dan posisi bola ATD.
3. **Audit Konsistensi Spatiotemporal Full Match (158.857 frame graded):**
   - **Cakupan posisi bola valid:** persentase frame graded dengan estimasi bola yang masuk akal.
   - **Statistik Spell Full Match (tanpa menyamakan ke target angka rahasia):** periksa apakah jumlah possession spell total, proporsi `H`/`A`/`D`, durasi rata-rata spell, dan jumlah fragmen pendek (`< 1 detik`) masuk akal secara taktis sepak bola serta jauh lebih stabil dari `baseline_raw`.
   - **Penyelarasan Restart dengan Smart Tagging:** ukur seberapa sering kejadian `SET PIECES` pada Smart Tagging diikuti transisi `D -> H/A` dalam jendela logis (misal sebelum vs sesudah akselerasi bola), dan seberapa sering kejadian `GOALS` diikuti periode `D` hingga kick-off berikutnya.

### Lapis 2: Validasi Supervised pada Pertandingan Lain (Hanya Jika Menggunakan Model Terlatih)

- Jika tim melatih classifier `P(H/A/D)` atau melakukan fine-tuning detector bola, dataset latih dan validasi **wajib berasal dari pertandingan lain** (misal dataset publik sepak bola seperti SoccerNet atau rekaman berlisensi terbuka yang dijelaskan sumbernya), **bukan** dari `BAR-ATH`.

### Lapis 3: Sanity Check & Regression Guard pada Public 5-Minute GT

- Dijalankan hanya untuk memastikan:
  1. Tidak ada pergeseran indeks (`off-by-one`) pada alignment video <-> ATD <-> Homography <-> Smart Tagging.
  2. Makna label `H` (Barcelona) dan `A` (Athletic Club) tidak tertukar.
  3. Tidak ada error format atau NaN yang lolos.
  4. Jika mengecek `run_evaluation.py`, selalu aktifkan `--bootstrap 1000` untuk mengingatkan tim bahwa interval ketidakpastian pada 17 possession sangat lebar sehingga selisih kecil di public GT tidak bermakna.
- Saat menginspeksi `evaluation/reports/<name>_transitions.csv` di public window (seperti disarankan `starter/03_first_submission.ipynb`), catat pola kesalahan **umum** (misal "defleksi saat bola melaju cepat memicu turnover palsu"), lalu perbaiki mekanisme algoritmiknya secara umum.

---

## 4. Matriks Eksperimen Wajib (`E0` hingga `E8`)

Setiap eksperimen dijalankan lewat `scripts/run_ablation.py` dengan config YAML tersimpan agar dapat diaudit ulang dan dicantumkan ke bagian pendekatan/aset pada dokumen teknis final.

| ID | Nama Konfigurasi | Perubahan Utama | Pertanyaan Teknis yang Dijawab | Status |
| --- | --- | --- | --- | --- |
| **E0** | `baseline_official` | Jalankan `starter/baseline.py` + `run_evaluation.py` | Apakah environment, data path, dan evaluator resmi berfungsi? | Wajib |
| **E1** | `atd_kalman_only` | ATD player + ball dengan interpolasi/Kalman tanpa CV | Seberapa jauh kinerja dan stabilitas bola naik tanpa CV? | Wajib |
| **E2** | `atd_plus_cv_ball` | E1 + CV ball detector (`cv_ball.parquet`) | Apakah CV ball detector meningkatkan recall dan menurunkan RMSE pada uji synthetic masking? | Wajib |
| **E3** | `confidence_fusion_rts` | E2 + confidence-aware fusion + RTS smoother | Apakah fusi berbobot ketidakpastian mengurangi outlier CV dan memperhalus transisi gap? | Wajib |
| **E4** | `plus_dead_ball_engine` | E3 + proyeksi batas lapangan + player stillness + Smart Tagging set-piece prior + deteksi pra-kickoff | Apakah kelas `D` memisahkan bola keluar/stoppage secara akurat dari bola hilang di dalam lapangan? | Wajib |
| **E5** | `plus_control_estimator` | E4 + radius kontrol + kecepatan relatif + dwell consistency (anti-defleksi) | Apakah turnover palsu saat bola melintas dekat lawan berhasil ditekan? | Wajib |
| **E6** | `plus_hmm_viterbi` | E5 + 3-State HMM Viterbi decoding per babak (**Kandidat Utama MVP**) | Apakah fragmentasi turun drastis dan distribusi spell menjadi realistis? | Wajib |
| **E7** | `plus_player_cv` | E6 + Player detection / ByteTrack / Team Clustering | Apakah perbaikan tracking pemain visual menutupi celah ATD secara sepadan dengan biaya komputasi? | Opsional |
| **E8** | `plus_hsmm_or_x` | E6 + HSMM duration modeling dan/atau kebijakan `X` terukur | Apakah pemodelan durasi eksplisit atau abstention memberikan perbaikan struktural yang aman? | Opsional |

---

## 5. Template Log Eksperimen Tim

Salin dan isi tabel ini di `docs/EXPERIMENT_LOG.md` atau catatan kerja internal tim setiap kali eksperimen selesai dijalankan:

```markdown
| Run ID | Tanggal | Commit / Config | % Ball Valid (Graded) | Masking RMSE 50fr (m) | Pred Spells (Full) | Mean Spell Dur (s) | Public Sanity (macroF1 / spellF1) | Catatan Error & Keputusan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0-smooth50 | YYYY-MM-DD | official baseline | 59.3% | N/A | [isi terukur] | [isi terukur] | 0.6724 / 0.3256 (terdokumentasi) | Floor resmi 0.2433 full match |
| E1 | ... | ... | ... | ... | ... | ... | ... | ... |
```

