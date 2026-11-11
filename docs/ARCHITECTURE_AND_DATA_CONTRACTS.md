# Spesifikasi Arsitektur Sistem dan Kontrak Data

Dokumen ini menjabarkan kontrak teknis level-implementasi bagi seluruh modul pada `src/`, `configs/`, dan `cache/`. Seluruh koordinat, indeks frame, dan aturan proyeksi mengacu langsung pada [games.py](../games.py), [DATA.md](../DATA.md), [starter/load_atd.py](../starter/load_atd.py), [starter/load_homography.py](../starter/load_homography.py), [starter/load_smart_tagging.py](../starter/load_smart_tagging.py), dan [evaluation/possession_metrics.py](../evaluation/possession_metrics.py).

---

## 1. Diagram Alir Sistem End-to-End

```text
                       data/tactical/ (Authority Timeline: 0..199318, 25 FPS)
   ┌───────────────────┬───────────────────────┬─────────────────────────┬──────────────────────┐
   ▼                   ▼                       ▼                         ▼                      ▼
BAR-ATH_           BAR-ATH_tactical_       BAR-ATH_tactical_       BAR-ATH_tactical_     games.half_video_
tactical.mp4       atd.txt + atd.xml       homography.csv          smart_tagging.xml     blocks("BAR-ATH")
(199,319 fr)       (frame_id 1..199319)    (frame 0..199318)       (638 XML instances)   [(1,11185,90651),
   │                   │                       │                         │                (2,110808,190197)]
   ▼                   ▼                       │                         │                      │
[video/decoder]    [tracking/atd_loader]       │                         │                      │
Chunk streaming    video_frame = frame_id - 1  │                         │                      │
   │               scale (105.3, 68.0) m       │                         │                      │
   ▼                   │                       ▼                         │                      │
[cv/ball_detector] ────┼────────────────► [geometry/homography]          │                      │
YOLO / RT-DETR         │                  unproject(inv(H), u, v)        │                      │
   │                   │                  -> norm (0..1) -> metres       │                      │
   ▼                   ▼                       │                         ▼                      │
cache/tactical/cv_ball.parquet                 │               [tracking/smart_tagging]         │
   │                                           │               round(sec * 25), norm team       │
   └───────────────────┬───────────────────────┘                         │                      │
                       ▼                                                 │                      │
         [fusion/kalman_fusion.py]                                       │                      │
         Kalman 2D [x, y, vx, vy] + confidence gating                    │                      │
                       │                                                 │                      │
                       ▼                                                 │                      │
         cache/tactical/fused_tracking.parquet                           │                      │
                       │                                                 │                      │
                       └───────────────────────┬─────────────────────────┴──────────────────────┘
                                               ▼
                                 [features/feature_builder.py]
                                 Spatial, kinematics, boundary, priors
                                               │
                                               ▼
                                 cache/tactical/features.parquet
                                               │
                             ┌─────────────────┴─────────────────┐
                             ▼                                   ▼
                [possession/dead_ball.py]           [possession/control_model.py]
                Boundary + stillness + set-piece    Dwell + radius + momentum + team
                             └─────────────────┬─────────────────┘
                                               ▼
                                 Frame Emissions P(H), P(A), P(D)
                                 cache/tactical/probabilities.parquet
                                               │
                                               ▼
                                 [temporal/hmm_viterbi.py]
                                 Per-half decoding (no cross-halftime merge)
                                               │
                                               ▼
                                 [possession/abstention.py] (optional X policy)
                                               │
                                               ▼
                                 outputs/<team_name>.csv (199,319 rows)
                                               │
                                               ▼
                                 evaluation/validate_submission.py
```

---

## 2. Kontrak Alignment Timeline dan Sistem Koordinat (Wajib Dipatuhi)

Bagian ini mengoreksi asumsi pada draf awal [PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md](../PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md) agar seluruh modul menggunakan pemetaan yang persis sama dengan evaluator resmi.

### 2.1 Tabel Pemetaan Indeks Waktu (`video_frame`)

| Sumber Aset | Indeks Asli Aset | Rumus Konversi ke `video_frame` (`0..199318`) | Catatan Kritis |
| --- | --- | --- | --- |
| **Submission CSV** | `frame` (`0..199318`) | Identitas (`frame = video_frame`) | Harus tepat 199.319 baris, tipe int, terurut tanpa celah. |
| **Video Taktis** (`BAR-ATH_tactical.mp4`) | Indeks dekode `0..199318` | `video_frame` | 25 FPS, 1920x1080 piksel. |
| **ATD Taktis** (`BAR-ATH_tactical_atd.txt`) | `frame_id` (`1..199319`) | **`video_frame = frame_id - 1`** | **JANGAN** gunakan `games.video_to_id()`. Konstanta `offsets` (`1702057`, `1702058`) di `games.py` adalah milik ID internal pembuat GT privat, bukan indeks ATD peserta. |
| **Homography Taktis** (`BAR-ATH_tactical_homography.csv`) | `frame` (`0..199318`) | Identitas (`frame = video_frame`) | `valid == 1` pada 100% frame taktis; median `reprojection_error` 0.24 saat bermain, ~1.5 saat halftime. |
| **Smart Tagging** (`BAR-ATH_tactical_smart_tagging.xml`) | `<start>`, `<end>` (detik dari frame 0) | `start_frame = round(start_s * 25)`, `end_frame = round(end_s * 25)` | Interval setengah terbuka `[start_frame, end_frame)`. Akurasi batas +/- 1 hingga 5 detik (opening kickoff ~3.5 detik lebih awal). |
| **Rentang Graded** (`games.half_video_blocks("BAR-ATH")`) | `halves` + `offsets` di `games.py` | Babak 1: `[11185, 90651]` (79.467 frame)<br>Babak 2: `[110808, 190197]` (79.390 frame) | Pra-laga: `0..11184`; Halftime: `90652..110807`; Pasca-laga: `190198..199318`. Total graded: 158.857 frame. |

**Jebakan Awal Babak Kedua (`110808..113741`):**
- Babak 1 mulai dinilai tepat pada frame kick-off (`11185`).
- Babak 2 mulai dinilai sejak `110808`, sedangkan kick-off babak kedua baru terjadi di **`113742`**. Seluruh 2.934 frame di antaranya (`110808..113741`) berstatus `D` di ground truth. Sistem tidak boleh menghardcode nomor frame tersebut, melainkan harus mendeteksinya secara algoritmik dari kondisi bola mati/diam, posisi pemain di separuh lapangan masing-masing, kecepatan kolektif rendah, atau jeda sebelum peristiwa kick-off.

### 2.2 Sistem Koordinat dan Transformasi Homografi

Terdapat tiga ruang koordinat yang wajib dibedakan secara eksplisit dalam penamaan variabel (`*_px`, `*_norm`, `*_m`):

1. **Image Pixel Space `(u_px, v_px)`:**
   - Resolusi video taktis: lebar `1920`, tinggi `1080`.
   - Origin `(0, 0)` di sudut kiri atas gambar.
   - Untuk pemain hasil CV bounding box, titik kontak tanah berada di **bottom-center** `(x_center, y_max)`, sedangkan untuk bola gunakan **center** `(x_center, y_center)`.
2. **Normalised Pitch Space `(x_norm, y_norm)`:**
   - Rentang nominal `[0.0, 1.0] x [0.0, 1.0]`. Nilai `< 0` atau `> 1` berarti objek berada di luar garis lapangan.
   - Inilah ruang koordinat asli seluruh baris di `BAR-ATH_tactical_atd.txt` dan domain input matriks `H[frame]`.
3. **Metric Pitch Space `(x_m, y_m)`:**
   - Gunakan konstanta resmi di `games.GAMES["BAR-ATH"]["pitch"]`: **`PITCH_LENGTH = 105.3` m**, **`PITCH_WIDTH = 68.0` m** (diutamakan daripada `105 x 68` di metadata XML karena 105.3 x 68.0 berasal dari pengukuran tracking terkoreksi).
   - Konversi wajib sebelum menghitung jarak: `x_m = x_norm * 105.3`, `y_m = y_norm * 68.0`.

**Aturan Matriks Homografi (`starter/load_homography.py`):**
- Matriks 3x3 `H[frame]` memetakan **normalised `(0..1)` ke pixels `(u_px, v_px)`**.
- Oleh karena itu, proyeksi titik deteksi CV piksel ke meter lapangan **wajib** mengikuti alur:
  1. Pastikan `valid[frame]` bernilai `True`.
  2. Hitung `(x_norm, y_norm) = unproject(H[frame], u_px, v_px)` yaitu `project(np.linalg.inv(H[frame]), u_px, v_px)`.
  3. Hitung `(x_m, y_m) = (x_norm * 105.3, y_norm * 68.0)`.
  4. Simpan juga `reprojection_error[frame]` dan flag `homography_ok` untuk membobot ketidakpastian observasi CV.

---

## 3. Spesifikasi Tiap Modul Pipeline (`src/`)

### 3.1 `src/tracking/` (ATD Loader & Smart Tagging Parser)

**ATD Parser (`atd_loader.py`):**
- Baca metadata XML (`atd_meta`): identifikasi `Team` indeks 0 sebagai `HOME (H)` (Barcelona) dan indeks 1 sebagai `AWAY (A)` (Athletic Club), serta susun array `team_of_track` sepanjang 46 slot (mengikuti pola `starter/baseline.py`).
- Baca stream raw `atd_raw` (`<frame_id>:<46 slots>:<ball>`). Jangan simpan DataFrame 46 kolom x 199k baris via `kloppy.to_df()` jika lambat/boros RAM; gunakan streaming parser numpy terstruktur yang menyimpan:
  - `player_xy_m`: array float32 `(199319, 46, 2)` dalam meter (atau `NaN` jika slot tidak aktif).
  - `player_speed`: array float32 `(199319, 46)` dalam m/s.
  - `player_team`: array int8 `(46,)` berisi `0` (`H`) atau `1` (`A`).
  - `ball_atd_m`: array float32 `(199319, 2)` dalam meter (`NaN` pada ~43.1% total frame).
  - `ball_atd_speed`: array float32 `(199319,)` dalam m/s.
- **Filter Anomali ATD:** Saat memproses slot pemain, tangani duplikasi tracklet (>22 pemain aktif) dengan tidak mengasumsikan indeks slot sebagai identitas pemain permanen. Hitung metrik kolektif atau jarak berbasis set pemain aktif per tim.

**Smart Tagging Parser (`smart_tagging.py`):**
- Gunakan XML parser standar (pola `starter/load_smart_tagging.py`) tanpa bergantung `kloppy`.
- Terapkan normalisasi wajib:
  - `code`: `.strip().upper()`
  - `Team`: `.strip().lower()`: `"barcelona"` -> `"H"`, `"atletic club"` atau `"athletic club"` -> `"A"`, `"n/a"` / `None` -> `"NONE"`.
  - `Type` (pada `SET PIECES`): `.strip().lower()` untuk membedakan `"throw-in"` (26), `"corner kick"` (12), `"goal kick"` (10), `"free kick"` (5), `"kick off (after goal)"` (4), `"kick off (start)"` (2).
  - `Max Players in the box`: parsing aman untuk nilai `"7+"` -> `7` (integer) plus flag `is_7_plus`.

### 3.2 `src/video/` dan `src/cv/` (Video Decoder & CV Ball Recovery)

**Chunked Video Decoder (`video/decoder.py`):**
- Membuka `BAR-ATH_tactical.mp4` dengan OpenCV `cv2.VideoCapture` (atau `PyAV`/`decord` jika lebih cepat dan terverifikasi frame count tepat `199.319`).
- Iterasi sekuensial dalam chunk (misal `chunk_size = 1000` atau `2000` frame) hanya pada rentang frame graded (`11185..90651` dan `110808..190197`) sambil mempertahankan nomor `video_frame` eksak (verifikasi tidak ada frame drop saat seek; idealnya baca sekuensial dari frame 0 atau verifikasi posisi seek sebelum chunk).
- **Wajib:** jangan pernah menuliskan frame video sebagai file `.png`/`.jpg` ke disk.

**CV Ball Detector (`cv/ball_detector.py`):**
- Model utama: detektor objek publik pretrained (misal keluarga Ultralytics YOLO atau RT-DETR) dengan opsi inferensi resolusi penuh / tiling / ROI-guided crop.
- **Strategi ROI / Multi-Candidate Efisien:**
  - Karena bola di kamera taktis 1920x1080 berukuran sangat kecil (sering hanya beberapa piksel), deteksi full-frame 640px sering gagal mendeteksi bola.
  - Solusi algoritmik legal yang disarankan:
    1. Jalankan detektor pada full-frame resolusi tinggi (misal 1280/1920) atau multi-tile (2x2 / 3x2 grid overlap), **atau** gunakan **ROI crop adaptif** di sekitar prediksi posisi Kalman sebelumnya dan di sekitar area kerumunan pemain dari ATD (diubah ke piksel menggunakan `project(H[frame], x_norm, y_norm)`!).
    2. Catat: memproyeksikan titik ATD ke piksel untuk memilih ROI crop sepenuhnya algoritmik dan sangat mempercepat serta meningkatkan recall deteksi bola kecil!
- **Candidate Filtering:**
  - Tolak kandidat dengan ukuran bounding box tidak masuk akal untuk bola pada jarak kamera tersebut.
  - Proyeksikan pusat kandidat ke koordinat pitch via `unproject()`; beri penalti jika jauh di luar area lapangan + margin stadium (kecuali sedang situasi bola keluar).
  - Jika ada beberapa kandidat, simpan top-K kandidat (misal top-3) agar Kalman / temporal association dapat memilih kandidat yang konsisten secara kinematika, bukan sekadar confidence tunggal terbesar (mengurangi false positive dari kepala pemain atau garis/sepatu putih).

*(Opsional Fase Advanced)*: `cv/optical_flow.py` atau `cv/player_tracker.py` (ByteTrack + HSV/Lab clustering) hanya diaktifkan melalui config bila gate MVP sudah terlewati.

### 3.3 `src/fusion/` (Kalman Filter & Confidence-Aware Fusion)

**Model State Kalman 2D (`fusion/kalman_fusion.py`):**
- State vector pada frame $t$: $\mathbf{x}_t = [x_m, y_m, v_x, v_y]^T$ dalam koordinat lapangan (meter dan m/s), dengan langkah waktu $\Delta t = 1/25 = 0.04\text{ s}$.
- Transition matrix constant-velocity dengan damping ringan (gesekan rumput) pada komponen kecepatan atau process noise $Q$ terkalibrasi untuk akselerasi tendangan.
- **Gating Fisik (Outlier Rejection):**
  - Hitung Mahalanobis distance atau Euclidean jump terhadap prediksi Kalman $\hat{\mathbf{x}}_{t|t-1}$.
  - Jika jarak lompatan dalam 1 frame menyiratkan kecepatan fisik mustahil (misal $> 45\text{ m/s}$ terus-menerus tanpa kontinuitas frame berikutnya), tolak sebagai outlier atau turunkan bobot observasi.
- **Matriks Keputusan Fusi per Frame:**
  1. `ATD valid & CV valid`: jika kedua sumber dekat (misal $< 2.5\text{ m}$), lakukan fusi berbobot invers varians (ATD noise $\sigma_{atd}$, CV noise $\sigma_{cv}$ diskalakan oleh $(1 - c_{cv})$ dan `reprojection_error`). Jika berbeda jauh, pilih kandidat yang konsisten dengan lintasan Kalman dan kedekatan kerumunan pemain.
  2. `Hanya ATD valid`: update Kalman dengan ATD (saring spike tunggal).
  3. `ATD hilang & CV valid`: update Kalman dengan hasil `unproject` CV (inilah nilai tambah langsung cabang CV terhadap gap 41%).
  4. `Keduanya hilang`: lakukan `kalman.predict()` (coasting) sampai batas `max_coast_frames` (misal 25 hingga 50 frame) sambil memperbesar `uncertainty`. Setelah melewati batas coasting tanpa observasi, tandai `ball_source = "missing"` dan gunakan fitur konteks pemain (centroid / jarak historis).
- **Backward Smoothing (RTS Smoother / Bidirectional):** Karena seluruh video dan tracking sudah tersedia secara offline per babak (bukan real-time streaming), **Rauch-Tung-Striebel (RTS) Kalman Smoother** (forward-backward pass di dalam masing-masing babak) sangat dianjurkan agar interpolasi gap bola mulus di kedua ujung (sebelum hilang dan saat muncul lagi).

### 3.4 `src/features/` (Feature Table per Tactical Frame)

Setiap baris di `features.parquet` memiliki primary key `video_frame` (`0..199318`) dan `half` (`1`, `2`, atau `0` untuk non-graded). Kelompok fitur yang dihasilkan:

1. **Ball Kinematics & Uncertainty:**
   - `ball_x_m`, `ball_y_m`, `ball_vx`, `ball_vy`, `ball_speed_mps`, `ball_accel_mps2`
   - `ball_direction_change_rad` (sudut perubahan vektor kecepatan bola antar jendela pendek, penanda sentuhan/operan/pantulan)
   - `ball_confidence`, `ball_uncertainty_m`, `ball_missing_run_len`, `ball_source_code`
2. **Pitch Geometry:**
   - `dist_to_sideline_m` (`min(y_m, 68.0 - y_m)`), `dist_to_goalline_m` (`min(x_m, 105.3 - x_m)`)
   - `outside_pitch_margin_m` (positif jika di luar kotak `0..105.3 x 0..68.0`)
   - `in_penalty_box_flag`
3. **Ball-Player Spatial Relations (Metrik Meter):**
   - `nearest_home_dist_m`, `nearest_away_dist_m`, `second_nearest_home_dist_m`, `second_nearest_away_dist_m`
   - `dist_margin_m` (`nearest_away_dist_m - nearest_home_dist_m`)
   - `home_players_within_2m`, `away_players_within_2m`, `home_players_within_5m`, `away_players_within_5m`
   - `nearest_home_rel_speed`, `nearest_away_rel_speed` (selisih vektor/skalar kecepatan bola vs pemain terdekat)
4. **Team Collective Kinematics (Untuk Situasi Bola Hilang & Dead-Ball):**
   - `mean_player_speed_all`, `p75_player_speed_all`, `frac_players_walking` (`speed < 1.5 m/s`)
   - `home_centroid_x`, `home_centroid_y`, `away_centroid_x`, `away_centroid_y`
   - `active_track_count_h`, `active_track_count_a`
   - `players_in_own_half_ratio` (sinyal algoritmik untuk deteksi persiapan kick-off seperti awal babak kedua atau pasca gol)
5. **Smart Tagging Priors (Dilakukan Sinkronisasi Frame Tanpa Overtrust):**
   - `st_set_piece_active` + one-hot / indikator jenis (`throw_in`, `corner`, `goal_kick`, `free_kick`, `kickoff_goal`, `kickoff_start`)
   - `st_phase_team_h_score`, `st_phase_team_a_score` (agregasi fase aktif seperti `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION` untuk `H` vs `A`; `DEFENDING IN ...` harus ditangani berlawanan atau hati-hati karena tag `Team` pada defending phase merujuk tim yang *bertahan*, bukan yang *menguasai bola*!)
   - `st_goal_window`

**Peringatan Penting Smart Tagging Team Polarity:**
- Pada fase ofensif (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `GOALS`), `Team` menunjukkan tim yang menyerang/menguasai bola.
- Pada fase defensif (`DEFENDING IN ATTACKING/MIDDLE/DEFENSIVE THIRD`, `DEFENSIVE TRANSITION`), `Team` menunjukkan tim yang *melakukan aksi bertahan*, artinya **lawannya** yang cenderung menguasai bola! Membedakan polaritas kode ini wajib diimplementasikan di `smart_tagging.py` dan diuji lewat unit test.

### 3.5 `src/possession/` dan `src/temporal/` (Dead-Ball, Control Estimator, HMM/Viterbi)

#### A. Dead-Ball Engine (`possession/dead_ball.py`)
Menghasilkan log-likelihood / skor `score_D(t)` dari gabungan sinyal fisik dan taktis:
1. **Bola keluar lapangan:** jika `outside_pitch_margin_m > margin` (misal `0.5 m` dengan `ball_confidence` tinggi atau posisi pemain terkluster jauh dari bola/menuju garis lemparan).
2. **Keheningan kinematika kolektif:** saat `mean_player_speed_all` atau `p75_player_speed_all` turun di bawah ambang berjalan kaki dalam durasi kontinu, disertai bola diam (`ball_speed_mps < 0.8 m/s`) atau hilang di dekat pinggir lapangan.
3. **Konteks Set-Piece & Gol:** kehadiran window `SET PIECES` / `GOALS` dari Smart Tagging menaikkan prior `D` sebelum momen akselerasi bola/restart terjadi (transisi keluar `D -> H/A` dipicu saat bola mulai bergerak cepat atau pemain bergerak aktif setelah jeda, bukan pada `start_frame` Smart Tagging yang sering lebih awal).
4. **Deteksi Pra-Kickoff (Umum/Algoritmik):** bila kedua tim terpisah di separuh lapangan masing-masing dan bola diam dekat titik tengah `(52.65, 34.0)` serta kecepatan pemain rendah (mendeteksi kick-off babak 1, awal babak 2 `110808..113741`, dan pasca gol secara seragam tanpa hardcoding frame).

#### B. Controlled Possession Estimator (`possession/control_model.py`)
Menghasilkan `score_H(t)` dan `score_A(t)` serta probabilitas terkalibrasi `[P(H), P(A), P(D)]`:
- **Stage A (Interpretable Probabilistic Engine - MVP):**
  - Basis jarak softmax/exponential decay dari bola ke pemain terdekat `H` vs `A` dalam jendela lokal, dimodulasi oleh kecepatan relatif bola-pemain dan konsistensi posisi selama beberapa frame (*dwell consistency*).
  - Jika bola meluncur bebas cepat (operan jauh/bola melambung dengan `nearest_dist > control_radius` dan tanpa perubahan arah di kaki lawan), **jangan** berikan evidence turnover ke tim lawan hanya karena pemain lawan dilewati bola selama 1 hingga 4 frame.
  - Tambahan prior lemah (bobot terkendali) dari Smart Tagging dengan polaritas yang benar (`offensive -> +tim`, `defensive -> +lawan`).
- **Stage B (Optional Classical ML Classifier - ADV-2):**
  - Jika tim menggunakan data berlabel dari **game lain yang legal** (atau pseudo-label algoritmik yang terverifikasi bebas GT publik), fitur di atas dapat dilatih dengan `LightGBM` / `XGBoost` / `LogisticRegression` untuk menghasilkan `P(H), P(A), P(D)` yang lebih terkalibrasi.

#### C. Temporal Sequence Decoder (`temporal/hmm_viterbi.py`)
- Dijalankan **terpisah per blok babak** (`half 1`: `11185..90651`, `half 2`: `110808..190197`).
- Frame di luar dua blok (`0..11184`, `90652..110807`, `190198..199318`) diisi eksplisit `X`.
- **3-State HMM (`H=0, A=1, D=2`):**
  - Emission log-probability: $\log P(O_t \mid s_t = k)$ diturunkan dari skor emisi terkalibrasi (atau $\log P(s_t = k \mid O_t) - \log \pi_k$).
  - Transition matrix $T_{i \to j}$ ($3 \times 3$):
    - Self-transition tinggi ($T_{H \to H}, T_{A \to A}, T_{D \to D}$) selaras dengan ekspektasi durasi rata-rata segmen sepak bola (pada 25 FPS, durasi 12 hingga 15 detik setara 300 hingga 375 frame, sehingga biaya perpindahan antar state cukup besar untuk mencegah flicker 2 hingga 10 frame, namun tetap memperbolehkan pergantian bila bukti kontrol lawan bertumpuk kuat).
    - Pertimbangkan penalti asimetris yang masuk akal atau fitur transisi dinamis (misalnya transisi `D -> H/A` lebih mudah saat terdeteksi tendangan restart / akselerasi bola setelah set-piece).
- **Post-Processing / Minimum Spell Guard (Hati-hati):**
  - Pada Viterbi, durasi minimum terwujud otomatis dari penalti transisi. Jika memakai **HSMM (Hidden Semi-Markov Model)** di fase lanjutan, model dapat memodelkan distribusi durasi state secara eksplisit (menyelaraskan langsung dengan metrik `spellF1`).

---

## 4. Skema File Cache dan Konfigurasi

Seluruh folder `cache/` dan `outputs/` **wajib ada di `.gitignore`** agar tidak pernah masuk ke repositori publik.

### 4.1 Skema Cache Lokal (`cache/tactical/`)

| File | Kolom Kunci | Tipe & Tujuan |
| --- | --- | --- |
| `cv_ball.parquet` | `video_frame (int32)`, `detected (bool)`, `u_px (float32)`, `v_px (float32)`, `pitch_x_m (float32)`, `pitch_y_m (float32)`, `cv_conf (float32)`, `bbox_w (float32)`, `bbox_h (float32)`, `reproj_err (float32)` | Hasil mentah detektor bola visual per frame video taktis. |
| `fused_tracking.parquet` | `video_frame (int32)`, `half (int8)`, `ball_x_m (float32)`, `ball_y_m (float32)`, `ball_vx (float32)`, `ball_vy (float32)`, `ball_speed_mps (float32)`, `ball_conf (float32)`, `ball_uncertainty_m (float32)`, `ball_source (category/int8)`, `gap_len (int32)` | Lintasan bola gabungan ATD + CV hasil Kalman Filter / RTS Smoother. |
| `features.parquet` | `video_frame (int32)`, `half (int8)`, seluruh fitur spasial, kinematika, boundary, dan prior Smart Tagging | Tabel fitur algoritmik lengkap sebagai input estimator `P(H/A/D)`. |
| `probabilities.parquet` | `video_frame (int32)`, `half (int8)`, `p_home (float32)`, `p_away (float32)`, `p_dead (float32)`, `uncertainty (float32)` | Probabilitas per frame sebelum dekoding HMM/Viterbi. |
| `cache/metadata/run_manifest.json` | timestamp, git commit hash, config hash, seed, versi model CV, parameter pipeline | Rekam jejak determinisme dan audit reproducibility. |

### 4.2 Struktur Konfigurasi (`configs/`)

Gunakan file YAML terpisah atau terkomposisi (`configs/final.yaml`) yang berisi seluruh konstanta agar nol magic number tersembunyi di kode:
- `configs/cv.yaml`: nama checkpoint model publik, sumber/URL download otomatis atau instruksi resmi, ukuran input/tiling, ambang confidence, filter ukuran bbox, ROI margin, batch size, device/seed.
- `configs/fusion.yaml`: parameter proses & noise pengukuran Kalman ($Q$, $R_{atd}$, $R_{cv}$), kecepatan maksimal gating (m/s), batas coasting (frame), flag aktif/nonaktif RTS smoother.
- `configs/possession.yaml`: batas radius kontrol (m), margin out-of-bounds (m), ambang kecepatan berjalan (m/s), bobot Smart Tagging (termasuk pemetaan polaritas fase), parameter skor kontrol/dead-ball.
- `configs/temporal.yaml`: matriks transisi HMM (atau parameter durasi HSMM), prior awal per babak, ambang kebijakan `X` opsional (`enabled: false` di MVP).
- `configs/final.yaml`: file konfigurasi master untuk reproduksi submission resmi satu perintah.

