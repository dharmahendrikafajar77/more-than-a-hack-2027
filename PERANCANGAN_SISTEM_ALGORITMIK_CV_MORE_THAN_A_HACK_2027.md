# PERANCANGAN SISTEM ALGORITMIK & COMPUTER VISION
## More Than a Hack 2027 — Metrica Sports & FC Barcelona

> Versi Markdown dari rancangan teknis final. Fokus utama: pipeline algoritmik yang reproducible untuk menghasilkan label possession H/A/D/X dari video dan data pertandingan.

PERANCANGAN SISTEM ALGORITMIK BERBASIS
COMPUTER VISION, TRACKING & TEMPORAL INFERENCE

More than a Hack 2027
Metrica Sports × FC Barcelona Innovation Hub

Revisi final: pipeline algoritmik dari video dan tracking, evaluasi resmi spellF1, serta guardrail reproducibility dan anti-hand-label

| Tujuan Dokumen<br>Memperbarui rancangan sebelumnya agar video pertandingan menjadi sumber observasi algoritmik yang dapat menghasilkan data turunan secara otomatis, kemudian difusikan dengan tracking ATD, homography, dan smart tagging untuk menghasilkan label H/A/D yang reproducible. |
| --- |

# 1. Pembaruan Utama Berdasarkan games.py

games.py menegaskan bahwa timeline submission mengikuti tactical video. Karena itu seluruh pipeline Computer Vision harus menghasilkan observasi yang akhirnya diindeks kembali ke frame tactical, bukan ke timeline lain.

| Aspek | Ketentuan Teknis | Implikasi Rancangan |
| --- | --- | --- |
| Sumber utama | BAR-ATH_tactical.mp4 | Seluruh output akhir diindeks terhadap frame tactical. |
| Jumlah frame tactical | 199.319 frame | Pipeline inference harus dapat menghasilkan status untuk frame 0..199318. |
| Frame rate | 25 FPS | Semua durasi temporal, dwell time, smoothing, dan transition tolerance dihitung dalam 25 FPS. |
| Tracking alignment | Offset per half: 1.702.057 dan 1.702.058 | Join tracking ↔ video harus memakai offset per babak, tidak boleh satu offset global. |
| Pitch | 105,3 × 68,0 m | Koordinat hasil homography dinormalisasi ke metrik lapangan. |
| Broadcast video | 164.488 frame, opsional | Boleh untuk additional evidence tetapi perlu alignment khusus; bukan timeline submission. |
| Smart tagging | Hanya tactical | Gunakan sebagai prior/event evidence pada timeline tactical. |

| Keputusan desain<br>Pipeline utama memakai tactical video + tactical ATD + tactical homography + smart tagging. Broadcast diperlakukan sebagai optional auxiliary source, bukan dependensi MVP. |
| --- |

# 2. Apakah Harus Menggunakan Computer Vision?

Ya, jika tujuan tim adalah membangun atau memperbaiki data dari video yang diberikan, Computer Vision adalah teknologi yang paling relevan. Namun CV tidak berdiri sendiri. Solusi yang disarankan adalah sistem multimodal: CV menangkap bukti visual, tracking menyediakan koordinat yang sudah tersedia, homography membawa observasi ke koordinat lapangan, dan model temporal menjaga konsistensi possession.

| Lapisan | Teknologi | Fungsi |
| --- | --- | --- |
| Video I/O | OpenCV / FFmpeg | Decode tactical video, sampling frame, sinkronisasi frame index. |
| Object Detection | YOLO / RT-DETR / detector publik setara | Mendeteksi player, ball, referee bila data tracking perlu diperbaiki/ditambah. |
| Multi-Object Tracking | ByteTrack / BoT-SORT | Menghubungkan detection antar-frame menjadi track visual. |
| Team Classification | Jersey color clustering / classifier / embedding | Memetakan pemain visual ke Home/Away tanpa hand-label pada provided game. |
| Ball Tracking | Detector + Kalman + optical flow | Memperbaiki continuity posisi bola saat detection/tracking hilang. |
| Field Geometry | Provided homography + OpenCV | Proyeksi titik visual/tracking ke pitch 105,3 × 68 m. |
| Feature Engineering | NumPy / pandas / SciPy | Distance, velocity, acceleration, control radius, team compactness, boundary state. |
| ML Classifier | XGBoost / LightGBM / Random Forest / small MLP | Menghasilkan probability P(H), P(A), P(D) per frame. |
| Temporal Model | HMM / HSMM / state machine + Viterbi | Menstabilkan sequence dan mencegah flickering. |
| Evaluation | Python evaluator kompetisi | spellF1, macroF1, transF1, latency. |

# 3. Arsitektur Sistem yang Diperbarui

TACTICAL VIDEO (25 FPS, 199,319 frames)
        │
        ├─────────────── Computer Vision Branch ───────────────┐
        │                                                       │
        │   Decode frame → Player/Ball Detection               │
        │              → Multi-Object Tracking                 │
        │              → Team Classification                   │
        │              → Ball Visual Track                     │
        │              → Visual Motion / Optical Flow          │
        │                                                       │
        └───────────────────────────────────────────────────────┤
                                                                │
ATD TRACKING ── frame alignment per half ── tracking features ──┤
HOMOGRAPHY ──────────────────────────────── pitch coordinates ──┤
SMART TAGGING ───────────────────────────── tactical priors ─────┤
                                                                ▼
                                                   MULTIMODAL FEATURE FUSION
                                                                │
                                 ┌──────────────────────────────┴────────────────────────────┐
                                 │                                                           │
                         DEAD-BALL ENGINE                                       CONTROL / POSSESSION ENGINE
                                 │                                                           │
                                 └─────────────── frame probabilities P(H/A/D) ───────────────┘
                                                                │
                                                     HMM / HSMM / STATE MACHINE
                                                                │
                                                        VITERBI DECODING
                                                                │
                                                     POST-PROCESSING / SPELLS
                                                                │
                                                   submission.csv (199,319 rows)

# 4. Pipeline Computer Vision dari Video

## 4.1 Frame Decoder dan Indexing

Jangan mengekstrak video menjadi jutaan gambar permanen. Decode secara streaming/chunk agar storage dan I/O efisien. Setiap frame harus membawa video_frame sebagai primary key.

FrameRecord:
- source = 'tactical'
- video_frame: int
- timestamp_sec = video_frame / 25.0
- image: in-memory frame
- half: 1 / 2 / None
- id_frame: hasil mapping games.video_to_id(...) bila tersedia

- Gunakan tactical frame sebagai key utama seluruh feature table.

- Gunakan mapping per-half untuk menyambungkan ATD IdFrame ke video frame.

- Halftime/outside match dapat memiliki tracking = None dan harus ditangani eksplisit.

## 4.2 Player Detection

Player detection diperlukan bila tim ingin memperbaiki atau mengganti tracking yang tersedia, atau memperoleh evidence visual tambahan seperti bounding box, visible player count, dan team appearance.

| Pilihan | Kapan Dipakai | Catatan |
| --- | --- | --- |
| MVP: gunakan ATD player tracking | Jika tracking pemain cukup baik | Lebih cepat; fokus CV pada bola dan validasi visual. |
| Pre-trained detector | Jika ingin bounding box visual otomatis | Boleh memakai public/pre-trained model sesuai rules; dokumentasikan model dan versi. |
| Fine-tuned detector dari game lain | Jika pre-trained lemah di tactical view | Training labels harus berasal dari game lain, bukan provided game. |

## 4.3 Ball Detection & Ball Recovery

Ball adalah objek paling sulit karena kecil, blur, tertutup, dan sering hilang. Rancangan harus memakai kombinasi detector visual dan temporal filtering, bukan detection tunggal.

candidate detections
      ↓
filter by size / confidence / pitch location
      ↓
associate with previous ball trajectory
      ↓
Kalman prediction
      ↓
visual confirmation (detector / optical flow)
      ↓
ball_track_visual(x, y, confidence, observed/predicted)

- Jika detection bola kuat → update Kalman state.

- Jika detection hilang singkat → Kalman predict / optical flow bridge.

- Jika hilang lama → turunkan confidence dan bergantung pada ATD + team context; jangan otomatis menjadi D.

## 4.4 Multi-Object Tracking

ByteTrack atau BoT-SORT dapat digunakan untuk mempertahankan identitas visual antar-frame. ID visual tidak perlu dianggap sebagai nomor pemain permanen; yang penting continuity lokal untuk motion dan team-control evidence.

- Output minimal: track_id, bbox, center_x, center_y, confidence, frame.

- Jangan membuat koreksi ID manual pada provided game.

## 4.5 Team Classification

Untuk possession, sistem harus mengetahui apakah pemain terdekat/terkontrol adalah Home atau Away. Ini dapat diperoleh dari tracking metadata bila tersedia, atau dari visual jersey classification.

| Metode | Kelebihan | Risiko / Mitigasi |
| --- | --- | --- |
| ATD team membership | Cepat dan stabil | Pastikan mapping slot/metadata benar. |
| HSV/Lab jersey color clustering | Tanpa training label manusia pada match | Terpengaruh cahaya/referee; gunakan cluster temporal. |
| Embedding + clustering | Lebih robust terhadap variasi warna | Lebih berat; cluster harus otomatis. |
| Classifier dilatih dari game lain | Bisa sangat akurat | Data training harus dapat diakses dan dijelaskan. |

## 4.6 Homography & Pitch Mapping

Gunakan homography yang disediakan sebagai default karena sudah tersedia per source. Jika CV menghasilkan center/bottom-center bounding box pemain atau titik bola dalam pixel, proyeksikan ke koordinat pitch. Ini memungkinkan feature berbasis meter, bukan pixel.

pixel point (u,v)
    ↓ H(frame)
homogeneous transform
    ↓ normalize
pitch point (x_m, y_m)
    ↓
inside_pitch, distance_to_sideline, distance_to_goal_line,
player-ball distance, team centroid, local density

## 4.7 Optical Flow / Motion Features (Opsional)

Optical flow bukan komponen wajib MVP, tetapi berguna sebagai evidence tambahan ketika bola tidak terdeteksi atau untuk mengukur arah gerak lokal. Gunakan hanya bila memberi peningkatan terukur pada validation framework.

# 5. Sensor / Data Fusion: CV + ATD + Smart Tagging

Rancangan yang paling kuat bukan mengganti semua data provided secara langsung, melainkan membangun confidence-aware fusion. Setiap sumber memberikan observation dan confidence.

| Feature Group | Sumber Utama | Contoh Feature |
| --- | --- | --- |
| Ball position | ATD + CV | x/y, source confidence, missing duration, Kalman uncertainty |
| Player positions | ATD + CV optional | nearest H/A player, distances, density |
| Motion | ATD + CV | velocity, acceleration, optical flow magnitude |
| Geometry | Homography | inside pitch, boundary distance, zones |
| Tactical context | Smart tagging | set piece / phase prior / event proximity |
| Visual context | Video CV | ball visible, player cluster, referee/bbox evidence |
| Temporal history | Previous states | last possession, dwell time, transition age |

## 5.1 Confidence-Aware Fusion

if ATD ball valid and CV ball confident:
    fused_ball = weighted_average(ATD, CV)
elif ATD valid:
    fused_ball = ATD
elif CV confident:
    fused_ball = CV
else:
    fused_ball = temporal_prediction
    uncertainty = high

Bobot tidak boleh ditentukan dengan melihat label manual pada provided game. Gunakan physical consistency, confidence detector, residual Kalman, dan data dari game lain bila perlu.

# 6. Feature Engineering untuk Possession

| Kategori | Feature yang Disarankan |
| --- | --- |
| Ball-player relation | nearest_home_dist, nearest_away_dist, distance margin, ball-to-player relative velocity |
| Control evidence | dwell frames, repeated nearest-team consistency, ball speed after touch, trajectory continuity |
| Team structure | home/away centroid, local player density, compactness around ball |
| Ball state | speed, acceleration, missing duration, confidence, in/out pitch |
| Boundary | distance to touchline/goal line, outside-pitch flag |
| Tactical | set-piece tag, phase tag, restart evidence |
| Temporal | previous predicted state, time since last stable possession, candidate turnover duration |

# 7. Model Possession: dari Evidence ke H/A/D

## 7.1 Rekomendasi MVP

Untuk waktu kompetisi yang pendek, mulai dari rule/probability engine yang interpretable, lalu tambahkan ML jika feature sudah stabil.

Stage A: heuristic probability
    P(H) ← home control evidence
    P(A) ← away control evidence
    P(D) ← dead-ball evidence

Stage B: optional classifier
    features → XGBoost/LightGBM → P(H),P(A),P(D)

Stage C: temporal decoder
    probabilities + transition constraints → HMM/HSMM/Viterbi → final labels

## 7.2 Intelligent Dead-Ball Engine

- Ball outside pitch dari homography → evidence D tinggi.

- Smart-tagging set-piece/restart → evidence D, tetapi tetap harus sinkron frame.

- Low ball visibility saja tidak cukup untuk D.

- Optional CV: motion freeze / player reset pattern sebagai feature tambahan.

## 7.3 Controlled Possession Estimator

Perpindahan possession harus menunggu evidence kontrol, bukan satu sentuhan atau satu frame nearest-player. Gunakan candidate state + dwell threshold dan/atau learned control probability.

stable_state = H
candidate = A
candidate_age = 0

for frame:
    if evidence(A) > threshold:
        candidate_age += 1
    else:
        candidate_age = 0

    if candidate_age >= min_control_frames:
        stable_state = A

## 7.4 HMM vs HSMM

| Model | Kelebihan | Rekomendasi |
| --- | --- | --- |
| State Machine | Sangat interpretable dan cepat | Baseline temporal pertama. |
| HMM | Probabilistik, Viterbi mudah | Pilihan utama setelah emission probabilities tersedia. |
| HSMM | Memodelkan durasi spell secara eksplisit | Advanced; menarik karena metric adalah spellF1. |
| Sequence NN (TCN/LSTM/Transformer) | Bisa menangkap pola kompleks | Gunakan hanya dengan training data legal dari game lain yang cukup. |

# 8. Teknologi yang Direkomendasikan

| Area | Rekomendasi Utama | Alternatif / Advanced |
| --- | --- | --- |
| Language | Python 3.11+ | C++ untuk komponen performa tinggi |
| Video | OpenCV + FFmpeg | PyAV / decord |
| Detection | Ultralytics YOLO family / RT-DETR public model | Custom detector trained on other-game labels |
| Tracking | ByteTrack | BoT-SORT / custom Kalman association |
| Image processing | OpenCV | scikit-image |
| Numerical | NumPy, SciPy, pandas | Polars untuk pipeline besar |
| Kalman | FilterPy / custom implementation | torch-based differentiable filter |
| Classical ML | XGBoost / LightGBM / scikit-learn | CatBoost |
| Temporal | hmmlearn / custom Viterbi | HSMM custom implementation |
| Deep Learning | PyTorch | TensorFlow |
| Experiment config | YAML + argparse/Hydra | MLflow/W&B jika accessible & documented |
| Testing | pytest | CI GitHub Actions |
| Packaging | requirements.txt / pyproject.toml | Docker untuk reproducibility |

# 9. Strategi MVP vs Advanced

| Level | Komponen | Tujuan |
| --- | --- | --- |
| MVP-0 | Provided ATD + homography + smart tagging + baseline | Reproduce score resmi. |
| MVP-1 | CV ball detector + Kalman fusion | Perbaiki observasi bola hilang. |
| MVP-2 | Control features + dead-ball engine | P(H/A/D) yang lebih benar. |
| MVP-3 | HMM/Viterbi | Optimasi spell continuity dan transisi. |
| Advanced-1 | Player detector + ByteTrack + team classifier | Redundant visual tracking untuk mengoreksi/menambah ATD secara algoritmik. |
| Advanced-2 | HSMM / duration model | Eksplisit mengoptimalkan durasi possession spell. |
| Advanced-3 | Event/action recognition dari game lain | Tambahan evidence untuk pass/tackle/restart bila waktunya cukup. |
| Advanced-4 | Broadcast auxiliary fusion | Hanya jika alignment dapat dibuat secara algoritmik dan memberi gain. |

| Rekomendasi prioritas<br>Jangan mulai dengan membangun ulang semua tracking pemain. Mulai dari provided tracking, gunakan CV terutama untuk ball recovery dan visual evidence. Hanya mengganti player tracking jika audit menunjukkan provided tracking menjadi bottleneck nyata. |
| --- |

# 10. Struktur Data Hasil Ekstraksi

Jangan commit derived frame-level data yang berpotensi merekonstruksi match ke public repository. Struktur berikut adalah local cache yang dihasilkan ulang oleh code.

cache/
├── tactical/
│   ├── cv_ball.parquet
│   ├── cv_players.parquet
│   ├── fused_tracking.parquet
│   ├── features.parquet
│   └── probabilities.parquet
└── metadata/
    └── run_manifest.json

Example frame features:
video_frame, half, id_frame,
ball_x, ball_y, ball_confidence, ball_source,
nearest_home_dist, nearest_away_dist,
home_control_score, away_control_score,
dead_ball_score,
p_home, p_away, p_dead

# 11. Struktur Repository Revisi

more-than-a-hack/
├── data/                         # GITIGNORED
├── cache/                        # GITIGNORED derived match data
├── configs/
│   ├── cv.yaml
│   ├── fusion.yaml
│   ├── possession.yaml
│   └── temporal.yaml
├── src/
│   ├── game/                    # games.py / frame alignment
│   ├── video/                   # decoder, chunk reader
│   ├── cv/
│   │   ├── detector.py
│   │   ├── ball_detector.py
│   │   ├── tracker.py
│   │   ├── team_classifier.py
│   │   └── optical_flow.py
│   ├── tracking/                # ATD parser + alignment
│   ├── geometry/                # homography / pitch mapping
│   ├── fusion/                  # CV + ATD + confidence fusion
│   ├── features/
│   ├── possession/
│   ├── temporal/
│   ├── evaluation/
│   └── pipeline.py
├── scripts/
│   ├── extract_cv.py
│   ├── build_features.py
│   ├── predict.py
│   ├── evaluate_public_check.py
│   └── generate_submission.py
├── tests/
├── outputs/                     # submission only; no restricted derived data
├── requirements.txt
├── README.md
└── run.py

# 12. Urutan Pengerjaan yang Diperbarui

| Tahap | Pekerjaan | Output / Gate |
| --- | --- | --- |
| 0 | Rules, repo, data path, .gitignore, environment | Tidak ada data/video/derived tracking ter-commit. |
| 1 | Jalankan games.py utilities + verifikasi tactical paths dan frame count | 199.319 tactical frames dikenali. |
| 2 | Reproduce official baseline + evaluator | Baseline dapat diregenerasi. |
| 3 | Audit ATD, homography, smart tagging, missingness | Laporan bottleneck data. |
| 4 | Implement video decoder per chunk | Frame index 0..199318 stabil. |
| 5 | CV ball detection prototype pada sampling otomatis | Precision/continuity diagnostics tanpa hand-label provided game. |
| 6 | Ball Kalman + ATD/CV fusion | Fused ball track + uncertainty. |
| 7 | Pitch/spatial features + dead-ball engine | Frame-level evidence D. |
| 8 | Control features + P(H/A/D) | Emission probabilities. |
| 9 | State machine → HMM/Viterbi | Stable label sequence. |
| 10 | Ablation: ATD only vs +CV ball vs +HMM | Bukti CV memang memberi gain. |
| 11 | Optional player CV / HSMM / broadcast | Hanya jika baseline utama sudah stabil. |
| 12 | Full inference + reproducibility test | Repository menghasilkan CSV identik. |

# 13. Validasi Tanpa Melanggar Rules

- Provided game boleh ditonton untuk debugging dan memahami data, tetapi jangan membuat label/correction/constant berdasarkan frame tertentu secara manual.

- 5 menit ground truth publik hanya untuk sanity checking alignment dan format, bukan training/tuning.

- Untuk melatih detector/classifier supervised, gunakan hand-labelled data dari game lain yang dapat diakses panitia.

- Pseudo-label pada provided game boleh jika seluruh keputusan dihasilkan algoritma.

- Semua parameter yang penting harus tercatat di config dan bukan hasil edit manual frame-by-frame.

- Ablation dan diagnostics harus bisa dijalankan ulang dari code.

# 14. Experiment Matrix yang Wajib

| ID | Eksperimen | Pertanyaan yang Dijawab |
| --- | --- | --- |
| E0 | Official baseline | Apakah environment/evaluator benar? |
| E1 | ATD + improved missing handling | Berapa gain tanpa CV? |
| E2 | E1 + CV ball detector | Apakah evidence visual membantu? |
| E3 | E2 + Kalman/fusion | Apakah continuity bola meningkat? |
| E4 | E3 + dead-ball engine | Apakah D lebih stabil? |
| E5 | E4 + controlled possession | Apakah turnover palsu turun? |
| E6 | E5 + HMM/Viterbi | Apakah spellF1 naik karena sequence stabil? |
| E7 | E6 + optional player CV | Apakah rebuild tracking pemain layak biaya komputasi? |
| E8 | E6 + HSMM | Apakah duration modeling menguntungkan spellF1? |

# 15. Pembagian Tim 4 Orang

| Peran | Fokus Utama | Deliverable |
| --- | --- | --- |
| Engineer A — Video/CV | decoder, detector, ball visual track, optional player tracking | cv_ball + cv diagnostics |
| Engineer B — Tracking/Geometry | ATD parser, per-half alignment, homography, pitch features | aligned tracking + spatial features |
| Engineer C — Possession/Temporal | control model, dead-ball, HMM/HSMM | P(H/A/D) + final sequence |
| Engineer D — Evaluation/Integration | experiment runner, ablation, reproducibility, technical doc | metrics, run pipeline, submission |

# 16. Acceptance Criteria Teknis

- Pipeline membaca tactical video dan mempertahankan frame index persis 0..199318.

- Tracking-video join memakai offset per-half, bukan satu offset global.

- CV tidak memerlukan koreksi manual pada provided game.

- Setiap output CV memiliki confidence/quality indicator.

- Fusion memiliki fallback ketika CV atau ATD hilang.

- Frame probabilities dapat diproduksi sebelum temporal decoding.

- Final CSV berisi 199.319 baris data dan dapat diregenerasi identik.

- Ablation membuktikan kontribusi CV, bukan hanya menambah kompleksitas.

- README/technical document menjelaskan model publik/API/external data yang dipakai dan cara mengaksesnya.

# 17. Rekomendasi Keputusan Teknologi

| Stack yang paling saya rekomendasikan untuk tim<br>Python + OpenCV/FFmpeg + provided ATD/homography + CV ball detector + Kalman fusion + NumPy/pandas/SciPy + XGBoost/heuristic possession probabilities + HMM/Viterbi. ByteTrack/player detection dibuat tahap kedua, bukan tahap pertama. |
| --- |

Alasannya: kompetisi menilai possession spells. Bottleneck utama bukan sekadar mendeteksi semua pemain dengan sempurna, tetapi memperoleh evidence bola/kontrol yang cukup stabil dan kemudian menghasilkan sequence possession yang temporal-consistent. CV harus dipakai untuk menambah informasi yang benar-benar hilang atau unreliable, terutama bola, bukan membangun ulang seluruh ekosistem tanpa bukti kebutuhan.

# 18. Checklist Mulai Coding

1. Pastikan data tactical berada di lokasi yang dibaca MTAH_DATA_DIR / data/tactical.

1. Jalankan asset_paths()/require() dan verifikasi video + ATD + homography + smart tagging tersedia.

1. Baca metadata/frame count dan pastikan 199.319 frame tactical.

1. Implement test video_frame ↔ id_frame untuk kedua babak.

1. Jalankan baseline resmi dan simpan metrics.

1. Buat decoder video chunked.

1. Buat modul cv/ball_detector.py dengan output confidence.

1. Buat Kalman fusion dengan ATD ball.

1. Bangun feature table per tactical frame.

1. Bangun P(H/A/D), lalu HMM/Viterbi.

1. Jalankan ablation E0–E6 sebelum menambah player detector atau broadcast.

# 19. Ketentuan Baru: Pipeline Harus Algoritmik dan Reproducible

Ketentuan evaluasi memperjelas bahwa yang dinilai bukan hasil pengamatan manusia, tetapi sebuah sistem yang dapat dijalankan ulang. Dengan demikian, rancangan final harus memperlakukan setiap label H/A/D/X sebagai keluaran algoritma. Tidak boleh ada frame, segmen, transisi, posisi bola, atau identity switch pada provided game yang diperbaiki secara manual.

| Keputusan desain final<br>Jalur inference submission tidak menggunakan LLM. Label H/A/D/X dihasilkan oleh Computer Vision, tracking, geometry, feature engineering, probabilistic/classical ML, serta temporal decoding yang seluruhnya dapat dijalankan ulang. LLM boleh saja dipakai di luar jalur scoring untuk membantu menulis dokumentasi/kode, tetapi bukan sebagai sumber keputusan label pada match. |
| --- |

| Komponen | Status dalam rancangan | Alasan |
| --- | --- | --- |
| OpenCV / FFmpeg | WAJIB / utama | Decode video dan indeks frame secara deterministik. |
| ATD tracking | WAJIB / utama | Sumber observasi posisi pemain/bola yang disediakan. |
| Homography | WAJIB / utama | Transformasi pixel/pitch untuk spatial reasoning. |
| Smart tagging | BOLEH / auxiliary | Prior taktis yang tersedia pada tactical timeline. |
| YOLO / RT-DETR | BOLEH / direkomendasikan untuk ball recovery | Model publik dapat memperkaya observasi visual secara algoritmik. |
| Kalman / interpolation | DIREKOMENDASIKAN | Recovery trajectory dan uncertainty secara reproducible. |
| XGBoost / RF / heuristic scoring | BOLEH | Mengubah features menjadi P(H/A/D). |
| HMM / HSMM / Viterbi | DIREKOMENDASIKAN | Mengoptimalkan kontinuitas possession spell. |
| LLM sebagai labeler | TIDAK DIGUNAKAN | Tidak dibutuhkan dan menyulitkan determinisme/reproduksi. |
| Human correction pada provided game | DILARANG | Melanggar aturan hand annotation. |

# 20. Environment dan Dependency Strategy

Repository resmi sengaja memiliki dependency minimum yang sangat kecil. Evaluator dan baseline hanya membutuhkan numpy dan pandas, sehingga tim sebaiknya mempertahankan jalur submission tetap ringan, terkontrol, dan mudah direproduksi. Dependency Computer Vision dipisahkan sebagai kebutuhan pipeline tim, bukan evaluator.

| Lapisan | Python / Dependency | Rekomendasi |
| --- | --- | --- |
| Evaluator resmi | Python 3.8+; numpy; pandas | Jangan memodifikasi evaluator. Jalankan sebagai sumber kebenaran skor. |
| Starter notebooks | Python 3.9+ disarankan; kloppy; matplotlib | Gunakan untuk eksplorasi dan verifikasi data. |
| CV pipeline tim | Python 3.10/3.11 disarankan; OpenCV; PyTorch/Ultralytics jika perlu | Pin versi di requirements.txt. |
| Temporal / ML | NumPy/SciPy/scikit-learn atau custom implementation | Utamakan komponen yang deterministik dan mudah diaudit. |
| Reproducibility | requirements.txt + seed + config + optional Docker | Satu command harus menghasilkan submission yang sama. |

python evaluation/run_evaluation.py
python evaluation/validate_submission.py evaluation/results/team_name.csv
python run.py --config configs/final.yaml --output evaluation/results/team_name.csv

# 21. Konsekuensi Metrik: Optimasi Utama adalah spellF1

Rancangan model tidak boleh diarahkan ke accuracy atau macroF1 saja. Ranking menggunakan spellF1: possession dipandang sebagai objek temporal (maximal run H atau A), lalu prediksi dan ground truth dicocokkan satu-ke-satu berdasarkan temporal IoU pada beberapa threshold. Karena itu, model yang benar di tengah possession tetapi sering flicker di transisi dapat memiliki macroF1 bagus namun spellF1 buruk.

| Metrik | Makna | Konsekuensi Engineering |
| --- | --- | --- |
| spellF1 | Kualitas possession spells; metrik ranking | Prioritas utama: continuity, boundary, turnover timing, hindari fragmentation. |
| macroF1 | Ketepatan H/A/D per frame | Dipakai sebagai diagnostic/tie-break; jangan menjadi tujuan utama. |
| Transition timing | Ketepatan perubahan status | Frame alignment dan latency penting. |
| Validation status | Format dan jumlah frame | Submission invalid tidak akan dinilai. |

| Threshold keputusan eksperimen<br>Perbedaan sangat kecil pada public 5-minute window tidak boleh dianggap sebagai bukti model lebih baik. Public GT dipakai untuk sanity check, bukan pemilihan hyperparameter/model. |
| --- |

# 22. Strategi Label X (Abstention)

X bukan label biasa. Pada macroF1, X pada graded frame dihitung sebagai miss. Pada spellF1, evaluator menghilangkan X sebelum temporal scoring dengan mengisi berdasarkan label non-X terdekat sesuai aturan evaluator. Artinya X dapat berguna saat evidence hilang di tengah possession, tetapi berbahaya di sekitar transition karena dapat mempertahankan state lama terlalu lama.

| Kondisi | Strategi | Alasan |
| --- | --- | --- |
| Occlusion / ball missing di tengah possession yang sudah stabil | X dapat dipertimbangkan | Menghindari guess yang memecah spell; temporal fill mempertahankan state. |
| Dekat kandidat turnover / restart | Hindari X; pilih state berdasarkan evidence algoritmik | X dapat membawa state lama melewati transition. |
| Awal half / sequence | Gunakan aturan awal yang eksplisit dan diuji | Back-fill evaluator perlu dipahami agar tidak menciptakan bias. |
| Confidence rendah tetapi state temporal kuat | Lebih baik pertahankan state via HMM/state machine | Lebih eksplisit dan mudah dianalisis daripada banyak X. |

# 23. Public Ground Truth: Hanya untuk Sanity Check

Ground truth publik hanya mencakup sekitar 5 menit dan 17 possession. Karena sampelnya sangat kecil, rancangan eksperimen harus melarang penggunaan public window untuk training, threshold selection, smoothing-window selection, early stopping, atau pemilihan versi model.

- BOLEH: cek video-frame alignment, arti label, file format, dan apakah pipeline berjalan end-to-end.

- BOLEH: menjalankan evaluator untuk memastikan skor dan submission valid secara teknis.

- TIDAK BOLEH: memilih smoothing 25 vs 50 vs 125 berdasarkan public spellF1.

- TIDAK BOLEH: melatih classifier dengan 5 menit label publik.

- TIDAK BOLEH: menyesuaikan threshold possession/dead-ball agar public score naik.

- Jika perlu supervised learning, gunakan data game lain yang legal, accessible, documented, dan bukan human-produced data tentang provided game.

# 24. Pipeline Final yang Direkomendasikan

TACTICAL VIDEO + ATD + HOMOGRAPHY + SMART TAGGING
                │
                ▼
       [A] Exact frame alignment
                │
                ▼
       [B] Video decoding (OpenCV/FFmpeg)
                │
        ┌───────┴────────┐
        ▼                ▼
 [C1] Ball CV      [C2] Provided ATD
 YOLO/RT-DETR       players + ball
        │                │
        └───────┬────────┘
                ▼
       [D] Ball fusion / Kalman
          + confidence / uncertainty
                │
                ▼
       [E] Pitch projection + spatial features
           nearest player/team, distance,
           velocity, boundary, dwell time
                │
        ┌───────┴────────────┐
        ▼                    ▼
 [F1] Dead-ball score   [F2] Control score H/A
        └───────┬────────────┘
                ▼
          [G] P(H), P(A), P(D)
                │
                ▼
       [H] HMM / HSMM / state machine
            + Viterbi decoding
                │
                ▼
       [I] Optional X policy based on
           uncertainty + transition context
                │
                ▼
       [J] validate_submission.py
                │
                ▼
        FINAL CSV 199,319 frames

# 25. Urutan Implementasi yang Harus Diikuti

| Prioritas | Implementasi | Gate sebelum lanjut |
| --- | --- | --- |
| P0 | Clone repo, Python env, data path, .gitignore | Evaluator dan baseline resmi berjalan. |
| P1 | Pelajari 03_first_submission.ipynb + baseline.py | Tim memahami data → CSV → score. |
| P2 | Unit test games.py alignment per-half | video_frame ↔ IdFrame tepat. |
| P3 | Audit ATD missingness dan smart tagging | Bottleneck data terukur tanpa membuat hand-label. |
| P4 | Ball CV prototype dari tactical video | Output detection per frame + confidence, tanpa correction manual. |
| P5 | Kalman + CV/ATD fusion | Trajectory stabil dan fallback jelas. |
| P6 | Spatial/control/dead-ball features | Feature table algoritmik per frame. |
| P7 | Emission P(H/A/D) | Classifier/heuristic menghasilkan probability. |
| P8 | HMM/Viterbi + transition constraints | Flickering turun dan spells stabil. |
| P9 | Implement X policy bila benar-benar bermanfaat | Policy berbasis confidence, bukan edit manual. |
| P10 | Ablation tanpa tuning ke public GT | Model dipilih berdasarkan reasoning, other-game validation, dan diagnostics non-GT. |
| P11 | Full run + reproducibility hash | CSV regenerated byte-identical / equivalent content. |

# 26. Checklist Compliance Final

- Label pada provided game seluruhnya berasal dari code/algoritma.

- Tidak ada hand label, hand correction, frame-specific manual constant, atau transition correction.

- Public 5-minute GT tidak digunakan untuk training atau model selection.

- Tidak ada human-produced data tentang provided game dari Opta/StatsBomb/Wyscout/LaLiga/dll.

- Jika memakai data dari game lain, sumber, lisensi/akses, proses labeling, dan jumlah data dijelaskan.

- Jika memakai model pretrained/public, nama, versi/checkpoint, sumber, dan cara mendapatkannya dijelaskan.

- Tactical video tetap menjadi timeline authority untuk submission.

- Alignment ATD menggunakan offset berbeda per half.

- Pipeline tidak mengandalkan broadcast tanpa alignment algoritmik.

- Data pertandingan dan derived reconstructable data tidak masuk public repository.

- requirements.txt/pinned versions dan konfigurasi inference tersedia.

- Seed dan mode deterministic ditetapkan jika framework mendukung.

- validate_submission.py lulus.

- run_evaluation.py dapat membaca submission.

- Satu command repository dapat menghasilkan ulang final CSV.
