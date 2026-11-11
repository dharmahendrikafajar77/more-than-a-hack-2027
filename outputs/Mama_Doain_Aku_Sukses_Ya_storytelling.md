# Laporan Data Storytelling Penguasaan Bola: FC Barcelona vs Athletic Club

**Tim Analis:** Mama Doain Aku Sukses Ya  
**Kesimpulan Utama:** Bola lebih dominan dan menguntungkan ke pihak **FC Barcelona (H)** dengan keunggulan porsi penguasaan **59.46% vs 40.54%** (selisih **18.92%** atau **11.91 menit** waktu bola hidup).

**Checksum Submission CSV (SHA-256):** `59f0cff74f3fa39df74c9bae3f724c9d9ef3d4dc8f3ebdb695e0d9ba6190a914`  

![Dashboard Data Storytelling](Mama_Doain_Aku_Sukses_Ya_storytelling.png)

---

## 1. Ringkasan Pertandingan dalam Bahasa Sederhana (*Data Storytelling*)

- **Waktu Efektif vs Bola Mati:** Dari total **158,857 frame** laga aktif (105.9 menit waktu berjalan di Babak 1 & 2, ditambah 40,462 frame luar laga berlabel `X`), bola benar-benar dimainkan selama **62.91 menit (59.4%)**. Sisanya **43.00 menit (40.6%)** adalah kondisi **Bola Mati (`D`)** seperti lemparan ke dalam, tendangan bebas, sepakan sudut, sepakan gawang, selebrasi gol, dan persiapan *kick-off*.
- **Kontras Gaya Bermain:** Total terjadi **358 rangkaian penguasaan bola** (*possession spells*) dengan rata-rata **10.5 detik**. **FC Barcelona (Home)** memegang bola **187 kali** dengan rata-rata **12.00 detik** per penguasaan (di mana **30.5%** di antaranya bertahan lama `>= 15 detik`). Sebaliknya, **Athletic Club (Away)** mencatat **171 kali** penguasaan dengan rata-rata lebih singkat (**8.95 detik**) dan banyak terhenti cepat (`< 5 detik` sebesar **39.8%**).
- **Pergeseran Momentum Pertandingan:** Dominasi **FC Barcelona** mencapai puncaknya pada fase **B2: 45-60'** (**70.1%** vs 29.9%). Namun **Athletic Club** sempat bangkit memberikan tekanan balik terkuatnya pada fase **B2: 75-90+'** (**55.2%** vs 44.8%).

---

## 2. Perbandingan Akhir: Home vs Away

| Metrik Lapangan | **Home (`H` — FC Barcelona)** | **Away (`A` — Athletic Club)** | **Bola Mati (`D`)** |
|---|---|---|---|
| **Jumlah Frame** | 56,110 | 38,249 | 64,498 |
| **Durasi Total** | **37.41 menit** | 25.50 menit | 43.00 menit |
| **Share Bola Hidup (%)** | **59.46%** | 40.54% | — |
| **Porsi Total Laga (%)** | 35.32% | 24.08% | 40.60% |
| **Jumlah Penguasaan (*Spells*)** | 187 kali | 171 kali | — |
| **Rata-rata / Median Durasi** | **12.00 s** / 7.96 s | 8.95 s / 6.32 s | — |
| **Penguasaan Terpanjang** | **88.24 detik** | 50.40 detik | — |
| **Build-Up Panjang (`>= 15s`)** | **57** (30.5%) | 32 (18.7%) | — |

### Perbandingan per Babak

| Babak | Durasi Berjalan | Bola Hidup | Home Share (`H`) | Away Share (`A`) | Bola Mati (`D`) |
|---|---|---|---|---|---|
| Babak 1 | 53.0 mnt | 32.1 mnt | **60.28%** | 39.72% | 39.42% |
| Babak 2 | 52.9 mnt | 30.8 mnt | **58.62%** | 41.38% | 41.78% |

### Dinamika Momentum per Segmen Waktu (~15 Menit)

| Segmen Waktu | Bola Hidup | Home (`H`) | Away (`A`) | Bola Mati (`D`) | Tim Dominan |
|---|---|---|---|---|---|
| B1: 0-15' | 12.0 mnt | 67.8% | 32.2% | 20.3% | **FC Barcelona** |
| B1: 15-30' | 9.5 mnt | 54.4% | 45.6% | 36.4% | **FC Barcelona** |
| B1: 30-45+' | 10.6 mnt | 57.1% | 42.9% | 53.9% | **FC Barcelona** |
| B2: 45-60' | 10.4 mnt | 70.1% | 29.9% | 31.0% | **FC Barcelona** |
| B2: 60-75' | 9.5 mnt | 62.1% | 37.9% | 36.7% | **FC Barcelona** |
| B2: 75-90+' | 11.0 mnt | 44.8% | 55.2% | 52.2% | **Athletic Club** |

### 5 Rangkaian Penguasaan Bola Terpanjang (*Top Spells*)

| Peringkat | Tim | Babak | Menit Perkiraan | Durasi | Rentang Frame |
|---|---|---|---|---|---|
| 1 | **FC Barcelona** (H) | Babak 2 | ~menit 64.6' | **88.24 detik** | `140217..142423` |
| 2 | **FC Barcelona** (H) | Babak 2 | ~menit 56.2' | **70.24 detik** | `127636..129392` |
| 3 | **Athletic Club** (A) | Babak 1 | ~menit 29.5' | **50.40 detik** | `55470..56730` |
| 4 | **FC Barcelona** (H) | Babak 1 | ~menit 36.6' | **44.64 detik** | `66040..67156` |
| 5 | **FC Barcelona** (H) | Babak 1 | ~menit 1.7' | **43.40 detik** | `13777..14862` |

---

## 3. Penjelasan Metrik Evaluasi & Arti Istilah

Berdasarkan evaluasi pada jendela publik 5 menit awal (`7.500` frame, `17` GT possessions):

| Metrik | Skor Sistem Kita | Arti & Makna Praktis |
|---|---|---|
| **`macroF1`** | **`0.7843`** | Rata-rata tak berbobot skor F1 kelas `H`, `A`, dan `D`. Mengukur ketepatan per frame tanpa bias ke kelas dominan seperti akurasi biasa. |
| **`spellF1`** | **`0.4000`** | **Metrik Peringkat Resmi Lomba.** Mengukur kesesuaian setiap *episode penguasaan utuh* lewat temporal IoU pada ambang `0.3..0.7`. |
| **`Alive/Dead F1`** | **`0.9416`** | Kemampuan model membedakan bola hidup (`H/A`) vs bola mati (`D`). |
| **`transF1` (±25f)** | **`0.2500`** | Ketepatan mendeteksi momen perpindahan penguasaan bola dalam toleransi ±1 detik (`lat_f`: `+9.0` frame). |
| **`shareE`** | **`0.0457`** | Selisih absolut porsi penguasaan Home prediksi (61.84%) vs ground truth (57.27%). |
| **`frag`** | **`1.38`** | Rasio jumlah segmen prediksi dibagi asli (ideal mendekati `1.0`; angka tinggi berarti hasil patah-patah). |
| **`kappa` / `acc`** | `0.6643` / `0.7903` | Kesesuaian frame terkoreksi peluang acak (*Cohen's Kappa*) dan akurasi per frame. |

> **Catatan Metodologi:** Seluruh statistik pertandingan penuh di atas dihitung otomatis dari keluaran inferensi model (karena ground truth penuh bersifat privat), sedangkan jendela publik 5 menit awal digunakan semata-mata sebagai pemeriksaan kewajaran (*sanity check*) sesuai aturan resmi kompetisi.
