# Dokumentasi Perencanaan dan Eksekusi Proyek: More Than a Hack 2027

Direktori ini berisi Product Requirements Document (PRD) serta dokumentasi teknis pendukung untuk menuntaskan kompetisi **More Than a Hack 2027: Phase 1 (Ball Possession)** pada laga **FC Barcelona vs Athletic Club (`BAR-ATH`)** sebelum tenggat **17 November 2026**.

Seluruh dokumen di sini menyatukan rancangan sistem pada [PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md](../PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md) dengan ketentuan resmi pada kode dan dokumentasi repositori ([README.md](../README.md), [DATA.md](../DATA.md), [EVALUATION.md](../EVALUATION.md), [RULES.md](../RULES.md), [SUBMISSION.md](../SUBMISSION.md), [FAQ.md](../FAQ.md), [games.py](../games.py), serta modul di [evaluation/](../evaluation/) dan [starter/](../starter/)).

## Daftar Dokumen

| Dokumen | Peruntukan | Kapan Dibaca |
| --- | --- | --- |
| [PRD.md](PRD.md) | Product Requirements Document: visi produk, batasan kompetisi, koreksi fakta teknis dari rancangan awal, kebutuhan fungsional/non-fungsional, cakupan MVP vs lanjutan, serta kriteria penerimaan. | Langkah pertama seluruh anggota tim sebelum mulai coding. |
| [ARCHITECTURE_AND_DATA_CONTRACTS.md](ARCHITECTURE_AND_DATA_CONTRACTS.md) | Spesifikasi arsitektur modul, kontrak skema data (`cache/`, `configs/`), koreksi alignment frame (`frame_id - 1`), transformasi homografi invers (`unproject`), formulasi Kalman/Fusion, fitur, dan model temporal HMM/HSMM. | Saat merancang dan mengimplementasikan modul di `src/`. |
| [ENVIRONMENT_AND_REPO_SETUP.md](ENVIRONMENT_AND_REPO_SETUP.md) | Panduan setup environment Python, pemisahan dependency, pengamanan `.gitignore`, struktur folder target, dan checklist verifikasi aset awal. | Hari pertama saat onboarding perangkat tim. |
| [EXECUTION_PLAN_AND_EXPERIMENTS.md](EXECUTION_PLAN_AND_EXPERIMENTS.md) | Jadwal eksekusi sprint 7 hari (10 hingga 17 November 2026), pembagian peran tim 3 sampai 5 orang, gerbang prioritas P0 hingga P11, protokol validasi legal tanpa overfitting ke public GT, dan matriks eksperimen E0 sampai E8. | Acuan harian standup, tracking progres, dan pengambilan keputusan eksperimen. |
| [COMPLIANCE_AND_SUBMISSION_GUIDE.md](COMPLIANCE_AND_SUBMISSION_GUIDE.md) | Matriks kepatuhan aturan resmi (`RULES.md`), aturan privasi LLM dan repositori publik, checklist audit final sebelum unggah ke Google Form, serta template dokumen teknis resmi (`SUBMISSION.md`). | Sepanjang pengerjaan dan wajib diaudit menjelang deadline 17 November 2026. |

## Prinsip Sumber Kebenaran (Authoritative Hierarchy)

Jika terjadi perbedaan angka, definisi, atau aturan antar dokumen, urutan acuan yang berlaku adalah:

1. Kode resmi evaluator dan konstanta: [evaluation/possession_metrics.py](../evaluation/possession_metrics.py), [evaluation/run_evaluation.py](../evaluation/run_evaluation.py), [evaluation/validate_submission.py](../evaluation/validate_submission.py), dan [games.py](../games.py).
2. Dokumentasi resmi kompetisi: [RULES.md](../RULES.md), [SUBMISSION.md](../SUBMISSION.md), [DATA.md](../DATA.md), [EVALUATION.md](../EVALUATION.md), [CHALLENGE.md](../CHALLENGE.md), [FAQ.md](../FAQ.md), dan [OVERVIEW.md](../OVERVIEW.md).
3. Dokumen perencanaan proyek di dalam folder `docs/` ini.
4. Catatan draf awal di [PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md](../PERANCANGAN_SISTEM_ALGORITMIK_CV_MORE_THAN_A_HACK_2027.md) dan [LAPORAN_ANALISIS_DAN_PANDUAN_KOMPETISI.md](../LAPORAN_ANALISIS_DAN_PANDUAN_KOMPETISI.md) (beberapa asumsi teknis pada draf awal telah dikoreksi di [PRD.md](PRD.md) dan [ARCHITECTURE_AND_DATA_CONTRACTS.md](ARCHITECTURE_AND_DATA_CONTRACTS.md)).

