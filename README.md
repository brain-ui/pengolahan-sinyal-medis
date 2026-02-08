# Pengolahan Sinyal Medis

**Biomedical Signal Processing** — Course materials for Universitas Indonesia.

Repositori ini berisi materi kuliah, demo, dan tugas untuk mata kuliah Pengolahan Sinyal Medis. Fokus pada sinyal **ECG**, **EEG**, serta **pengolahan citra medis** (X-ray, CT, MRI, ultrasound) dengan konteks klinis.

## Prasyarat

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — package manager

## Setup

```bash
# Install uv (jika belum)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
git clone <repo-url>
cd pengolahan-sinyal-medis

# Install dependencies
uv sync

# Install dev dependencies (untuk testing)
uv sync --extra dev
```

## Menjalankan Demo

```bash
# Buka Jupyter notebook
uv run jupyter notebook

# Atau jalankan notebook tertentu
uv run jupyter notebook demos/01_pengenalan_sinyal/demo_pengenalan_sinyal.ipynb
```

## Download Data

```bash
# Generate data sintetik
uv run python scripts/generate_synthetic_data.py

# Download data PhysioNet (ECG, EEG)
uv run python data/download_physionet.py

# Download data citra medis
uv run python data/download_images.py
```

## Struktur Repositori

```
├── demos/              # Notebook demo untuk kuliah
├── assignments/        # Tugas mahasiswa
├── src/medsinyal/      # Library utilitas bersama
├── scripts/            # Script contoh standalone
├── data/               # Data sintetik dan script download
└── tests/              # Unit tests
```

## Jadwal Kuliah

| Minggu | Topik | Demo | Tugas |
|--------|-------|------|-------|
| 1 | Pengenalan Sinyal Biomedis | `demos/01_pengenalan_sinyal/` | Tugas 1 |
| 2 | Sampling, Kuantisasi, Analisis Waktu | `demos/02_sampling_dan_kuantisasi/` | Tugas 1 (lanjutan) |
| 3 | Analisis Frekuensi | `demos/03_analisis_frekuensi/` | Tugas 2 |
| 4 | Filter Digital | `demos/04_filter_digital/` | Tugas 2 (lanjutan) |
| 5 | Sinyal ECG & Deteksi QRS | `demos/05_sinyal_ecg/` | Tugas 3 |
| 6 | Fitur ECG & Klasifikasi | `demos/06_fitur_ecg_dan_klasifikasi/` | Tugas 3 (lanjutan) |
| 7 | Sinyal EEG & Pita Frekuensi | `demos/07_sinyal_eeg/` | Tugas 4 |
| 8 | **UTS (Ujian Tengah Semester)** | Review session | Ujian |
| 9 | Analisis EEG Lanjutan | `demos/09_analisis_eeg_lanjutan/` | Tugas 4 (lanjutan) |
| 10 | Analisis Waktu-Frekuensi | `demos/10_analisis_waktu_frekuensi/` | — |
| 11 | Dasar Citra Medis | `demos/11_dasar_citra_medis/` | Tugas 5 |
| 12 | Pengolahan Citra Medis | `demos/12_pengolahan_citra_medis/` | Tugas 5 (lanjutan) |
| 13 | Segmentasi Citra Medis | `demos/13_segmentasi_citra_medis/` | Tugas 5 (lanjutan) |
| 14 | Machine Learning & Pipeline | `demos/14_ml_dan_pipeline/` | Tugas 6 |
| 15 | Presentasi Proyek Akhir (Grup A) | — | Tugas 6 due |
| 16 | Presentasi Proyek Akhir (Grup B) | — | — |

## Progesi Tugas

| Tugas | Format | Deskripsi |
|-------|--------|-----------|
| Tugas 1 | Jupyter notebook | Dasar sinyal, sampling, aliasing |
| Tugas 2 | Jupyter notebook | Analisis frekuensi, desain filter |
| Tugas 3 | Jupyter + script | Pipeline ECG (transisi ke script) |
| Tugas 4 | Python script (CLI) | Analisis band power EEG |
| Tugas 5 | Python script | Pipeline pengolahan citra medis |
| Tugas 6 | Python module/package | Klasifikasi ML |
| Proyek Akhir | Aplikasi lengkap | Pipeline end-to-end |

## Library Utilitas

Package `medsinyal` (`src/medsinyal/`) menyediakan fungsi-fungsi yang digunakan di seluruh kuliah:

- `medsinyal.io` — Loading data (sintetik, PhysioNet, DICOM)
- `medsinyal.viz` — Plotting standar untuk ECG/EEG/citra
- `medsinyal.filters` — Implementasi filter digital
- `medsinyal.ecg` — Pemrosesan sinyal ECG
- `medsinyal.eeg` — Pemrosesan sinyal EEG
- `medsinyal.imaging` — Pemrosesan citra medis

## Lisensi

MIT
