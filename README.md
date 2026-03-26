# magisterka

System do automatycznego wykrywania filmów wideo wygenerowanych przez modele AI
(Sora, Runway Gen-3/4.5, Pika, Luma, InVideo itp.) na podstawie pięciu niezależnych sygnałów:
widoczny watermark (OCR / YOLO), niewidzialny watermark, Optical Flow, zero-variance ROI
oraz metadane C2PA.

## Spis treści

- [Funkcje](#funkcje)
- [Architektura detektorów](#architektura-detektorów)
- [Struktura projektu](#struktura-projektu)
- [Wymagania systemowe](#wymagania-systemowe)
- [Instalacja](#instalacja)
- [Uruchomienie GUI](#uruchomienie-gui)
- [Uruchomienie benchmarku](#uruchomienie-benchmarku)
- [Dataset](#dataset)
- [Wyniki i ograniczenia](#wyniki-i-ograniczenia)

---

## Funkcje

- Analiza pojedynczego wideo przez interfejs graficzny (PyQt6)
- Detekcja widocznych watermarków (OCR + YOLO, słownik fraz generatorów AI)
- Detekcja niewidzialnych watermarków (DWT/DWT-DCT/RivaGAN przez `imwatermark`)
- Analiza Optical Flow (Farneback CPU/GPU) — wykrywa statyczne overlaye mimo ruchu kamery
- Detekcja zero-variance ROI — statyczne narożniki kadru
- Analiza FFT noise — artefakty upsamplingu AI
- Detekcja metadanych C2PA (Coalition for Content Provenance and Authenticity)
- Benchmark na czterech zestawach: `ai_baseline`, `adv_compressed`, `adv_cropped`, `adv_fp_trap`
- Diagnostyka persistent FN (`kod/tools/fn_diagnosis_v2.py`) z liczbową skalą problemu i wzorcami awarii
- Skrypt do budowy zróżnicowanego benchmarku watermarków (`kod/dataset/download_watermark_benchmark.py`)
- Skrypt do porównania metryk z narzędziami zewnętrznymi (`kod/tools/compare_external_apps.py`)
- Zapis surowych wyników każdego detektora do CSV — umożliwia strojenie progów bez ponownego przetwarzania

---

## Architektura detektorów

System stosuje **dwustopniową fuzję sygnałów**:

1. **Sygnały wysokiej precyzji** (samodzielnie wystarczające do klasyfikacji):
   - C2PA manifest z polem generatora AI
   - OCR / YOLO z dopasowaniem do słownika znanych fraz (`runway`, `sora`, `pika`, `luma` itp.)
   - Niewidzialny watermark z `best_similarity ≥ 0.85` do znanych sygnatur

2. **Sygnały heurystyczne / kontekstowe** (wymagają koroboracji):
   - Optical Flow overlay (statyczne kontury mimo ruchu kamery)
   - Zero-variance ROI (narożniki bez zmian przez cały film)
   - FFT noise artifacts (periodyczne artefakty upsamplingu)

Heurystyki geometryczno-ruchowe mają dobrą czułość, ale niską specyficzność — w finalnej
decyzji pełnią rolę pomocniczą wobec detektorów semantycznych i metadanych.

Decyzja finalna opiera się na ważonym `score` kalibrowanym eksperymentalnie na zbiorze
kalibracyjnym (threshold sweep → patrz `evaluate.py`).

---

## Struktura projektu

```
magisterka/
├── docs/
│   ├── opis pracy magisterskiej.pdf
│   └── źródła.txt
├── kod/
│   ├── main.py                  # launcher GUI
│   ├── gui.py                   # interfejs PyQt6
│   ├── advanced_detectors.py    # OF, zero-variance, invisible WM, FFT
│   ├── ocr_detector.py          # OCR + YOLO watermark detector
│   ├── c2pa_detector.py         # C2PA metadata detector
│   ├── watermark_detector.py    # wrapper
│   ├── config.py                # stałe konfiguracyjne
│   ├── super_resolution.py      # EDSR/FSRCNN upscaling przed OCR
│   ├── bulk_download.py         # pobieranie filmów (yt-dlp)
│   ├── diag_ocr.py              # narzędzie diagnostyczne OCR
│   ├── download_sota_weights.py # pobieranie wag modeli
│   ├── dataset/
│   │   ├── download_ai_baseline.py   # pobiera 27 filmów AI (YouTube)
│   │   ├── download_fp_traps.py      # pobiera 37 filmów-pułapek FP (YouTube)
│   │   ├── evaluate.py               # benchmark + surowy CSV
│   │   └── generate_adversarial.py   # wersje compressed/cropped
│   └── tools/
│       ├── quick_test.py        # test pojedynczego wideo z CLI
│       ├── fn_diagnosis_v2.py   # persistent FN: skala + wzorce awarii
│       ├── compare_external_apps.py # porównanie metryk z innymi aplikacjami
│       └── sample_videos.py     # sampling klatek
└── README.md
```

---

## Wymagania systemowe

- Python 3.10+
- **ffmpeg** — wymagane w PATH (`ffmpeg --version`)
- **Tesseract OCR** — wymagane w PATH, na Windows dodaj do zmiennej środowiskowej
- NVIDIA GPU + CUDA (opcjonalne — OF automatycznie wykrywa dostępność)

---

## Instalacja

```bash
git clone https://github.com/bartpom/magisterka.git
cd magisterka
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

---

## Uruchomienie GUI

```bash
cd kod
python main.py
```

---

## Uruchomienie benchmarku

### 1. Pobierz dataset

```bash
cd kod/dataset
python download_ai_baseline.py    # ~27 filmów AI
python download_fp_traps.py       # ~37 filmów-pułapek
python generate_adversarial.py    # wersje compressed + cropped
python download_watermark_benchmark.py --per-query 8
```

### 1b. Zdiagnozuj persistent false negatives

```bash
python ../tools/fn_diagnosis_v2.py
```

Skrypt raportuje m.in. odsetek źródłowych filmów AI, które są FN we wszystkich dodatnich splitach
oraz liczniki dominujących wzorców błędu (`no zero-variance ROIs`, `small static contour area` itd.).

### 2. Uruchom ewaluację

```bash
python evaluate.py
```

Wyniki trafiają do `kod/results/`:
- `raw_signals.csv` — surowe wyniki każdego detektora (do strojenia progów)
- `evaluation_results.csv` — decyzja finalna per wideo
- `metrics_summary.csv` — TP/TN/FP/FN, Accuracy, F1, FPR per kategoria
- `threshold_sweep.csv` — sweep progów `invisible_wm` i `optical_flow`

---

## Dataset

| Zestaw | Etykieta | Opis |
|---|---|---|
| `ai_baseline` | AI=1 | Filmy AI z watermarkiem, pobrane z YouTube |
| `adv_compressed` | AI=1 | Wersje silnie skompresowane (H.264, niszczą WM) |
| `adv_cropped` | AI=1 | Wersje przycięte o kilka % (niszczą overlaye narożnikowe) |
| `adv_fp_trap` | Real=0 | Prawdziwe nagrania z grafikami TV, billboardy, napisy |

Dataset nie jest dołączony do repo (pliki .mp4 są w .gitignore).
Skrypty pobierające korzystają z `yt-dlp`.

---


## Porównanie z innymi aplikacjami

Po uruchomieniu `evaluate.py` możesz porównać metryki z narzędziem zewnętrznym:

```bash
python kod/tools/compare_external_apps.py \
  --external-csv tmp/tool_x_results.csv \
  --external-name ToolX \
  --ext-filename-col filename \
  --ext-pred-col detected
```

Wynik per plik zapisuje się do `kod/results/latest/external_comparison.csv`.
To ułatwia rozdział porównawczy w pracy (TP/TN/FP/FN, F1, FPR dla tych samych próbek).

## Wyniki i ograniczenia

| Kategoria | Acc | F1 | FPR | Uwagi |
|---|---|---|---|---|
| ai_baseline | 92.6% | wysoki | 0% | Detekcja watermarków AI |
| adv_compressed | 92.6% | wysoki | 0% | Odporna na rekompresję |
| adv_cropped | 85.2% | dobry | 0% | Lekki spadek przy przycinaniu |
| adv_fp_trap | ~19% | ~0.00 | 81% | **Główny problem — fałszywe alarmy** |

**Główne ograniczenie:** heurystyki OF i zero-variance dają fałszywe alarmy na
filmach z grafikami TV, billboardami i stałymi overlayami mediów społecznościowych.
Planowana poprawa: fuzja ważona sygnałów z kalibrowanymi progami.
