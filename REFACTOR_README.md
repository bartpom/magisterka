# Refactor API Layout - Dokumentacja

## Przegląd zmian

Refactor reorganizuje kod z płaskiej struktury `kod/` do modularnej architektury z czystym API.

### Przed refactorem
```
kod/
├── ai_detector.py          # 1500+ linii - wszystko w jednym pliku
├── enhanced_detector.py
├── videomae_detector.py
├── advanced_detectors.py
├── gui.py                  # GUI importuje bezpośrednio ai_detector
└── ...
```

### Po refactorze
```
src/magisterka_detector/
├── __init__.py             # Eksportuje publiczne API
├── types.py                # Dataclasses i typy
├── api.py                  # Główny entrypoint (analyze_media, analyze_batch, begin_run)
├── extractors/             # Ekstrakcja cech
│   ├── video_frames.py
│   ├── hf_image.py         # TODO
│   ├── videomae.py         # TODO
│   ├── forensic.py         # TODO
│   └── enhanced.py         # TODO
├── scoring/                # Scoring i decyzje
│   ├── fuse.py
│   ├── policy.py
│   └── calibration.py      # TODO
├── reporting/              # Raporty
│   ├── run_dir.py
│   └── writers.py
└── core/                   # Orkiestracja
    └── pipeline.py

gui/
└── gui.py                  # Przerobimy na import z magisterka_detector.api
```

## Publiczne API

### Podstawowe użycie (GUI/CLI)

```python
from magisterka_detector import analyze_media, begin_run, AnalyzeOptions

# Utwórz katalog runu
run_dir = begin_run(reports_root="reports")

# Konfiguracja analizy
opts = AnalyzeOptions(
    mode="combined",             # "ai" | "deepfake" | "combined"
    max_frames=60,
    enable_hf_face=True,
    enable_hf_scene=True,
    enable_videomae=True,
    enable_forensic=True,
    decision_policy="high_precision",  # "high_precision" | "balanced" | "high_recall"
    write_txt=True,
    write_json=False,
)

# Analizuj plik
result = analyze_media(
    "path/to/video.mp4",
    opts=opts,
    progress=lambda curr, tot: print(f"{curr}/{tot}"),
    cancel=lambda: False,  # Zwróć True, żeby przerwać
    run_dir=run_dir,
)

print(f"Verdict: {result.verdict}")
print(f"Score: {result.final_score:.2f}%")
print(f"Report: {result.report_txt_path}")
```

### Analiza wsadowa

```python
from magisterka_detector import analyze_batch

files = ["video1.mp4", "video2.mp4", "image.jpg"]
results = analyze_batch(
    files,
    opts=opts,
    progress=lambda curr, tot: print(f"Global: {curr}/{tot}"),
    run_dir=run_dir,
)

for result in results:
    print(f"{result.features.path}: {result.verdict} ({result.final_score:.2f}%)")
```

### Typy danych

#### AnalyzeOptions
```python
@dataclass
class AnalyzeOptions:
    mode: DetectMode = "combined"              # Tryb detekcji
    max_frames: int = 60                       # Max klatek do analizy
    
    # Feature extractors
    enable_hf_face: bool = True                # HF model twarzy/postaci
    enable_hf_scene: bool = True               # HF model sceny
    enable_videomae: bool = True               # VideoMAE
    enable_forensic: bool = True               # Forensic (ELA/FFT/jitter/...)
    enable_enhanced: bool = False              # Enhanced detector
    enable_watermark: bool = False             # OCR watermark
    
    # Scoring
    decision_policy: PolicyName = "high_precision"
    calibration_thresholds_path: Optional[str] = None
    
    # Reporting
    reports_root: str = "reports"
    write_txt: bool = True
    write_json: bool = False
```

#### AnalysisResult
```python
@dataclass
class AnalysisResult:
    verdict: str                               # "FAKE (PRAWDOPODOBNE)" | "REAL (...)" | "NIEPEWNE"
    final_score: float                         # 0-100
    features: AnalysisFeatures                 # Szczegółowe cechy
    flags: List[str]                           # Ostrzeżenia/flagi
    report_txt_path: Optional[str]             # Ścieżka do TXT
    report_json_path: Optional[str]            # Ścieżka do JSON
    report_folder: Optional[str]               # Katalog raportu
```

#### AnalysisFeatures
```python
@dataclass
class AnalysisFeatures:
    media_kind: MediaKind                      # "video" | "image"
    path: str                                  # Pełna ścieżka
    
    # AI signals (0-100)
    ai_face_score: Optional[float] = None
    ai_scene_score: Optional[float] = None
    ai_video_score: Optional[float] = None
    
    # Forensic
    jitter_px: Optional[float] = None
    ela_score: Optional[float] = None
    fft_score: Optional[float] = None
    border_artifacts: Optional[float] = None
    face_sharpness: Optional[float] = None
    face_ratio: Optional[float] = None
    face_frames: int = 0
    total_frames: int = 0
    
    raw: Dict[str, Any] = field(default_factory=dict)  # Debug/raw data
```

## Migracja GUI

### Przed (obecny `kod/gui.py`)

```python
import ai_detector

# W workerze:
res = ai_detector.scan_for_deepfake(
    path,
    progress_callback=cb,
    check_stop=lambda: self._stop,
    do_face_ai=self._do_face_ai,
    do_forensic=self._do_forensic,
    run_dir=self._run_dir,
)

# Potem GUI ręcznie normalizuje/fuzuje:
if isinstance(res, tuple) and len(res) == 2:
    ai_res, for_res = res
    # ... ekstrakcja pól ...
    # ... fuzja score'ów ...
    # ... werdykt z progów ...
```

### Po (nowy `gui/gui.py`)

```python
from magisterka_detector import analyze_media, begin_run, AnalyzeOptions

# W workerze:
run_dir = begin_run()
opts = AnalyzeOptions(
    mode="combined",
    enable_forensic=self._do_forensic,
    enable_watermark=self._do_watermark,
    write_txt=True,
)

result = analyze_media(
    path,
    opts=opts,
    progress=cb,
    cancel=lambda: self._stop,
    run_dir=run_dir,
)

# result.verdict już gotowe
# result.final_score już gotowe
# result.features.* - wszystkie cechy
```

**Usuń z GUI:**
- `_load_local_ai_detector()` - nie potrzebne
- `_normalize_details()` - API zwraca znormalizowane
- `_fuse_ai_score()` - scoring w core
- `_compute_deepfake_score()` - scoring w core
- `_verdict_from_fallback_thresholds()` - policy w core

**Dodaj do GUI:**
- `from magisterka_detector import analyze_media, begin_run, AnalyzeOptions`

## Status implementacji

### ✅ Zrobione (obecny commit)
- [x] Struktura katalogów
- [x] `types.py` - wszystkie dataclasses
- [x] `api.py` - publiczny interface
- [x] `extractors/video_frames.py` - sampling klatek
- [x] `scoring/fuse.py` - fuzja sygnałów
- [x] `scoring/policy.py` - werdykty i progi
- [x] `reporting/run_dir.py` - katalogi runów
- [x] `reporting/writers.py` - TXT/JSON
- [x] `core/pipeline.py` - orkiestracja (placeholder extractors)

### 🚧 TODO (następne commity)
- [ ] `extractors/hf_image.py` - przenieść HF pipelines z `ai_detector.py`
- [ ] `extractors/videomae.py` - wrapper na `videomae_detector.py`
- [ ] `extractors/forensic.py` - przenieść ELA/FFT/jitter z `ai_detector.py`
- [ ] `extractors/enhanced.py` - opcjonalny wrapper na `enhanced_detector.py`
- [ ] `scoring/calibration.py` - ładowanie progów per-run
- [ ] `cli.py` - CLI zamiast logiki w `ai_detector.py`
- [ ] Migracja `gui.py` - import z `magisterka_detector.api`
- [ ] Testy jednostkowe
- [ ] Dokumentacja API (docstringi + sphinx)

## Jak pobrać i przetestować

### Pobranie brancha

```bash
# Fetch nowego brancha
git fetch origin refactor/api-layout

# Checkout
git checkout refactor/api-layout

# Lub bezpośrednio:
git checkout -b refactor/api-layout origin/refactor/api-layout
```

### Instalacja (edytowalny tryb)

```bash
# Z katalogu głównego repo:
pip install -e .

# Albo jeśli nie ma setup.py (tymczasowo):
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

### Test podstawowego API

```python
import sys
sys.path.insert(0, "src")  # jeśli nie zainstalowałeś przez pip

from magisterka_detector import analyze_media, begin_run, AnalyzeOptions

run_dir = begin_run()
opts = AnalyzeOptions(write_txt=True)

# Placeholder - zwróci wynik z dummy scores
result = analyze_media(
    "test_video.mp4",
    opts=opts,
    run_dir=run_dir,
)

print(result.verdict)
print(result.final_score)
```

**UWAGA:** Obecny commit ma placeholder extractors (zwracają dummy scores 50-55%), więc wyniki nie są prawdziwe. Kolejne commity dodadzą prawdziwe ekstraktory.

## Architektura

### Przepływ danych

```
GUI/CLI
  |
  v
api.analyze_media(path, opts)
  |
  v
core.pipeline.run_pipeline()
  |
  +---> extractors.video_frames (sampling)
  |       |
  |       v
  +---> extractors.hf_image (HF models)
  |       |
  |       v
  +---> extractors.videomae (VideoMAE)
  |       |
  |       v
  +---> extractors.forensic (ELA/FFT/...)
  |       |
  |       v
  +---> scoring.fuse (combine signals)
  |       |
  |       v
  +---> scoring.policy (verdict)
  |       |
  |       v
  +---> reporting.writers (TXT/JSON)
  |
  v
AnalysisResult
```

### Separation of Concerns

| Warstwa | Odpowiedzialność |
|---------|------------------|
| `api.py` | Publiczny interface, walidacja argumentów |
| `types.py` | Dataclasses, typy, sygnatury |
| `core/pipeline.py` | Orkiestracja: kolejność wywołań, przepływ danych |
| `extractors/*` | Ekstrakcja cech z mediów (bez scoring) |
| `scoring/*` | Fuzja sygnałów, decyzje, progi |
| `reporting/*` | Zapis wyników (TXT/JSON), katalogi |
| `gui.py` | Wyświetlanie, interakcja użytkownika (zero logiki biznesowej) |

## FAQ

### Czy stare `kod/ai_detector.py` będzie działać?

Tak, przez okres przejściowy (`kod/` zostanie) — ale docelowo `gui.py` przestawi się na nowe API, a `ai_detector.py` będzie deprecated.

### Czy mogę używać obu równocześnie?

Tak, ale GUI powinno używać **tylko** nowego API. Stare moduły zostaną jako legacy/reference.

### Co z `calibrate_thresholds.py`?

Przenosimy do `scoring/calibration.py` i ładowanie progów będzie w `AnalyzeOptions.calibration_thresholds_path`.

### Co z `tools/`?

Skrypty pomocnicze (`bulk_download.py`, `calibrate_ensamble.py`, itp.) zostaną w `kod/tools/` lub przepiszemy na CLI komendy.

### Jak dodać nowy ekstraktor?

1. Stwórz plik w `src/magisterka_detector/extractors/`
2. Zdefiniuj funkcję zwracającą Dict[str, float] (znormalizowane 0-100)
3. Dodaj wywołanie w `core/pipeline.py`
4. Dodaj flagę `enable_X` w `AnalyzeOptions`

## Kontakt

Pytania/problemy: otwórz issue na GitHubie lub napisz do @bartpom.
