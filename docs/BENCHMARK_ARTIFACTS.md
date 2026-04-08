# Benchmark artifacts and thesis workflow

This short note explains how to treat files in `kod/results/latest/` as the canonical source
for thesis tables, confusion matrices, and benchmark summaries.

## Canonical files after `python kod/dataset/evaluate.py`

The latest benchmark run should populate `kod/results/latest/` with at least:

- `evaluation_results.csv` — final binary decision per video
- `metrics_summary.csv` — per-split TP/TN/FP/FN and aggregate metrics

Depending on the active evaluation configuration, the run may also generate:

- `raw_signals.csv` — raw detector outputs before fusion
- `run_info.txt` — selected detector version and threshold configuration
- `best_config_selection.csv` — best parameter selection from sweep
- `pareto_frontier.csv` — non-dominated operating points
- `feature_activation_summary.csv` — feature activation rates per split

## Recommended workflow for thesis writing

1. Run the benchmark:

```bash
python kod/dataset/evaluate.py
```

2. Build a compact markdown summary:

```bash
python kod/tools/summarize_latest_results.py
```

This writes:

- `kod/results/latest/thesis_summary.md`
- `kod/results/latest/global_confusion_matrix.csv`

3. Use those generated files as the primary source when writing:

- global TP / TN / FP / FN,
- per-split recall and FPR,
- selected operating point,
- confusion matrix table,
- short benchmark summary for README or thesis text.

## Why this matters

This workflow reduces manual transcription errors. It is especially useful when the detector is
under active iteration and metrics can change across commits.
