Title: Early Warning of Intraoperative Hypotension (VitalDB)

Summary
- This repository accompanies a Master’s thesis on short‑horizon prediction of intraoperative hypotension (MAP < 65 mmHg in the next 5 minutes) using VitalDB. It provides a reproducible pipeline for cohort construction, feature and sequence generation, baseline (logistic regression) and sequence (TCN) training, and evaluation (PR curves, calibration, bootstrap confidence intervals). The code is organized for end‑to‑end execution and artifact review.

Data Sources and Cohort
- Index endpoints: VitalDB `/cases` and `/trks` are downloaded to `data/raw/`.
- Cohort selection: Cases must contain invasive MAP (ART_MBP) and propofol rate (PPF20_RATE), and at least one remifentanil rate (RFTN20_RATE or RFTN50_RATE). An optional duration filter (≥ 3600 s) is applied if timestamps are present.
- Final cohort size: 1,959 cases. One case (3807) was excluded after alignment because it lacked a contiguous 30‑minute MAP history on the 10 s grid (under 60 s LOCF), preventing window construction.

Directory Layout (Key)
- `src/`: Python source (no single‑line comments; docstrings retained)
  - Index + cohort: `download_index.py`, `select_cohort.py`
  - Downloads: `download_tracks_parallel.py`
  - Alignment + labels: `align_and_label.py`
  - Baseline features + LR: `features_baseline.py`, `train_logreg.py`
  - Sequences + TCN: `build_sequences.py`, `train_tcn.py`
  - No‑MAP ablation: `no_map/build_sequences_no_map.py`, `no_map/train_tcn_no_map.py`
  - Plots + CIs: `plots/plot_pr_curves.py`, `plots/plot_pr_tcn_only.py`, `plots/plot_calibration_lr_only.py`, `plots/plot_auprc_bars.py`, `plots/bootstrap_cis.py`, `plots/bootstrap_cis_caselevel.py`
- `data/raw/`: index files (`cases.csv.gz`, `trks.csv.gz`) and downloaded `tracks/`
- `data/interim/`: aligned per‑case 10 s grids with labels (`case{ID}__aligned_10s.parquet`)
- `data/features/`: per‑case feature parquet files for LR
- `data/seq_cache/`: cached TCN sequences `{train,val,test}.npz` (full input set)
- `data/seq_cache_nomap/`: cached TCN sequences for no‑MAP ablation
- `models/`: trained models (TCN `.pt`, LR `.joblib` and scalers)
- `metrics/`: metrics JSON and figures (PR, calibration, bar charts, bootstrap CIs)
- `logs/`: run logs

Environment
- Python: 3.10–3.12
- Recommended packages: numpy, pandas, pyarrow, scikit‑learn, matplotlib, joblib, requests, torch (2.5.x CPU or CUDA/MPS as available)
- Example (CPU):
  - `pip install numpy pandas pyarrow scikit-learn matplotlib joblib requests`
  - `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1`

Reproducibility Controls
- Case‑level splits are frozen in `data/splits/case_splits_seed42.json` and reused across models.
- TCN cached sequences are built once and reused by `train_tcn.py` (no on‑the‑fly building).
- TCN checkpoints persist per‑channel normalization statistics (`mu`, `sd`) used for inference.
- Sequence cache folders include a short signature of channels and window config.

TCN Model (Implementation Details)
- Inputs: 30 minutes at 10 s resolution (180 steps), channels: MAP, HR, SpO2, ETCO2, RR_CO2, BT, PPF20_RATE, Remi_RATE_any, and missingness flags (PPF20_RATE_missing, Remi_missing_any).
- Architecture: 4 causal residual 1D conv blocks, kernel size 3, dilations [1, 2, 4, 8], channels [64, 64, 64, 64], dropout 0.2; global average pooling → linear head.
- Training: Adam (lr=1e‑3), batch size 64, BCEWithLogits with class weighting (neg/pos), 10 epochs, model selection by validation AUPRC; final test evaluated with the best checkpoint.

Workflow (End‑to‑End)
1) Download VitalDB indices
   - `python src/download_index.py`
   - Writes `data/raw/cases.csv.gz` and `data/raw/trks.csv.gz`.

2) Select cohort
   - `python src/select_cohort.py`
   - Writes `data/raw/cohort_caseids.txt` (expected 1,960 cases before alignment filtering).

3) Download tracks (parallel)
   - `python src/download_tracks_parallel.py`
   - Saves per‑track files under `data/raw/tracks/` for the required signals.

4) Align signals and label windows
   - `python src/align_and_label.py`
   - Writes `data/interim/case{ID}__aligned_10s.parquet`. All 1,960 cases aligned; case 3807 later excluded during feature/window construction.

5) Baseline features and LR
   - Build features: `python src/features_baseline.py`
   - Train LR: `python src/train_logreg.py --seed 42`
   - Artifacts: `models/logreg_seed42.joblib`, `models/logreg_scaler_seed42.json`, metrics in `metrics/logreg_metrics_seed42.json`.

6) Build TCN sequences (full inputs)
   - `python src/build_sequences.py --seed 42 --stride 3 --cache_dir data/seq_cache`
   - Produces `train.npz`, `val.npz`, `test.npz` and `meta.json` in a signatured subfolder.

7) Train TCN (load cached sequences)
   - `python src/train_tcn.py --seed 42 --epochs 10 --batch_size 64 --stride 3 --cache_dir data/seq_cache`
   - Artifacts: `models/tcn_seed42.pt`, metrics in `metrics/tcn_metrics_seed42.json`.

8) No‑MAP ablation (optional)
   - Build: `python src/no_map/build_sequences_no_map.py --seed 42 --stride 3 --cache_dir data/seq_cache_nomap`
   - Train: `python src/no_map/train_tcn_no_map.py --seed 42 --epochs 10 --batch_size 64 --stride 3 --cache_dir data/seq_cache_nomap`
   - Artifacts: `models/tcn_nomap_seed42.pt`, metrics in `metrics/tcn_nomap_metrics_seed42.json`.

9) Plots and Evaluation
   - PR curves (TCN‑only):
     - `python src/plots/plot_pr_tcn_only.py --seed 42 --stride 3 --model models/tcn_seed42.pt --cache_dir data/seq_cache --out metrics/pr_curve_tcn_seed42.png`
   - Combined PR (LR + TCN):
     - `python src/plots/plot_pr_curves.py --seed 42 --stride 3 --features_dir data/features --splits data/splits/case_splits_seed42.json --models_dir models --cache_dir_full data/seq_cache --cache_dir_nomap data/seq_cache_nomap --out metrics/pr_curves_with_tcn_seed42.png`
   - Calibration (LR only):
     - `python src/plots/plot_calibration_lr_only.py --seed 42 --out metrics/calibration_lr_seed42.png`
   - Bootstrap CIs (window‑level):
     - `python src/plots/bootstrap_cis.py --seed 42 --stride 3 --n_boot 1000 --out metrics/bootstrap_ci_seed42.json`
   - Bootstrap CIs (case‑level):
     - `python src/plots/bootstrap_cis_caselevel.py --seed 42 --stride 3 --n_boot 1000 --out metrics/bootstrap_caselevel_seed42.json`
   - AUPRC bar chart with error bars:
     - `python src/plots/plot_auprc_bars.py --ci_json metrics/bootstrap_caselevel_seed42.json --out metrics/auprc_bars_caselevel_seed42.png`

Key Results (Seed 42, Stride 3)
- LR (test): AUROC ≈ 0.867, AUPRC ≈ 0.668, Brier ≈ 0.155; test size 339,840 windows.
- TCN (test): AUROC ≈ 0.996, AUPRC ≈ 0.988, Brier ≈ 0.016; test size 119,937 windows.
- No‑MAP ablations: LR and TCN performance decreases substantially (see metrics JSON and bar chart with CIs).
- Bootstrap CIs (B=1,000): Provided in `metrics/bootstrap_ci_seed42.json` (window‑level) and `metrics/bootstrap_caselevel_seed42.json` (case‑level). Case‑level CIs are wider and account for within‑case correlation; conclusions are unchanged.

Notes and Limitations
- MAP‑only experiments are omitted from the final submission; associated code and artifacts have been removed. The no‑MAP ablation remains to contextualize MAP’s contribution.
- TCN and LR rely on the same case‑level split (frozen in `data/splits/`), but the number of test windows differs by modeling pipeline.
- Alignment uses 10 s grid with 60 s last‑observation‑carried‑forward tolerance inside each 30‑minute window; no information after time t is used for the next‑5‑minute label.
- Hardware: CPU is sufficient; GPU/Metal accelerates TCN training and inference.

Provenance and Traceability
- Cohort construction: `data/raw/cohort_caseids.txt` lists the 1,960 selected cases. `data/interim/` contains 1,960 aligned outputs; `data/features/` contains 1,959 feature files after excluding case 3807.
- All model artifacts, metrics, and plots are stored under `models/` and `metrics/`, with logs under `logs/`. Filenames encode seed and ablation where applicable.

