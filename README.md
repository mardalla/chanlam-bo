# chanlam-bo

Companion code for the manuscript on Bayesian optimisation of Chan–Evans–Lam coupling, comparing pure exploration of new conditions against replication of existing ones.

## Install

```
pip install -r requirements.txt
```

Python 3.10+. A CUDA GPU is recommended for full-scale runs.

## Reproduce

Build features from the raw data ([Open Reaction Database](https://open-reaction-database.org/)):

```
python src/ingest_csv.py --in_csv data/raw/dataset.csv \
    --out_raw data/processed/chanlam_raw.csv \
    --out_tidy data/processed/chanlam_tidy.csv \
    --drop_incomplete_pairs
python src/featurize.py
```

Run the experiments:

```
python experiments/run_forced_repeat.py --lambda_di 0.0 --outdir results_forced_lam0
python experiments/run_forced_repeat.py --lambda_di 1.0 --outdir results_forced_lam1
python experiments/run_mandatory_reps.py --outdir results_mandatory
```

Defaults match the manuscript (budget 200, 30 seeds). Open `experiments/compare_all.ipynb` to regenerate the figures.

## License

MIT. See `LICENSE`.
