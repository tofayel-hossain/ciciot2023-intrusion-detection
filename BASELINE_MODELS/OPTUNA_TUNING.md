
# Optuna tuning for GAT and GIN (CICIoT2023)

This adds a Kaggle-compatible tuning pipeline without overwriting the baseline notebook.

## Search space

- k-NN neighbours: 3, 4, 5, 6, 7
- hidden dimension: 32, 64, 128
- dropout: 0.1–0.5
- learning rate and weight decay
- optimizer: Adam, AdamW, RAdam
- loss: ordinary cross-entropy or inverse-frequency class-weighted cross-entropy
- GAT attention heads: 2, 4, 8

The objective is validation macro-F1. Optuna TPE search and median pruning are enabled, with early stopping. Class weights are calculated from training labels only. The test split is evaluated only after parameter selection. A fixed equal-weight GIN+GAT soft-voting hybrid is reported separately.

## Kaggle steps

1. Open the existing baseline notebook and enable a GPU if available.
2. Run the cells for imports, data loading, preprocessing, split, graph/model definitions. Stop before the original baseline training/evaluation cells.
3. Install Optuna if needed:

   ~~~python
   %pip install -q optuna
   ~~~

4. Upload this Python file to the Kaggle session or clone the repository, then execute the file from its actual location. For example:

   ~~~python
   %run /kaggle/working/optuna_tuning.py
   studies, best_configs, tuned_models, tuned_results = run_gat_gin_optuna_pipeline()
   ~~~

5. Results and artifacts are written to /kaggle/working/optuna_gat_gin/.

## Budget and methodology

The default is 30 trials per model, up to 50 epochs per trial, patience 10. Start with 10 trials to smoke-test, then use 20–30 or more if Kaggle time permits. This budget does not guarantee a global optimum.

The original notebook remains unchanged. The split and preprocessing are inherited from that notebook, and each k builds separate train/validation/test graphs as in the baseline. The current GAT and GIN classes are two-layer architectures, so this script does not claim to tune num_layers; it tunes parameters the existing classes actually expose. For thesis reporting, repeat selected configurations with multiple seeds and report mean and standard deviation.
