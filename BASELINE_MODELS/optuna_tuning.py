
"""
Optuna tuning for the CICIoT2023 GAT/GIN baseline notebook.

Run this after the baseline notebook's imports, preprocessing/split, model-class,
and graph-construction cells have executed, and before its original test evaluation.
See OPTUNA_TUNING.md for Kaggle instructions.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import optuna

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

OPTUNA_SETTINGS = {
    "seed": int(SEED),
    "n_trials_per_model": 30,
    "max_epochs": 50,
    "early_stopping_patience": 10,
    "timeout_seconds_per_model": None,
    "models": ["GIN", "GAT"],
    "output_dir": "/kaggle/working/optuna_gat_gin",
}
OUT_DIR = Path(OPTUNA_SETTINGS["output_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

_REQUIRED_GLOBALS = [
    "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
    "class_names", "n_classes", "device", "set_seed",
    "make_split_graph", "GINBaseline", "GATBaseline",
]
_missing = [name for name in _REQUIRED_GLOBALS if name not in globals()]
if _missing:
    raise RuntimeError(
        "Run the setup, preprocessing, split, and model-definition cells in the "
        "baseline notebook first. Missing variables: " + str(_missing)
    )

IN_DIM = int(X_train.shape[1])
OUT_DIM = int(n_classes)
_GRAPH_CACHE = {}


def _get_graphs(k):
    """Cache split-specific graphs for each k value."""
    k = int(k)
    if k not in _GRAPH_CACHE:
        _GRAPH_CACHE[k] = {
            "train": make_split_graph(X_train, y_train, k),
            "val": make_split_graph(X_val, y_val, k),
            "test": make_split_graph(X_test, y_test, k),
        }
    return _GRAPH_CACHE[k]


def _make_model(model_name, params):
    hidden = int(params["hidden_dim"])
    dropout = float(params["dropout"])
    if model_name == "GIN":
        return GINBaseline(IN_DIM, hidden, OUT_DIM, dropout)
    if model_name == "GAT":
        return GATBaseline(IN_DIM, hidden, OUT_DIM, dropout, int(params["gat_heads"]))
    raise ValueError("Unsupported model: " + str(model_name))


def _class_weights(mode):
    """Compute class weights from training labels only."""
    if mode == "none":
        return None
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=OUT_DIM)
    if np.any(counts == 0):
        raise ValueError("A class is absent from the training split; review stratification.")
    weights = len(y_train) / (OUT_DIM * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_optimizer(name, model, lr, weight_decay):
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "RAdam":
        return torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError("Unsupported optimizer: " + str(name))


def _macro_f1(y_true, y_pred):
    return precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]


def _fit_trial(model_name, params, graphs, trial=None, final_fit=False):
    """Train on train graph and keep the checkpoint with highest validation macro-F1."""
    trial_seed = 10000 if final_fit else int(trial.number)
    set_seed(int(OPTUNA_SETTINGS["seed"]) + trial_seed)
    model = _make_model(model_name, params).to(device)
    train_data = graphs["train"].to(device)
    val_data = graphs["val"].to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(params["class_weight_mode"]))
    optimizer = _make_optimizer(
        params["optimizer"], model, float(params["learning_rate"]),
        float(params["weight_decay"])
    )

    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = []
    for epoch in range(1, int(OPTUNA_SETTINGS["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_data.x, train_data.edge_index)
        loss = criterion(logits, train_data.y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_data.x, val_data.edge_index)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_true = val_data.y.cpu().numpy()
            val_f1 = float(_macro_f1(val_true, val_pred))
            val_loss = float(criterion(val_logits, val_data.y).item())

        history.append({
            "epoch": epoch, "train_loss": float(loss.item()),
            "val_loss": val_loss, "val_macro_f1": val_f1,
        })
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1

        if trial is not None:
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned("Pruned at epoch " + str(epoch))
        if stale >= int(OPTUNA_SETTINGS["early_stopping_patience"]):
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint produced for " + model_name)
    model.load_state_dict(best_state)
    return model, best_f1, best_epoch, history


def _suggest_params(trial, model_name):
    params = {
        "k_neighbors": trial.suggest_categorical("k_neighbors", [3, 4, 5, 6, 7]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RAdam"]),
        "class_weight_mode": trial.suggest_categorical(
            "class_weight_mode", ["none", "balanced"]
        ),
    }
    if model_name == "GAT":
        params["gat_heads"] = trial.suggest_categorical("gat_heads", [2, 4, 8])
    return params


def run_optuna_study(model_name):
    sampler = optuna.samplers.TPESampler(seed=int(OPTUNA_SETTINGS["seed"]))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=10, interval_steps=1
    )
    study = optuna.create_study(
        study_name="ciciot2023_" + model_name.lower() + "_tuning",
        direction="maximize", sampler=sampler, pruner=pruner,
    )

    def objective(trial):
        params = _suggest_params(trial, model_name)
        _, score, epoch, _ = _fit_trial(
            model_name, params, _get_graphs(params["k_neighbors"]), trial=trial
        )
        trial.set_user_attr("best_epoch", int(epoch))
        return float(score)

    study.optimize(
        objective,
        n_trials=int(OPTUNA_SETTINGS["n_trials_per_model"]),
        timeout=OPTUNA_SETTINGS["timeout_seconds_per_model"],
        gc_after_trial=True,
        show_progress_bar=True,
    )
    study.trials_dataframe(attrs=("number", "value", "params", "state")).to_csv(
        OUT_DIR / (model_name.lower() + "_optuna_trials.csv"), index=False
    )
    best = dict(study.best_trial.params)
    best["best_validation_macro_f1"] = float(study.best_value)
    best["best_epoch"] = int(study.best_trial.user_attrs.get("best_epoch", 0))
    (OUT_DIR / (model_name.lower() + "_best_config.json")).write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    print(model_name, "best validation macro-F1:", round(study.best_value, 5))
    print(json.dumps(best, indent=2))
    return study, best


def _evaluate_test(model, graph):
    """Evaluate once on held-out test data after parameter selection."""
    model.eval()
    data = graph.to(device)
    start = time.time()
    with torch.no_grad():
        probabilities = F.softmax(model(data.x, data.edge_index), dim=1).cpu().numpy()
    inference_time = time.time() - start
    y_true = data.y.cpu().numpy()
    y_pred = probabilities.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r),
        "macro_f1": float(f1), "weighted_f1": float(weighted_f1),
        "inference_time_sec": float(inference_time),
    }
    return metrics, y_true, y_pred, probabilities


def run_gat_gin_optuna_pipeline():
    """Tune both models, refit best configurations, and report equal-weight soft voting."""
    studies, best_configs = {}, {}
    for name in OPTUNA_SETTINGS["models"]:
        studies[name], best_configs[name] = run_optuna_study(name)

    final_models, probabilities_by_model, test_graphs = {}, {}, {}
    result_rows = []
    for name, best in best_configs.items():
        params = {k: v for k, v in best.items()
                  if k not in ("best_validation_macro_f1", "best_epoch")}
        graphs = _get_graphs(params["k_neighbors"])
        model, val_f1, epoch, history = _fit_trial(
            name, params, graphs, trial=None, final_fit=True
        )
        metrics, y_true, y_pred, proba = _evaluate_test(model, graphs["test"])
        final_models[name] = model
        probabilities_by_model[name] = proba
        test_graphs[name] = graphs["test"]
        result_rows.append({
            "model": name, "selected_val_macro_f1": float(val_f1),
            "selected_epoch": int(epoch), **metrics,
        })
        torch.save(model.state_dict(), OUT_DIR / (name.lower() + "_best_model.pt"))
        pd.DataFrame(history).to_csv(
            OUT_DIR / (name.lower() + "_final_fit_history.csv"), index=False
        )
        pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(range(OUT_DIM)))).to_csv(
            OUT_DIR / (name.lower() + "_test_confusion_matrix.csv"), index=False
        )
        (OUT_DIR / (name.lower() + "_test_classification_report.txt")).write_text(
            classification_report(
                y_true, y_pred, labels=list(range(OUT_DIM)),
                target_names=[str(c) for c in class_names], zero_division=0,
            ), encoding="utf-8"
        )

    if "GIN" in probabilities_by_model and "GAT" in probabilities_by_model:
        # Equal weights are fixed in advance; never optimize ensemble weights on test data.
        hybrid_proba = (probabilities_by_model["GIN"] + probabilities_by_model["GAT"]) / 2.0
        hybrid_pred = hybrid_proba.argmax(axis=1)
        hybrid_true = test_graphs["GIN"].y.cpu().numpy()
        p, r, f1, _ = precision_recall_fscore_support(
            hybrid_true, hybrid_pred, average="macro", zero_division=0
        )
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            hybrid_true, hybrid_pred, average="weighted", zero_division=0
        )
        result_rows.append({
            "model": "GIN_GAT_soft_voting", "selected_val_macro_f1": np.nan,
            "selected_epoch": np.nan,
            "accuracy": float(accuracy_score(hybrid_true, hybrid_pred)),
            "precision_macro": float(p), "recall_macro": float(r),
            "macro_f1": float(f1), "weighted_f1": float(weighted_f1),
            "inference_time_sec": np.nan,
        })
        pd.DataFrame(confusion_matrix(
            hybrid_true, hybrid_pred, labels=list(range(OUT_DIM))
        )).to_csv(OUT_DIR / "hybrid_test_confusion_matrix.csv", index=False)
        (OUT_DIR / "hybrid_test_classification_report.txt").write_text(
            classification_report(
                hybrid_true, hybrid_pred, labels=list(range(OUT_DIM)),
                target_names=[str(c) for c in class_names], zero_division=0,
            ), encoding="utf-8"
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(OUT_DIR / "final_test_results.csv", index=False)
    (OUT_DIR / "run_settings.json").write_text(
        json.dumps(OPTUNA_SETTINGS, indent=2), encoding="utf-8"
    )
    print("\nHeld-out test results (test set was not used for tuning):")
    print(results.to_string(index=False))
    print("Artifacts saved to:", OUT_DIR)
    return studies, best_configs, final_models, results


print("Optuna tuning helpers loaded.")
print("Run: studies, best_configs, tuned_models, tuned_results = run_gat_gin_optuna_pipeline()")
