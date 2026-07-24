"""
训练 Random Forest 双目标模型用于 AC_MS_ITS_GUI

使用当前 389 样本 MC50 方案指定的 RF 超参数，在全量数据集上训练。
输出：models/best_model_MS_ITS_RF.joblib
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 当前 GUI 基座模型参数：与 revision_dataset389_table4params_randomcv_20260724 一致。
RF_PARAMS = {
    "n_estimators": 175,
    "max_depth": 19,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

NUMERIC_FEATURES = [
    "Pe", "Du", "SP", "AC", "AV", "VMA", "VFA",
    "Ag2.36", "Ag4.75", "Ag9.5", "FC", "FL", "TS", "MT"
]
CATEGORICAL_FEATURES = ["FT"]
TARGETS = ["MS", "ITS"]

def make_preprocessor():
    """构建预处理器（与主项目一致）"""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    try:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    except TypeError:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-json", default=None, help="selected_model_params.json exported by the current Optuna run")
    parser.add_argument("--data-path", default=None, help="Dataset workbook to train from; defaults to GUI data/Dataset_cleaned.xlsx.")
    parser.add_argument("--models-dir", default=None, help="Directory for the model bundle; defaults to GUI models/.")
    return parser.parse_args()


def load_rf_params(params_json):
    if not params_json:
        return RF_PARAMS
    with open(params_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return payload["model_params"]["RF"]
    except KeyError as exc:
        raise ValueError("Parameter JSON does not contain model_params.RF") from exc


def main():
    args = parse_args()
    rf_params = load_rf_params(args.params_json)
    base_dir = Path(__file__).parent
    data_path = Path(args.data_path) if args.data_path else base_dir / "data" / "Dataset_cleaned.xlsx"
    models_dir = Path(args.models_dir) if args.models_dir else base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_excel(data_path)
    print(f"  Loaded {len(df)} samples")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGETS].values

    print("\nBuilding full-dataset RF model...")
    preprocessor = make_preprocessor()
    X_prep = preprocessor.fit_transform(X)
    print(f"  Fitted preprocessor on all {len(X)} samples")

    # Train separate RF for MS and ITS
    models = {}
    for idx, target in enumerate(TARGETS):
        print(f"\nTraining {target}...")
        rf = RandomForestRegressor(**rf_params)

        rf.fit(X_prep, y[:, idx])
        print(f"  Fitted on all {len(X)} samples")

        models[target] = rf

    # Save as dict structure (compatible with existing code)
    model_bundle = {
        "preprocessor": preprocessor,
        "models": models,
        "device": "cpu",
        "rf_params": rf_params,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "training_sample_count": int(len(X)),
        "training_method": "full_dataset_refit",
    }

    output_path = models_dir / "best_model_MS_ITS_RF.joblib"
    joblib.dump(model_bundle, output_path)
    (models_dir / "best_model_MS_ITS_RF_metadata.json").write_text(
        json.dumps(
            {
                "dataset_sha256": model_bundle["dataset_sha256"],
                "rf_params": rf_params,
                "training_sample_count": model_bundle["training_sample_count"],
                "training_method": model_bundle["training_method"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved RF model to: {output_path}")
    print(f"Model structure: dict with 'preprocessor', 'models' (MS/ITS), 'device'")

if __name__ == "__main__":
    main()
