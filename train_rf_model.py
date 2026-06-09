"""
训练 Random Forest 双目标模型用于 AC_MS_ITS_GUI

使用主项目中 Optuna 调优的 RF 超参数，在全量数据集上训练。
输出：models/best_model_MS_ITS_RF.joblib
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Optuna-tuned hyperparameters from main project
RF_PARAMS = {
    "n_estimators": 194,
    "max_depth": 28,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
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

def main():
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "Dataset_cleaned.xlsx"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = pd.read_excel(data_path)
    print(f"  Loaded {len(df)} samples")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGETS].values

    # 80/20 split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nBuilding RF model...")
    preprocessor = make_preprocessor()

    # Train separate RF for MS and ITS
    models = {}
    for idx, target in enumerate(TARGETS):
        print(f"\nTraining {target}...")
        rf = RandomForestRegressor(**RF_PARAMS)

        # Fit preprocessor and model
        X_train_prep = preprocessor.fit_transform(X_train)
        rf.fit(X_train_prep, y_train[:, idx])

        # Evaluate
        X_test_prep = preprocessor.transform(X_test)
        y_pred = rf.predict(X_test_prep)

        r2 = r2_score(y_test[:, idx], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[:, idx], y_pred))
        mae = mean_absolute_error(y_test[:, idx], y_pred)
        mape = mean_absolute_percentage_error(y_test[:, idx], y_pred) * 100

        print(f"  Test R2 = {r2:.4f}")
        print(f"  Test RMSE = {rmse:.4f}")
        print(f"  Test MAE = {mae:.4f}")
        print(f"  Test MAPE = {mape:.2f}%")

        models[target] = rf

    # Save as dict structure (compatible with existing code)
    model_bundle = {
        "preprocessor": preprocessor,
        "models": models,
        "device": "cpu",
    }

    output_path = models_dir / "best_model_MS_ITS_RF.joblib"
    joblib.dump(model_bundle, output_path)
    print(f"\nSaved RF model to: {output_path}")
    print(f"Model structure: dict with 'preprocessor', 'models' (MS/ITS), 'device'")

if __name__ == "__main__":
    main()
