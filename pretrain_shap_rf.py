"""
预训练 TreeExplainer 用于 RF 模型

使用全量 401 样本作为 background，为 MS 和 ITS 分别构建 TreeExplainer。
输出：models/shap_explainer_MS_RF.pkl, models/shap_explainer_ITS_RF.pkl
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import warnings
warnings.filterwarnings('ignore')

def main():
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "Dataset_cleaned.xlsx"
    model_path = base_dir / "models" / "best_model_MS_ITS_RF.joblib"
    models_dir = base_dir / "models"

    print("Loading dataset...")
    df = pd.read_excel(data_path)
    print(f"  Loaded {len(df)} samples")

    print("\nLoading RF model...")
    model_bundle = joblib.load(model_path)
    preprocessor = model_bundle["preprocessor"]
    rf_models = model_bundle["models"]
    print(f"  Loaded models: {list(rf_models.keys())}")

    # Prepare full dataset as background
    FEATURES = [
        "Pe", "Du", "SP", "AC", "AV", "VMA", "VFA",
        "Ag2.36", "Ag4.75", "Ag9.5", "FT", "FC", "FL", "TS", "MT"
    ]
    X = df[FEATURES].copy()
    print(f"\nPreprocessing full dataset ({len(X)} samples)...")
    X_prep = preprocessor.transform(X)
    print(f"  Preprocessed shape: {X_prep.shape}")

    # Train TreeExplainer for each target
    for target in ["MS", "ITS"]:
        print(f"\nBuilding TreeExplainer for {target}...")
        rf_model = rf_models[target]

        print(f"  Creating explainer with {len(X_prep)} background samples...")
        explainer = shap.TreeExplainer(rf_model, X_prep)

        output_path = models_dir / f"shap_explainer_{target}_RF.pkl"
        print(f"  Saving to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(explainer, f)

        print(f"  [OK] {target} TreeExplainer saved")

    print("\nDone! TreeExplainer files created:")
    print(f"  - {models_dir / 'shap_explainer_MS_RF.pkl'}")
    print(f"  - {models_dir / 'shap_explainer_ITS_RF.pkl'}")

if __name__ == "__main__":
    main()
