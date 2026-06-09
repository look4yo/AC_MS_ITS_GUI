"""
工具函数：SHAP 解释分析（快速版）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import traceback


def build_fallback_background(preprocessor, default_input, dataset_df=None, n_samples=150):
    """
    构造 background 数据。

    优先使用真实数据集中的样本作为 SHAP background；如果数据集不可用，
    则使用默认输入的扰动版本作为回退。
    """
    from .prediction import normalize_raw_input, transform_input

    if dataset_df is not None and len(dataset_df) > 0:
        n_samples = min(n_samples, len(dataset_df))
        sampled = dataset_df.sample(n=n_samples, random_state=42)
        return transform_input(preprocessor, sampled)

    default_raw = pd.DataFrame([default_input.copy()])
    default_raw = normalize_raw_input(default_raw)
    bg_single = transform_input(preprocessor, default_raw)

    fallback_samples = 100
    bg_repeated = np.tile(bg_single.values, (fallback_samples, 1))

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, bg_repeated.shape)
    bg_noisy = bg_repeated + noise * np.abs(bg_repeated)

    return pd.DataFrame(bg_noisy, columns=bg_single.columns)


def get_runtime_fallback_explainer(model, preprocessor, default_input, dataset_df=None):
    """
    运行时临时构建 SHAP explainer（快速版）

    策略：使用真实数据集中的 150 个样本作为 background；数据集不可用时，
    使用默认输入的扰动版本回退。
    """
    background_df = build_fallback_background(
        preprocessor,
        default_input,
        dataset_df=dataset_df,
        n_samples=150,
    )

    # 定义预测函数
    def predict_func(X):
        """
        批量预测，返回 shape: (n_samples, 2)
        """
        from .prediction import predict_ms_its

        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=background_df.columns)
        else:
            X_df = X

        n_samples = len(X_df)
        results = np.zeros((n_samples, 2))

        for i in range(n_samples):
            row_df = X_df.iloc[[i]]
            ms_pred, its_pred = predict_ms_its(model, row_df)
            results[i, 0] = ms_pred
            results[i, 1] = its_pred

        return results

    # 使用 Permutation Explainer。显式设置 masker 的 max_samples=150，避免
    # SHAP 默认把 background 截断到 100 个样本。
    try:
        masker = shap.maskers.Independent(background_df, max_samples=len(background_df))
        explainer = shap.explainers.Permutation(predict_func, masker)
        return explainer, "runtime_permutation"
    except Exception:
        try:
            explainer = shap.Explainer(predict_func, background_df)
            return explainer, "runtime_auto"
        except Exception:
            # 最终回退：创建一个简单的 explainer
            explainer = shap.explainers.Exact(predict_func, background_df)
            return explainer, "runtime_exact"


def make_local_shap_explanation(explainer, X_df, output_index=None):
    """
    生成单样本 SHAP 解释
    """
    if explainer is None:
        raise RuntimeError("未能加载 SHAP explainer")

    shap_values = explainer(X_df)

    # 处理多输出
    if output_index is not None and hasattr(shap_values, 'values'):
        if shap_values.values.ndim == 3:
            shap_values.values = shap_values.values[:, :, output_index]
            if isinstance(shap_values.base_values, np.ndarray):
                if shap_values.base_values.ndim > 1:
                    shap_values.base_values = shap_values.base_values[:, output_index]
                elif len(shap_values.base_values) > 1:
                    shap_values.base_values = shap_values.base_values[output_index]

    return shap_values


def plot_waterfall_from_explanation(sample_exp, max_display=12, title=None):
    """
    绘制 SHAP waterfall plot
    """
    plt.close("all")
    fig = plt.figure(figsize=(5, 3.5))

    try:
        shap.plots.waterfall(sample_exp, max_display=max_display, show=False)
        if title:
            plt.title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"绘制 waterfall plot 失败: {str(e)}")

    return fig


def analyze_feature_consistency(shap_values_ms, shap_values_its, feature_names):
    """
    分析特征对两个目标的作用方向一致性
    """
    consistency_data = []

    for i, feat in enumerate(feature_names):
        ms_val = float(shap_values_ms[i])
        its_val = float(shap_values_its[i])

        threshold = 0.01
        if abs(ms_val) < threshold and abs(its_val) < threshold:
            label = "Neutral"
            emoji = "⚪"
        elif ms_val * its_val > 0:
            label = "Aligned"
            emoji = "✅"
        else:
            label = "Opposite"
            emoji = "⚠️"

        consistency_data.append({
            'Feature': feat,
            'MS_SHAP': ms_val,
            'ITS_SHAP': its_val,
            'abs(MS_SHAP)': abs(ms_val),
            'abs(ITS_SHAP)': abs(its_val),
            'Consistency': f"{emoji} {label}"
        })

    df = pd.DataFrame(consistency_data)
    df = df.sort_values('abs(MS_SHAP)', ascending=False)

    return df
