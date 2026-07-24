"""
工具函数：预测逻辑
"""
import numpy as np
import pandas as pd


def normalize_raw_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化原始输入数据
    - 数值列转换为 float
    - 分类列（FT）去除首尾空白，保留原始大小写以匹配训练时的 OneHotEncoder 类别
    """
    df = raw_df.copy()

    numeric_cols = [
        "Pe", "Du", "SP", "AC", "AV", "VMA", "VFA",
        "Ag2.36", "Ag4.75", "Ag9.5", "FC", "FL", "TS", "MT"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "FT" in df.columns:
        df["FT"] = df["FT"].astype(str).str.strip()

    return df


def transform_input(preprocessor, raw_df):
    """
    使用预处理器转换输入数据
    Returns: DataFrame with transformed features
    """
    if preprocessor is None:
        raise RuntimeError("未能加载预处理器，无法进行模型输入转换。")

    raw_df = normalize_raw_input(raw_df)
    X = preprocessor.transform(raw_df)

    # 如果是稀疏矩阵，转为密集矩阵
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    # 获取特征名
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if feature_names is not None and len(feature_names) == X.shape[1]:
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    return X_df


def get_feature_names_from_preprocessor(preprocessor):
    """
    从预处理器提取特征名
    """
    if preprocessor is None:
        return None
    try:
        names = preprocessor.get_feature_names_out()
        names = [str(x) for x in names]
        cleaned = []
        for n in names:
            # 移除管道前缀
            n2 = n.replace("num__", "").replace("cat__", "").replace("ft__", "")
            cleaned.append(n2)
        return cleaned
    except Exception:
        return None


def predict_ms_its(model, X_df):
    """
    双目标预测：MS 和 ITS

    Args:
        model: 训练好的多输出模型（可能是模型对象或包含模型的字典）
        X_df: 转换后的特征 DataFrame

    Returns:
        (ms_pred, its_pred): 两个浮点数
    """
    if model is None:
        raise RuntimeError("未能加载模型，无法进行预测。")

    # 如果 model 是字典（如 TabICLv2 的保存格式），提取实际的模型
    if isinstance(model, dict):
        if 'models' in model and isinstance(model['models'], dict):
            # TabICLv2 格式：{'preprocessor': ..., 'models': {'MS': model_ms, 'ITS': model_its}, 'device': ...}
            ms_model = model['models'].get('MS')
            its_model = model['models'].get('ITS')

            if ms_model is None or its_model is None:
                raise ValueError(f"无法从字典中提取 MS 或 ITS 模型。可用键: {list(model['models'].keys())}")

            # 分别预测
            ms_pred_arr = ms_model.predict(X_df)
            its_pred_arr = its_model.predict(X_df)

            ms_pred = float(ms_pred_arr.reshape(-1)[0])
            its_pred = float(its_pred_arr.reshape(-1)[0])

            return ms_pred, its_pred

        elif 'model' in model:
            actual_model = model['model']
        else:
            raise ValueError(f"无法从字典中提取模型。可用键: {list(model.keys())}")
    else:
        actual_model = model

    # 标准多输出模型预测
    pred = actual_model.predict(X_df)

    # 处理多输出
    if isinstance(pred, np.ndarray) and pred.ndim == 2:
        # shape: (1, 2) -> MS, ITS
        ms_pred = float(pred[0, 0])
        its_pred = float(pred[0, 1])
    elif isinstance(pred, np.ndarray) and pred.ndim == 1 and len(pred) == 2:
        ms_pred = float(pred[0])
        its_pred = float(pred[1])
    else:
        raise ValueError(f"意外的预测输出格式: {type(pred)}, shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")

    return ms_pred, its_pred


def get_ft_options_from_preprocessor(preprocessor):
    """
    从预处理器中提取 FT（纤维类型）的所有可能值
    """
    fallback_options = [
        "no_fiber",
        "plastic_fiber",
        "mineral_fiber",
        "bio-fiber",
        "carbon_fiber",
        "glass_fiber",
        "steel_fiber",
    ]

    if preprocessor is None:
        return fallback_options

    try:
        if hasattr(preprocessor, "transformers_"):
            for name, transformer, cols in preprocessor.transformers_:
                if transformer == "drop":
                    continue
                if isinstance(cols, (list, tuple)) and "FT" in list(cols):
                    # 查找 OneHotEncoder
                    if hasattr(transformer, "categories_") and len(transformer.categories_) > 0:
                        return [str(x) for x in transformer.categories_[0].tolist()]
                    # 查找 Pipeline 中的 OneHotEncoder
                    if hasattr(transformer, "named_steps"):
                        for _, step in transformer.named_steps.items():
                            if hasattr(step, "categories_") and len(step.categories_) > 0:
                                return [str(x) for x in step.categories_[0].tolist()]
    except Exception:
        pass

    return fallback_options
