"""
工具函数：SHAP 特征聚合

将 one-hot 编码的分类特征聚合回原始特征
"""
import numpy as np
import shap


def aggregate_onehot_features(shap_values, feature_names, user_input_df):
    """
    将 one-hot 编码的 FT 特征聚合为单个原始 FT 特征

    Args:
        shap_values: SHAP values array, shape (n_features,)
        feature_names: List of feature names (21 features)
        user_input_df: 用户原始输入 DataFrame，包含 'FT' 列

    Returns:
        aggregated_values: np.array (15 features)
        aggregated_names: list of str (15 feature names)
    """
    # 获取用户输入的 FT 类型
    user_ft = user_input_df['FT'].iloc[0] if 'FT' in user_input_df.columns else 'Unknown'

    # 识别 FT one-hot 特征的索引
    ft_indices = []
    non_ft_indices = []

    for i, name in enumerate(feature_names):
        name_lower = str(name).lower()
        # 识别 FT 相关特征（可能的前缀：cat__FT_, cat__ft_, FT_, ft_）
        if 'ft_' in name_lower or '_ft' in name_lower:
            # 进一步确认是否是 one-hot 编码的 FT
            if any(fiber_type in name_lower for fiber_type in
                   ['no_fiber', 'plastic', 'bio', 'carbon', 'steel', 'basalt', 'glass', 'mineral']):
                ft_indices.append(i)
                continue
        non_ft_indices.append(i)

    # 计算聚合的 FT SHAP 值（所有 one-hot 特征的和）
    ft_shap_sum = sum(shap_values[i] for i in ft_indices) if ft_indices else 0.0

    # 构建聚合后的特征名和 SHAP 值
    aggregated_values = []
    aggregated_names = []

    # 添加非 FT 特征（保持原顺序）
    for i in non_ft_indices:
        aggregated_values.append(shap_values[i])
        # 清理特征名（移除 num__ 前缀）
        clean_name = str(feature_names[i]).replace('num__', '').replace('cat__', '')
        aggregated_names.append(clean_name)

    # 添加聚合的 FT 特征
    aggregated_values.append(ft_shap_sum)
    aggregated_names.append(f"FT ({user_ft})")

    return np.array(aggregated_values), aggregated_names


def create_aggregated_shap_explanation(shap_explanation, user_input_df):
    """
    从原始 SHAP explanation 创建聚合后的 explanation 对象

    Args:
        shap_explanation: SHAP Explanation 对象（21 个特征）
        user_input_df: 用户原始输入 DataFrame

    Returns:
        aggregated_explanation: 聚合后的 SHAP Explanation 对象（15 个特征）
    """
    # 提取原始数据
    original_values = shap_explanation.values  # (n_features,) 或 (1, n_features)
    original_names = shap_explanation.feature_names
    base_value = shap_explanation.base_values

    # 处理维度
    if original_values.ndim == 2:
        # Shape: (1, n_features) → (n_features,)
        original_values = original_values[0]

    if isinstance(base_value, np.ndarray):
        if base_value.ndim > 0:
            base_value = base_value[0] if len(base_value) > 0 else base_value

    # 聚合特征
    aggregated_values, aggregated_names = aggregate_onehot_features(
        original_values,
        original_names,
        user_input_df
    )

    # 创建新的 SHAP Explanation 对象
    aggregated_explanation = shap.Explanation(
        values=aggregated_values,
        base_values=base_value,
        data=aggregated_values,  # 使用 SHAP 值作为 data（简化）
        feature_names=aggregated_names
    )

    return aggregated_explanation
