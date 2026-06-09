"""
工具函数：可视化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def plot_ms_its_tradeoff(ms_pred, its_pred, dataset_df=None, pareto_df=None):
    """
    绘制 MS vs ITS 权衡图

    Args:
        ms_pred: 当前预测的 MS 值
        its_pred: 当前预测的 ITS 值
        dataset_df: 数据集 DataFrame，需包含 'MS' 和 'ITS' 列
        pareto_df: Pareto 前沿 DataFrame，需包含 'MS' 和 'ITS' 列

    Returns:
        matplotlib figure
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))

    # 背景样本分布
    if dataset_df is not None and 'MS' in dataset_df.columns and 'ITS' in dataset_df.columns:
        ax.scatter(dataset_df['MS'], dataset_df['ITS'],
                   alpha=0.3, s=30, c='gray', label='Dataset samples')

    # Pareto 前沿
    if pareto_df is not None and 'MS' in pareto_df.columns and 'ITS' in pareto_df.columns:
        pareto_sorted = pareto_df.sort_values('MS')
        ax.plot(pareto_sorted['MS'], pareto_sorted['ITS'],
                'b-', linewidth=2.5, alpha=0.7, label='Pareto Front')
        ax.scatter(pareto_sorted['MS'], pareto_sorted['ITS'],
                   s=60, c='blue', alpha=0.5, zorder=5)

    # 当前预测点（红星）
    ax.scatter(ms_pred, its_pred, s=400, c='red',
               marker='*', edgecolors='black', linewidths=2.5,
               label='Current Prediction', zorder=10)

    # 在预测点旁边标注数值
    ax.annotate(f'MS={ms_pred:.2f}\nITS={its_pred:.2f}',
                xy=(ms_pred, its_pred),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('Marshall Stability (kN)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Indirect Tensile Strength (MPa)', fontsize=13, fontweight='bold')
    ax.set_title('MS vs ITS Performance Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    return fig


def plot_dual_waterfall(shap_exp_ms, shap_exp_its, max_display=10):
    """
    并排绘制 MS 和 ITS 的 waterfall plot

    Args:
        shap_exp_ms: MS 的 SHAP explanation 对象
        shap_exp_its: ITS 的 SHAP explanation 对象
        max_display: 显示的最大特征数

    Returns:
        matplotlib figure
    """
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MS waterfall
    plt.sca(ax1)
    try:
        import shap
        shap.plots.waterfall(shap_exp_ms, max_display=max_display, show=False)
        ax1.set_title('MS Contribution', fontsize=12, fontweight='bold')
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax1.transAxes)

    # ITS waterfall
    plt.sca(ax2)
    try:
        import shap
        shap.plots.waterfall(shap_exp_its, max_display=max_display, show=False)
        ax2.set_title('ITS Contribution', fontsize=12, fontweight='bold')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    return fig


def configure_matplotlib():
    """
    配置 matplotlib 全局样式
    """
    matplotlib.use("Agg")  # 非交互式后端
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 100
