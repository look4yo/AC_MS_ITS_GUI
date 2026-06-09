# MS_ITS_GUI_APP

多目标预测 GUI 应用：Marshall Stability (MS) 和 Indirect Tensile Strength (ITS)

## 功能

### Phase 1 (MVP)
- **Tab 1: Multi-target Prediction** - 双目标预测 + 性能权衡可视化
- **Tab 2: SHAP Interpretation** - 双目标 SHAP 解释 + 特征一致性分析

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型文件

```bash
python prepare_models.py
```

此脚本将：
- 从训练输出目录复制最佳模型和预处理器
- 生成 SHAP explainer（如果不存在）
- 复制数据集和 Pareto 前沿数据

### 3. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中自动打开（默认地址：`http://localhost:8501`）

## 文件结构

```
MS_ITS_GUI_APP/
├── app.py                          # 主应用入口
├── prepare_models.py               # 模型准备脚本
├── requirements.txt                # 依赖清单
├── models/
│   ├── best_model_MS_ITS.joblib    # 最佳模型（双输出）
│   ├── preprocessor_MS_ITS.joblib  # 数据预处理器
│   └── shap_explainer_MS_ITS.joblib # SHAP explainer（可选）
├── data/
│   ├── Dataset_cleaned.xlsx        # 数据集（用于背景显示）
│   └── pareto_solutions.xlsx       # Pareto 前沿（可选）
└── utils/
    ├── prediction.py               # 预测逻辑
    ├── shap_analysis.py            # SHAP 解释
    └── visualization.py            # 可视化函数
```

## 输入特征（15个）

**沥青特性：**
- Pe: 针入度 (0.1 mm)
- Du: 延度 (cm)
- SP: 软化点 (°C)
- AC: 沥青含量 (wt.%)

**集料和体积特性：**
- AV: 空隙率 (%)
- VMA: 矿料间隙率 (%)
- VFA: 沥青饱和度 (%)
- Ag2.36, Ag4.75, Ag9.5: 集料级配 (%)

**纤维和生产特性：**
- FT: 纤维类型（分类变量）
- FC: 纤维含量 (wt.%)
- FL: 纤维长度 (mm)
- TS: 纤维抗拉强度 (MPa)
- MT: 混合温度 (°C)

## 输出

- **MS**: Marshall Stability（马歇尔稳定度，kN）
- **ITS**: Indirect Tensile Strength（间接拉伸强度，MPa）

## 技术栈

- **框架**: Streamlit
- **ML 模型**: TabPFN / TabICLv2
- **解释性**: SHAP (SHapley Additive exPlanations)
- **可视化**: Matplotlib
- **数据处理**: Pandas, NumPy, scikit-learn

## 部署

### 本地部署

```bash
streamlit run app.py --server.port 8501
```

### Streamlit Cloud 部署

1. 推送代码到 GitHub
2. 连接 Streamlit Cloud
3. 选择仓库和分支
4. 自动部署

## License

MIT
