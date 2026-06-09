# MS_ITS_GUI_APP 使用指南

## 📋 项目概述

**MS_ITS_GUI_APP** 是一个基于 Streamlit 的交互式 Web 应用，用于预测纤维增强沥青混凝土的双目标性能：
- **MS (Marshall Stability)** - 马歇尔稳定度
- **ITS (Indirect Tensile Strength)** - 间接拉伸强度

应用集成了 SHAP 解释框架，提供透明的模型决策过程。

---

## 🚀 快速开始

### 1. 环境准备

确保已安装 Anaconda 并创建了 `AC_MS_ITS` 环境：

```bash
conda activate AC_MS_ITS
```

### 2. 安装依赖

如果是首次运行，安装必要的依赖：

```bash
pip install streamlit shap
```

### 3. 准备模型文件

运行准备脚本（只需执行一次）：

```bash
python prepare_models.py
```

这将：
- 从训练输出目录复制最佳模型
- 重新构建数据预处理器
- 复制数据集和 Pareto 前沿数据

### 4. 启动应用

**方法 1：使用批处理脚本（Windows）**

双击运行 `run_app.bat`

**方法 2：手动启动**

```bash
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址：`http://localhost:8501`

---

## 🎯 功能模块

### Tab 1: 多目标预测

**输入 15 个特征参数，同时预测 MS 和 ITS**

### Tab 2: SHAP 解释

**双目标 SHAP 分析 + 特征一致性分析**

---

## 📧 支持

如有问题或建议，请联系项目维护者。
