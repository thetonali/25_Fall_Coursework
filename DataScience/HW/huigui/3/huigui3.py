# -*- coding: utf-8 -*-
"""
回归问题实验：波士顿房价预测
说明：
1. 加载波士顿房价数据（兼容旧版和新版 sklearn）
2. 单个/多个特征与房价关系可视化
3. 使用线性回归与非线性模型（随机森林、SVR）进行预测
4. 输出模型评估指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 尝试加载波士顿数据
def load_boston_data():
    """
    自动兼容：
    sklearn <= 1.0: load_boston 可直接使用
    sklearn > 1.2 : 需要从 openml 数据库下载
    """
    try:
        # 老版本
        from sklearn.datasets import load_boston
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target, name="MEDV")
        return X, y

    except Exception:
        # 新版本 sklearn，从 openml 下载
        from sklearn.datasets import fetch_openml
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        df = boston.frame
        X = df.drop(columns=["MEDV"])
        y = df["MEDV"]
        return X, y


# ====================== 数据加载 ======================
X, y = load_boston_data()
print("数据维度：", X.shape)

# ====================== 单个特征与房价关系可视化 ======================
def plot_single_feature_relationship(feature):
    """
    绘制单个特征与房价 MEDV 的关系
    """
    plt.figure(figsize=(6,4))
    plt.scatter(X[feature], y, s=10)
    plt.xlabel(feature)
    plt.ylabel("MEDV")
    plt.title(f"Feature vs Price: {feature}")
    plt.tight_layout()
    plt.savefig(f"single_feature_{feature}.png")
    plt.close()

# 示例：选择几个典型特征
selected_features = ["RM", "LSTAT", "PTRATIO"]
for f in selected_features:
    plot_single_feature_relationship(f)


# ====================== 多特征关系可视化（矩阵图） ======================
import seaborn as sns

def plot_multi_feature_relationship(features):
    """
    画多特征 + 房价的相关性热力图
    """
    df = X[features].copy()
    df["MEDV"] = y

    plt.figure(figsize=(7,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("multi_feature_heatmap.png")
    plt.close()

plot_multi_feature_relationship(selected_features)


# ====================== 模型训练：线性回归 vs 非线性回归 ======================
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 定义模型
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest (Nonlinear)": RandomForestRegressor(n_estimators=200, random_state=42),
    "SVR (Nonlinear)": SVR(kernel="rbf", C=80, epsilon=0.3)
}

results = {}

# 训练模型 + 评估
for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    results[model_name] = {"MSE": mse, "R2": r2}

    print(f"\n===== {model_name} =====")
    print("MSE:", mse)
    print("R2 :", r2)


# ====================== 特征重要性（非线性模型） ======================
def plot_feature_importance(model, name):
    """
    仅对支持 feature_importances_ 的模型（如随机森林）
    """
    if not hasattr(model, "feature_importances_"):
        return

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10,5))
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), X.columns[indices], rotation=90)
    plt.title(f"Feature Importance - {name}")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{name}.png")
    plt.close()

# 绘制随机森林特征重要性
plot_feature_importance(models["Random Forest (Nonlinear)"], "RandomForest")


print("\n实验完成！图像已生成：")
print(" - 单特征关系图：single_feature_*.png")
print(" - 多特征热力图：multi_feature_heatmap.png")
print(" - 特征重要性：feature_importance_RandomForest.png")
