import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置 Matplotlib 样式和字体
plt.style.use('seaborn-v0_8-whitegrid')
# 注意：若环境不支持中文，请注释掉下方两行
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False 

def load_and_prepare_data_from_url():
    """
    通过公开 URL 加载波士顿房价数据集，并手动设置特征名称。
    这是为了避免使用 scikit-learn 中已被移除的 load_boston 函数。
    """
    # 公开可用的波士顿房价数据集 CSV 文件 URL
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    
    # 原始数据集的特征名称（13个特征 + 1个目标变量）
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
        'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
    ]
    
    try:
        # 使用 pandas 读取 CSV 文件，并指定列名
        df = pd.read_csv(data_url, names=feature_names, header=0)
    except Exception as e:
        print(f"致命错误：无法从 URL 加载数据，请检查网络连接或 URL 有效性: {e}")
        return None

    print("--- 数据集信息 (通过 URL 原始加载) ---")
    print(df.head())
    print(df.info())
    print("-" * 30)
    
    return df

def feature_relationship_analysis(df):
    """
    第一问：针对波士顿房价数据，通过实验观测其他特征(单个和多个)和房价之间的关系。
    """
    print("## 1. 特征与房价之间的关系观测 ##")

    # 目标变量 (房价)
    target = 'MEDV'
    
    # --- 1.1. 单个特征与房价的关系 ---

    print("\n### 1.1. 单个特征与房价的关系：相关系数 ###")
    # 计算所有特征与房价 (MEDV) 的皮尔逊相关系数
    correlation_with_medv = df.corr()[target].sort_values(ascending=False)
    print(correlation_with_medv)

    # 可视化：使用散点图观察两个最有代表性的特征
    # 选择：正相关最强 (RM: 平均房间数) 和 负相关最强 (LSTAT: 低收入人群比例)
    
    plt.figure(figsize=(15, 6))
    
    # 房间数 (RM) vs 房价 (MEDV)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='RM', y=target, data=df)
    plt.title(f'RM (平均房间数) vs {target} (房价)')
    plt.xlabel('RM')
    plt.ylabel(target)
    
    # 低收入人群比例 (LSTAT) vs 房价 (MEDV)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='LSTAT', y=target, data=df)
    plt.title(f'LSTAT (低收入人群比例) vs {target} (房价)')
    plt.xlabel('LSTAT')
    plt.ylabel(target)
    
    plt.tight_layout()
    plt.savefig('single_feature_vs_price.png')
    plt.show()
    print("-> 已保存 'single_feature_vs_price.png'：RM 和 LSTAT 的散点图。")

    # --- 1.2. 多个特征之间的关系 ---

    print("\n### 1.2. 多个特征之间的关系：相关系数矩阵 (热力图) ###")
    # 观察特征之间是否存在共线性 (多重共线性)

    # 计算所有特征（包括MEDV）间的相关系数矩阵
    correlation_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('所有特征之间的相关系数矩阵')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()
    print("-> 已保存 'correlation_heatmap.png'：特征间的相关系数热力图。")

def polynomial_regression_analysis(df, degree=2):
    """
    第二问：采用非线性回归模型来分析特征和房价之间的关系。
    这里采用多项式回归 (Polynomial Regression) 作为非线性模型。
    """
    print("\n## 2. 非线性回归模型分析：多项式回归 (Polynomial Regression) ##")
    
    target = 'MEDV'
    # 确保分离特征 (X) 和目标变量 (y)
    X = df.drop(columns=[target])
    y = df[target]

    # 1. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 2. 特征缩放 (标准化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. 创建多项式特征
    # degree=2 表示生成二次多项式特征
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    print(f"-> 原始特征数: {X_train.shape[1]}")
    print(f"-> {degree} 阶多项式特征数: {X_train_poly.shape[1]}")

    # 4. 训练多项式回归模型
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # 5. 模型预测与评估
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # 测试集评估
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n### 多项式回归模型 (Degree: 2) 性能 ###")
    print(f"测试集均方误差 (MSE): {test_mse:.2f}")
    print(f"测试集 R² 分数: {test_r2:.4f}")
    print("-" * 40)

    # 6. 对比线性回归性能 (基准)
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    linear_test_pred = linear_model.predict(X_test_scaled)
    linear_test_r2 = r2_score(y_test, linear_test_pred)
    
    print("### 线性回归模型 (作为基准) 性能 ###")
    print(f"测试集 R² 分数: {linear_test_r2:.4f}")
    print("-" * 40)
    
    print(f"比较结果：多项式回归 (R²={test_r2:.4f}) vs 线性回归 (R²={linear_test_r2:.4f})")


if __name__ == '__main__':
    # 1. 加载和准备数据
    df_boston = load_and_prepare_data_from_url()
    
    if df_boston is not None:
        # 2. 执行第一问：特征关系观测
        feature_relationship_analysis(df_boston)

        # 3. 执行第二问：非线性回归分析
        polynomial_regression_analysis(df_boston, degree=2)