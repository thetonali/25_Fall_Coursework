# ==================== MARKDOWN ====================
# # 胎心宫缩监护(CTG)数据随机森林分类实验报告
# 
# ## 1. 实验目的
# 
# 本实验旨在：
# 1. 使用随机森林算法对CTG数据进行分类分析
# 2. 比较随机森林与Voting、Bagging等集成学习方法的性能
# 3. 评估不同模型在胎儿健康状态分类任务中的表现
# 4. 通过特征重要性分析，识别对胎儿健康状态最有影响的指标
# 
# ## 2. 数据集介绍
# 
# CTG（Cardiotocography，胎心宫缩监护）数据集记录了胎儿心率监护的多项指标。
# 
# **数据集特征说明：**
# - LB: 基线胎心率（Baseline Fetal Heart Rate）
# - AC: 加速次数（Accelerations）
# - FM: 胎动次数（Fetal Movements）
# - UC: 子宫收缩次数（Uterine Contractions）
# - DL: 轻度减速（Light Decelerations）
# - DS: 严重减速（Severe Decelerations）
# - DP: 延长减速（Prolonged Decelerations）
# - ASTV: 短期变异异常百分比（Abnormal Short Term Variability）
# - MSTV: 短期变异均值（Mean Short Term Variability）
# - ALTV: 长期变异异常百分比（Abnormal Long Term Variability）
# - MLTV: 长期变异均值（Mean Long Term Variability）
# - Width: 直方图宽度
# - Min: 直方图最小值
# - Max: 直方图最大值
# - Nmax: 直方图最大值出现次数
# - Nzeros: 直方图零值数量
# - Mode: 直方图众数
# - Mean: 直方图均值
# - Median: 直方图中位数
# - Variance: 直方图方差
# - Tendency: 直方图趋势
# - CLASS: 胎心率模式分类代码
# - **NSP: 胎儿状态分类（目标变量）**
#   - 1 = Normal（正常）
#   - 2 = Suspect（可疑）
#   - 3 = Pathologic（病理）

# ==================== CODE CELL ====================
# 导入所有必要的Python库
# 数据处理相关库
import numpy as np  # 用于数值计算和数组操作
import pandas as pd  # 用于数据读取、处理和分析

# 数据可视化相关库
import matplotlib.pyplot as plt  # 用于绘制各种图表
import seaborn as sns  # 用于绘制统计图表，基于matplotlib

# 机器学习模型相关库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 模型评估指标相关库
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)

# 其他工具
from math import pi
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保实验结果可重复
np.random.seed(42)

# 设置matplotlib的中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*70)
print("所有必要的库已成功导入！")
print("实验环境准备完毕，可以开始数据分析")
print("="*70)

# ==================== OUTPUT ====================
# 运行输出：库导入成功信息


# ==================== MARKDOWN ====================
# ## 3. 数据加载与预处理
# ### 3.1 数据加载

# ==================== CODE CELL ====================
# 从本地文件加载CTG数据集
print("\n" + "="*70)
print("开始加载CTG数据集")
print("="*70)

# 读取CSV文件（文件应该与脚本在同一目录下）
ctg_data = pd.read_csv('CTG.NAOMIT.csv')

print(f"\n✓ 数据加载成功！")
print(f"数据集形状: {ctg_data.shape}")
print(f"  - 样本数量: {ctg_data.shape[0]} 个")
print(f"  - 特征数量: {ctg_data.shape[1]} 个")

print(f"\n数据集包含的所有列（共{len(ctg_data.columns)}列）:")
for i, col in enumerate(ctg_data.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\n数据集前5行预览:")
print(ctg_data.head())

print(f"\n数据集基本信息:")
print(ctg_data.info())

# ==================== OUTPUT ====================
# 运行输出：数据加载信息和前5行预览


# ==================== MARKDOWN ====================
# ### 3.2 数据探索性分析(EDA)

# ==================== CODE CELL ====================
print("\n" + "="*70)
print("数据探索性分析（EDA）")
print("="*70)

# 检查缺失值
print("\n【1. 缺失值检查】")
print("-"*70)
missing_values = ctg_data.isnull().sum()
if missing_values.sum() == 0:
    print("✓ 数据集中没有缺失值")
else:
    print("数据集中存在缺失值:")
    print(missing_values[missing_values > 0])

# 检查重复值
print("\n【2. 重复值检查】")
print("-"*70)
duplicate_count = ctg_data.duplicated().sum()
print(f"重复行数量: {duplicate_count}")
if duplicate_count > 0:
    print(f"重复行占比: {duplicate_count/len(ctg_data)*100:.2f}%")

# 目标变量分布
print("\n【3. 目标变量（NSP）分布】")
print("-"*70)
target_counts = ctg_data['NSP'].value_counts().sort_index()
print("各类别样本数量:")
print(target_counts)

print("\n各类别样本占比:")
target_proportions = ctg_data['NSP'].value_counts(normalize=True).sort_index() * 100
for category, proportion in target_proportions.items():
    category_name = {1: 'Normal(正常)', 2: 'Suspect(可疑)', 3: 'Pathologic(病理)'}[category]
    print(f"  类别 {category} ({category_name}): {proportion:.2f}%")

# 类别不平衡检查
max_proportion = target_proportions.max()
min_proportion = target_proportions.min()
imbalance_ratio = max_proportion / min_proportion
print(f"\n类别不平衡比: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 3:
    print("⚠ 警告: 数据存在较严重的类别不平衡问题")
else:
    print("✓ 数据类别分布相对平衡")

# ==================== OUTPUT ====================
# 运行输出：缺失值、重复值、目标变量分布统计


# ==================== CODE CELL ====================
# 可视化目标变量分布
print("\n正在生成目标变量分布图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 柱状图
target_counts.plot(kind='bar', ax=axes[0], color=['green', 'orange', 'red'],
                   edgecolor='black', alpha=0.7)
axes[0].set_title('胎儿健康状态分布', fontsize=14, fontweight='bold')
axes[0].set_xlabel('健康状态', fontsize=12)
axes[0].set_ylabel('样本数量', fontsize=12)
axes[0].set_xticklabels(['Normal\n(正常)', 'Suspect\n(可疑)', 'Pathologic\n(病理)'], rotation=0)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=11)

# 饼图
colors = ['#66b3ff', '#ffcc99', '#ff9999']
explode = (0.05, 0.05, 0.05)
axes[1].pie(target_counts, labels=['Normal\n(正常)', 'Suspect\n(可疑)', 'Pathologic\n(病理)'],
            autopct='%1.1f%%', colors=colors, startangle=90, explode=explode, shadow=True,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1].set_title('胎儿健康状态比例', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
print("✓ 目标变量分布可视化完成！")

# ==================== OUTPUT ====================
# 运行输出：目标变量的柱状图和饼图


# ==================== CODE CELL ====================
# 特征统计描述
print("\n【4. 特征统计描述】")
print("="*70)

statistics = ctg_data.describe().T
statistics_sorted = statistics.sort_values('mean', ascending=False)

print("\n各特征的统计描述（按均值降序排列）:")
print(statistics_sorted)

# ==================== OUTPUT ====================
# 运行输出：所有特征的统计描述


# ==================== CODE CELL ====================
# 可视化关键特征分布
print("\n正在生成特征分布对比图...")

important_features = ['LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV']
feature_names_chinese = {
    'LB': '基线胎心率', 'AC': '加速次数', 'FM': '胎动次数',
    'UC': '子宫收缩次数', 'ASTV': '短期变异异常%', 'MSTV': '短期变异均值'
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

colors_dict = {1: 'green', 2: 'orange', 3: 'red'}
labels_dict = {1: 'Normal(正常)', 2: 'Suspect(可疑)', 3: 'Pathologic(病理)'}

for idx, feature in enumerate(important_features):
    ax = axes[idx]
    for health_status in [1, 2, 3]:
        data_subset = ctg_data[ctg_data['NSP'] == health_status][feature]
        ax.hist(data_subset, alpha=0.6, bins=30, label=labels_dict[health_status],
                color=colors_dict[health_status], edgecolor='black', linewidth=0.5)
    
    chinese_name = feature_names_chinese.get(feature, feature)
    ax.set_title(f'{feature} - {chinese_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('数值', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
print("✓ 特征分布对比图完成！")

# ==================== OUTPUT ====================
# 运行输出：6个关键特征的分布直方图


# ==================== CODE CELL ====================
# 绘制特征相关性热力图
print("\n正在生成特征相关性热力图...")

numeric_features = ctg_data.drop('CLASS', axis=1)
correlation_matrix = numeric_features.corr()

plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8})
plt.title('CTG特征相关性热力图', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("✓ 相关性热力图完成！")
print("\n【关键观察点】")
print("- 深红色表示强正相关")
print("- 深蓝色表示强负相关")
print("- 接近白色表示弱相关或无相关")
print("- 注意观察NSP（目标变量）与其他特征的相关性")

# ==================== OUTPUT ====================
# 运行输出：特征相关性热力图


# ==================== MARKDOWN ====================
# ### 3.3 数据预处理

# ==================== CODE CELL ====================
print("\n" + "="*70)
print("数据预处理")
print("="*70)

# 分离特征和目标变量
print("\n【步骤1：分离特征和目标变量】")
print("-"*70)

X = ctg_data.drop(['NSP', 'CLASS'], axis=1)
y = ctg_data['NSP']

print(f"特征矩阵 X 的形状: {X.shape}")
print(f"  - 样本数: {X.shape[0]}")
print(f"  - 特征数: {X.shape[1]}")
print(f"\n目标变量 y 的形状: {y.shape}")
print(f"  - 样本数: {y.shape[0]}")

print(f"\n特征列表（共{len(X.columns)}个）:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")

# 划分训练集和测试集
print("\n【步骤2：划分训练集和测试集】")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"测试集大小: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\n训练集类别分布:")
train_dist = y_train.value_counts().sort_index()
for category, count in train_dist.items():
    category_name = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}[category]
    print(f"  类别 {category} ({category_name:10s}): {count:4d} ({count/len(y_train)*100:5.2f}%)")

print(f"\n测试集类别分布:")
test_dist = y_test.value_counts().sort_index()
for category, count in test_dist.items():
    category_name = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}[category]
    print(f"  类别 {category} ({category_name:10s}): {count:4d} ({count/len(y_test)*100:5.2f}%)")

# 特征标准化
print("\n【步骤3：特征标准化】")
print("-"*70)
print("标准化方法: Z-score标准化 (均值=0, 标准差=1)")
print("公式: z = (x - μ) / σ")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n标准化完成！")
print(f"标准化后训练集形状: {X_train_scaled.shape}")
print(f"标准化后测试集形状: {X_test_scaled.shape}")

print(f"\n标准化效果示例（第一个训练样本的前5个特征）:")
print(f"{'特征名':<30s} {'标准化前':>12s} {'标准化后':>12s}")
print("-"*56)
for i in range(min(5, X_train.shape[1])):
    feature_name = X.columns[i]
    before_value = X_train.iloc[0, i]
    after_value = X_train_scaled[0, i]
    print(f"{feature_name:<30s} {before_value:>12.4f} {after_value:>12.4f}")

print("\n✓ 数据预处理完成！数据已准备好用于模型训练")

# ==================== OUTPUT ====================
# 运行输出：数据划分、标准化信息