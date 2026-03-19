# ============================================================================
# 第一部分：导入必要的库和数据加载
# ============================================================================

# 导入数据处理相关库
import pandas as pd  # 用于数据框操作和CSV文件读取
import numpy as np  # 用于数值计算和数组操作

# 导入数据可视化库
import matplotlib.pyplot as plt  # 基础绘图库
import seaborn as sns  # 高级统计可视化库，提供更美观的图表

# 导入机器学习相关库
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
from sklearn.preprocessing import StandardScaler  # 用于特征标准化
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier  # 三种集成学习分类器
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器，作为Bagging的基学习器
from sklearn.linear_model import LogisticRegression  # 逻辑回归，作为Voting的一个基学习器
from sklearn.svm import SVC  # 支持向量机，作为Voting的另一个基学习器

# 导入模型评估相关库
from sklearn.metrics import (
    classification_report,  # 生成分类报告，包含精确率、召回率、F1分数
    confusion_matrix,  # 生成混淆矩阵
    accuracy_score,  # 计算准确率
    roc_auc_score,  # 计算ROC曲线下面积（多分类需要特殊处理）
    roc_curve  # 计算ROC曲线的坐标点
)

# 导入交叉验证相关库
from sklearn.model_selection import cross_val_score, GridSearchCV  # 交叉验证和网格搜索

# 设置随机种子，确保实验可重复性
RANDOM_STATE = 42

# 设置matplotlib显示中文（如果需要中文标签）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置图表风格
sns.set_style("whitegrid")  # 使用白色网格背景
plt.rcParams['figure.figsize'] = (12, 8)  # 设置默认图表大小

print("=" * 80)
print("胎心宫缩监护(CTG)数据分类实验")
print("=" * 80)
print("\n实验目标：使用随机森林、Voting和Bagging三种集成学习方法对CTG数据进行分类")
print("目标变量：NSP（胎儿状态分类）\n")

# ============================================================================
# 加载数据
# ============================================================================
print("-" * 80)
print("步骤1：数据加载")
print("-" * 80)

# 从本地CSV文件读取数据
# 文件名：CTG.NAOMIT.csv，位于当前目录下
data = pd.read_csv('CTG.NAOMIT.csv')

print(f"数据集成功加载！")
print(f"数据集形状：{data.shape[0]} 行（样本数）× {data.shape[1]} 列（特征数）")
print(f"\n前5行数据预览：")
print(data.head())

# ============================================================================
# 数据基本信息探索
# ============================================================================
print("\n" + "-" * 80)
print("步骤2：数据基本信息探索")
print("-" * 80)

# 显示数据集的基本信息：列名、非空值数量、数据类型
print("\n数据集基本信息：")
print(data.info())

# 显示数值型特征的统计描述：均值、标准差、最小值、四分位数、最大值
print("\n数值特征统计描述：")
print(data.describe())

# 检查缺失值
print("\n缺失值检查：")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "没有缺失值")

# 检查目标变量的分布
print("\n目标变量(NSP)的分布情况：")
print(data['NSP'].value_counts().sort_index())
print("\n目标变量的百分比分布：")
print(data['NSP'].value_counts(normalize=True).sort_index() * 100)


# ============================================================================
# 第二部分：数据可视化分析
# ============================================================================
print("\n" + "-" * 80)
print("步骤3：数据可视化分析")
print("-" * 80)

# ============================================================================
# 3.1 目标变量分布可视化
# ============================================================================
print("\n3.1 目标变量(NSP)分布可视化")

# --------------------
# 1. 确保中文字体生效（可重复设置，避免 Notebook 字体缓存问题）
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示

# --------------------
# 创建一个包含两个子图的画布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1：柱状图显示各类别的样本数量
nsp_counts = data['NSP'].value_counts().sort_index()
axes[0].bar(nsp_counts.index, nsp_counts.values, color=['#2ecc71', '#f39c12', '#e74c3c'])
axes[0].set_xlabel('NSP类别', fontsize=12)
axes[0].set_ylabel('样本数量', fontsize=12)
axes[0].set_title('NSP类别分布（柱状图）', fontsize=14, fontweight='bold', fontname='SimHei')
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(['正常(1)', '可疑(2)', '病理(3)'], fontname='SimHei')

# 在柱状图上添加数值标签（显式指定中文字体）
for i, v in enumerate(nsp_counts.values):
    axes[0].text(i + 1, v + 20, str(v),
                 ha='center', fontsize=11, fontweight='bold', fontname='SimHei')

# 子图2：饼图显示各类别的占比
colors = ['#2ecc71', '#f39c12', '#e74c3c']
axes[1].pie(
    nsp_counts.values,
    labels=['正常(1)', '可疑(2)', '病理(3)'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 11, 'fontname': 'SimHei'}  # 显式字体
)
axes[1].set_title('NSP类别占比（饼图）', fontsize=14, fontweight='bold', fontname='SimHei')

plt.tight_layout()
plt.show()

# --------------------
# 输出类别不平衡信息
print(f"\n数据集类别分布分析：")
print(f"- 正常样本(NSP=1)：{nsp_counts[1]} 个，占比 {nsp_counts[1]/len(data)*100:.2f}%")
print(f"- 可疑样本(NSP=2)：{nsp_counts[2]} 个，占比 {nsp_counts[2]/len(data)*100:.2f}%")
print(f"- 病理样本(NSP=3)：{nsp_counts[3]} 个，占比 {nsp_counts[3]/len(data)*100:.2f}%")

if nsp_counts.max() / nsp_counts.min() > 3:
    print(f"\n⚠️ 注意：数据集存在类别不平衡问题，最大类与最小类的比例为 {nsp_counts.max() / nsp_counts.min():.2f}:1")
    print("   这可能影响模型性能，建议在模型训练时考虑使用class_weight参数或采样技术")
