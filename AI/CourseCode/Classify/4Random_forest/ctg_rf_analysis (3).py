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
# train_test_split: 用于划分训练集和测试集
# cross_val_score: 用于交叉验证
# GridSearchCV: 用于网格搜索超参数优化
# learning_curve: 用于绘制学习曲线

from sklearn.preprocessing import StandardScaler, label_binarize
# StandardScaler: 用于特征标准化（均值为0，标准差为1）
# label_binarize: 用于标签二值化，用于多分类ROC曲线

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
# RandomForestClassifier: 随机森林分类器
# VotingClassifier: 投票分类器，组合多个不同的分类器
# BaggingClassifier: Bagging集成分类器

from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器
from sklearn.svm import SVC  # 支持向量机分类器
from sklearn.neighbors import KNeighborsClassifier  # K近邻分类器

# 模型评估指标相关库
from sklearn.metrics import (
    classification_report,  # 生成分类报告（精确率、召回率、F1分数等）
    confusion_matrix,  # 生成混淆矩阵
    accuracy_score,  # 计算准确率
    precision_score,  # 计算精确率
    recall_score,  # 计算召回率
    f1_score,  # 计算F1分数
    roc_curve,  # 计算ROC曲线的点
    auc  # 计算AUC面积
)

# 其他工具
from math import pi  # 用于绘制雷达图时计算角度
from itertools import cycle  # 用于循环迭代颜色等元素
import warnings  # 用于控制警告信息
warnings.filterwarnings('ignore')  # 忽略所有警告信息，使输出更清晰

# 设置随机种子，确保实验结果可重复
np.random.seed(42)  # NumPy的随机种子
# 42是机器学习领域常用的随机种子数字，来自《银河系漫游指南》

# 设置matplotlib的中文显示（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置pandas显示选项，使数据框显示更完整
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)  # 设置显示宽度

print("="*70)
print("所有必要的库已成功导入！")
print("实验环境准备完毕，可以开始数据分析")
print("="*70)

# ==================== OUTPUT ====================
# 运行此单元格，输出：
# ======================================================================
# 所有必要的库已成功导入！
# 实验环境准备完毕，可以开始数据分析
# ======================================================================


# ==================== MARKDOWN ====================
# ## 3. 数据加载与预处理
# 
# ### 3.1 数据加载
# 
# 从本地CSV文件加载CTG数据集，该数据集已经过预处理，移除了缺失值。

# ==================== CODE CELL ====================
# 从本地文件加载CTG数据集
print("\n" + "="*70)
print("开始加载CTG数据集")
print("="*70)

# 读取CSV文件
# 文件说明：CTG.NAOMIT.csv 是已经去除缺失值的CTG数据集
# 该文件应该与当前Python脚本在同一目录下
ctg_data = pd.read_csv('CTG.NAOMIT.csv')

# 输出数据加载成功的信息
print(f"\n✓ 数据加载成功！")
print(f"数据集形状: {ctg_data.shape}")  # (行数, 列数)
print(f"  - 样本数量: {ctg_data.shape[0]} 个")  # 总共有多少条记录
print(f"  - 特征数量: {ctg_data.shape[1]} 个")  # 总共有多少列（包括目标变量）

# 显示数据集包含的所有列名
print(f"\n数据集包含的所有列（共{len(ctg_data.columns)}列）:")
for i, col in enumerate(ctg_data.columns, 1):
    print(f"  {i:2d}. {col}")  # 按序号显示每一列的名称

# 显示数据的前5行，用于预览数据结构
print(f"\n数据集前5行预览:")
print(ctg_data.head())

# 显示数据的基本统计信息
print(f"\n数据集基本信息:")
print(ctg_data.info())

# ==================== OUTPUT ====================
# 运行此单元格，输出：
# - 数据加载成功的确认信息
# - 数据集的维度（行数和列数）
# - 所有列名的列表
# - 前5行数据的详细内容
# - 数据类型和非空值信息


# ==================== MARKDOWN ====================
# ### 3.2 数据探索性分析(EDA)
# 
# 对数据进行全面的统计分析和可视化，了解数据的分布特征。

# ==================== CODE CELL ====================
# 数据基本统计分析
print("\n" + "="*70)
print("数据探索性分析（EDA）")
print("="*70)

# 检查数据中是否存在缺失值
print("\n【1. 缺失值检查】")
print("-"*70)
missing_values = ctg_data.isnull().sum()  # 统计每列的缺失值数量
if missing_values.sum() == 0:
    print("✓ 数据集中没有缺失值")
else:
    print("数据集中存在缺失值:")
    print(missing_values[missing_values > 0])  # 只显示有缺失值的列

# 检查数据中是否存在重复行
print("\n【2. 重复值检查】")
print("-"*70)
duplicate_count = ctg_data.duplicated().sum()  # 统计重复行的数量
print(f"重复行数量: {duplicate_count}")
if duplicate_count > 0:
    print(f"重复行占比: {duplicate_count/len(ctg_data)*100:.2f}%")

# 显示目标变量的分布情况
print("\n【3. 目标变量（NSP）分布】")
print("-"*70)
# NSP是我们要预测的目标变量，表示胎儿健康状态
target_counts = ctg_data['NSP'].value_counts().sort_index()  # 统计每个类别的数量
print("各类别样本数量:")
print(target_counts)

# 计算每个类别的占比
print("\n各类别样本占比:")
target_proportions = ctg_data['NSP'].value_counts(normalize=True).sort_index() * 100
for category, proportion in target_proportions.items():
    category_name = {1: 'Normal(正常)', 2: 'Suspect(可疑)', 3: 'Pathologic(病理)'}[category]
    print(f"  类别 {category} ({category_name}): {proportion:.2f}%")

# 检查数据是否存在类别不平衡问题
# 如果某个类别的样本数量远少于其他类别，可能需要特殊处理
max_proportion = target_proportions.max()
min_proportion = target_proportions.min()
imbalance_ratio = max_proportion / min_proportion
print(f"\n类别不平衡比: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 3:
    print("⚠ 警告: 数据存在较严重的类别不平衡问题")
else:
    print("✓ 数据类别分布相对平衡")

# ==================== OUTPUT ====================
# 运行此单元格，输出：
# - 缺失值统计信息
# - 重复值统计信息
# - 目标变量的类别分布
# - 类别不平衡情况分析


# ==================== CODE CELL ====================
# 可视化目标变量的分布
print("\n正在生成目标变量分布图...")

# 创建一个包含2个子图的图形，figsize设置图形大小
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：柱状图显示每个类别的数量
target_counts.plot(
    kind='bar',  # 柱状图
    ax=axes[0],  # 绘制在第一个子图上
    color=['green', 'orange', 'red'],  # 分别为正常、可疑、病理设置颜色
    edgecolor='black',  # 柱子的边框颜色
    alpha=0.7  # 透明度
)
axes[0].set_title('胎儿健康状态分布', fontsize=14, fontweight='bold')
axes[0].set_xlabel('健康状态', fontsize=12)
axes[0].set_ylabel('样本数量', fontsize=12)
axes[0].set_xticklabels(['Normal\n(正常)', 'Suspect\n(可疑)', 'Pathologic\n(病理)'], rotation=0)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')  # 添加网格线

# 在每个柱子上方显示具体数值
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=11)

# 右图：饼图显示每个类别的占比
colors = ['#66b3ff', '#ffcc99', '#ff9999']  # 设置饼图的颜色
explode = (0.05, 0.05, 0.05)  # 将每个扇区稍微分离，使图形更清晰
axes[1].pie(
    target_counts,  # 数据
    labels=['Normal\n(正常)', 'Suspect\n(可疑)', 'Pathologic\n(病理)'],  # 标签
    autopct='%1.1f%%',  # 显示百分比，保留1位小数
    colors=colors,  # 颜色
    startangle=90,  # 起始角度
    explode=explode,  # 分离程度
    shadow=True,  # 添加阴影效果
    textprops={'fontsize': 11, 'fontweight': 'bold'}  # 文本属性
)
axes[1].set_title('胎儿健康状态比例', fontsize=14, fontweight='bold')

# 调整子图之间的间距
plt.tight_layout()
# 显示图形
plt.show()

print("✓ 目标变量分布可视化完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示：
# - 柱状图：每个类别的样本数量
# - 饼图：每个类别的占比


# ==================== CODE CELL ====================
# 显示所有特征的统计描述
print("\n【4. 特征统计描述】")
print("="*70)

# describe()函数会自动计算每列的统计指标
# 包括：count(数量)、mean(均值)、std(标准差)、min(最小值)、
#      25%(第一四分位数)、50%(中位数)、75%(第三四分位数)、max(最大值)
statistics = ctg_data.describe().T  # .T表示转置，使特征名称作为行显示

# 按照均值从大到小排序，便于观察
statistics_sorted = statistics.sort_values('mean', ascending=False)

print("\n各特征的统计描述（按均值降序排列）:")
print(statistics_sorted)

# 保存统计结果到CSV文件（可选）
# statistics_sorted.to_csv('ctg_statistics.csv', encoding='utf-8-sig')
# print("\n统计描述已保存到: ctg_statistics.csv")

# ==================== OUTPUT ====================
# 运行此单元格，输出：
# - 所有特征的详细统计描述（均值、标准差、分位数等）


# ==================== CODE CELL ====================
# 可视化关键特征在不同健康状态下的分布
print("\n正在生成特征分布对比图...")

# 选择6个最重要的特征进行可视化
# 这些特征在临床上对判断胎儿健康状态有重要意义
important_features = ['LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV']
feature_names_chinese = {
    'LB': '基线胎心率',
    'AC': '加速次数',
    'FM': '胎动次数',
    'UC': '子宫收缩次数',
    'ASTV': '短期变异异常%',
    'MSTV': '短期变异均值'
}

# 创建2行3列的子图布局
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()  # 将2D数组展平为1D，方便循环

# 为三个类别设置不同的颜色
colors_dict = {1: 'green', 2: 'orange', 3: 'red'}
labels_dict = {1: 'Normal(正常)', 2: 'Suspect(可疑)', 3: 'Pathologic(病理)'}

# 遍历每个特征，绘制其在不同健康状态下的分布
for idx, feature in enumerate(important_features):
    ax = axes[idx]
    
    # 为每个类别绘制直方图
    for health_status in [1, 2, 3]:
        # 筛选出该健康状态的所有样本
        data_subset = ctg_data[ctg_data['NSP'] == health_status][feature]
        
        # 绘制直方图
        ax.hist(
            data_subset,  # 数据
            alpha=0.6,  # 透明度，使重叠部分可见
            bins=30,  # 分箱数量
            label=labels_dict[health_status],  # 图例标签
            color=colors_dict[health_status],  # 颜色
            edgecolor='black',  # 边框颜色
            linewidth=0.5  # 边框宽度
        )
    
    # 设置子图标题和标签
    chinese_name = feature_names_chinese.get(feature, feature)
    ax.set_title(f'{feature} - {chinese_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('数值', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)  # 添加图例
    ax.grid(axis='y', alpha=0.3, linestyle='--')  # 添加网格线

# 调整子图布局
plt.tight_layout()
plt.show()

print("✓ 特征分布对比图完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示：
# - 6个关键特征在三种健康状态下的分布直方图


# ==================== CODE CELL ====================
# 绘制特征相关性热力图
print("\n正在生成特征相关性热力图...")

# 选择所有数值型特征（排除CLASS列，因为它是分类代码）
# NSP是目标变量，也包含在内以观察其与特征的相关性
numeric_features = ctg_data.drop('CLASS', axis=1)

# 计算皮尔逊相关系数矩阵
# 相关系数范围在-1到1之间：
# 1表示完全正相关，-1表示完全负相关，0表示不相关
correlation_matrix = numeric_features.corr()

# 创建图形
plt.figure(figsize=(16, 14))

# 绘制热力图
sns.heatmap(
    correlation_matrix,  # 相关系数矩阵
    annot=True,  # 在每个格子中显示数值
    fmt='.2f',  # 数值格式：保留2位小数
    cmap='coolwarm',  # 颜色映射：蓝色表示负相关，红色表示正相关
    center=0,  # 将0值设为中心颜色
    square=True,  # 每个格子为正方形
    linewidths=0.5,  # 格子之间的线宽
    cbar_kws={"shrink": 0.8},  # 颜色条的大小
    annot_kws={"size": 8}  # 注释文字大小
)

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
# 运行此单元格，显示：
# - 所有数值特征之间的相关性热力图


# ==================== MARKDOWN ====================
# ### 3.3 数据预处理
# 
# 将数据分为特征矩阵和目标变量，然后划分训练集和测试集，最后进行特征标准化。

# ==================== CODE CELL ====================
print("\n" + "="*70)
print("数据预处理")
print("="*70)

# 【步骤1】分离特征和目标变量
print("\n【步骤1：分离特征和目标变量】")
print("-"*70)

# X: 特征矩阵，包含所有输入特征
# 我们需要删除NSP（目标变量）和CLASS（分类代码，与NSP相关）
X = ctg_data.drop(['NSP', 'CLASS'], axis=1)

# y: 目标变量，即我们要预测的胎儿健康状态
y = ctg_data['NSP']

print(f"特征矩阵 X 的形状: {X.shape}")  # (样本数, 特征数)
print(f"  - 样本数: {X.shape[0]}")
print(f"  - 特征数: {X.shape[1]}")
print(f"\n目标变量 y 的形状: {y.shape}")  # (样本数,)
print(f"  - 样本数: {y.shape[0]}")

print(f"\n特征列表（共{len(X.columns)}个）:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")

# 【步骤2】划分训练集和测试集
print("\n【步骤2：划分训练集和测试集】")
print("-"*70)

# 使用train_test_split函数将数据划分为训练集（80%）和测试集（20%）
# test_size=0.2: 测试集占20%
# random_state=42: 设置随机种子，保证每次划分结果相同
# stratify=y: 分层采样，保证训练集和测试集中各类别的比例与原数据集相同
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 测试集占20%
    random_state=42,  # 随机种子
    stratify=y  # 分层采样，保持类别比例
)

print(f"训练集大小: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"测试集大小: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)")

# 检查训练集和测试集中的类别分布是否平衡
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

# 【步骤3】特征标准化
print("\n【步骤3：特征标准化】")
print("-"*70)

# 特征标准化的目的：
# 1. 将所有特征缩放到相同的尺度（均值为0，标准差为1）
# 2. 避免数值范围大的特征主导模型训练
# 3. 加速模型收敛
print("标准化方法: Z-score标准化 (均值=0, 标准差=1)")
print("公式: z = (x - μ) / σ")
print("  其中 μ 是均值，σ 是标准差")

# 创建标准化器对象
scaler = StandardScaler()

# 在训练集上拟合标准化器（计算均值和标准差）
# 然后对训练集进行标准化
X_train_scaled = scaler.fit_transform(X_train)

# 使用训练集的均值和标准差对测试集进行标准化
# 注意：不能在测试集上重新fit，这会导致数据泄露
X_test_scaled = scaler.transform(X_test)

print(f"\n标准化完成！")
print(f"标准化后训练集形状: {X_train_scaled.shape}")
print(f"标准化后测试集形状: {X_test_scaled.shape}")

# 显示标准化前后的对比（以第一个样本的前5个特征为例）
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
# 运行此单元格，输出：
# - 特征和目标变量的形状信息
# - 训练集和测试集的大小和类别分布
# - 特征标准化的说明和效果示例


# ==================== MARKDOWN ====================
# ## 4. 模型构建与训练
# 
# ### 4.1 随机森林模型（基础版本）
# 
# 随机森林是一种集成学习方法，它通过构建多个决策树并综合它们的预测结果来进行分类。
# 
# **随机森林的核心思想：**
# 1. **Bootstrap采样**：从训练数据中有放回地随机抽取多个子集
# 2. **随机特征选择**：在每个决策树的每次分裂时，随机选择部分特征
# 3. **多数投票**：综合所有决策树的预测结果，采用多数投票的方式得到最终预测
# 
# **随机森林的优点：**
# - 具有很高的准确率
# - 对异常值和噪声有很好的容忍度
# - 不容易过拟合
# - 可以处理高维数据
# - 可以评估特征的重要性

# ==================== CODE CELL ====================
print("\n" + "="*70)
print("4.1 随机森林模型训练（基础版本）")
print("="*70)

# 初始化随机森林分类器
# 设置各项超参数
print("\n【模型参数设置】")
print("-"*70)
print("n_estimators=100      : 决策树的数量，更多的树通常能提高性能但也增加计算时间")
print("max_depth=10          : 每棵树的最大深度，限制树的生长可以防止过拟合")
print("min_samples_split=10  : 分裂内部节点所需的最小样本数")
print("min_samples_leaf=4    : 叶节点所需的最小样本数")
print("random_state=42       : 随机种子，确保结果可重复")
print("n_jobs=-1             : 使用所有CPU核心进行并行计算，加快训练速度")

# 创建随机森林分类器对象
rf_model = RandomForestClassifier(
    n_estimators=100,      # 构建100棵决策树
    max_depth=10,          # 每棵树最多10层深
    min_samples_split=10,  # 至少10个样本才能继续分裂节点
    min_samples_leaf=4,    # 叶节点至少包含4个样本
    random_state=42,       # 设置随机种子
    n_jobs=-1              # 使用所有可用的CPU核心
)

# 训练模型
print("\n【开始训练模型】")
print("-"*70)
print("正在训练随机森林模型...")
print("（根据数据大小，这可能需要几秒到几分钟）")

# fit()方法用于训练模型，输入标准化后的训练数据和对应的标签
rf_model.fit(X_train_scaled, y_train)

print("✓ 随机森林模型训练完成！")

# 使用训练好的模型进行预测
print("\n【模型预测】")
print("-"*70)

# 对测试集进行预测，得到预测的类别标签
y_pred_rf = rf_model.predict(X_test_scaled)

# 获取每个类别的预测概率
# predict_proba()返回每个样本属于每个类别的概率
# 返回形状为(n_samples, n_classes)的数组
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)

print(f"预测完成！共预测 {len(y_pred_rf)} 个样本")
print(f"预测结果示例（前10个样本）:")
print(f"{'真实值':<10s} {'预测值':<10s} {'预测正确?':<12s} {'Normal概率':<12s} {'Suspect概率':<13s} {'Pathologic概率':<15s}")
print("-"*80)
for i in range(min(10, len(y_pred_rf))):
    correct = "✓" if y_test.iloc[i] == y_pred_rf[i] else "✗"
    print(f"{y_test.iloc[i]:<10d} {y_pred_rf[i]:<10d} {correct:<12s} "
          f"{y_pred_proba_rf[i][0]:>11.4f} {y_pred_proba_rf[i][1]:>12.4f} {y_pred_proba_rf[i][2]:>14.4f}")

# 评估模型性能
print("\n【模型性能评估】")
print("="*70)

# 计算各项评估指标
# accuracy_score: 准确率 = 正确预测的样本数 / 总样本数
accuracy = accuracy_score(y_test, y_pred_rf)

# precision_score: 精确率 = 真正例 / (真正例 + 假正例)
# average='weighted': 按各类别样本数加权平均
precision = precision_score(y_test, y_pred_rf, average='weighted')

# recall_score: 召回率 = 真正例 / (真正例 + 假负例)
recall = recall_score(y_test, y_pred_rf, average='weighted')

# f1_score: F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
# F1分数是精确率和召回率的调和平均数
f1 = f1_score(y_test, y_pred_rf, average='weighted')

print("\n整体性能指标:")
print(f"  准确率 (Accuracy) : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  召回率 (Recall)   : {recall:.4f}")
print(f"  F1分数 (F1-Score) : {f1:.4f}")

# 生成详细的分类报告
# classification_report会为每个类别单独计算精确率、召回率和F1分数
print("\n详细分类报告:")
print("-"*70)
target_names = ['Normal(正常)', 'Suspect(可疑)', 'Pathologic(病理)']
print(classification_report(y_test, y_pred_rf, target_names=target_names))

# ==================== OUTPUT ====================
# 运行此单元格，输出：
# - 模型参数设置说明
# - 训练过程信息
# - 预测结果示例
# - 整体性能指标（准确率、精确率、召回率、F1分数）
# - 每个类别的详细分类报告


# ==================== CODE CELL ====================
# 绘制混淆矩阵
print("\n正在生成混淆矩阵...")

# 混淆矩阵：展示模型预测结果与真实标签的对比
# 行表示真实类别，列表示预测类别
# 对角线上的值表示正确预测的数量
cm_rf = confusion_matrix(y_test, y_pred_rf)

# 创建图形
plt.figure(figsize=(10, 8))

# 使用seaborn的heatmap函数绘制混淆矩阵热力图
sns.heatmap(
    cm_rf,  # 混淆矩阵数据
    annot=True,  # 在每个格子中显示数值
    fmt='d',  # 数值格式为整数
    cmap='Blues',  # 使用蓝色系配色
    xticklabels=['Normal', 'Suspect', 'Pathologic'],  # x轴标签
    yticklabels=['Normal', 'Suspect', 'Pathologic'],  # y轴标签
    cbar_kws={'label': '样本数量'},  # 颜色条标签
    annot_kws={'size': 14, 'weight': 'bold'}  # 注释文字样式
)

plt.title('随机森林模型混淆矩阵', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('真实标签', fontsize=13, fontweight='bold')
plt.xlabel('预测标签', fontsize=13, fontweight='bold')

# 添加说明文字
plt.text(1.5, -0.3, '说明：对角线上的值表示正确预测的数量', 
         ha='center', fontsize=10, style='italic', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.show()

print("✓ 混淆矩阵绘制完成！")

# 计算并显示每个类别的分类准确率
print("\n【各类别分类准确率】")
print("-"*70)
for i, class_name in enumerate(['Normal(正常)', 'Suspect(可疑)', 'Pathologic(病理)']):
    # 该类别的总样本数
    total = cm_rf[i].sum()
    # 该类别正确预测的数量
    correct = cm_rf[i, i]
    # 该类别的准确率
    class_accuracy = correct / total if total > 0 else 0
    print(f"{class_name:<20s}: {correct:>4d}/{total:<4d} = {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

# ==================== OUTPUT ====================
# 运行此单元格，显示：
# - 混淆矩阵热力图
# - 每个类别的分类准确率


# ==================== CODE CELL ====================
# 特征重要性分析
print("\n" + "="*70)
print("特征重要性分析")
print("="*70)

# 随机森林可以计算每个特征的重要性
# 特征重要性基于该特征在所有决策树中用于分裂时带来的信息增益
# 重要性越高，说明该特征对模型预测的贡献越大
feature_importance = rf_model.feature_importances_

# 创建包含特征名和重要性的DataFrame，便于排序和显示
importance_df = pd.DataFrame({
    'feature': X.columns,  # 特征名称
    'importance': feature_importance  # 重要性得分
}).sort_values('importance', ascending=False)  # 按重要性降序排列

print("\n【特征重要性排名 Top 15】")
print("-"*70)
print(f"{'排名':<6s} {'特征名':<30s} {'重要性得分':<12s} {'累计贡献率':<12s}")
print("-"*70)

cumulative = 0  # 累计贡献率
for rank, (idx, row) in enumerate(importance_df.head(15).iterrows(), 1):
    cumulative += row['importance']
    print(f"{rank:<6d} {row['feature']:<30s} {row['importance']:<12.6f} {cumulative:<12.4f}")

print("\n【特征重要性解读】")
print("-"*70)
top3_features = importance_df.head(3)
print("最重要的3个特征:")
for i, (idx, row) in enumerate(top3_features.iterrows(), 1):
    print(f"  {i}. {row['feature']}: {row['importance']:.6f}")
    
total_importance_top3 = top3_features['importance'].sum()
print(f"\n前3个特征的累计贡献率: {total_importance_top3:.4f} ({total_importance_top3*100:.2f}%)")

# 可视化特征重要性
print("\n正在生成特征重要性图...")

# 选取前15个最重要的特征进行可视化
top_n = 15
top_features = importance_df.head(top_n)

plt.figure(figsize=(12, 8))

# 绘制水平条形图
bars = plt.barh(
    range(len(top_features)),  # y轴位置
    top_features['importance'],  # 条形长度（重要性得分）
    color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))  # 渐变色
)

# 设置y轴刻度和标签
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('重要性得分', fontsize=13, fontweight='bold')
plt.ylabel('特征名称', fontsize=13, fontweight='bold')
plt.title(f'随机森林特征重要性（Top {top_n}）', fontsize=15, fontweight='bold', pad=20)

# 在每个条形上添加数值标签
for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
    plt.text(
        importance + 0.001,  # x位置：条形末端稍右
        i,  # y位置
        f'{importance:.4f}',  # 文本内容
        va='center',  # 垂直居中对齐
        fontsize=9,
        fontweight='bold'
    )

# 反转y轴，使重要性最高的特征显示在最上方
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("✓ 特征重要性可视化完成！")

# ==================== OUTPUT ====================
# 运行此单元格，输出：
# - 特征重要性排名表
# - 特征重要性柱状图


# ==================== MARKDOWN ====================
# ### 4.2 随机森林超参数优化
# 
# 使用网格搜索（Grid Search）结合交叉验证来寻找最优的超参数组合。

# ==================== CODE CELL ====================
print("\n" + "="*70)
print("4.2 随机森林超参数优化")
print("="*70)

# 定义要搜索的超参数网格
# GridSearchCV会尝试所有参数组合，找出性能最好的那组
param_grid = {
    'n_estimators': [50, 100, 200],  # 决策树数量：50、100或200棵
    'max_depth': [5, 10, 15, None],  # 树的最大深度：5层、10层、15层或不限制
    'min_samples_split': [5, 10, 15],  # 分裂节点所需最小样本数
    'min_samples_leaf': [2, 4, 6]  # 叶节点最小样本数
}

# 计算总共要尝试的参数组合数
total_combinations = 1
for param_values in param_grid.values():
    total_combinations *= len(param_values)

print(f"\n【超参数网格】")
print("-"*70)
print(f"将要搜索的参数组合总数: {total_combinations}")
print("\n参数网格详情:")
for param_name, param_values in param_grid.items():
    print(f"  {param_name:<20s}: {param_values}")

print(f"\n注意：使用5折交叉验证，总共需要训练 {total_combinations * 5} 个模型")
print("这可能需要较长时间（根据数据大小，可能需要几分钟到十几分钟）")

# 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),  # 基础模型
    param_grid=param_grid,  # 参数网格
    cv=5,  # 5折交叉验证
    scoring='accuracy',  # 评估指标：准确率
    n_jobs=-1,  # 使用所有CPU核心并行计算
    verbose=2,  # 显示详细的训练过程信息
    return_train_score=True  # 返回训练集得分
)

# 执行网格搜索
print("\n【开始网格搜索】")
print("="*70)
grid_search.fit(X_train_scaled, y_train)

print("\n" + "="*70)
print("✓ 网格搜索完成！")
print("="*70)

# 输出最佳参数和得分
print(f"\n【最佳参数组合】")
print("-"*70)
for param_name, param_value in grid_search.best_params_.items():
    print(f"  {param_name:<20s}: {param_value}")

print(f"\n【最佳模型性能】")
print("-"*70)
print(f"  最佳交叉验证得分: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# 获取最优模型
best_rf_model = grid_search.best_estimator_

# 在测试集上评估最优模型
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

print(f"\n【优化后的随机森林在测试集上的性能】")
print("-"*70)
accuracy_best = accuracy_score(y_test, y_pred_best_rf)
precision_best = precision_score(y_test, y_pred_best_rf, average='weighted')
recall_best = recall_score(y_test, y_pred_best_rf, average='weighted')
f1_best = f1_score(y_test, y_pred_best_rf, average='weighted')

print(f"  准确率 (Accuracy) : {accuracy_best:.4f} ({accuracy_best*100:.2f}%)")
print(f"  精确率 (Precision): {precision_best:.4f}")
print(f"  召回率 (Recall)   : {recall_best:.4f}")
print(f"  F1分数 (F1-Score) : {f1_best:.4f}")

# 对比优化前后的性能提升
print(f"\n【性能对比】")
print("-"*70)
print(f"{'指标':<15s} {'基础模型':<15s} {'优化后模型':<15s} {'提升':<15s}")
print("-"*60)
print(f"{'准确率':<15s} {accuracy:<15.4f} {accuracy_best:<15.4f} {accuracy_best-accuracy:+.4f}")
print(f"{'F1分数':<15s} {f1:<15.4f} {f1_best:<15.4f} {f1_best-f1:+.4f}")

# ==================== OUTPUT ====================
# 运行此单元格，输出：
# - 超参数网格详情
# - 网格搜索过程（可能需要几分钟）
# - 最佳参数组合
# - 优化后模型的性能
# - 优化前后的性能对比


# ==================== MARKDOWN ====================
# ### 4.3 Voting分类器
# 
# Voting分类器通过组合多个不同类型的基分类器，利用"集体智慧"提高预测准确率。
# 
# **两种投票方式：**
# 1. **硬投票（Hard Voting）**：每个基分类器投一票，最终选择得票最多的类别
# 2. **软投票（Soft Voting）**：基于每个分类器输出的概率进行加权平均，通常效果更好

# ==================== CODE CELL ====================
print("\n" + "="*70)
print("4.3 Voting分类器训练")
print("="*70)

print("\n【Voting分类器原理】")
print("-"*70)
print("Voting分类器组合多个不同类型的分类器：")
print("  1. 逻辑回归 (Logistic Regression)   - 线性模型")
print("  2. 支持向量机 (SVM)                 - 基于间隔最大化")
print("  3. K近邻 (KNN)                      - 基于相似度")
print("  4. 随机森林 (Random Forest)         - 基于决策树集成")
print("\n每个模型有不同的假设和优势，组合后可以互补不足")

# 创建基分类器
print("\n【创建基分类器】")
print("-"*70)

# 1. 逻辑回归
lr = LogisticRegression(
    max_iter=1000,  # 最大迭代次数
    random_state=42,
    solver='lbfgs',  # 优化算法
    multi_class='multinomial'  # 多分类策略
)
print("✓ 逻辑回归分类器已创建")

# 2. 支持向量机
svm = SVC(
    kernel='rbf',  # 径向基函数核
    probability=True,  # 启用概率估计（软投票需要）
    random_state=42,
    C=1.0,  # 正则化参数
    gamma='scale'  # 核函数系数
)
print("✓ 支持向量机分类器已创建")

# 3. K近邻
knn = KNeighborsClassifier(
    n_neighbors=5,  # 考虑5个最近的邻居
    weights='distance',  # 根据距离加权
    metric='euclidean'  # 欧氏距离
)
print("✓ K近邻分类器已创建")

# 4. 随机森林（用于Voting）
rf_for_voting = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
print("✓ 随机森林分类器已创建")

# 创建硬投票分类器
print("\n【创建Voting分类器】")
print("-"*70)

voting_hard = VotingClassifier(
    estimators=[
        ('lr', lr),  # 逻辑回归
        ('svm', svm),  # 支持向量机
        ('knn', knn),  # K近邻
        ('rf', rf_for_voting)  # 随机森林
    ],
    voting='hard'  # 硬投票模式
)
print("✓ 硬投票(Hard Voting)分类器已创建")

# 创建软投票分类器
voting_soft = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svm', svm),
        ('knn', knn),
        ('rf', rf_for_voting)
    ],
    voting='soft'  # 软投票模式
)
print("✓ 软投票(Soft Voting)分类器已创建")

# 训练硬投票模型
print("\n【训练模型】")
print("="*70)
print("正在训练硬投票(Hard Voting)模型...")
print("（需要训练4个基分类器，可能需要一些时间）")
voting_hard.fit(X_train_scaled, y_train)
y_pred_voting_hard = voting_hard.predict(X_test_scaled)
print("✓ 硬投票模型训练完成！")

# 训练软投票模型
print("\n正在训练软投票(Soft Voting)模型...")
voting_soft.fit(X_train_scaled, y_train)
y_pred_voting_soft = voting_soft.predict(X_test_scaled)
print("✓ 软投票模型训练完成！")

# 评估硬投票模型
print("\n【硬投票模型性能】")
print("-"*70)
accuracy_vh = accuracy_score(y_test, y_pred_voting_hard)
precision_vh = precision_score(y_test, y_pred_voting_hard, average='weighted')
recall_vh = recall_score(y_test, y_pred_voting_hard, average='weighted')
f1_vh = f1_score(y_test, y_pred_voting_hard, average='weighted')

print(f"  准确率 (Accuracy) : {accuracy_vh:.4f} ({accuracy_vh*100:.2f}%)")
print(f"  精确率 (Precision): {precision_vh:.4f}")
print(f"  召回率 (Recall)   : {recall_vh:.4f}")
print(f"  F1分数 (F1-Score) : {f1_vh:.4f}")

# 评估软投票模型
print("\n【软投票模型性能】")
print("-"*70)
accuracy_vs = accuracy_score(y_test, y_pred_voting_soft)
precision_vs = precision_score(y_test, y_pred_voting_soft, average='weighted')
recall_vs = recall_score(y_test, y_pred_voting_soft, average='weighted')
f1_vs = f1_score(y_test, y_pred_voting_soft, average='weighted')

print(f"  准确率 (Accuracy) : {accuracy_vs:.4f} ({accuracy_vs*100:.2f}%)")
print(f"  精确率 (Precision): {precision_vs:.4f}")
print(f"  召回率 (Recall)   : {recall_vs:.4f}")
print(f"  F1分数 (F1-Score) : {f1_vs:.4f}")

# 对比两种投票方式
print("\n【两种投票方式对比】")
print("-"*70)
print(f"{'指标':<15s} {'硬投票':<15s} {'软投票':<15s} {'差异':<15s}")
print("-"*60)
print(f"{'准确率':<15s} {accuracy_vh:<15.4f} {accuracy_vs:<15.4f} {accuracy_vs-accuracy_vh:+.4f}")
print(f"{'F1分数':<15s} {f1_vh:<15.4f} {f1_vs:<15.4f} {f1_vs-f1_vh:+.4f}")

if accuracy_vs > accuracy_vh:
    print("\n✓ 软投票模型性能更优！")
else:
    print("\n✓ 硬投票模型性能更优！")

# ==================== OUTPUT ====================
# 运行此单元格，输出：
# - Voting分类器原理说明
# - 基分类器创建过程
# - 两种投票模式的训练过程
# - 硬投票和软投票模型的性能对比


# ==================== MARKDOWN ====================
# ### 4.4 Bagging分类器
# 
# Bagging（Bootstrap Aggregating）通过对训练数据进行有放回抽样（Bootstrap），
# 训练多个基分类器，然后将它们的预测结果进行聚合。
# 
# **Bagging的核心思想：**
# 1. 从原始训练集中有放回地随机抽取多个子集
# 2. 在每个子集上训练一个基分类器
# 3. 将所有基分类器的预测结果进行平均或投票

# ==================== CODE CELL ====================
print("\n" + "="*70)
print("4.4 Bagging分类器训练")
print("="*70)

print("\n【Bagging分类器原理】")
print("-"*70)
print("Bagging通过Bootstrap采样减少模型方差：")
print("  1. 有放回抽样：从训练集中随机抽取样本（可重复）")
print("  2. 训练多个模型：在每个抽样子集上训练基分类器")
print("  3. 聚合预测：通过投票或平均综合所有模型的预测")
print("\n优点：降低过拟合风险，提高模型稳定性")

# 创建Bagging分类器（使用决策树作为基分类器）
print("\n【创建Bagging分类器】")
print("-"*70)

print("\n1. Bagging + 决策树")
print("   基分类器：决策树（max_depth=10）")
print("   集成数量：100个")
print("   采样比例：80%的样本，80%的特征")

bagging_dt = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10, random_state=42),  # 基分类器
    n_estimators=100,        # 集成100个基分类器
    max_samples=0.8,         # 每个基分类器使用80%的样本
    max_features=0.8,        # 每个基分类器使用80%的特征
    bootstrap=True,          # 有放回抽样
    random_state=42,
    n_jobs=-1  # 并行计算
)
print("✓ Bagging-DecisionTree分类器已创建")

# 创建Bagging分类器（使用KNN作为基分类器）
print("\n2. Bagging + KNN")
print("   基分类器：K近邻（n_neighbors=5）")
print("   集成数量：100个")
print("   采样比例：80%的样本，80%的特征")

bagging_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),  # 基分类器
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
print("✓ Bagging-KNN分类器已创建")

# 训练Bagging-DecisionTree模型
print("\n【训练模型】")
print("="*70)
print("正在训练Bagging-DecisionTree模型...")
print("（需要训练100个决策树，可能需要一些时间）")
bagging_dt.fit(X_train_scaled, y_train)
y_pred_bagging_dt = bagging_dt.predict(X_test_scaled)
print("✓ Bagging-DecisionTree模型训练完成！")

# 训练Bagging-KNN模型
print("\n正在训练Bagging-KNN模型...")
print("（需要训练100个KNN模型，可能需要较长时间）")
bagging_knn.fit(X_train_scaled, y_train)
y_pred_bagging_knn = bagging_k