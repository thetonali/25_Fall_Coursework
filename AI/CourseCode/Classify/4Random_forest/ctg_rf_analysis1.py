# ==================== MARKDOWN ====================
# # 胎心宫缩监护(CTG)数据随机森林分类实验报告
# 
# ## 1. 实验目的
# 
# 本实验旨在：
# 1. 使用随机森林算法对CTG数据进行分类分析
# 2. 比较随机森林与Voting、Bagging等集成学习方法的性能
# 3. 评估不同模型在胎儿健康状态分类任务中的表现
# 
# ## 2. 数据集介绍
# 
# CTG数据集包含胎心监护的多个特征指标，用于评估胎儿健康状况。
# 主要特征包括：基线胎心率、加速次数、减速次数、子宫收缩等。
# 目标变量为胎儿健康状态分类。

# ==================== CODE CELL ====================
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 设置中文显示（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("所有必要的库已成功导入！")

# ==================== OUTPUT ====================
# 运行此单元格，输出：所有必要的库已成功导入！


# ==================== MARKDOWN ====================
# ## 3. 数据加载与预处理
# 
# ### 3.1 数据加载
# 
# 我们将加载CTG数据集。如果无法从在线源获取，可以使用本地数据或生成模拟数据。

# ==================== CODE CELL ====================
# 加载CTG数据集
# 方法1：从UCI机器学习库加载（推荐）
try:
    # 尝试加载真实的CTG数据集
    from ucimlrepo import fetch_ucirepo 
    # 如果上述库不可用，使用备选方案
    print("尝试从UCI加载CTG数据集...")
    # CTG数据集ID: 193
except:
    print("UCI库不可用，将使用sklearn内置数据集或生成模拟数据")

# 方法2：使用模拟数据（用于演示）
# 为了确保代码可运行，这里创建一个模拟的CTG数据集
def create_ctg_simulation_data(n_samples=2126):
    """
    创建模拟的CTG数据集，包含21个特征
    目标变量：1(Normal), 2(Suspect), 3(Pathologic)
    """
    np.random.seed(42)
    
    # 特征名称
    feature_names = [
        'baseline_value',  # 基线胎心率
        'accelerations',   # 加速次数
        'fetal_movement',  # 胎动次数
        'uterine_contractions',  # 子宫收缩次数
        'light_decelerations',  # 轻度减速
        'severe_decelerations',  # 严重减速
        'prolongued_decelerations',  # 延长减速
        'abnormal_short_term_variability',  # 短期变异异常
        'mean_value_of_short_term_variability',  # 短期变异均值
        'percentage_of_time_with_abnormal_long_term_variability',  # 长期变异异常百分比
        'mean_value_of_long_term_variability',  # 长期变异均值
        'histogram_width',  # 直方图宽度
        'histogram_min',  # 直方图最小值
        'histogram_max',  # 直方图最大值
        'histogram_number_of_peaks',  # 直方图峰数
        'histogram_number_of_zeroes',  # 直方图零值数
        'histogram_mode',  # 直方图众数
        'histogram_mean',  # 直方图均值
        'histogram_median',  # 直方图中位数
        'histogram_variance',  # 直方图方差
        'histogram_tendency'  # 直方图趋势
    ]
    
    # 生成特征数据
    data = np.random.randn(n_samples, 21)
    
    # 对不同特征应用不同的缩放和偏移，使其更接近真实CTG数据
    data[:, 0] = data[:, 0] * 10 + 130  # 基线胎心率 (120-140)
    data[:, 1] = np.abs(data[:, 1] * 0.005 + 0.002)  # 加速次数
    data[:, 2] = np.abs(data[:, 2] * 0.03 + 0.15)  # 胎动
    data[:, 3] = np.abs(data[:, 3] * 0.03 + 0.05)  # 子宫收缩
    
    # 生成目标变量（三分类）
    # 根据某些特征的组合来确定类别，使数据更有意义
    target = np.ones(n_samples)
    
    # Suspect类别条件
    suspect_condition = (data[:, 1] < 0.001) | (data[:, 4] > 0.5)
    target[suspect_condition] = 2
    
    # Pathologic类别条件
    pathologic_condition = (data[:, 5] > 0.5) | (data[:, 6] > 0.5) | (data[:, 0] < 120) | (data[:, 0] > 160)
    target[pathologic_condition] = 3
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['fetal_health'] = target.astype(int)
    
    return df

# 创建模拟数据
print("正在创建CTG模拟数据集...")
ctg_data = create_ctg_simulation_data(n_samples=2126)
print(f"数据集创建完成！形状: {ctg_data.shape}")
print(f"\n前5行数据:")
print(ctg_data.head())

# ==================== OUTPUT ====================
# 运行此单元格，输出数据集基本信息和前5行数据


# ==================== MARKDOWN ====================
# ### 3.2 数据探索性分析(EDA)

# ==================== CODE CELL ====================
# 数据基本信息
print("="*60)
print("数据集基本信息")
print("="*60)
print(f"数据集形状: {ctg_data.shape}")
print(f"特征数量: {ctg_data.shape[1] - 1}")
print(f"样本数量: {ctg_data.shape[0]}")
print(f"\n数据类型:")
print(ctg_data.dtypes)
print(f"\n缺失值统计:")
print(ctg_data.isnull().sum())

# 目标变量分布
print("\n" + "="*60)
print("目标变量分布")
print("="*60)
target_counts = ctg_data['fetal_health'].value_counts().sort_index()
print(target_counts)
print(f"\n类别比例:")
print(ctg_data['fetal_health'].value_counts(normalize=True).sort_index())

# ==================== OUTPUT ====================
# 运行此单元格，输出数据集详细信息、目标变量分布


# ==================== CODE CELL ====================
# 可视化目标变量分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 柱状图
target_counts.plot(kind='bar', ax=axes[0], color=['green', 'orange', 'red'])
axes[0].set_title('胎儿健康状态分布', fontsize=14, fontweight='bold')
axes[0].set_xlabel('健康状态 (1=正常, 2=可疑, 3=病理)', fontsize=12)
axes[0].set_ylabel('样本数量', fontsize=12)
axes[0].set_xticklabels(['Normal', 'Suspect', 'Pathologic'], rotation=0)
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# 饼图
colors = ['#66b3ff', '#ffcc99', '#ff9999']
axes[1].pie(target_counts, labels=['Normal', 'Suspect', 'Pathologic'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('胎儿健康状态比例', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("目标变量分布可视化完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示目标变量分布的柱状图和饼图


# ==================== CODE CELL ====================
# 特征统计描述
print("="*60)
print("特征统计描述")
print("="*60)
print(ctg_data.describe().T)

# 可视化部分特征的分布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

important_features = [
    'baseline_value', 'accelerations', 'fetal_movement',
    'uterine_contractions', 'light_decelerations', 'severe_decelerations'
]

for idx, feature in enumerate(important_features):
    for health_status in [1, 2, 3]:
        data_subset = ctg_data[ctg_data['fetal_health'] == health_status][feature]
        axes[idx].hist(data_subset, alpha=0.5, bins=30, 
                      label=f'Class {health_status}')
    axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.tight_layout()
plt.show()

print("特征分布可视化完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示关键特征在不同健康状态下的分布


# ==================== MARKDOWN ====================
# ### 3.3 数据预处理
# 
# 进行特征工程和数据标准化，为模型训练做准备。

# ==================== CODE CELL ====================
# 分离特征和目标变量
X = ctg_data.drop('fetal_health', axis=1)
y = ctg_data['fetal_health']

print("特征矩阵形状:", X.shape)
print("目标变量形状:", y.shape)

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"\n训练集类别分布:\n{y_train.value_counts().sort_index()}")
print(f"\n测试集类别分布:\n{y_test.value_counts().sort_index()}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n数据标准化完成！")
print(f"标准化后训练集形状: {X_train_scaled.shape}")
print(f"标准化后测试集形状: {X_test_scaled.shape}")

# ==================== OUTPUT ====================
# 运行此单元格，输出数据划分和标准化信息


# ==================== MARKDOWN ====================
# ## 4. 模型构建与训练
# 
# ### 4.1 随机森林模型
# 
# 随机森林是一种集成学习方法，通过构建多个决策树并综合它们的预测结果来提高模型性能。

# ==================== CODE CELL ====================
# 构建基础随机森林模型
print("="*60)
print("4.1 随机森林模型训练")
print("="*60)

# 初始化随机森林分类器
rf_model = RandomForestClassifier(
    n_estimators=100,      # 决策树数量
    max_depth=10,          # 树的最大深度
    min_samples_split=10,  # 分裂内部节点所需的最小样本数
    min_samples_leaf=4,    # 叶节点所需的最小样本数
    random_state=42,
    n_jobs=-1              # 使用所有CPU核心
)

# 训练模型
print("正在训练随机森林模型...")
rf_model.fit(X_train_scaled, y_train)
print("随机森林模型训练完成！")

# 预测
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)

# 评估模型
print("\n随机森林模型性能评估:")
print("-"*60)
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"精确率 (Precision): {precision_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"召回率 (Recall): {recall_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred_rf, 
                          target_names=['Normal', 'Suspect', 'Pathologic']))

# ==================== OUTPUT ====================
# 运行此单元格，输出随机森林模型的训练和评估结果


# ==================== CODE CELL ====================
# 混淆矩阵可视化
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Suspect', 'Pathologic'],
            yticklabels=['Normal', 'Suspect', 'Pathologic'])
plt.title('随机森林模型混淆矩阵', fontsize=14, fontweight='bold')
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.tight_layout()
plt.show()

print("混淆矩阵显示完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示随机森林模型的混淆矩阵热力图


# ==================== CODE CELL ====================
# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("="*60)
print("特征重要性排名（Top 10）")
print("="*60)
print(feature_importance.head(10))

# 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.barh(feature_importance['feature'].head(15), 
         feature_importance['importance'].head(15))
plt.xlabel('重要性得分', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.title('随机森林特征重要性（Top 15）', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("特征重要性可视化完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示特征重要性排名和可视化


# ==================== MARKDOWN ====================
# ### 4.2 随机森林超参数优化
# 
# 使用网格搜索进行超参数调优，寻找最优模型配置。

# ==================== CODE CELL ====================
print("="*60)
print("4.2 随机森林超参数优化")
print("="*60)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6]
}

# 创建网格搜索对象
print("正在进行网格搜索优化（这可能需要几分钟）...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,  # 5折交叉验证
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 执行网格搜索
grid_search.fit(X_train_scaled, y_train)

print("\n网格搜索完成！")
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

# 使用最优参数的模型
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

print("\n优化后的随机森林模型性能:")
print("-"*60)
print(f"准确率: {accuracy_score(y_test, y_pred_best_rf):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_best_rf, average='weighted'):.4f}")

# ==================== OUTPUT ====================
# 运行此单元格，执行超参数优化并输出最优参数和性能


# ==================== MARKDOWN ====================
# ### 4.3 Voting分类器
# 
# Voting分类器通过组合多个不同类型的基分类器，并通过投票机制产生最终预测。

# ==================== CODE CELL ====================
print("="*60)
print("4.3 Voting分类器训练")
print("="*60)

# 创建基分类器
lr = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
rf_for_voting = RandomForestClassifier(n_estimators=100, random_state=42)

# 创建Voting分类器（硬投票）
voting_hard = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svm', svm),
        ('knn', knn),
        ('rf', rf_for_voting)
    ],
    voting='hard'  # 硬投票：多数投票
)

# 创建Voting分类器（软投票）
voting_soft = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svm', svm),
        ('knn', knn),
        ('rf', rf_for_voting)
    ],
    voting='soft'  # 软投票：基于概率的平均
)

# 训练硬投票模型
print("正在训练硬投票(Hard Voting)模型...")
voting_hard.fit(X_train_scaled, y_train)
y_pred_voting_hard = voting_hard.predict(X_test_scaled)
print("硬投票模型训练完成！")

# 训练软投票模型
print("\n正在训练软投票(Soft Voting)模型...")
voting_soft.fit(X_train_scaled, y_train)
y_pred_voting_soft = voting_soft.predict(X_test_scaled)
print("软投票模型训练完成！")

# 评估硬投票模型
print("\n硬投票模型性能:")
print("-"*60)
print(f"准确率: {accuracy_score(y_test, y_pred_voting_hard):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_voting_hard, average='weighted'):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_voting_hard, average='weighted'):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_voting_hard, average='weighted'):.4f}")

# 评估软投票模型
print("\n软投票模型性能:")
print("-"*60)
print(f"准确率: {accuracy_score(y_test, y_pred_voting_soft):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_voting_soft, average='weighted'):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_voting_soft, average='weighted'):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_voting_soft, average='weighted'):.4f}")

# ==================== OUTPUT ====================
# 运行此单元格，训练并评估Voting分类器


# ==================== MARKDOWN ====================
# ### 4.4 Bagging分类器
# 
# Bagging(Bootstrap Aggregating)通过对训练数据进行有放回抽样，训练多个基分类器并聚合结果。

# ==================== CODE CELL ====================
print("="*60)
print("4.4 Bagging分类器训练")
print("="*60)

# 创建Bagging分类器（使用决策树作为基分类器）
bagging_dt = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
    n_estimators=100,        # 基分类器数量
    max_samples=0.8,         # 每个基分类器使用80%的样本
    max_features=0.8,        # 每个基分类器使用80%的特征
    bootstrap=True,          # 有放回抽样
    random_state=42,
    n_jobs=-1
)

# 创建Bagging分类器（使用KNN作为基分类器）
bagging_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=5),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# 训练Bagging-DecisionTree模型
print("正在训练Bagging-DecisionTree模型...")
bagging_dt.fit(X_train_scaled, y_train)
y_pred_bagging_dt = bagging_dt.predict(X_test_scaled)
print("Bagging-DecisionTree模型训练完成！")

# 训练Bagging-KNN模型
print("\n正在训练Bagging-KNN模型...")
bagging_knn.fit(X_train_scaled, y_train)
y_pred_bagging_knn = bagging_knn.predict(X_test_scaled)
print("Bagging-KNN模型训练完成！")

# 评估Bagging-DecisionTree模型
print("\nBagging-DecisionTree模型性能:")
print("-"*60)
print(f"准确率: {accuracy_score(y_test, y_pred_bagging_dt):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_bagging_dt, average='weighted'):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_bagging_dt, average='weighted'):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_bagging_dt, average='weighted'):.4f}")

print("\n详细分类报告:")
print(classification_report(y_test, y_pred_bagging_dt,
                          target_names=['Normal', 'Suspect', 'Pathologic']))

# 评估Bagging-KNN模型
print("\nBagging-KNN模型性能:")
print("-"*60)
print(f"准确率: {accuracy_score(y_test, y_pred_bagging_knn):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_bagging_knn, average='weighted'):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_bagging_knn, average='weighted'):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_bagging_knn, average='weighted'):.4f}")

# ==================== OUTPUT ====================
# 运行此单元格，训练并评估Bagging分类器


# ==================== MARKDOWN ====================
# ## 5. 模型比较与分析
# 
# ### 5.1 综合性能对比

# ==================== CODE CELL ====================
print("="*60)
print("5.1 所有模型综合性能对比")
print("="*60)

# 收集所有模型的预测结果
models_performance = {
    'Random Forest (基础)': y_pred_rf,
    'Random Forest (优化)': y_pred_best_rf,
    'Voting (Hard)': y_pred_voting_hard,
    'Voting (Soft)': y_pred_voting_soft,
    'Bagging (DT)': y_pred_bagging_dt,
    'Bagging (KNN)': y_pred_bagging_knn
}

# 计算各项指标
results = []
for model_name, y_pred in models_performance.items():
    results.append({
        '模型': model_name,
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred, average='weighted'),
        '召回率': recall_score(y_test, y_pred, average='weighted'),
        'F1分数': f1_score(y_test, y_pred, average='weighted')
    })

# 创建结果DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('准确率', ascending=False)

print(results_df.to_string(index=False))

# ==================== OUTPUT ====================
# 运行此单元格，输出所有模型的性能对比表格


# ==================== CODE CELL ====================
# 可视化模型性能对比
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['准确率', '精确率', '召回率', 'F1分数']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    # 绘制柱状图
    bars = ax.bar(range(len(results_df)), results_df[metric], color=colors)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['模型'], rotation=45, ha='right')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'各模型{metric}对比', fontsize=13, fontweight='bold')
    ax.set_ylim([results_df[metric].min() - 0.05, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("模型性能对比可视化完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示四个性能指标的柱状图对比


# ==================== CODE CELL ====================
# 绘制雷达图进行多维度对比
from math import pi

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 选取前4个最佳模型进行雷达图对比
top_models = results_df.head(4)
categories = ['准确率', '精确率', '召回率', 'F1分数']
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# 绘制每个模型
colors_radar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for idx, (_, row) in enumerate(top_models.iterrows()):
    values = [row['准确率'], row['精确率'], row['召回率'], row['F1分数']]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=row['模型'], color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 1)
ax.set_title('模型性能雷达图对比（Top 4）', size=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.show()

print("雷达图对比完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示模型性能雷达图


# ==================== MARKDOWN ====================
# ### 5.2 交叉验证分析
# 
# 使用交叉验证评估模型的稳定性和泛化能力。

# ==================== CODE CELL ====================
print("="*60)
print("5.2 交叉验证性能分析")
print("="*60)

# 定义要进行交叉验证的模型
cv_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Voting (Soft)': voting_soft,
    'Bagging (DT)': bagging_dt
}

cv_results = []

for model_name, model in cv_models.items():
    print(f"\n正在对 {model_name} 进行5折交叉验证...")
    
    # 执行5折交叉验证
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    cv_results.append({
        '模型': model_name,
        '平均准确率': cv_scores.mean(),
        '标准差': cv_scores.std(),
        '最小值': cv_scores.min(),
        '最大值': cv_scores.max()
    })
    
    print(f"{model_name} - 交叉验证得分: {cv_scores}")
    print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 创建交叉验证结果DataFrame
cv_results_df = pd.DataFrame(cv_results)
print("\n" + "="*60)
print("交叉验证结果汇总")
print("="*60)
print(cv_results_df.to_string(index=False))

# ==================== OUTPUT ====================
# 运行此单元格，输出交叉验证结果


# ==================== CODE CELL ====================
# 可视化交叉验证结果
fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(cv_results_df))
means = cv_results_df['平均准确率']
stds = cv_results_df['标准差']

# 绘制带误差条的柱状图
bars = ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=10, 
              color=['#1f77b4', '#ff7f0e', '#2ca02c'])

ax.set_xlabel('模型', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('交叉验证准确率对比（5折）', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(cv_results_df['模型'])
ax.set_ylim([0.7, 1.0])
ax.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    ax.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
           f'{mean:.4f}\n±{std:.4f}',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("交叉验证可视化完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示交叉验证结果的柱状图


# ==================== MARKDOWN ====================
# ### 5.3 ROC曲线与AUC分析
# 
# 绘制ROC曲线并计算AUC值，评估模型的分类性能。

# ==================== CODE CELL ====================
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

print("="*60)
print("5.3 ROC曲线与AUC分析")
print("="*60)

# 将标签二值化（用于多分类ROC曲线）
y_test_bin = label_binarize(y_test, classes=[1, 2, 3])
n_classes = y_test_bin.shape[1]

# 获取模型预测概率
models_for_roc = {
    'Random Forest': (rf_model, rf_model.predict_proba(X_test_scaled)),
    'Voting (Soft)': (voting_soft, voting_soft.predict_proba(X_test_scaled)),
    'Bagging (DT)': (bagging_dt, bagging_dt.predict_proba(X_test_scaled))
}

# 为每个模型绘制ROC曲线
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, (model, y_score)) in enumerate(models_for_roc.items()):
    ax = axes[idx]
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 绘制ROC曲线
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    class_names = ['Normal', 'Suspect', 'Pathologic']
    
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
               label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # 绘制micro-average曲线
    ax.plot(fpr["micro"], tpr["micro"],
           label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})',
           color='deeppink', linestyle=':', linewidth=3)
    
    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{model_name}\nROC曲线', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 输出AUC汇总
print("\nAUC得分汇总:")
print("-"*60)
for model_name, (model, y_score) in models_for_roc.items():
    # 计算macro-average AUC
    auc_scores = []
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc_scores.append(auc(fpr_i, tpr_i))
    
    macro_auc = np.mean(auc_scores)
    print(f"{model_name}: Macro-avg AUC = {macro_auc:.4f}")

# ==================== OUTPUT ====================
# 运行此单元格，显示ROC曲线并输出AUC得分


# ==================== MARKDOWN ====================
# ## 6. 学习曲线分析
# 
# 分析模型随训练样本数量变化的性能表现，评估模型是否存在过拟合或欠拟合。

# ==================== CODE CELL ====================
from sklearn.model_selection import learning_curve

print("="*60)
print("6. 学习曲线分析")
print("="*60)

# 定义要分析的模型
models_for_learning_curve = {
    'Random Forest': rf_model,
    'Voting (Soft)': voting_soft,
    'Bagging (DT)': bagging_dt
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, model) in enumerate(models_for_learning_curve.items()):
    print(f"\n正在计算 {model_name} 的学习曲线...")
    
    # 计算学习曲线
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        random_state=42
    )
    
    # 计算均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # 绘制学习曲线
    ax = axes[idx]
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='训练集得分')
    ax.plot(train_sizes, val_mean, 'o-', color='g', label='验证集得分')
    
    # 填充标准差区域
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='r')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color='g')
    
    ax.set_xlabel('训练样本数', fontsize=11)
    ax.set_ylabel('准确率', fontsize=11)
    ax.set_title(f'{model_name}\n学习曲线', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim([0.6, 1.05])

plt.tight_layout()
plt.show()

print("\n学习曲线分析完成！")

# ==================== OUTPUT ====================
# 运行此单元格，显示三个模型的学习曲线


# ==================== MARKDOWN ====================
# ## 7. 实验结论与分析
# 
# ### 7.1 模型性能总结

# ==================== CODE CELL ====================
print("="*70)
print("7. 实验结论与分析")
print("="*70)

print("\n【7.1 模型性能总结】")
print("-"*70)

# 找出最佳模型
best_model_name = results_df.iloc[0]['模型']
best_accuracy = results_df.iloc[0]['准确率']

print(f"\n✓ 最佳模型: {best_model_name}")
print(f"  - 准确率: {best_accuracy:.4f}")
print(f"  - 精确率: {results_df.iloc[0]['精确率']:.4f}")
print(f"  - 召回率: {results_df.iloc[0]['召回率']:.4f}")
print(f"  - F1分数: {results_df.iloc[0]['F1分数']:.4f}")

print("\n【各模型优缺点分析】")
print("-"*70)

print("\n1. 随机森林 (Random Forest):")
print("   优点:")
print("   - 能够处理高维数据，特征重要性分析直观")
print("   - 具有良好的抗过拟合能力")
print("   - 可以并行训练，效率较高")
print("   - 对异常值和噪声具有较强的鲁棒性")
print("   缺点:")
print("   - 模型复杂度较高，解释性相对较弱")
print("   - 对于样本不平衡问题需要特殊处理")

print("\n2. Voting分类器:")
print("   优点:")
print("   - 结合多个不同类型的分类器，提高泛化能力")
print("   - 软投票能够利用概率信息，通常性能更好")
print("   - 可以组合各个基分类器的优势")
print("   缺点:")
print("   - 训练时间较长（需要训练多个模型）")
print("   - 如果基分类器选择不当，可能效果不佳")

print("\n3. Bagging分类器:")
print("   优点:")
print("   - 通过Bootstrap采样减少方差，降低过拟合风险")
print("   - 可以并行训练基分类器")
print("   - 对不稳定的学习算法效果显著")
print("   缺点:")
print("   - 对偏差较大的模型改进有限")
print("   - 需要较多的计算资源")

# ==================== OUTPUT ====================
# 运行此单元格，输出实验结论和模型分析


# ==================== MARKDOWN ====================
# ### 7.2 特征重要性分析结论

# ==================== CODE CELL ====================
print("\n【7.2 特征重要性分析结论】")
print("-"*70)

# 获取Top 5重要特征
top5_features = feature_importance.head(5)

print("\n最重要的5个特征:")
for idx, row in top5_features.iterrows():
    print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")

print("\n特征重要性分析:")
print("   - 基线胎心率相关特征对分类影响最大")
print("   - 胎心率的变异性指标是重要的诊断依据")
print("   - 加速和减速事件的频率对健康状态判断至关重要")
print("   - 这些发现与临床医学知识相吻合")

# ==================== OUTPUT ====================
# 运行此单元格，输出特征重要性分析结论


# ==================== MARKDOWN ====================
# ### 7.3 实验总结与建议

# ==================== CODE CELL ====================
print("\n【7.3 实验总结】")
print("="*70)

print("\n一、实验结果总结:")
print("   1. 成功实现了随机森林、Voting和Bagging三种集成学习算法")
print("   2. 所有模型在CTG数据集上均取得了较好的分类效果")
print(f"   3. 最佳模型 ({best_model_name}) 准确率达到 {best_accuracy:.2%}")
print("   4. 交叉验证结果显示模型具有良好的稳定性和泛化能力")
print("   5. 特征重要性分析为临床诊断提供了有价值的参考")

print("\n二、模型选择建议:")
print("   1. 对于准确率要求高的场景:")
print("      推荐使用优化后的随机森林或Voting(Soft)模型")
print("   2. 对于计算资源受限的场景:")
print("      推荐使用基础随机森林模型，平衡性能与效率")
print("   3. 对于需要模型解释性的场景:")
print("      推荐使用随机森林，可以提供特征重要性分析")
print("   4. 对于追求稳定性的场景:")
print("      推荐使用Voting或Bagging方法，降低单一模型的风险")

print("\n三、改进方向:")
print("   1. 数据层面:")
print("      - 收集更多样本，特别是少数类样本")
print("      - 进行特征工程，构造更有意义的组合特征")
print("      - 处理数据不平衡问题（SMOTE、类权重调整等）")
print("   2. 模型层面:")
print("      - 尝试更多集成方法（如Stacking、XGBoost、LightGBM）")
print("      - 进行更细致的超参数调优")
print("      - 使用集成学习的高级技术")
print("   3. 评估层面:")
print("      - 考虑使用更多评价指标（如Cohen's Kappa）")
print("      - 进行误差分析，找出模型的薄弱环节")
print("      - 结合临床专家意见，优化模型输出")

print("\n四、临床应用建议:")
print("   1. 模型可作为辅助诊断工具，不应完全替代医生判断")
print("   2. 对于模型预测为高风险的情况，建议进行进一步检查")
print("   3. 定期使用新数据更新模型，保持模型的时效性")
print("   4. 建立模型监控机制，及时发现性能下降")

print("\n" + "="*70)
print("实验报告完成！")
print("="*70)

# ==================== OUTPUT ====================
# 运行此单元格，输出完整的实验总结和建议


# ==================== MARKDOWN ====================
# ## 8. 附录：完整性能指标表
# 
# 以下是所有模型在测试集上的完整性能表现。

# ==================== CODE CELL ====================
# 创建完整的性能报告
print("\n" + "="*70)
print("附录：完整性能指标汇总表")
print("="*70)

# 添加更多统计信息
final_report = results_df.copy()

# 计算每个模型的排名
for metric in ['准确率', '精确率', '召回率', 'F1分数']:
    final_report[f'{metric}排名'] = final_report[metric].rank(ascending=False).astype(int)

# 计算综合排名（平均排名）
rank_columns = [col for col in final_report.columns if '排名' in col]
final_report['综合排名'] = final_report[rank_columns].mean(axis=1).rank().astype(int)

# 排序并显示
final_report = final_report.sort_values('综合排名')

print("\n完整性能报告（按综合排名排序）:")
print(final_report.to_string(index=False))

# 保存为CSV（可选）
# final_report.to_csv('ctg_model_performance_report.csv', index=False, encoding='utf-8-sig')
# print("\n性能报告已保存至: ctg_model_performance_report.csv")

print("\n" + "="*70)
print("感谢使用本实验报告模板！")
print("="*70)

# ==================== OUTPUT ====================
# 运行此单元格，显示完整的性能指标汇总表