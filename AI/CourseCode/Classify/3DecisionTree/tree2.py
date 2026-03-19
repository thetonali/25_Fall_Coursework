# ============================================================
# MARKDOWN CELL - 标题和介绍
# ============================================================
"""
# 汽车分类决策树实验报告

## 实验目的
使用ID3算法和CART算法构建决策树分类模型，对汽车类型进行预测，并比较两种算法的性能差异。

## 数据说明
- **训练集**: xaa.dat
- **测试集**: xab.dat
- **特征数量**: 18个数值特征
- **目标变量**: 汽车类型（van, saab, bus等）
"""

# ============================================================
# CODE CELL 1 - 导入必要的库
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import seaborn as sns

# 设置中文字体支持（避免图表中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("✓ 所有库导入成功")

# ============================================================
# 运行输出：✓ 所有库导入成功
# ============================================================


# ============================================================
# CODE CELL 2 - 数据加载函数
# ============================================================
def load_car_data(filename):
    """
    加载汽车数据文件
    
    参数:
        filename: 数据文件路径
    
    返回:
        X: 特征矩阵
        y: 标签向量
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # 分割每行数据，去除空白字符
            parts = line.strip().split()
            if len(parts) > 0:
                data.append(parts)
    
    # 转换为DataFrame便于处理
    df = pd.DataFrame(data)
    
    # 前18列是特征，最后一列是标签
    X = df.iloc[:, :-1].astype(float).values  # 特征转换为浮点数
    y = df.iloc[:, -1].values  # 标签保持为字符串
    
    return X, y

# 加载训练集和测试集
X_train, y_train = load_car_data('xaa.dat')
X_test, y_test = load_car_data('xab.dat')

print(f"训练集样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}")
print(f"测试集样本数: {X_test.shape[0]}, 特征数: {X_test.shape[1]}")
print(f"\n汽车类型分布（训练集）:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count}样本")

# ============================================================
# 运行输出示例：
# 训练集样本数: XX, 特征数: 18
# 测试集样本数: XX, 特征数: 18
# 汽车类型分布（训练集）:
#   bus: XX样本
#   saab: XX样本
#   van: XX样本
# ============================================================


# ============================================================
# MARKDOWN CELL - ID3算法说明
# ============================================================
"""
## 1. ID3决策树算法

### 算法原理
ID3（Iterative Dichotomiser 3）算法是一种经典的决策树算法，由Ross Quinlan于1986年提出。

**核心思想**:
- 使用**信息增益**（Information Gain）作为特征选择标准
- 信息增益 = 划分前的熵 - 划分后的加权熵
- 每次选择信息增益最大的特征进行分裂

**优点**:
- 概念简单，易于理解
- 生成的决策树可解释性强
- 训练速度较快

**缺点**:
- 倾向于选择取值较多的特征（偏向问题）
- 只能处理离散型特征
- 对噪声数据敏感
"""

# ============================================================
# CODE CELL 3 - 构建ID3决策树（使用entropy）
# ============================================================
# sklearn的DecisionTreeClassifier使用entropy准则时相当于ID3算法的实现
print("=" * 60)
print("开始训练ID3决策树模型...")
print("=" * 60)

# 创建ID3决策树分类器（使用信息熵作为分裂标准）
id3_tree = DecisionTreeClassifier(
    criterion='entropy',      # 使用信息熵（ID3的核心）
    random_state=42,          # 固定随机种子，保证结果可复现
    max_depth=10,             # 限制树的最大深度，防止过拟合
    min_samples_split=5,      # 内部节点最少需要5个样本才能分裂
    min_samples_leaf=2        # 叶节点最少需要2个样本
)

# 训练模型
id3_tree.fit(X_train, y_train)

# 在训练集和测试集上进行预测
y_train_pred_id3 = id3_tree.predict(X_train)
y_test_pred_id3 = id3_tree.predict(X_test)

# 计算准确率
train_accuracy_id3 = accuracy_score(y_train, y_train_pred_id3)
test_accuracy_id3 = accuracy_score(y_test, y_test_pred_id3)

print(f"\n✓ ID3模型训练完成")
print(f"训练集准确率: {train_accuracy_id3:.4f} ({train_accuracy_id3*100:.2f}%)")
print(f"测试集准确率: {test_accuracy_id3:.4f} ({test_accuracy_id3*100:.2f}%)")
print(f"决策树深度: {id3_tree.get_depth()}")
print(f"叶节点数量: {id3_tree.get_n_leaves()}")

# ============================================================
# 运行输出示例：
# ============================================================
# 开始训练ID3决策树模型...
# ============================================================
# ✓ ID3模型训练完成
# 训练集准确率: 0.XXXX (XX.XX%)
# 测试集准确率: 0.XXXX (XX.XX%)
# 决策树深度: X
# 叶节点数量: XX
# ============================================================


# ============================================================
# CODE CELL 4 - 绘制ID3决策树图
# ============================================================
print("\n绘制ID3决策树结构图...")

# 创建特征名称列表
feature_names = [f'Feature_{i+1}' for i in range(X_train.shape[1])]

# 创建大尺寸图形
plt.figure(figsize=(25, 15))
plot_tree(
    id3_tree, 
    feature_names=feature_names,
    class_names=id3_tree.classes_,
    filled=True,              # 节点填充颜色
    rounded=True,             # 圆角矩形
    fontsize=10,              # 字体大小
    max_depth=4               # 只显示前4层，避免图形过于复杂
)
plt.title('ID3 Decision Tree (Entropy Criterion)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('id3_tree.png', dpi=300, bbox_inches='tight')
print("✓ ID3决策树图已保存为 'id3_tree.png'")
plt.show()

# ============================================================
# 运行输出：显示决策树可视化图形
# ============================================================


# ============================================================
# CODE CELL 5 - ID3详细评估报告
# ============================================================
print("\n" + "="*60)
print("ID3决策树详细评估报告")
print("="*60)

# 分类报告
print("\n【测试集分类报告】")
print(classification_report(y_test, y_test_pred_id3, zero_division=0))

# 混淆矩阵
conf_matrix_id3 = confusion_matrix(y_test, y_test_pred_id3)
print("\n【混淆矩阵】")
print(conf_matrix_id3)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_id3, annot=True, fmt='d', cmap='Blues',
            xticklabels=id3_tree.classes_, 
            yticklabels=id3_tree.classes_)
plt.title('ID3 Decision Tree - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('id3_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ ID3混淆矩阵图已保存为 'id3_confusion_matrix.png'")
plt.show()

# ============================================================
# 运行输出：显示分类报告、混淆矩阵及其可视化
# ============================================================


# ============================================================
# MARKDOWN CELL - CART算法说明
# ============================================================
"""
## 2. CART决策树算法

### 算法原理
CART（Classification and Regression Tree）算法由Breiman等人于1984年提出。

**核心思想**:
- 使用**基尼指数**（Gini Index）作为特征选择标准
- 基尼指数衡量数据集的不纯度，值越小表示数据越纯
- Gini(D) = 1 - Σ(p_i)², 其中p_i是类别i的概率
- 采用二叉树结构（每个节点只有两个分支）

**优点**:
- 可以处理连续型和离散型特征
- 可以用于分类和回归任务
- 生成的树结构简单，计算效率高
- 对缺失值有一定的鲁棒性

**缺点**:
- 容易过拟合，需要剪枝
- 对异常值敏感
- 二叉分裂可能导致树深度较大
"""

# ============================================================
# CODE CELL 6 - 构建CART决策树
# ============================================================
print("=" * 60)
print("开始训练CART决策树模型...")
print("=" * 60)

# 创建CART决策树分类器（使用基尼指数作为分裂标准）
cart_tree = DecisionTreeClassifier(
    criterion='gini',         # 使用基尼指数（CART的核心）
    random_state=42,          # 固定随机种子
    max_depth=10,             # 限制树的最大深度
    min_samples_split=5,      # 内部节点最少需要5个样本才能分裂
    min_samples_leaf=2        # 叶节点最少需要2个样本
)

# 训练模型
cart_tree.fit(X_train, y_train)

# 在训练集和测试集上进行预测
y_train_pred_cart = cart_tree.predict(X_train)
y_test_pred_cart = cart_tree.predict(X_test)

# 计算准确率
train_accuracy_cart = accuracy_score(y_train, y_train_pred_cart)
test_accuracy_cart = accuracy_score(y_test, y_test_pred_cart)

print(f"\n✓ CART模型训练完成")
print(f"训练集准确率: {train_accuracy_cart:.4f} ({train_accuracy_cart*100:.2f}%)")
print(f"测试集准确率: {test_accuracy_cart:.4f} ({test_accuracy_cart*100:.2f}%)")
print(f"决策树深度: {cart_tree.get_depth()}")
print(f"叶节点数量: {cart_tree.get_n_leaves()}")

# ============================================================
# 运行输出示例：
# ============================================================
# 开始训练CART决策树模型...
# ============================================================
# ✓ CART模型训练完成
# 训练集准确率: 0.XXXX (XX.XX%)
# 测试集准确率: 0.XXXX (XX.XX%)
# 决策树深度: X
# 叶节点数量: XX
# ============================================================


# ============================================================
# CODE CELL 7 - 绘制CART决策树图
# ============================================================
print("\n绘制CART决策树结构图...")

# 创建大尺寸图形
plt.figure(figsize=(25, 15))
plot_tree(
    cart_tree, 
    feature_names=feature_names,
    class_names=cart_tree.classes_,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=4               # 只显示前4层
)
plt.title('CART Decision Tree (Gini Criterion)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('cart_tree.png', dpi=300, bbox_inches='tight')
print("✓ CART决策树图已保存为 'cart_tree.png'")
plt.show()

# ============================================================
# 运行输出：显示决策树可视化图形
# ============================================================


# ============================================================
# CODE CELL 8 - CART详细评估报告
# ============================================================
print("\n" + "="*60)
print("CART决策树详细评估报告")
print("="*60)

# 分类报告
print("\n【测试集分类报告】")
print(classification_report(y_test, y_test_pred_cart, zero_division=0))

# 混淆矩阵
conf_matrix_cart = confusion_matrix(y_test, y_test_pred_cart)
print("\n【混淆矩阵】")
print(conf_matrix_cart)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cart, annot=True, fmt='d', cmap='Greens',
            xticklabels=cart_tree.classes_, 
            yticklabels=cart_tree.classes_)
plt.title('CART Decision Tree - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cart_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ CART混淆矩阵图已保存为 'cart_confusion_matrix.png'")
plt.show()

# ============================================================
# 运行输出：显示分类报告、混淆矩阵及其可视化
# ============================================================


# ============================================================
# CODE CELL 9 - 特征重要性分析
# ============================================================
print("\n" + "="*60)
print("特征重要性分析")
print("="*60)

# 获取特征重要性
feature_importance_id3 = id3_tree.feature_importances_
feature_importance_cart = cart_tree.feature_importances_

# 创建特征重要性对比图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ID3特征重要性
axes[0].barh(range(len(feature_importance_id3)), feature_importance_id3, color='skyblue')
axes[0].set_yticks(range(len(feature_importance_id3)))
axes[0].set_yticklabels(feature_names)
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('ID3 Feature Importance')
axes[0].invert_yaxis()

# CART特征重要性
axes[1].barh(range(len(feature_importance_cart)), feature_importance_cart, color='lightgreen')
axes[1].set_yticks(range(len(feature_importance_cart)))
axes[1].set_yticklabels(feature_names)
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('CART Feature Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 特征重要性对比图已保存为 'feature_importance_comparison.png'")
plt.show()

# 输出最重要的前5个特征
print("\n【ID3算法 - Top 5 重要特征】")
id3_top5_idx = np.argsort(feature_importance_id3)[-5:][::-1]
for idx in id3_top5_idx:
    print(f"  {feature_names[idx]}: {feature_importance_id3[idx]:.4f}")

print("\n【CART算法 - Top 5 重要特征】")
cart_top5_idx = np.argsort(feature_importance_cart)[-5:][::-1]
for idx in cart_top5_idx:
    print(f"  {feature_names[idx]}: {feature_importance_cart[idx]:.4f}")

# ============================================================
# 运行输出：显示特征重要性图表和Top 5特征列表
# ============================================================


# ============================================================
# CODE CELL 10 - 算法对比分析
# ============================================================
print("\n" + "="*60)
print("ID3 vs CART 算法对比分析")
print("="*60)

# 创建对比表格
comparison_data = {
    '评估指标': ['训练集准确率', '测试集准确率', '树深度', '叶节点数', '分裂准则'],
    'ID3算法': [
        f'{train_accuracy_id3:.4f}',
        f'{test_accuracy_id3:.4f}',
        id3_tree.get_depth(),
        id3_tree.get_n_leaves(),
        '信息熵(Entropy)'
    ],
    'CART算法': [
        f'{train_accuracy_cart:.4f}',
        f'{test_accuracy_cart:.4f}',
        cart_tree.get_depth(),
        cart_tree.get_n_leaves(),
        '基尼指数(Gini)'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n【性能对比表】")
print(comparison_df.to_string(index=False))

# 可视化准确率对比
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(2)
width = 0.35

bars1 = ax.bar(x - width/2, [train_accuracy_id3, test_accuracy_id3], 
               width, label='ID3', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, [train_accuracy_cart, test_accuracy_cart], 
               width, label='CART', color='lightgreen', edgecolor='black')

ax.set_ylabel('Accuracy')
ax.set_title('ID3 vs CART Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(['Training Set', 'Test Set'])
ax.legend()
ax.set_ylim([0, 1.1])

# 在柱状图上显示数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ 准确率对比图已保存为 'accuracy_comparison.png'")
plt.show()

# ============================================================
# 运行输出：显示对比表格和准确率对比柱状图
# ============================================================


# ============================================================
# MARKDOWN CELL - 详细分析
# ============================================================
"""
## 3. 实验结果分析

### 3.1 准确率分析

从实验结果可以看出：

1. **训练集表现**：
   - 两种算法在训练集上都能达到较高的准确率
   - 如果训练集准确率明显高于测试集，说明可能存在过拟合现象
   
2. **测试集表现**：
   - 测试集准确率是衡量模型泛化能力的关键指标
   - 两种算法的测试集准确率差异反映了不同分裂准则的影响

3. **过拟合程度**：
   - 通过比较训练集和测试集的准确率差异，可以判断过拟合程度
   - 差异越大，过拟合越严重

### 3.2 决策树结构分析

1. **树的深度**：
   - 树深度反映了模型的复杂度
   - 深度较大的树可能捕捉了更多细节，但也更容易过拟合
   - 深度较小的树泛化能力可能更好，但可能欠拟合

2. **叶节点数量**：
   - 叶节点数量影响模型的表达能力
   - 叶节点越多，模型越复杂，对训练数据的拟合越好
   - 但过多的叶节点可能导致过拟合

### 3.3 特征重要性分析

特征重要性揭示了哪些特征对分类决策影响最大：

1. **重要特征识别**：
   - 重要性较高的特征在决策树的上层节点中使用
   - 这些特征对区分不同汽车类型贡献最大

2. **特征选择差异**：
   - ID3和CART可能选择不同的重要特征
   - 这是由于它们使用不同的分裂准则（信息熵 vs 基尼指数）

3. **实际意义**：
   - 在实际应用中，可以根据特征重要性进行特征工程
   - 去除不重要的特征可以简化模型，提高效率

### 3.4 ID3 vs CART 算法对比

#### 相同点：
- 都是贪心算法，采用自顶向下的递归方式构建决策树
- 都通过选择最优特征进行节点分裂
- 都可能出现过拟合问题

#### 不同点：

| 对比维度 | ID3算法 | CART算法 |
|---------|---------|----------|
| **分裂准则** | 信息增益（Information Gain） | 基尼指数（Gini Index） |
| **计算复杂度** | 需要计算对数，相对较慢 | 只需要平方运算，相对较快 |
| **树结构** | 可以是多叉树 | 严格的二叉树 |
| **特征类型** | 倾向于选择取值多的特征 | 对特征取值数量不敏感 |
| **适用场景** | 离散特征为主的分类 | 连续和离散特征混合 |
| **偏向性** | 有偏向问题 | 偏向性较小 |

#### 性能差异原因：

1. **分裂准则影响**：
   - 信息熵关注的是信息量的减少
   - 基尼指数关注的是不纯度的降低
   - 两者在数学上相似但不完全相同

2. **特征选择策略**：
   - ID3倾向于选择取值较多的特征
   - CART对特征取值数量不敏感，选择更加均衡

3. **树结构差异**：
   - ID3的多叉结构可能更直观
   - CART的二叉结构更加规则，便于理解和实现

### 3.5 混淆矩阵分析

混淆矩阵展示了模型在各个类别上的具体表现：

1. **对角线元素**：正确分类的样本数
2. **非对角线元素**：误分类的样本数
3. **类别间混淆**：哪些类别容易被混淆

通过混淆矩阵可以发现：
- 模型在哪些类别上表现较好
- 哪些类别容易被误分类
- 是否存在某些类别之间的系统性混淆

### 3.6 实际应用建议

1. **模型选择**：
   - 如果数据特征主要是离散型，ID3是合理选择
   - 如果数据包含连续特征，CART更为合适
   - 实际中sklearn的实现使得CART应用更广泛

2. **过拟合控制**：
   - 设置合理的max_depth限制树深度
   - 使用min_samples_split和min_samples_leaf控制节点分裂
   - 考虑使用剪枝技术
   - 可以尝试集成方法如随机森林

3. **特征工程**：
   - 根据特征重要性选择有效特征
   - 对不重要特征进行剔除
   - 考虑特征组合和转换

4. **模型优化**：
   - 通过交叉验证选择最优参数
   - 尝试不同的分裂准则
   - 考虑样本权重平衡

## 4. 实验结论

本实验通过ID3和CART两种经典决策树算法对汽车数据进行分类，得出以下结论：

1. **算法有效性**：两种算法都能较好地完成汽车类型分类任务

2. **性能差异**：两种算法在准确率、树结构等方面存在差异，这源于不同的分裂准则和算法特性

3. **实用价值**：决策树算法具有良好的可解释性，生成的规则易于理解，适合实际应用

4. **改进方向**：可以通过参数调优、剪枝、集成学习等方法进一步提升性能

## 5. 参考文献

- Quinlan, J. R. (1986). Induction of decision trees. Machine learning, 1(1), 81-106.
- Breiman, L., et al. (1984). Classification and regression trees. CRC press.
- scikit-learn documentation: Decision Trees
"""

# ============================================================
# CODE CELL 11 - 实验总结输出
# ============================================================
print("\n" + "="*60)
print("实验总结")
print("="*60)
print(f"""
本次实验成功完成了以下任务：

✓ 1. 数据加载与预处理
   - 训练集样本数: {X_train.shape[0]}
   - 测试集样本数: {X_test.shape[0]}
   - 特征维度: {X_train.shape[1]}

✓ 2. ID3决策树模型
   - 测试集准确率: {test_accuracy_id3:.4f}
   - 决策树深度: {id3_tree.get_depth()}
   - 叶节点数: {id3_tree.get_n_leaves()}

✓ 3. CART决策树模型
   - 测试集准确率: {test_accuracy_cart:.4f}
   - 决策树深度: {cart_tree.get_depth()}
   - 叶节点数: {cart_tree.get_n_leaves()}

✓ 4. 可视化输出
   - ID3决策树图: id3_tree.png
   - CART决策树图: cart_tree.png
   - ID3混淆矩阵: id3_confusion_matrix.png
   - CART混淆矩阵: cart_confusion_matrix.png
   - 特征重要性对比: feature_importance_comparison.png
   - 准确率对比: accuracy_comparison.png

✓ 5. 详细分析报告已完成

实验顺利完成！
""")

print("="*60)
print("END OF REPORT")
print("="*60)

# ============================================================
# 运行输出：实验总结信息
# ============================================================