# ============================================================
# 【MARKDOWN】实验标题
# ============================================================
# # 皮马印第安人糖尿病数据集朴素贝叶斯分类实验报告
# 
# ## 实验目的
# 1. 使用皮马印第安人糖尿病数据集进行分类任务
# 2. 对比三种朴素贝叶斯算法的性能：
#    - 伯努利朴素贝叶斯（BernoulliNB）
#    - 高斯朴素贝叶斯（GaussianNB）
#    - 多项式朴素贝叶斯（MultinomialNB）
# 3. 分析不同算法的适用场景和性能差异
# 
# ## 实验环境
# - Python 3.x
# - 主要库：pandas, numpy, scikit-learn, matplotlib, seaborn
# ============================================================


# ============================================================
# 【CODE】导入必要的库
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体显示（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("所有库导入成功！")
# 【运行输出】显示"所有库导入成功！"


# ============================================================
# 【MARKDOWN】数据加载与探索
# ============================================================
# ## 1. 数据加载与探索性分析
# 
# ### 1.1 数据集介绍
# 皮马印第安人糖尿病数据集包含768个样本，9个特征：
# - Pregnancies: 怀孕次数
# - Glucose: 口服葡萄糖耐量试验中2小时的血浆葡萄糖浓度
# - BloodPressure: 舒张压（mm Hg）
# - SkinThickness: 三头肌皮褶厚度（mm）
# - Insulin: 2小时血清胰岛素（mu U/ml）
# - BMI: 身体质量指数（体重kg/身高m²）
# - DiabetesPedigreeFunction: 糖尿病家族影响函数
# - Age: 年龄
# - Outcome: 目标变量（0=无糖尿病，1=有糖尿病）
# ============================================================


# ============================================================
# 【CODE】加载数据
# ============================================================
# 注意：请确保数据文件在当前目录，或修改为正确的路径
# 如果从Kaggle下载，文件名通常为 'diabetes.csv'
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# 定义列名
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# 加载数据
try:
    df = pd.read_csv(url, names=column_names)
    print("数据加载成功！")
except:
    # 如果在线加载失败，尝试本地加载
    df = pd.read_csv('diabetes.csv')
    print("从本地文件加载数据成功！")

# 【运行输出】显示数据加载成功信息


# ============================================================
# 【CODE】数据基本信息查看
# ============================================================
print("\n" + "="*60)
print("数据集基本信息")
print("="*60)
print(f"数据集形状: {df.shape}")
print(f"总样本数: {df.shape[0]}")
print(f"特征数量: {df.shape[1] - 1}")  # 减去目标变量
print("\n前5行数据:")
print(df.head())
# 【运行输出】显示数据集形状和前5行数据

print("\n" + "="*60)
print("数据类型和缺失值信息")
print("="*60)
print(df.info())
# 【运行输出】显示数据类型和缺失值信息

print("\n" + "="*60)
print("数据统计描述")
print("="*60)
print(df.describe())
# 【运行输出】显示数据的统计描述信息


# ============================================================
# 【CODE】目标变量分布分析
# ============================================================
print("\n" + "="*60)
print("目标变量分布")
print("="*60)
print(df['Outcome'].value_counts())
print("\n类别比例:")
print(df['Outcome'].value_counts(normalize=True))
# 【运行输出】显示目标变量的分布情况

# 可视化目标变量分布
plt.figure(figsize=(8, 5))
df['Outcome'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('目标变量分布（0=无糖尿病, 1=有糖尿病）', fontsize=14)
plt.xlabel('类别', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
# 【运行输出】显示目标变量分布的柱状图


# ============================================================
# 【CODE】检查异常值和零值
# ============================================================
print("\n" + "="*60)
print("异常值检查")
print("="*60)
# 某些特征（如Glucose, BloodPressure等）不应该为0
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("各特征中零值的数量:")
for col in zero_columns:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} ({zero_count/len(df)*100:.2f}%)")
# 【运行输出】显示各特征中零值的统计


# ============================================================
# 【CODE】数据预处理
# ============================================================
print("\n" + "="*60)
print("数据预处理")
print("="*60)

# 将异常的零值替换为该列的中位数（排除零值后的中位数）
df_processed = df.copy()
for col in zero_columns:
    # 计算非零值的中位数
    median_value = df_processed[df_processed[col] != 0][col].median()
    # 替换零值
    df_processed.loc[df_processed[col] == 0, col] = median_value
    print(f"{col} 的零值已替换为中位数: {median_value:.2f}")

print("\n预处理完成！")
# 【运行输出】显示各特征零值的替换情况


# ============================================================
# 【CODE】特征和目标变量分离
# ============================================================
# 分离特征和目标变量
X = df_processed.drop('Outcome', axis=1)
y = df_processed['Outcome']

print("\n" + "="*60)
print("特征和目标变量分离完成")
print("="*60)
print(f"特征矩阵 X 的形状: {X.shape}")
print(f"目标变量 y 的形状: {y.shape}")
# 【运行输出】显示特征矩阵和目标变量的形状


# ============================================================
# 【CODE】数据集划分
# ============================================================
# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*60)
print("数据集划分完成")
print("="*60)
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"\n训练集目标变量分布:")
print(y_train.value_counts())
print(f"\n测试集目标变量分布:")
print(y_test.value_counts())
# 【运行输出】显示训练集和测试集的划分情况


# ============================================================
# 【MARKDOWN】朴素贝叶斯算法理论
# ============================================================
# ## 2. 朴素贝叶斯算法理论基础
# 
# ### 2.1 基本原理
# 朴素贝叶斯算法基于贝叶斯定理，假设特征之间相互独立。
# 贝叶斯定理：P(Y|X) = P(X|Y) * P(Y) / P(X)
# 
# ### 2.2 三种朴素贝叶斯算法
# 
# #### 2.2.1 高斯朴素贝叶斯（GaussianNB）
# - **适用场景**: 连续型特征，假设特征服从高斯（正态）分布
# - **特点**: 最常用于连续数值特征，如本数据集的血压、BMI等
# - **概率计算**: 使用高斯概率密度函数
# 
# #### 2.2.2 多项式朴素贝叶斯（MultinomialNB）
# - **适用场景**: 离散型特征，通常用于文本分类（词频统计）
# - **特点**: 特征值必须是非负的（通常是计数）
# - **本实验说明**: 需要对数据进行缩放到非负范围
# 
# #### 2.2.3 伯努利朴素贝叶斯（BernoulliNB）
# - **适用场景**: 二值特征（0/1）
# - **特点**: 假设特征是布尔值
# - **本实验说明**: 需要将连续特征二值化
# ============================================================


# ============================================================
# 【CODE】方法1：高斯朴素贝叶斯（GaussianNB）
# ============================================================
print("\n" + "="*60)
print("方法1：高斯朴素贝叶斯（GaussianNB）")
print("="*60)

# 高斯朴素贝叶斯适用于连续型数据，假设特征服从正态分布
# 对数据进行标准化（可选，但有助于提高性能）
scaler_gaussian = StandardScaler()
X_train_gaussian = scaler_gaussian.fit_transform(X_train)
X_test_gaussian = scaler_gaussian.transform(X_test)

# 创建并训练高斯朴素贝叶斯模型
gnb = GaussianNB()
gnb.fit(X_train_gaussian, y_train)

# 预测
y_pred_gnb = gnb.predict(X_test_gaussian)
y_pred_proba_gnb = gnb.predict_proba(X_test_gaussian)[:, 1]

# 计算评估指标
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
precision_gnb = precision_score(y_test, y_pred_gnb)
recall_gnb = recall_score(y_test, y_pred_gnb)
f1_gnb = f1_score(y_test, y_pred_gnb)

print("\n高斯朴素贝叶斯模型训练完成！")
print("\n模型评估指标:")
print(f"准确率 (Accuracy):  {accuracy_gnb:.4f}")
print(f"精确率 (Precision): {precision_gnb:.4f}")
print(f"召回率 (Recall):    {recall_gnb:.4f}")
print(f"F1分数 (F1-Score):  {f1_gnb:.4f}")
# 【运行输出】显示高斯朴素贝叶斯的评估指标

# 混淆矩阵
print("\n混淆矩阵:")
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
print(cm_gnb)
# 【运行输出】显示混淆矩阵

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred_gnb, target_names=['无糖尿病', '有糖尿病']))
# 【运行输出】显示详细的分类报告

# 交叉验证评估
cv_scores_gnb = cross_val_score(gnb, X_train_gaussian, y_train, cv=5, scoring='accuracy')
print(f"\n5折交叉验证准确率: {cv_scores_gnb.mean():.4f} (+/- {cv_scores_gnb.std() * 2:.4f})")
# 【运行输出】显示交叉验证结果


# ============================================================
# 【CODE】方法2：多项式朴素贝叶斯（MultinomialNB）
# ============================================================
print("\n" + "="*60)
print("方法2：多项式朴素贝叶斯（MultinomialNB）")
print("="*60)

# 多项式朴素贝叶斯要求特征值为非负
# 使用MinMaxScaler将数据缩放到[0,1]范围
scaler_multinomial = MinMaxScaler()
X_train_multinomial = scaler_multinomial.fit_transform(X_train)
X_test_multinomial = scaler_multinomial.transform(X_test)

# 创建并训练多项式朴素贝叶斯模型
mnb = MultinomialNB()
mnb.fit(X_train_multinomial, y_train)

# 预测
y_pred_mnb = mnb.predict(X_test_multinomial)
y_pred_proba_mnb = mnb.predict_proba(X_test_multinomial)[:, 1]

# 计算评估指标
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
precision_mnb = precision_score(y_test, y_pred_mnb)
recall_mnb = recall_score(y_test, y_pred_mnb)
f1_mnb = f1_score(y_test, y_pred_mnb)

print("\n多项式朴素贝叶斯模型训练完成！")
print("\n模型评估指标:")
print(f"准确率 (Accuracy):  {accuracy_mnb:.4f}")
print(f"精确率 (Precision): {precision_mnb:.4f}")
print(f"召回率 (Recall):    {recall_mnb:.4f}")
print(f"F1分数 (F1-Score):  {f1_mnb:.4f}")
# 【运行输出】显示多项式朴素贝叶斯的评估指标

# 混淆矩阵
print("\n混淆矩阵:")
cm_mnb = confusion_matrix(y_test, y_pred_mnb)
print(cm_mnb)
# 【运行输出】显示混淆矩阵

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred_mnb, target_names=['无糖尿病', '有糖尿病']))
# 【运行输出】显示详细的分类报告

# 交叉验证评估
cv_scores_mnb = cross_val_score(mnb, X_train_multinomial, y_train, cv=5, scoring='accuracy')
print(f"\n5折交叉验证准确率: {cv_scores_mnb.mean():.4f} (+/- {cv_scores_mnb.std() * 2:.4f})")
# 【运行输出】显示交叉验证结果


# ============================================================
# 【CODE】方法3：伯努利朴素贝叶斯（BernoulliNB）
# ============================================================
print("\n" + "="*60)
print("方法3：伯努利朴素贝叶斯（BernoulliNB）")
print("="*60)

# 伯努利朴素贝叶斯假设特征是二值的（0或1）
# 我们需要将连续特征二值化，使用中位数作为阈值
X_train_bernoulli = (X_train > X_train.median()).astype(int)
X_test_bernoulli = (X_test > X_train.median()).astype(int)

print("特征二值化完成（使用训练集中位数作为阈值）")

# 创建并训练伯努利朴素贝叶斯模型
bnb = BernoulliNB()
bnb.fit(X_train_bernoulli, y_train)

# 预测
y_pred_bnb = bnb.predict(X_test_bernoulli)
y_pred_proba_bnb = bnb.predict_proba(X_test_bernoulli)[:, 1]

# 计算评估指标
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
precision_bnb = precision_score(y_test, y_pred_bnb)
recall_bnb = recall_score(y_test, y_pred_bnb)
f1_bnb = f1_score(y_test, y_pred_bnb)

print("\n伯努利朴素贝叶斯模型训练完成！")
print("\n模型评估指标:")
print(f"准确率 (Accuracy):  {accuracy_bnb:.4f}")
print(f"精确率 (Precision): {precision_bnb:.4f}")
print(f"召回率 (Recall):    {recall_bnb:.4f}")
print(f"F1分数 (F1-Score):  {f1_bnb:.4f}")
# 【运行输出】显示伯努利朴素贝叶斯的评估指标

# 混淆矩阵
print("\n混淆矩阵:")
cm_bnb = confusion_matrix(y_test, y_pred_bnb)
print(cm_bnb)
# 【运行输出】显示混淆矩阵

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred_bnb, target_names=['无糖尿病', '有糖尿病']))
# 【运行输出】显示详细的分类报告

# 交叉验证评估
cv_scores_bnb = cross_val_score(bnb, X_train_bernoulli, y_train, cv=5, scoring='accuracy')
print(f"\n5折交叉验证准确率: {cv_scores_bnb.mean():.4f} (+/- {cv_scores_bnb.std() * 2:.4f})")
# 【运行输出】显示交叉验证结果


# ============================================================
# 【CODE】综合对比分析
# ============================================================
print("\n" + "="*60)
print("三种朴素贝叶斯算法综合对比")
print("="*60)

# 创建对比表格
comparison_df = pd.DataFrame({
    '算法': ['GaussianNB', 'MultinomialNB', 'BernoulliNB'],
    '准确率': [accuracy_gnb, accuracy_mnb, accuracy_bnb],
    '精确率': [precision_gnb, precision_mnb, precision_bnb],
    '召回率': [recall_gnb, recall_mnb, recall_bnb],
    'F1分数': [f1_gnb, f1_mnb, f1_bnb],
    '交叉验证准确率': [cv_scores_gnb.mean(), cv_scores_mnb.mean(), cv_scores_bnb.mean()]
})

print("\n详细指标对比表:")
print(comparison_df.to_string(index=False))
# 【运行输出】显示三种算法的详细指标对比表

# 找出最佳模型
best_model_idx = comparison_df['准确率'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, '算法']
best_accuracy = comparison_df.loc[best_model_idx, '准确率']

print(f"\n最佳模型: {best_model_name}")
print(f"最高准确率: {best_accuracy:.4f}")
# 【运行输出】显示最佳模型


# ============================================================
# 【CODE】可视化对比分析
# ============================================================
print("\n生成可视化对比图...")

# 图1：性能指标对比柱状图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['准确率', '精确率', '召回率', 'F1分数']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    values = comparison_df[metric].values
    bars = ax.bar(comparison_df['算法'], values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric}对比', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
# 【运行输出】显示四个性能指标的对比柱状图

# 图2：混淆矩阵可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

confusion_matrices = [cm_gnb, cm_mnb, cm_bnb]
titles = ['GaussianNB', 'MultinomialNB', 'BernoulliNB']

for idx, (cm, title) in enumerate(zip(confusion_matrices, titles)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['预测:无', '预测:有'],
                yticklabels=['实际:无', '实际:有'])
    axes[idx].set_title(f'{title} 混淆矩阵', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('实际类别', fontsize=10)
    axes[idx].set_xlabel('预测类别', fontsize=10)

plt.tight_layout()
plt.show()
# 【运行输出】显示三种算法的混淆矩阵热力图

# 图3：ROC曲线对比
plt.figure(figsize=(10, 8))

# 计算ROC曲线
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred_proba_gnb)
fpr_mnb, tpr_mnb, _ = roc_curve(y_test, y_pred_proba_mnb)
fpr_bnb, tpr_bnb, _ = roc_curve(y_test, y_pred_proba_bnb)

# 计算AUC值
auc_gnb = auc(fpr_gnb, tpr_gnb)
auc_mnb = auc(fpr_mnb, tpr_mnb)
auc_bnb = auc(fpr_bnb, tpr_bnb)

# 绘制ROC曲线
plt.plot(fpr_gnb, tpr_gnb, color='#3498db', lw=2, 
         label=f'GaussianNB (AUC = {auc_gnb:.4f})')
plt.plot(fpr_mnb, tpr_mnb, color='#e74c3c', lw=2, 
         label=f'MultinomialNB (AUC = {auc_mnb:.4f})')
plt.plot(fpr_bnb, tpr_bnb, color='#2ecc71', lw=2, 
         label=f'BernoulliNB (AUC = {auc_bnb:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
plt.title('ROC曲线对比', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# 【运行输出】显示三种算法的ROC曲线对比图

print("\n所有可视化图表生成完成！")


# ============================================================
# 【MARKDOWN】实验结果分析
# ============================================================
# ## 3. 实验结果详细分析
# 
# ### 3.1 性能指标解释
# 
# #### 3.1.1 准确率 (Accuracy)
# - **定义**: 正确预测的样本占总样本的比例
# - **计算公式**: (TP + TN) / (TP + TN + FP + FN)
# - **优