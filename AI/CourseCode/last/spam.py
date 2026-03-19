# -*- coding: utf-8 -*-
"""
垃圾邮件过滤系统
使用Scikit-Learn库对SMS垃圾信息进行分类
数据集：UCI机器学习仓库中的SMS Spam Collection数据集
"""

# ============================================================================
# 第一部分：导入必要的库
# ============================================================================

import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns  # 用于高级数据可视化
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # 用于文本特征提取
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯分类器
from sklearn.svm import SVC  # 支持向量机分类器
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器
from sklearn.metrics import (
    accuracy_score,  # 准确率
    precision_score,  # 精确率
    recall_score,  # 召回率
    f1_score,  # F1分数
    confusion_matrix,  # 混淆矩阵
    classification_report,  # 分类报告
    roc_curve,  # ROC曲线
    auc  # AUC值
)
import re  # 用于正则表达式文本处理
import warnings  # 用于控制警告信息
warnings.filterwarnings('ignore')  # 忽略警告信息，使输出更清晰

# 设置中文字体支持（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子，确保结果可重复
np.random.seed(42)

print("=" * 80)
print("垃圾邮件过滤系统 - 基于Scikit-Learn")
print("=" * 80)
print()


# ============================================================================
# 第二部分：数据加载和初步探索
# ============================================================================

print("【步骤1】数据加载")
print("-" * 80)

# 读取CSV文件
# sep='\t' 表示使用制表符作为分隔符
# names 指定列名，因为原始数据没有表头
# encoding 指定编码格式，避免乱码
df = pd.read_csv('SMSSpamCollection.csv', sep='\t', names=['label', 'message'], encoding='utf-8')

print(f"数据集成功加载！")
print(f"数据集形状：{df.shape[0]} 行, {df.shape[1]} 列")
print()

# 显示前几条数据
print("前5条数据预览：")
print(df.head())
print()

# 显示数据基本信息
print("数据集基本信息：")
print(df.info())
print()

# 检查缺失值
print("缺失值统计：")
print(df.isnull().sum())
print()

# 检查重复值
duplicate_count = df.duplicated().sum()
print(f"重复数据条数：{duplicate_count}")
if duplicate_count > 0:
    # 删除重复数据
    df = df.drop_duplicates()
    print(f"删除重复数据后，数据集形状：{df.shape}")
print()


# ============================================================================
# 第三部分：数据探索性分析（EDA）
# ============================================================================

print("\n" + "=" * 80)
print("【步骤2】探索性数据分析（EDA）")
print("-" * 80)

# 统计标签分布
label_counts = df['label'].value_counts()
print("标签分布统计：")
print(label_counts)
print()
print(f"正常邮件（ham）占比：{label_counts['ham'] / len(df) * 100:.2f}%")
print(f"垃圾邮件（spam）占比：{label_counts['spam'] / len(df) * 100:.2f}%")
print()

# 可视化标签分布
plt.figure(figsize=(10, 5))

# 第一个子图：柱状图
plt.subplot(1, 2, 1)
label_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('SMS Type Distribution (Bar Chart)', fontsize=12, fontweight='bold')
plt.xlabel('Message Type', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

# 第二个子图：饼图
plt.subplot(1, 2, 2)
colors = ['#2ecc71', '#e74c3c']
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 10})
plt.title('SMS Type Distribution (Pie Chart)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
print("标签分布图已保存为：label_distribution.png")
plt.close()

# 计算消息长度
df['message_length'] = df['message'].apply(len)  # 计算每条消息的字符数
df['word_count'] = df['message'].apply(lambda x: len(x.split()))  # 计算每条消息的单词数

# 按类别统计消息长度
print("\n消息长度统计（按类别）：")
print(df.groupby('label')[['message_length', 'word_count']].describe())
print()

# 可视化消息长度分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 字符长度箱线图
axes[0, 0].boxplot([df[df['label'] == 'ham']['message_length'],
                     df[df['label'] == 'spam']['message_length']],
                    labels=['Ham', 'Spam'])
axes[0, 0].set_title('Message Length Distribution (Characters)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Number of Characters', fontsize=10)
axes[0, 0].grid(axis='y', alpha=0.3)

# 单词数箱线图
axes[0, 1].boxplot([df[df['label'] == 'ham']['word_count'],
                     df[df['label'] == 'spam']['word_count']],
                    labels=['Ham', 'Spam'])
axes[0, 1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Number of Words', fontsize=10)
axes[0, 1].grid(axis='y', alpha=0.3)

# 字符长度直方图
axes[1, 0].hist(df[df['label'] == 'ham']['message_length'], bins=50, alpha=0.7, 
                label='Ham', color='#2ecc71', edgecolor='black')
axes[1, 0].hist(df[df['label'] == 'spam']['message_length'], bins=50, alpha=0.7, 
                label='Spam', color='#e74c3c', edgecolor='black')
axes[1, 0].set_title('Message Length Histogram', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Number of Characters', fontsize=10)
axes[1, 0].set_ylabel('Frequency', fontsize=10)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# 单词数直方图
axes[1, 1].hist(df[df['label'] == 'ham']['word_count'], bins=50, alpha=0.7, 
                label='Ham', color='#2ecc71', edgecolor='black')
axes[1, 1].hist(df[df['label'] == 'spam']['word_count'], bins=50, alpha=0.7, 
                label='Spam', color='#e74c3c', edgecolor='black')
axes[1, 1].set_title('Word Count Histogram', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Words', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('message_length_analysis.png', dpi=300, bbox_inches='tight')
print("消息长度分析图已保存为：message_length_analysis.png")
plt.close()


# ============================================================================
# 第四部分：文本预处理
# ============================================================================

print("\n" + "=" * 80)
print("【步骤3】文本预处理")
print("-" * 80)

def preprocess_text(text):
    """
    文本预处理函数
    
    参数：
        text: 输入的文本字符串
    
    返回：
        processed_text: 处理后的文本字符串
    
    处理步骤：
        1. 转换为小写
        2. 去除特殊字符和数字（保留字母和空格）
        3. 去除多余的空格
    """
    # 转换为小写，统一文本格式
    text = text.lower()
    
    # 使用正则表达式去除非字母字符（保留空格）
    # [^a-z\s] 表示匹配所有非小写字母和非空格的字符
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 去除多余的空格，将多个连续空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

# 应用预处理函数到所有消息
print("正在进行文本预处理...")
df['processed_message'] = df['message'].apply(preprocess_text)

# 显示预处理前后的对比示例
print("\n预处理前后对比示例：")
for i in range(3):
    print(f"\n示例 {i+1}:")
    print(f"原始消息：{df['message'].iloc[i]}")
    print(f"处理后：{df['processed_message'].iloc[i]}")

print("\n文本预处理完成！")
print()


# ============================================================================
# 第五部分：特征工程 - 文本向量化
# ============================================================================

print("\n" + "=" * 80)
print("【步骤4】特征工程 - 文本向量化")
print("-" * 80)

# 准备特征和标签
X = df['processed_message']  # 特征：预处理后的文本
y = df['label']  # 标签：ham 或 spam

# 将标签转换为数值：ham=0, spam=1
# 这是因为机器学习算法需要数值型的标签
y_numeric = y.map({'ham': 0, 'spam': 1})

print(f"特征数量：{len(X)}")
print(f"标签分布：")
print(f"  - Ham (0): {(y_numeric == 0).sum()}")
print(f"  - Spam (1): {(y_numeric == 1).sum()}")
print()

# 划分训练集和测试集
# test_size=0.2 表示20%的数据作为测试集，80%作为训练集
# random_state=42 确保每次划分结果相同，便于结果复现
# stratify=y_numeric 确保训练集和测试集中各类别的比例与原始数据集相同
X_train, X_test, y_train, y_test = train_test_split(
    X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
)

print("数据集划分完成：")
print(f"训练集大小：{len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"测试集大小：{len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print()

# 方法1：使用CountVectorizer（词袋模型）
print("方法1：CountVectorizer（词袋模型）")
print("-" * 40)
# CountVectorizer将文本转换为词频矩阵
# max_features=3000 表示只保留出现频率最高的3000个词
# stop_words='english' 表示去除英文停用词（如the, is, at等常见但无意义的词）
cv = CountVectorizer(max_features=3000, stop_words='english')

# fit_transform：在训练集上学习词汇表并转换
X_train_cv = cv.fit_transform(X_train)
# transform：使用训练集学到的词汇表转换测试集
X_test_cv = cv.transform(X_test)

print(f"词汇表大小：{len(cv.vocabulary_)}")
print(f"训练集特征矩阵形状：{X_train_cv.shape}")
print(f"测试集特征矩阵形状：{X_test_cv.shape}")
print(f"特征矩阵类型：{type(X_train_cv)}")
print()

# 方法2：使用TfidfVectorizer（TF-IDF）
print("方法2：TfidfVectorizer（TF-IDF）")
print("-" * 40)
# TF-IDF (Term Frequency-Inverse Document Frequency) 考虑了词的重要性
# TF：词频，衡量词在文档中出现的频率
# IDF：逆文档频率，衡量词的稀有程度（越稀有的词权重越高）
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"词汇表大小：{len(tfidf.vocabulary_)}")
print(f"训练集特征矩阵形状：{X_train_tfidf.shape}")
print(f"测试集特征矩阵形状：{X_test_tfidf.shape}")
print()

# 显示一些高频词汇
print("CountVectorizer提取的部分高频词汇：")
feature_names_cv = cv.get_feature_names_out()
print(feature_names_cv[:20])
print()


# ============================================================================
# 第六部分：模型训练与评估
# ============================================================================

print("\n" + "=" * 80)
print("【步骤5】模型训练与评估")
print("-" * 80)

# 创建一个字典来存储所有模型的评估结果
results = []

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, vectorizer_name):
    """
    训练模型并评估性能
    
    参数：
        model: 机器学习模型对象
        model_name: 模型名称（用于显示）
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
        vectorizer_name: 向量化方法名称
    
    返回：
        result_dict: 包含模型评估指标的字典
    """
    print(f"\n正在训练：{model_name} + {vectorizer_name}")
    print("-" * 60)
    
    # 训练模型
    # fit方法用于在训练数据上训练模型
    model.fit(X_train, y_train)
    
    # 预测
    # predict方法用于对测试数据进行预测
    y_pred = model.predict(X_test)
    
    # 对于支持概率预测的模型，获取预测概率（用于ROC曲线）
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # SVM默认不支持概率预测，使用decision_function代替
        y_pred_proba = model.decision_function(X_test)
    
    # 计算评估指标
    # accuracy：准确率 = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy_score(y_test, y_pred)
    # precision：精确率 = TP / (TP + FP)，表示预测为正的样本中实际为正的比例
    precision = precision_score(y_test, y_pred)
    # recall：召回率 = TP / (TP + FN)，表示实际为正的样本中被预测为正的比例
    recall = recall_score(y_test, y_pred)
    # f1：F1分数 = 2 * (precision * recall) / (precision + recall)，精确率和召回率的调和平均
    f1 = f1_score(y_test, y_pred)
    
    # 计算混淆矩阵
    # 混淆矩阵显示分类结果的详细情况
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    
    # 打印评估结果
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print()
    print("混淆矩阵：")
    print(cm)
    print()
    print("分类报告：")
    # classification_report提供详细的分类指标，包括各类别的precision、recall、f1-score
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # 保存结果
    result_dict = {
        'Model': model_name,
        'Vectorizer': vectorizer_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion_Matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return result_dict


# 定义要测试的模型列表
models = [
    (MultinomialNB(), 'Naive Bayes'),  # 朴素贝叶斯：基于贝叶斯定理，适合文本分类
    (LogisticRegression(max_iter=1000, random_state=42), 'Logistic Regression'),  # 逻辑回归：线性分类模型
    (SVC(kernel='linear', random_state=42), 'SVM (Linear)'),  # 支持向量机（线性核）：寻找最优分类超平面
    (RandomForestClassifier(n_estimators=100, random_state=42), 'Random Forest')  # 随机森林：集成多个决策树
]

# 对每个模型分别使用CountVectorizer和TfidfVectorizer进行训练和评估
print("\n使用CountVectorizer特征：")
print("=" * 80)
for model, name in models:
    result = train_and_evaluate_model(model, name, X_train_cv, X_test_cv, y_train, y_test, 'CountVectorizer')
    results.append(result)

print("\n" + "=" * 80)
print("\n使用TfidfVectorizer特征：")
print("=" * 80)
for model, name in models:
    result = train_and_evaluate_model(model, name, X_train_tfidf, X_test_tfidf, y_train, y_test, 'TfidfVectorizer')
    results.append(result)


# ============================================================================
# 第七部分：结果对比与可视化
# ============================================================================

print("\n" + "=" * 80)
print("【步骤6】模型性能对比")
print("-" * 80)

# 创建结果对比DataFrame
results_df = pd.DataFrame([{
    'Model': r['Model'],
    'Vectorizer': r['Vectorizer'],
    'Accuracy': r['Accuracy'],
    'Precision': r['Precision'],
    'Recall': r['Recall'],
    'F1-Score': r['F1-Score']
} for r in results])

print("\n所有模型性能对比表：")
print(results_df.to_string(index=False))
print()

# 保存结果到CSV文件
results_df.to_csv('model_comparison_results.csv', index=False)
print("模型对比结果已保存为：model_comparison_results.csv")
print()

# 找出最佳模型
best_model_idx = results_df['F1-Score'].idxmax()
best_model_info = results_df.iloc[best_model_idx]
print(f"最佳模型：{best_model_info['Model']} + {best_model_info['Vectorizer']}")
print(f"F1-Score: {best_model_info['F1-Score']:.4f}")
print()

# 可视化模型性能对比
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    # 按模型分组数据
    cv_data = results_df[results_df['Vectorizer'] == 'CountVectorizer']
    tfidf_data = results_df[results_df['Vectorizer'] == 'TfidfVectorizer']
    
    x = np.arange(len(cv_data))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, cv_data[metric], width, label='CountVectorizer', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, tfidf_data[metric], width, label='TfidfVectorizer', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # 在柱状图上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=10, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cv_data['Model'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.85, 1.0])  # 设置y轴范围，使差异更明显

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("模型性能对比图已保存为：model_performance_comparison.png")
plt.close()


# ============================================================================
# 第八部分：混淆矩阵可视化
# ============================================================================

print("\n" + "=" * 80)
print("【步骤7】混淆矩阵可视化")
print("-" * 80)

# 为每个模型绘制混淆矩阵热图
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for idx, result in enumerate(results):
    ax = axes[idx]
    cm = result['Confusion_Matrix']
    
    # 使用seaborn绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Ham', 'Spam'],
               yticklabels=['Ham', 'Spam'],
               ax=ax, cbar=True, annot_kws={"size": 12})
    
    ax.set_title(f"{result['Model']}\n{result['Vectorizer']}", 
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=9)
    ax.set_ylabel('True Label', fontsize=9)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("混淆矩阵图已保存为：confusion_matrices.png")
plt.close()


# ============================================================================
# 第九部分：ROC曲线绘制
# ============================================================================

print("\n" + "=" * 80)
print("【步骤8】ROC曲线分析")
print("-" * 80)

# 绘制ROC曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# CountVectorizer的ROC曲线
for result in results[:4]:  # 前4个是CountVectorizer的结果
    fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, linewidth=2, 
            label=f"{result['Model']} (AUC = {roc_auc:.3f})")

ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax1.set_title('ROC Curves - CountVectorizer', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(alpha=0.3)

# TfidfVectorizer的ROC曲线
for result in results[4:]:  # 后4个是TfidfVectorizer的结果
    fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, linewidth=2,
            label=f"{result['Model']} (AUC = {roc_auc:.3f})")

ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax2.set_title('ROC Curves - TfidfVectorizer', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("ROC曲线图已保存为：roc_curves.png")
print()

# 计算并显示AUC值
print("各模型AUC值：")
for result in results:
    fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    print(f"{result['Model']} + {result['Vectorizer']}: AUC = {roc_auc:.4f}")

plt.close()


# ============================================================================
# 第十部分：实际应用示例
# ============================================================================

print("\n" + "=" * 80)
print("【步骤9】实际应用示例 - 使用最佳模型预测新消息")
print("-" * 80)

# 使用最佳模型（通常是Naive Bayes + TfidfVectorizer）进行演示
# 重新训练最佳模型
best_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_best = best_vectorizer.fit_transform(X_train)
X_test_best = best_vectorizer.transform(X_test)

best_model = MultinomialNB()
best_model.fit(X_train_best, y_train)

# 定义一些测试消息
test_messages = [
    "Congratulations! You've won a $1000 gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow at 12pm?",
    "FREE entry to win iPhone! Text WIN to 12345 now!",
    "Don't forget to submit your assignment by Friday.",
    "URGENT! Your account will be suspended. Click this link immediately.",
    "Can you pick up some milk on your way home?"
]

print("\n测试新消息预测：")
print("=" * 80)

# 对每条测试消息进行预测
for i, message in enumerate(test_messages, 1):
    # 预处理消息
    processed = preprocess_text(message)
    
    # 向量化
    message_vectorized = best_vectorizer.transform([processed])
    
    # 预测
    prediction = best_model.predict(message_vectorized)[0]
    prediction_proba = best_model.predict_proba(message_vectorized)[0]
    
    # 显示结果
    print(f"\n消息 {i}:")
    print(f"原文: {message}")
    print(f"预测结果: {'垃圾邮件 (SPAM)' if prediction == 1 else '正常邮件 (HAM)'}")
    print(f"置信度: Ham={prediction_proba[0]:.2%}, Spam={prediction_proba[1]:.2%}")
    print("-" * 80)


# ============================================================================
# 第十一部分：特征重要性分析
# ============================================================================

print("\n" + "=" * 80)
print("【步骤10】特征重要性分析")
print("-" * 80)

# 获取特征名称
feature_names = best_vectorizer.get_feature_names_out()

# 获取朴素贝叶斯模型的特征对数概率
# feature_log_prob_[0]是Ham类别的对数概率，feature_log_prob_[1]是Spam类别的对数概率
log_prob_ham = best_model.feature_log_prob_[0]
log_prob_spam = best_model.feature_log_prob_[1]

# 计算特征对分类的重要性（Spam概率 - Ham概率）
# 值越大，该词越倾向于出现在垃圾邮件中
feature_importance = log_prob_spam - log_prob_ham

# 创建特征重要性DataFrame
feature_importance_df = pd.DataFrame({
    'word': feature_names,
    'importance': feature_importance
})

# 排序并获取最能表明垃圾邮件的词
top_spam_words = feature_importance_df.nlargest(20, 'importance')
print("\n最能表明垃圾邮件的20个词：")
print(top_spam_words.to_string(index=False))
print()

# 排序并获取最能表明正常邮件的词
top_ham_words = feature_importance_df.nsmallest(20, 'importance')
print("\n最能表明正常邮件的20个词：")
print(top_ham_words.to_string(index=False))
print()

# 可视化特征重要性
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 垃圾邮件关键词
ax1.barh(range(len(top_spam_words)), top_spam_words['importance'], color='#e74c3c', edgecolor='black')
ax1.set_yticks(range(len(top_spam_words)))
ax1.set_yticklabels(top_spam_words['word'])
ax1.set_xlabel('Feature Importance Score', fontsize=11, fontweight='bold')
ax1.set_title('Top 20 Spam-Indicating Words', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# 正常邮件关键词
ax2.barh(range(len(top_ham_words)), top_ham_words['importance'], color='#2ecc71', edgecolor='black')
ax2.set_yticks(range(len(top_ham_words)))
ax2.set_yticklabels(top_ham_words['word'])
ax2.set_xlabel('Feature Importance Score', fontsize=11, fontweight='bold')
ax2.set_title('Top 20 Ham-Indicating Words', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("特征重要性图已保存为：feature_importance.png")
plt.close()


# ============================================================================
# 第十二部分：错误分析
# ============================================================================

print("\n" + "=" * 80)
print("【步骤11】错误分析")
print("-" * 80)

# 获取最佳模型的预测结果
y_pred_best = best_model.predict(X_test_best)

# 找出错误分类的样本
errors_mask = (y_pred_best != y_test)
errors_df = pd.DataFrame({
    'message': X_test[errors_mask].values,
    'true_label': y_test[errors_mask].map({0: 'Ham', 1: 'Spam'}).values,
    'predicted_label': pd.Series(y_pred_best[errors_mask]).map({0: 'Ham', 1: 'Spam'}).values
})

print(f"\n错误分类样本总数：{len(errors_df)}")
print(f"错误率：{len(errors_df) / len(y_test) * 100:.2f}%")
print()

# 分析错误类型
false_positives = errors_df[errors_df['predicted_label'] == 'Spam']  # 误报：实际是Ham，预测为Spam
false_negatives = errors_df[errors_df['predicted_label'] == 'Ham']   # 漏报：实际是Spam，预测为Ham

print(f"假阳性（False Positives - 正常邮件被误判为垃圾邮件）：{len(false_positives)}")
print(f"假阴性（False Negatives - 垃圾邮件被误判为正常邮件）：{len(false_negatives)}")
print()

# 显示一些错误分类的示例
if len(false_positives) > 0:
    print("\n假阳性示例（正常邮件被误判为垃圾邮件）：")
    print("-" * 80)
    for i, row in false_positives.head(5).iterrows():
        print(f"消息: {row['message']}")
        print(f"真实标签: {row['true_label']}, 预测标签: {row['predicted_label']}")
        print()

if len(false_negatives) > 0:
    print("\n假阴性示例（垃圾邮件被误判为正常邮件）：")
    print("-" * 80)
    for i, row in false_negatives.head(5).iterrows():
        print(f"消息: {row['message']}")
        print(f"真实标签: {row['true_label']}, 预测标签: {row['predicted_label']}")
        print()

# 保存错误分类样本
errors_df.to_csv('misclassified_messages.csv', index=False, encoding='utf-8')
print("错误分类样本已保存为：misclassified_messages.csv")
print()


# ============================================================================
# 第十三部分：总结报告
# ============================================================================

print("\n" + "=" * 80)
print("【完整分析总结】")
print("=" * 80)

summary = f"""
数据集信息：
  - 总样本数：{len(df)}
  - 正常邮件（Ham）：{(y_numeric == 0).sum()} ({(y_numeric == 0).sum()/len(df)*100:.1f}%)
  - 垃圾邮件（Spam）：{(y_numeric == 1).sum()} ({(y_numeric == 1).sum()/len(df)*100:.1f}%)
  
数据划分：
  - 训练集：{len(X_train)} 样本
  - 测试集：{len(X_test)} 样本
  
特征工程：
  - 使用了两种向量化方法：CountVectorizer 和 TfidfVectorizer
  - 词汇表大小：{len(best_vectorizer.vocabulary_)} 个特征词
  
模型评估：
  - 共测试了4种机器学习算法
  - 每种算法使用2种特征表示方法，共8个模型配置
  
最佳模型：
  - 模型：{best_model_info['Model']}
  - 向量化方法：{best_model_info['Vectorizer']}
  - 准确率：{best_model_info['Accuracy']:.4f}
  - 精确率：{best_model_info['Precision']:.4f}
  - 召回率：{best_model_info['Recall']:.4f}
  - F1分数：{best_model_info['F1-Score']:.4f}
  
输出文件：
  1. label_distribution.png - 标签分布图
  2. message_length_analysis.png - 消息长度分析图
  3. model_performance_comparison.png - 模型性能对比图
  4. confusion_matrices.png - 混淆矩阵图
  5. roc_curves.png - ROC曲线图
  6. feature_importance.png - 特征重要性图
  7. model_comparison_results.csv - 模型对比结果CSV
  8. misclassified_messages.csv - 错误分类样本CSV
"""

print(summary)

print("=" * 80)
print("程序运行完成！所有结果已保存到当前目录。")
print("=" * 80)