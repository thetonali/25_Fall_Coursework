# 导入必要的库
# 数据处理与分析库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 机器学习模型库
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
# 预处理库
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer # 计算中位数
from sklearn.preprocessing import KBinsDiscretizer
# 评估指标库
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
from tabulate import tabulate
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import Binarizer
import warnings
warnings.filterwarnings('ignore')
# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# 定义列名
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# 加载数据
df = pd.read_csv(url, names=column_names)
# 数据基本信息查看
print(f"数据集形状: {df.shape}")
print(f"总样本数: {df.shape[0]}")
print(f"特征数量: {df.shape[1] - 1}")  # 减去目标变量
# 显示数据集形状和前5行数据
print(df.head())

# 显示目标变量的分布情况
print("\n目标变量分布")
print(df['Outcome'].value_counts())
print("\n类别比例:")
print(df['Outcome'].value_counts(normalize=True))

# 检查异常值和零值
# 某些特征（如Glucose, BloodPressure等）不应该为0
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 1.将0替换为NaN
df[zero_columns] = df[zero_columns].replace(0, np.nan)

#2.检查缺失值情况
print("缺失值统计（填充前）：")
print(df.isnull().sum())

#3.使用中位数填充缺失值 (中位数比均值更抗异常值干扰)
imputer = SimpleImputer(strategy='median')
df[zero_columns] = imputer.fit_transform(df[zero_columns])

print("\n缺失值填充完成。")

# 特征相关性分析
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5)
plt.title('特征相关性热图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#  输出与目标变量相关性最强的特征
print("\n特征与目标变量的相关性:")
target_corr = correlation_matrix['Outcome'].sort_values(ascending=False)
print(target_corr)

# 分离特征和目标变量
X = df.drop('Outcome', axis=1) # 特征矩阵
y = df['Outcome'] # 特征矩阵
print(f"特征矩阵 X 的形状: {X.shape}")
print(f"目标变量 y 的形状: {y.shape}")

# 划分训练集和测试集（80%训练，20%测试）
# 使用stratify参数确保训练集和测试集中的类别比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"\n训练集目标变量分布:")
print(y_train.value_counts())
print(f"\n测试集目标变量分布:")
print(y_test.value_counts())

# 数据标准化处理
# 为不同的朴素贝叶斯算法准备不同的数据格式

# 1.高斯朴素贝叶斯：使用标准化数据
scaler_gaussian = StandardScaler()
X_train_gaussian = scaler_gaussian.fit_transform(X_train)
X_test_gaussian = scaler_gaussian.transform(X_test)

# 2.伯努利朴素贝叶斯：需要二值化数据（使用中位数阈值）
X_train_bernoulli = (X_train > X_train.median()).astype(int)
X_test_bernoulli = (X_test > X_train.median()).astype(int)

# 3.多项式朴素贝叶斯：需要非负数据，使用MinMax缩放到[0,1]再乘以100转为计数型
scaler_multinomial = MinMaxScaler()
X_train_multinomial = scaler_multinomial.fit_transform(X_train) * 100
X_test_multinomial = scaler_multinomial.transform(X_test) * 100

# 定义模型评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    完整的模型评估函数
    参数:
        model: 训练好的模型
        X_train, X_test: 训练集和测试集特征
        y_train, y_test: 训练集和测试集标签
        model_name: 模型名称
    返回:
        results: 包含各项评估指标的字典
    """
    # 训练集预测
    y_train_pred = model.predict(X_train)
    # 测试集预测
    y_test_pred = model.predict(X_test)
    
    # 计算各项指标
    results = {
        '模型名称': model_name,
        '训练集准确率': accuracy_score(y_train, y_train_pred),
        '测试集准确率': accuracy_score(y_test, y_test_pred),
        '精确率(Precision)': precision_score(y_test, y_test_pred),
        '召回率(Recall)': recall_score(y_test, y_test_pred),
        'F1分数': f1_score(y_test, y_test_pred)
    }
    
    # 打印详细评估报告
    print(f"{model_name} - 模型评估报告")
    print(f"\n训练集准确率: {results['训练集准确率']:.4f}")
    print(f"测试集准确率: {results['测试集准确率']:.4f}")
    print(f"精确率: {results['精确率(Precision)']:.4f}")
    print(f"召回率: {results['召回率(Recall)']:.4f}")
    print(f"F1分数: {results['F1分数']:.4f}")
    
    # 分类报告
    print(f"\n详细分类报告:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['0', '1']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    
    return results, y_test_pred, cm

# 训练高斯朴素贝叶斯模型
# 创建并训练模型
gnb = GaussianNB()
gnb.fit(X_train_gaussian, y_train)

# 评估模型
results_gnb, y_pred_gnb, cm_gnb = evaluate_model(
    gnb, X_train_gaussian, X_test_gaussian, y_train, y_test, 
    "高斯朴素贝叶斯(GaussianNB)"
)

# 可视化高斯朴素贝叶斯的混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['无糖尿病', '有糖尿病'],
            yticklabels=['无糖尿病', '有糖尿病'])
plt.title('高斯朴素贝叶斯 - 混淆矩阵', fontsize=14, fontweight='bold')
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.tight_layout()
plt.show()

# 训练伯努利朴素贝叶斯模型
# 创建并训练模型
bnb = BernoulliNB()
bnb.fit(X_train_bernoulli, y_train)

# 评估模型
results_bnb, y_pred_bnb, cm_bnb = evaluate_model(
    bnb, X_train_bernoulli, X_test_bernoulli, y_train, y_test, 
    "伯努利朴素贝叶斯(BernoulliNB)"
)

# 可视化伯努利朴素贝叶斯的混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bnb, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['无糖尿病', '有糖尿病'],
            yticklabels=['无糖尿病', '有糖尿病'])
plt.title('伯努利朴素贝叶斯 - 混淆矩阵', fontsize=14, fontweight='bold')
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.tight_layout()
plt.show()

# 训练多项式朴素贝叶斯模型
# 创建并训练模型
mnb = MultinomialNB()
mnb.fit(X_train_multinomial, y_train)

# 评估模型
results_mnb, y_pred_mnb, cm_mnb = evaluate_model(
    mnb, X_train_multinomial, X_test_multinomial, y_train, y_test, 
    "多项式朴素贝叶斯(MultinomialNB)"
)

# 可视化多项式朴素贝叶斯的混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mnb, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['无糖尿病', '有糖尿病'],
            yticklabels=['无糖尿病', '有糖尿病'])
plt.title('多项式朴素贝叶斯 - 混淆矩阵', fontsize=14, fontweight='bold')
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.tight_layout()
plt.show()

print("综合对比分析——")
# 1.汇总所有模型的性能指标
results_df = pd.DataFrame([results_gnb, results_bnb, results_mnb])
results_df = results_df.set_index('模型名称')

print("\n【性能指标汇总表】")
print(tabulate(results_df.round(4), headers='keys', tablefmt='grid'))

# 2.可视化性能对比 - 柱状图
metrics = ['测试集准确率', '精确率(Precision)', '召回率(Recall)', 'F1分数']
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(metrics))
width = 0.25

bars1 = ax.bar(x - width, results_df.iloc[0][metrics], width, 
               label='高斯朴素贝叶斯', color='skyblue', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, results_df.iloc[1][metrics], width, 
               label='伯努利朴素贝叶斯', color='lightgreen', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, results_df.iloc[2][metrics], width, 
               label='多项式朴素贝叶斯', color='lightsalmon', alpha=0.8, edgecolor='black')

ax.set_xlabel('评估指标', fontsize=13, fontweight='bold')
ax.set_ylabel('分数', fontsize=13, fontweight='bold')
ax.set_title('三种朴素贝叶斯算法性能对比', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.1])

# 在柱状图上添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
plt.show()


# 3.训练集vs测试集准确率对比（检测过拟合）
model_names = ['高斯NB', '伯努利NB', '多项式NB']
train_accs = [results_gnb['训练集准确率'], 
              results_bnb['训练集准确率'], 
              results_mnb['训练集准确率']]
test_accs = [results_gnb['测试集准确率'], 
             results_bnb['测试集准确率'], 
             results_mnb['测试集准确率']]

x_pos = np.arange(len(model_names))
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
bars1 = ax.bar(x_pos - bar_width/2, train_accs, bar_width, 
               label='训练集准确率', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x_pos + bar_width/2, test_accs, bar_width, 
               label='测试集准确率', color='#e74c3c', alpha=0.8, edgecolor='black')

ax.set_xlabel('模型', fontsize=13, fontweight='bold')
ax.set_ylabel('准确率', fontsize=13, fontweight='bold')
ax.set_title('训练集 vs 测试集准确率对比（过拟合检测）', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.1])

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# 准备数据 (仅取Glucose和BMI)

feature_cols = ['Glucose', 'BMI']
X_vis = df[feature_cols].values
y_vis = df['Outcome'].values

# 绘图设置
h = 0.5  # 网格步长
x_min, x_max = X_vis[:, 0].min() - 5, X_vis[:, 0].max() + 5
y_min, y_max = X_vis[:, 1].min() - 5, X_vis[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 定义三个模型及其对应的预处理器
models = [
    ("GaussianNB", GaussianNB(), StandardScaler()),
    (
        "MultinomialNB", 
        MultinomialNB(fit_prior=False), 
        KBinsDiscretizer(n_bins=500, encode='ordinal', strategy='uniform')
    ),
    ("BernoulliNB", BernoulliNB(), Binarizer(threshold=np.median(X_vis))) # 使用全局中位数做简单二值化
]

# 开始绘图
plt.figure(figsize=(18, 6)) # 设置画布大小，宽一点放三张图
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00AA00'])

for i, (name, model, preprocessor) in enumerate(models):
    # 1. 创建子图
    plt.subplot(1, 3, i + 1)
    
    # 2.数据预处理 (训练集)
    # MultinomialNB 必须非负，BernoulliNB 必须二值，GaussianNB 最好标准化
    if name == "BernoulliNB":
        # Bernoulli 特殊处理：先标准化再二值化，或者直接二值化。这里直接对原始数据二值化
        # 为了画图方便对每一列单独计算阈值（中位数）
        X_train_trans = X_vis.copy()
        thresholds = np.median(X_vis, axis=0)
        X_train_trans = (X_train_trans > thresholds).astype(int)
        model.fit(X_train_trans, y_vis)
        
        # 网格数据预处理
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_trans = (grid_points > thresholds).astype(int)
        
    else:
        # 其他两个模型使用各自的 scaler
        X_train_trans = preprocessor.fit_transform(X_vis)
        model.fit(X_train_trans, y_vis)
        
        # 网格数据预处理
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_trans = preprocessor.transform(grid_points)

    # 3.预测网格
    Z = model.predict(grid_trans)
    Z = Z.reshape(xx.shape)

    # 4.绘制边界
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.6, shading='auto')

    # 5.绘制散点 (画原始数据，方便观察)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=cmap_bold, edgecolor='k', s=20)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"{name} Decision Boundary")
    plt.xlabel('Glucose')
    if i == 0: plt.ylabel('BMI') # 只在第一张图显示Y轴标签

plt.tight_layout()
plt.show()