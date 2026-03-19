# 1. n_neighbors (邻居数量): 测试 K=1, 3, 5, 7, 9, 11, 15, 20
#    - K值太小容易过拟合,太大容易欠拟合

# 2. weights (权重方法):
#    - uniform: 所有邻居权重相同
#    - distance: 距离越近权重越大

# 3. metric (距离度量):
#    - euclidean: 欧几里得距离
#    - manhattan: 曼哈顿距离
#    - minkowski: 闵可夫斯基距离

# 4. algorithm (搜索算法):
#    - auto: 自动选择
#    - ball_tree: 球树
#    - kd_tree: KD树
#    - brute: 暴力搜索

# 5. p (Minkowski距离的幂参数):
#    - p=1 等同于曼哈顿距离
#    - p=2 等同于欧几里得距离

# 脚本会输出每个参数组合的:
# - 准确率
# - 详细的分类报告(precision, recall, f1-score)
# - 最后汇总所有结果并按准确率排序


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import sys

# 创建输出文件
output_file = 'knn_analysis_results.txt'
f = open(output_file, 'w', encoding='utf-8')

# 定义一个打印函数,同时输出到控制台和文件
def print_both(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=f)

# 加载数据
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=33
)

# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

print_both("=" * 80)
print_both("K近邻分类器参数影响分析")
print_both("=" * 80)

# 存储结果
results = []

# 1. 测试不同的 n_neighbors (K值)
print_both("\n【1. 测试不同的邻居数量 (n_neighbors)】")
print_both("-" * 80)
for k in [1, 3, 5, 7, 9, 11, 15, 20]:
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    accuracy = knc.score(X_test, y_test)
    
    print_both(f"\nn_neighbors = {k}")
    print_both(f"准确率: {accuracy:.4f}")
    print_both(classification_report(y_test, y_predict, target_names=iris.target_names, zero_division=0))
    
    results.append({
        '参数': f'n_neighbors={k}',
        '准确率': accuracy
    })

# 2. 测试不同的权重方法
print_both("\n" + "=" * 80)
print_both("【2. 测试不同的权重方法 (weights)】")
print_both("-" * 80)
for weight in ['uniform', 'distance']:
    knc = KNeighborsClassifier(n_neighbors=5, weights=weight)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    accuracy = knc.score(X_test, y_test)
    
    print_both(f"\nweights = '{weight}'")
    print_both(f"准确率: {accuracy:.4f}")
    print_both(classification_report(y_test, y_predict, target_names=iris.target_names, zero_division=0))
    
    results.append({
        '参数': f'weights={weight}',
        '准确率': accuracy
    })

# 3. 测试不同的距离度量方法
print_both("\n" + "=" * 80)
print_both("【3. 测试不同的距离度量方法 (metric)】")
print_both("-" * 80)
for metric in ['euclidean', 'manhattan', 'minkowski']:
    knc = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    accuracy = knc.score(X_test, y_test)
    
    print_both(f"\nmetric = '{metric}'")
    print_both(f"准确率: {accuracy:.4f}")
    print_both(classification_report(y_test, y_predict, target_names=iris.target_names, zero_division=0))
    
    results.append({
        '参数': f'metric={metric}',
        '准确率': accuracy
    })

# 4. 测试不同的算法
print_both("\n" + "=" * 80)
print_both("【4. 测试不同的算法 (algorithm)】")
print_both("-" * 80)
for algo in ['auto', 'ball_tree', 'kd_tree', 'brute']:
    knc = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    accuracy = knc.score(X_test, y_test)
    
    print_both(f"\nalgorithm = '{algo}'")
    print_both(f"准确率: {accuracy:.4f}")
    print_both(classification_report(y_test, y_predict, target_names=iris.target_names, zero_division=0))
    
    results.append({
        '参数': f'algorithm={algo}',
        '准确率': accuracy
    })

# 5. 测试不同的 p 值（用于 Minkowski 距离）
print_both("\n" + "=" * 80)
print_both("【5. 测试不同的 p 值 (Minkowski距离的幂参数)】")
print_both("-" * 80)
for p in [1, 2, 3, 4]:
    knc = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=p)
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)
    accuracy = knc.score(X_test, y_test)
    
    print_both(f"\np = {p} ({'manhattan' if p==1 else 'euclidean' if p==2 else f'minkowski-{p}'})")
    print_both(f"准确率: {accuracy:.4f}")
    print_both(classification_report(y_test, y_predict, target_names=iris.target_names, zero_division=0))
    
    results.append({
        '参数': f'p={p}',
        '准确率': accuracy
    })

# 汇总结果
print_both("\n" + "=" * 80)
print_both("【参数对比汇总】")
print_both("=" * 80)
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('准确率', ascending=False)
print_both(df_results.to_string(index=False))

print_both("\n最佳参数组合:")
print_both(df_results.iloc[0]['参数'], f"- 准确率: {df_results.iloc[0]['准确率']:.4f}")

# 关闭文件
f.close()
print(f"\n结果已保存到文件: {output_file}")