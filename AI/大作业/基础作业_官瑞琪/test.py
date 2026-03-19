"""
声纳矿物数据集分类
本程序实现了6种机器学习算法对声纳数据的分类，并展示优化前后的性能对比。
"""
# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, randint, loguniform
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 声纳数据分类器类
class SonarClassifierWithOptimization:
    # 初始化分类器
    def __init__(self, data_path='sonar data.csv'):
        self.data_path = data_path
        self.baseline_models = {}  # 优化前的模型
        self.optimized_models = {}  # 优化后的模型
        self.best_params = {}  # 保存每个模型的最佳参数
        self.baseline_results = {}
        self.optimized_results = {}
        
    # 加载并预处理数据    
    def load_data(self):
        print("\n(一)数据加载与预处理")
        print("=" * 80)
        # 读取数据（声纳数据集没有列名）
        column_names = [f'feature_{i}' for i in range(60)] + ['label']
        self.df = pd.read_csv(self.data_path, header=None, names=column_names)
        print(f"数据集形状: {self.df.shape}")
        print(f"类别分布:")
        print(self.df['label'].value_counts())
        print(f"  - Rock (岩石): {sum(self.df['label']=='R')} 样本")
        print(f"  - Mine (矿物): {sum(self.df['label']=='M')} 样本")
        print(f"\n缺失值检查: {self.df.isnull().sum().sum()} (无缺失值)")
        
        # 分离特征和标签
        X = self.df.drop('label', axis=1)
        y = self.df['label'].map({'M': 1, 'R': 0})
        
        # 划分训练集和测试集（80:20）
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"数据划分:")
        print(f"  训练集: {self.X_train.shape[0]} 样本 (80%)")
        print(f"  测试集: {self.X_test.shape[0]} 样本 (20%)")
        print(f"  特征维度: {self.X_train.shape[1]}")
        print()
        
        # 特征标准化（对某些模型很重要）
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    # 分析特征相关性
    def plot_feature_correlation(self):
        print("(二)特征相关性分析与可视化")
        print("=" * 80)
        # 仅使用特征列
        feature_df = self.df.drop(columns=['label'])

        # 计算皮尔逊相关系数矩阵
        corr_matrix = feature_df.corr(method='pearson')
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={'label': 'Pearson Correlation'},
            xticklabels=False,
            yticklabels=False
        )
        plt.title(
            '声纳数据集特征相关性热力图（60维特征）',
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(
            '特征相关性热力图.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()
        print("特征相关性热力图已保存: 特征相关性热力图.png")
        
    # 构建基线模型（优化前） 
    # 使用默认或简单参数
    def build_baseline_models(self):
        # 1.k-NN——默认参数
        # 参数n_neighbors=默认值5
        self.baseline_models['k-NN'] = {
            'model': KNeighborsClassifier(),
            'use_scaled': True
        }
        
        # 2.朴素贝叶斯——默认参数
        self.baseline_models['Naive Bayes'] = {
            'model': GaussianNB(),
            'use_scaled': False
        }
        
        # 3.决策树——无深度限制
        self.baseline_models['Decision Tree'] = {
            'model': DecisionTreeClassifier(random_state=42), # 设置随机种子=42
            'use_scaled': False
        }
        
        # 4.随机森林——默认参数
        # 默认参数n_estimators=100, max_depth=None
        self.baseline_models['Random Forest'] = {
            'model': RandomForestClassifier(random_state=42),# 设置随机种子=42
            'use_scaled': False
        }
        
        # 5.逻辑回归
        # 默认参数: C=1.0, penalty='l2'
        self.baseline_models['Logistic Regression'] = {
            'model': LogisticRegression(random_state=42),# 设置随机种子=42
            'use_scaled': True
        }
        
        # 6.多层感知机
        # 默认参数:hidden_layer_sizes=(64,), alpha=0.0001
        self.baseline_models['MLP'] = {
            'model': MLPClassifier(random_state=42),
            'use_scaled': True
        }
        print()
      
    # 定义每个模型的搜索空间
        param_distributions = {
            'k-NN': {
                'n_neighbors': randint(1, 15),
                'weights': ['uniform', 'distance'],
                'p': [1, 2]   # 只保留 minkowski + p
            },
            'Naive Bayes': {
                'var_smoothing': loguniform(1e-10, 1e-6)
            },
            'Decision Tree': {
                'max_depth': randint(3, 20),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 10),
                'criterion': ['gini', 'entropy']
            },
            'Random Forest': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(5, 30),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'Logistic Regression': [
                {
                    'solver': ['liblinear'],
                    'penalty': ['l1', 'l2'],
                    'C': loguniform(0.01, 100)
                },
                {
                    'solver': ['saga'],
                    'penalty': ['elasticnet'],
                    'C': loguniform(0.01, 100),
                    'l1_ratio': uniform(0, 1)
                }
            ],
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': loguniform(1e-5, 1e-2),
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [800]
            }
        }
        
        for name in param_distributions.keys():
            print(f"\n正在优化 {name} ...")
            config = self.baseline_models[name]
            model = config['model']
            use_scaled = config['use_scaled']
            X_train = self.X_train_scaled if use_scaled else self.X_train
            
            search = RandomizedSearchCV(
                model,
                param_distributions[name],
                n_iter=50,              # 随机搜索次数（可根据计算资源调整）
                cv=5,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(X_train, self.y_train)
            
            self.best_params[name] = search.best_params_
            print(f"  最佳参数: {search.best_params_}")
            print(f"  最佳交叉验证准确率: {search.best_score_:.4f}")
            
            # 创建优化后的模型实例
            best_params = search.best_params_.copy()
            if name in ['Decision Tree', 'Random Forest', 'Logistic Regression', 'MLP']:
                best_params['random_state'] = 42
            self.optimized_models[name] = {
                'model': type(model)(**best_params),
                'use_scaled': use_scaled
            }
        
        print("\n所有模型优化完成！\n")
    
    # 训练并评估模型
    def train_and_evaluate(self, models, results_dict):        
        for name, config in models.items():
            print(f"{name}——")
            model = config['model']
            
            # 选择是否使用标准化数据
            if config['use_scaled']:
                X_train = self.X_train_scaled
                X_test = self.X_test_scaled
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # 训练模型
            model.fit(X_train, self.y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # 交叉验证分数
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=5)
            
            # 混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)
            
            # 保存结果
            results_dict[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': cm,
                'predictions': y_pred
            }
            print(f"Acc: {acc:.4f}     Prec: {prec:.4f}     Rec: {rec:.4f}")
        print()
        
    # 绘制优化前后对比图 
    def plot_optimization_comparison(self):
        print("(四)可视化优化效果")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        models = list(self.baseline_results.keys())
        
        # 1.准确率对比柱状图
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        
        baseline_acc = [self.baseline_results[m]['accuracy'] for m in models]
        optimized_acc = [self.optimized_results[m]['accuracy'] for m in models]
        
        bars1 = ax1.bar(x - width/2, baseline_acc, width, label='优化前', 
                       color='lightcoral', alpha=0.8)
        bars2 = ax1.bar(x + width/2, optimized_acc, width, label='优化后', 
                       color='lightgreen', alpha=0.8)
        
        ax1.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax1.set_ylabel('准确率', fontsize=12, fontweight='bold')
        ax1.set_title('优化前后准确率对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0.85, color='red', linestyle='--', linewidth=2, 
                   label='目标线 (0.85)')
        
        # 在柱子上添加数值和提升百分比
        for i, (b, o) in enumerate(zip(baseline_acc, optimized_acc)):
            improvement = ((o - b) / b * 100) if b > 0 else 0
            ax1.text(i - width/2, b, f'{b:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
            ax1.text(i + width/2, o, f'{o:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
            if improvement > 0:
                ax1.text(i, max(b, o) + 0.02, f'+{improvement:.1f}%', 
                        ha='center', va='bottom', fontsize=9, color='green', 
                        fontweight='bold')
        
        # 2.精准率和召回率对比
        ax2 = axes[0, 1]
        baseline_prec = [self.baseline_results[m]['precision'] for m in models]
        optimized_prec = [self.optimized_results[m]['precision'] for m in models]
        baseline_rec = [self.baseline_results[m]['recall'] for m in models]
        optimized_rec = [self.optimized_results[m]['recall'] for m in models]
        
        x2 = np.arange(len(models))
        width2 = 0.2
        
        ax2.bar(x2 - 1.5*width2, baseline_prec, width2, label='优化前-精准率', 
               color='lightblue', alpha=0.8)
        ax2.bar(x2 - 0.5*width2, optimized_prec, width2, label='优化后-精准率', 
               color='blue', alpha=0.8)
        ax2.bar(x2 + 0.5*width2, baseline_rec, width2, label='优化前-召回率', 
               color='yellow', alpha=0.8)
        ax2.bar(x2 + 1.5*width2, optimized_rec, width2, label='优化后-召回率', 
               color='orange', alpha=0.8)
        
        ax2.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax2.set_ylabel('分数', fontsize=12, fontweight='bold')
        ax2.set_title('精准率和召回率对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3.性能提升热图
        ax3 = axes[1, 0]
        improvements = []
        metrics = ['准确率', '精准率', '召回率', 'F1分数']
        for model in models:
            model_improvements = [
                (self.optimized_results[model]['accuracy'] - 
                 self.baseline_results[model]['accuracy']) * 100,
                (self.optimized_results[model]['precision'] - 
                 self.baseline_results[model]['precision']) * 100,
                (self.optimized_results[model]['recall'] - 
                 self.baseline_results[model]['recall']) * 100,
                (self.optimized_results[model]['f1_score'] - 
                 self.baseline_results[model]['f1_score']) * 100
            ]
            improvements.append(model_improvements)
        improvements_array = np.array(improvements)
        im = ax3.imshow(improvements_array, cmap='RdYlGn', aspect='auto', 
                       vmin=-5, vmax=15)
        
        ax3.set_xticks(np.arange(len(metrics)))
        ax3.set_yticks(np.arange(len(models)))
        ax3.set_xticklabels(metrics, fontsize=11)
        ax3.set_yticklabels(models, fontsize=11)
        ax3.set_title('性能提升热图 (百分点)', fontsize=14, fontweight='bold')
        
        # 添加数值标注
        for i in range(len(models)):
            for j in range(len(metrics)):
                value = improvements_array[i, j]
                color = 'white' if abs(value) > 5 else 'black'
                text = ax3.text(j, i, f'{value:.1f}%', ha='center', va='center',
                              color=color, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax3, label='提升百分点 (%)')
        
        # 4.交叉验证稳定性对比
        ax4 = axes[1, 1]
        baseline_cv = [self.baseline_results[m]['cv_mean'] for m in models]
        optimized_cv = [self.optimized_results[m]['cv_mean'] for m in models]
        baseline_std = [self.baseline_results[m]['cv_std'] for m in models]
        optimized_std = [self.optimized_results[m]['cv_std'] for m in models]
        
        x4 = np.arange(len(models))
        ax4.errorbar(x4 - 0.15, baseline_cv, yerr=baseline_std, fmt='o-', 
                    capsize=5, capthick=2, markersize=8, label='优化前', 
                    color='coral', linewidth=2)
        ax4.errorbar(x4 + 0.15, optimized_cv, yerr=optimized_std, fmt='s-', 
                    capsize=5, capthick=2, markersize=8, label='优化后', 
                    color='green', linewidth=2)
        
        ax4.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax4.set_ylabel('交叉验证分数 (均值±标准差)', fontsize=12, fontweight='bold')
        ax4.set_title('模型稳定性对比 (5折交叉验证)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x4)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend(fontsize=11)
        ax4.grid(alpha=0.3)
        ax4.axhline(y=0.85, color='red', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig('优化对比图表.png', dpi=300, bbox_inches='tight')
        print("优化对比图表已保存: 优化对比图表.png")
        plt.show()
        
    # 绘制混淆矩阵（优化后的模型）
    def plot_confusion_matrices(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.optimized_results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Mine(0)', 'Rock(1)'],
                       yticklabels=['Mine(0)', 'Rock(1)'],
                       ax=axes[idx], cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{name} (优化后)\nAccuracy: {result["accuracy"]:.4f}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('混淆矩阵.png', dpi=300, bbox_inches='tight')
        print("混淆矩阵已保存: 混淆矩阵.png\n")
        plt.show()
    
    # 生成优化对比报告
    def generate_comparison_report(self):
        print("(五)优化对比报告")
        print("=" * 80)
        
        report = []   
        # 数据集信息
        report.append("1.数据集信息")
        report.append(f"  样本总数: {len(self.df)}")
        report.append(f"  特征数量: 60个声纳回波频率特征")
        report.append(f"  训练集: {len(self.X_train)} 样本 (80%)")
        report.append(f"  测试集: {len(self.X_test)} 样本 (20%)")
        report.append(f"  类别分布: Rock={sum(self.df['label']=='R')}, Mine={sum(self.df['label']=='M')}")
        report.append("")
        
        # 优化前后对比表
        report.append("2.优化前后性能对比")
        report.append(f"{'模型':<20} {'版本':<10} {'准确率':<10} {'精准率':<10} {'召回率':<10} {'F1分数':<10} {'交叉验证':<15}")
        report.append("-" * 100)
        
        models = list(self.baseline_results.keys())
        for name in models:
            # 优化前
            b = self.baseline_results[name]
            report.append(
                f"{name:<20} {'优化前':<10} "
                f"{b['accuracy']:<10.4f} "
                f"{b['precision']:<10.4f} "
                f"{b['recall']:<10.4f} "
                f"{b['f1_score']:<10.4f} "
                f"{b['cv_mean']:.4f}±{b['cv_std']:.3f}"
            )
            
            # 优化后
            o = self.optimized_results[name]
            acc_imp = (o['accuracy'] - b['accuracy']) * 100
            prec_imp = (o['precision'] - b['precision']) * 100
            rec_imp = (o['recall'] - b['recall']) * 100
            f1_imp = (o['f1_score'] - b['f1_score']) * 100
            
            report.append(
                f"{name:<20} {'优化后':<10} "
                f"{o['accuracy']:<10.4f} "
                f"{o['precision']:<10.4f} "
                f"{o['recall']:<10.4f} "
                f"{o['f1_score']:<10.4f} "
                f"{o['cv_mean']:.4f}±{o['cv_std']:.3f}"
            )
            
            report.append(
                f"{'提升':<20} {'':<10} "
                f"{acc_imp:>+9.2f}% "
                f"{prec_imp:>+9.2f}% "
                f"{rec_imp:>+9.2f}% "
                f"{f1_imp:>+9.2f}% "
                f"{'':>15}"
            )
            report.append("-" * 100)
        
        report.append("")
        
        # 关键发现
        report.append("3.小结")
        
        # 找出提升最大的模型
        max_improvement = 0
        best_improved_model = ""
        for name in models:
            improvement = (self.optimized_results[name]['accuracy'] - 
                          self.baseline_results[name]['accuracy'])
            if improvement > max_improvement:
                max_improvement = improvement
                best_improved_model = name
        
        report.append(f"(1)提升最显著的模型: {best_improved_model}")
        report.append(f"   准确率提升: {max_improvement*100:.2f} 百分点")
        
        # 找出性能最好的模型
        best_model = max(self.optimized_results.items(), 
                        key=lambda x: x[1]['accuracy'])
        report.append(f"(2)最佳模型: {best_model[0]}")
        report.append(f"   准确率: {best_model[1]['accuracy']:.4f}")
        report.append(f"   精准率: {best_model[1]['precision']:.4f}")
        report.append(f"   召回率: {best_model[1]['recall']:.4f}")
        
        # 达标模型
        models_above_95 = [name for name, r in self.optimized_results.items() 
                          if r['accuracy'] > 0.85]
        report.append(f"(3)达到0.85准确率目标的模型: {len(models_above_95)}个")
        for model in models_above_95:
            report.append(f"{model:<20} : {self.optimized_results[model]['accuracy']:.4f}")
        
        # 打印并保存报告
        report_text = "\n".join(report)
        print(report_text)
        
        with open('优化对比报告.txt', 'w', encoding='utf-8')as f:
            f.write(report_text)
        
        print("\n优化对比报告已保存: 优化对比报告.txt\n")
    
    # 运行完整的分析流程
    def run_complete_pipeline(self):
        self.load_data()
        self.plot_feature_correlation()
        self.build_baseline_models()
        self.build_optimized_models()
        

        print("(三)训练模型")
        print("=" * 80)
        print("1.基线")
        self.train_and_evaluate(self.baseline_models, self.baseline_results)
        print("2.优化")
        self.train_and_evaluate(self.optimized_models, self.optimized_results)
        
        self.plot_optimization_comparison()
        self.plot_confusion_matrices()
        self.generate_comparison_report()


# 主程序
if __name__ == "__main__":
    # 创建分类器实例并运行
    classifier = SonarClassifierWithOptimization('sonar data.csv')
    classifier.run_complete_pipeline()