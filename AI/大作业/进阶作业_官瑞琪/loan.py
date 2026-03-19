"""
贷款审批数据集分类
本程序实现了4种机器学习算法对贷款数据的分类，并展示优化前后的性能对比。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score)
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 贷款状态分类器（含优化对比）
class LoanClassifierWithOptimization:
    # 初始化    
    def __init__(self, data_path='Loan_approval_data_2025.csv'):
        self.data_path = data_path
        self.baseline_models = {}     # 基线模型
        self.optimized_models = {}    # 优化模型
        self.baseline_results = {}    # 基线结果
        self.optimized_results = {}   # 优化结果
    
    # 加载并预处理数据
    def load_data(self):
        print("\n(一)数据加载与预处理")
        print("=" * 80)
        
        # 读取数据
        self.df = pd.read_csv(self.data_path)
        print(f"1. 数据集形状: {self.df.shape}")
        
        # 删除customer_id列
        if 'customer_id' in self.df.columns:
            self.df = self.df.drop('customer_id', axis=1)
            print("2. 已删除 customer_id 列")
        
        # 检查缺失值
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n发现缺失值:\n{missing_values[missing_values > 0]}")
            # 数值型用中位数填充，分类型用众数填充
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if self.df[col].dtype in ['float64', 'int64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            print("缺失值已处理")
        else:
            print("3. 无缺失值")
        
        # 编码分类变量
        print("4. 特征编码:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        if 'loan_status' in categorical_cols:
            categorical_cols.remove('loan_status')
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            print(f"  {col}: {len(le.classes_)} 个类别")
        
        # 分离特征和标签
        X = self.df.drop('loan_status', axis=1)
        y = self.df['loan_status']
        print(f"5. ")
        print(f"样本数量: {len(X)}")
        print(f"特征数量: {X.shape[1]}")
        print(f"类别分布:\n{y.value_counts()}")
        
        # 划分训练集和测试集（80:20，分层采样）
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"6. ")
        print(f"训练集: {self.X_train.shape[0]} 样本 (80%)")
        print(f"测试集: {self.X_test.shape[0]} 样本 (20%)")
        
        # 特征标准化
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print()
    
    # 分析并绘制特征相关性
    def plot_feature_correlation(self):
        print("(二)特征相关性分析")
        print("=" * 80)
        
        # 计算相关性矩阵
        correlation_matrix = self.df.corr()
        
        # 与目标变量的相关性
        target_corr = correlation_matrix['loan_status'].abs().sort_values(ascending=False)
        print("与目标变量(loan_status)相关性最强的前10个特征:")
        print(target_corr.head(11))
        
        # 绘制热力图
        plt.figure(figsize=(14, 12))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('特征相关性热力图', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('特征相关性热力图.png', dpi=300, bbox_inches='tight')
        print("相关性热力图已保存: 特征相关性热力图.png")
        plt.show()
        plt.close()
        print()
    
    # 构建基线模型（优化前）
    def build_baseline_models(self):
        # k-NN（默认n_neighbors=5）
        self.baseline_models['k-NN'] = {
            'model': KNeighborsClassifier(),
            'use_scaled': True
        }
        
        # 逻辑回归（默认C=1.0）
        self.baseline_models['Logistic Regression'] = {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'use_scaled': True
        }
        
        # 随机森林（默认n_estimators=100）
        self.baseline_models['Random Forest'] = {
            'model': RandomForestClassifier(random_state=42),
            'use_scaled': False
        }
        
        # MLP（默认hidden_layer_sizes=(100,)）
        self.baseline_models['MLP'] = {
            'model': MLPClassifier(max_iter=1000, random_state=42),
            'use_scaled': True
        }
    # 使用随机搜索找到每个模型的最佳参数
    # 运行一次后注释掉，直接使用找到的最佳参数
    def find_best_hyperparameters(self):     
        best_params = {}
        # 定义搜索空间
        param_distributions = {
            'k-NN': {
                'n_neighbors': randint(1, 20),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'cosine'],
                'p': [1, 2]
            },
            'Logistic Regression': {
                'C': uniform(0.01, 10),
                'penalty': ['l2', None],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [1000, 2000]
            },
            'Random Forest': {
                'n_estimators': randint(50, 300),
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': uniform(0.0001, 0.01),
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        # 为每个模型执行随机搜索
        for model_name in self.baseline_models.keys():
            print(f"搜索 {model_name} 的最佳参数...")
            
            base_model = self.baseline_models[model_name]['model']
            use_scaled = self.baseline_models[model_name]['use_scaled']
            X_train = self.X_train_scaled if use_scaled else self.X_train
            
            # 随机搜索
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions[model_name],
                n_iter=50,
                cv=3,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            random_search.fit(X_train, self.y_train)
            best_params[model_name] = random_search.best_params_
            
            print(f"  最佳参数: {random_search.best_params_}")
            print(f"  交叉验证得分: {random_search.best_score_:.4f}\n")
        return best_params
    
    # 构建优化后的模型
    def build_optimized_models(self):
        # k-NN优化
        self.optimized_models['k-NN'] = {
            'model': KNeighborsClassifier(
                n_neighbors=18,
                weights='distance',
                metric='manhattan',
                p=2
            ),
            'use_scaled': True
        }
        
        # 逻辑回归优化
        self.optimized_models['Logistic Regression'] = {
            'model': LogisticRegression(
                C=3.7554011,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            ),
            'use_scaled': True
        }
        
        # 随机森林优化
        self.optimized_models['Random Forest'] = {
            'model': RandomForestClassifier(
                n_estimators=196,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features=None,
                random_state=42
            ),
            'use_scaled': False
        }
        
        # MLP优化
        self.optimized_models['MLP'] = {
            'model': MLPClassifier(
                hidden_layer_sizes=(50,),
                activation='tanh',
                alpha=0.008181203,
                learning_rate='constant',
                random_state=42
            ),
            'use_scaled': True
        }
    
    # 训练并评估模型
    def train_and_evaluate(self, models, results_dict):
        for name, config in models.items():
            print(f"{name}:")
            model = config['model']
            
            # 选择数据
            if config['use_scaled']:
                X_train = self.X_train_scaled
                X_test = self.X_test_scaled
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # 训练
            model.fit(X_train, self.y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='binary', zero_division=0)
            rec = recall_score(self.y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='binary', zero_division=0)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, n_jobs=-1)
            
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
            print(f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}")
        print()
    
    # 绘制优化前后对比图
    def plot_optimization_comparison(self):
        print("(五)可视化优化效果")
        print("=" * 80)
        
        models = list(self.baseline_results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 准备数据
        baseline_acc = [self.baseline_results[m]['accuracy'] for m in models]
        optimized_acc = [self.optimized_results[m]['accuracy'] for m in models]
        baseline_prec = [self.baseline_results[m]['precision'] for m in models]
        optimized_prec = [self.optimized_results[m]['precision'] for m in models]
        baseline_rec = [self.baseline_results[m]['recall'] for m in models]
        optimized_rec = [self.optimized_results[m]['recall'] for m in models]
        
        # 1.准确率对比
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_acc, width, label='优化前', 
                       color='lightcoral', alpha=0.8)
        bars2 = ax1.bar(x + width/2, optimized_acc, width, label='优化后', 
                       color='lightgreen', alpha=0.8)
        
        ax1.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax1.set_ylabel('准确率', fontsize=12, fontweight='bold')
        ax1.set_title('准确率对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15, ha='right')
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0.90, color='red', linestyle='--', linewidth=2)
        
        # 添加数值和提升百分比
        for i, (b, o) in enumerate(zip(baseline_acc, optimized_acc)):
            improvement = ((o - b) / b * 100) if b > 0 else 0
            ax1.text(i - width/2, b, f'{b:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, o, f'{o:.3f}', ha='center', va='bottom', fontsize=9)
            if improvement > 0:
                ax1.text(i, max(b, o) + 0.02, f'+{improvement:.1f}%', 
                        ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
        
        # 2.精准率和召回率对比
        ax2 = axes[0, 1]
        width2 = 0.2
        
        ax2.bar(x - 1.5*width2, baseline_prec, width2, label='优化前-精准率', color='lightblue', alpha=0.8)
        ax2.bar(x - 0.5*width2, optimized_prec, width2, label='优化后-精准率', color='blue', alpha=0.8)
        ax2.bar(x + 0.5*width2, baseline_rec, width2, label='优化前-召回率', color='yellow', alpha=0.8)
        ax2.bar(x + 1.5*width2, optimized_rec, width2, label='优化后-召回率', color='orange', alpha=0.8)
        
        ax2.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax2.set_ylabel('分数', fontsize=12, fontweight='bold')
        ax2.set_title('精准率和召回率对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=15, ha='right')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3.性能提升热图
        ax3 = axes[1, 0]
        improvements = []
        metrics = ['准确率', '精准率', '召回率', 'F1分数']
        
        for model in models:
            improvements.append([
                (self.optimized_results[model]['accuracy'] - 
                 self.baseline_results[model]['accuracy']) * 100,
                (self.optimized_results[model]['precision'] - 
                 self.baseline_results[model]['precision']) * 100,
                (self.optimized_results[model]['recall'] - 
                 self.baseline_results[model]['recall']) * 100,
                (self.optimized_results[model]['f1_score'] - 
                 self.baseline_results[model]['f1_score']) * 100
            ])
        
        improvements_array = np.array(improvements)
        im = ax3.imshow(improvements_array, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=15)
        
        ax3.set_xticks(np.arange(len(metrics)))
        ax3.set_yticks(np.arange(len(models)))
        ax3.set_xticklabels(metrics, fontsize=11)
        ax3.set_yticklabels(models, fontsize=11)
        ax3.set_title('性能提升热图 (百分点)', fontsize=14, fontweight='bold')
        
        for i in range(len(models)):
            for j in range(len(metrics)):
                value = improvements_array[i, j]
                color = 'white' if abs(value) > 5 else 'black'
                ax3.text(j, i, f'{value:.1f}%', ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax3, label='提升百分点 (%)')
        
        # 4.交叉验证稳定性对比
        ax4 = axes[1, 1]
        baseline_cv = [self.baseline_results[m]['cv_mean'] for m in models]
        optimized_cv = [self.optimized_results[m]['cv_mean'] for m in models]
        baseline_std = [self.baseline_results[m]['cv_std'] for m in models]
        optimized_std = [self.optimized_results[m]['cv_std'] for m in models]
        
        ax4.errorbar(x - 0.15, baseline_cv, yerr=baseline_std, fmt='o-', 
                    capsize=5, capthick=2, markersize=8, label='优化前', 
                    color='coral', linewidth=2)
        ax4.errorbar(x + 0.15, optimized_cv, yerr=optimized_std, fmt='s-', 
                    capsize=5, capthick=2, markersize=8, label='优化后', 
                    color='green', linewidth=2)
        
        ax4.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax4.set_ylabel('交叉验证分数', fontsize=12, fontweight='bold')
        ax4.set_title('模型稳定性对比 (5折交叉验证)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=15, ha='right')
        ax4.legend(fontsize=11)
        ax4.grid(alpha=0.3)
        ax4.axhline(y=0.90, color='red', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig('优化对比图表.png', dpi=300, bbox_inches='tight')
        print("优化对比图表已保存: 优化对比图表.png")
        plt.show()
        print()
        
    def plot_confusion_matrices(self):
        """绘制混淆矩阵（优化后）"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.optimized_results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Rejected', 'Approved'],
                       yticklabels=['Rejected', 'Approved'],
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
        print("(六)生成优化对比报告")
        print("=" * 80)

        report = []
        # 数据集信息
        report.append("1.数据集信息")
        report.append(f"  样本总数: {len(self.df)}")
        report.append(f"  特征数量: {self.X_train.shape[1]}")
        report.append(f"  训练集: {len(self.X_train)} 样本 (80%)")
        report.append(f"  测试集: {len(self.X_test)} 样本 (20%)")
        report.append("")
        
        # 优化前后对比表
        report.append("2.优化前后性能对比")
        report.append(f"{'模型':<20} {'版本':<10} {'准确率':<10} {'精准率':<10} {'召回率':<10} {'F1分数':<10}")
        report.append("-" * 80)
        
        models = list(self.baseline_results.keys())
        for name in models:
            # 优化前
            b = self.baseline_results[name]
            report.append(
                f"{name:<20} {'优化前':<10} "
                f"{b['accuracy']:<10.4f} {b['precision']:<10.4f} "
                f"{b['recall']:<10.4f} {b['f1_score']:<10.4f}"
            )
            
            # 优化后
            o = self.optimized_results[name]
            acc_imp = (o['accuracy'] - b['accuracy']) * 100
            report.append(
                f"{name:<20} {'优化后':<10} "
                f"{o['accuracy']:<10.4f} {o['precision']:<10.4f} "
                f"{o['recall']:<10.4f} {o['f1_score']:<10.4f}"
            )
            
            report.append(f"{'提升':<20} {'':<10} {acc_imp:>+9.2f}%")
            report.append("-" * 80)
        
        report.append("")
        
        # 关键发现
        report.append("3.小结")
        
        # 最佳模型
        best_model = max(self.optimized_results.items(), key=lambda x: x[1]['accuracy'])
        report.append(f"(1)最佳模型: {best_model[0]}")
        report.append(f"  准确率: {best_model[1]['accuracy']:.4f}")
        report.append(f"  精准率: {best_model[1]['precision']:.4f}")
        report.append(f"  召回率: {best_model[1]['recall']:.4f}")
        
        # 达标模型
        models_above_90 = [name for name, r in self.optimized_results.items() 
                          if r['accuracy'] > 0.90]
        report.append(f"(2)达到0.90准确率的模型: {len(models_above_90)}个")
        for model in models_above_90:
            report.append(f"{model:<20} : {self.optimized_results[model]['accuracy']:.4f}")
        
        # 打印并保存
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        with open('优化对比报告.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n优化对比报告已保存: 优化对比报告.txt")
        print()
        
    # 运行完整分析流程
    def run_complete_pipeline(self):
        self.load_data()
        self.plot_feature_correlation()
        
        self.build_baseline_models()
        self.build_optimized_models()
        
        # 超参数搜索（运行一次后注释掉）
        print("(三)超参数搜索")
        # self.find_best_hyperparameters()
                
        print("(四)训练模型")
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
    classifier = LoanClassifierWithOptimization('Loan_approval_data_2025.csv')
    classifier.run_complete_pipeline()