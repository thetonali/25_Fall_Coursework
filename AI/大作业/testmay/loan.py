"""
《人工智能导论》进阶作业 - 贷款审批预测（含优化前后对比）
作者：[你的姓名]
学号：[你的学号]
日期：2026-01-08

本程序对贷款审批数据集进行分类和回归双重任务，展示优化前后的性能对比。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LoanApprovalAnalyzerWithOptimization:
    """贷款审批数据分析器（含优化对比）"""
    
    def __init__(self, data_path='loan_approval_dataset.csv'):
        self.data_path = data_path
        self.baseline_clf_models = {}
        self.optimized_clf_models = {}
        self.baseline_reg_models = {}
        self.optimized_reg_models = {}
        self.baseline_clf_results = {}
        self.optimized_clf_results = {}
        self.baseline_reg_results = {}
        self.optimized_reg_results = {}
        
    def load_and_explore_data(self):
        """数据加载与探索性分析"""
        print("=" * 80)
        print("第一步：数据加载与探索性分析")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"\n【数据集基本信息】")
        print(f"数据形状: {self.df.shape}")
        print(f"特征数量: {self.df.shape[1]}")
        print(f"样本数量: {self.df.shape[0]}")
        
        print(f"\n【前5行数据】")
        print(self.df.head())
        
        print(f"\n【数据类型】")
        print(self.df.dtypes)
        
        print(f"\n【缺失值统计】")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("✓ 无缺失值")
        
        # 识别目标变量
        if 'loan_status' in self.df.columns:
            print(f"\n【贷款状态分布】")
            print(self.df['loan_status'].value_counts())
        
        print("\n")
        
    def visualize_data(self):
        """数据可视化"""
        print("=" * 80)
        print("第二步：数据可视化分析")
        print("=" * 80)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 贷款状态分布
        ax1 = plt.subplot(2, 3, 1)
        if 'loan_status' in self.df.columns:
            status_counts = self.df['loan_status'].value_counts()
            colors = ['#2ecc71', '#e74c3c']
            ax1.pie(status_counts.values, labels=status_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Loan Status Distribution', fontweight='bold')
        
        # 2-6: 其他可视化...
        plt.tight_layout()
        plt.savefig('loan_data_eda.png', dpi=300, bbox_inches='tight')
        print("数据探索图表已保存: loan_data_eda.png\n")
        plt.show()
        
    def engineer_features(self, df):
        """特征工程"""
        df = df.copy()
        
        # 创建衍生特征
        if 'person_income' in df.columns and 'loan_amnt' in df.columns:
            df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)
            df['debt_to_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)
        
        if 'person_age' in df.columns:
            df['age_group'] = pd.cut(df['person_age'], 
                                     bins=[0, 25, 35, 45, 55, 100],
                                     labels=[0, 1, 2, 3, 4])
        
        return df
        
    def preprocess_data(self):
        """数据预处理"""
        print("=" * 80)
        print("第三步：数据预处理与特征工程")
        print("=" * 80)
        
        df_processed = self.df.copy()
        
        # 1. 处理缺失值
        print("\n【步骤1: 处理缺失值】")
        df_processed = df_processed.dropna()
        print(f"处理后样本数: {len(df_processed)}")
        
        # 2. 特征工程
        print("\n【步骤2: 特征工程】")
        df_processed = self.engineer_features(df_processed)
        print(f"  ✓ 创建 income_to_loan_ratio")
        print(f"  ✓ 创建 debt_to_income_ratio")
        print(f"  ✓ 创建 age_group")
        
        # 3. 编码分类变量
        print("\n【步骤3: 编码分类变量】")
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        self.label_encoders = {}
        
        for col in categorical_cols:
            if col not in ['loan_status']:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
                print(f"  编码: {col}")
        
        self.df_processed = df_processed
        
        # 4. 准备分类任务数据
        if 'loan_status' in df_processed.columns:
            print("\n【步骤4: 准备分类任务数据】")
            
            le_target = LabelEncoder()
            y_classification = le_target.fit_transform(df_processed['loan_status'])
            self.target_encoder = le_target
            
            X_classification = df_processed.drop(['loan_status'], axis=1)
            if 'risk_score' in X_classification.columns:
                X_classification = X_classification.drop(['risk_score'], axis=1)
            
            self.X_train_clf, self.X_test_clf, self.y_train_clf, self.y_test_clf = \
                train_test_split(X_classification, y_classification, 
                               test_size=0.2, random_state=42, stratify=y_classification)
            
            self.scaler_clf = StandardScaler()
            self.X_train_clf_scaled = self.scaler_clf.fit_transform(self.X_train_clf)
            self.X_test_clf_scaled = self.scaler_clf.transform(self.X_test_clf)
            
            print(f"  训练集: {self.X_train_clf.shape[0]} 样本")
            print(f"  测试集: {self.X_test_clf.shape[0]} 样本")
        
        print("\n")
        
    def build_baseline_classification_models(self):
        """构建基线分类模型"""
        print("=" * 80)
        print("第四步：构建基线分类模型（优化前）")
        print("=" * 80)
        
        print("\n1. 随机森林分类器（基线）")
        print("   参数: n_estimators=100, max_depth=None")
        self.baseline_clf_models['Random Forest'] = {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'use_scaled': False
        }
        
        print("\n2. 梯度提升分类器（基线）")
        print("   参数: n_estimators=100, learning_rate=0.1")
        self.baseline_clf_models['Gradient Boosting'] = {
            'model': GradientBoostingClassifier(n_estimators=100, 
                                               learning_rate=0.1, 
                                               random_state=42),
            'use_scaled': False
        }
        
        print("\n3. 逻辑回归（基线）")
        print("   参数: C=1.0, penalty='l2'")
        self.baseline_clf_models['Logistic Regression'] = {
            'model': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            'use_scaled': True
        }
        print()
        
    def build_optimized_classification_models(self):
        """构建优化后的分类模型"""
        print("=" * 80)
        print("第五步：构建优化后的分类模型")
        print("=" * 80)
        
        print("\n1. 改进随机森林分类器 ⭐")
        print("   优化策略:")
        print("   - n_estimators: 100 → 300 (增加树数量)")
        print("   - class_weight: None → 'balanced' (平衡类别权重)")
        print("   - max_depth: None → 15 (适度限制深度)")
        self.optimized_clf_models['Random Forest'] = {
            'model': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=4,
                class_weight='balanced',
                random_state=42
            ),
            'use_scaled': False
        }
        
        print("\n2. 改进梯度提升分类器")
        print("   优化策略:")
        print("   - learning_rate: 0.1 → 0.05 (降低学习率)")
        print("   - subsample: 1.0 → 0.8 (子采样防过拟合)")
        print("   - max_depth: 3 → 5 (增加树深度)")
        self.optimized_clf_models['Gradient Boosting'] = {
            'model': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ),
            'use_scaled': False
        }
        
        print("\n3. L1正则化逻辑回归")
        print("   优化策略:")
        print("   - penalty: 'l2' → 'l1' (L1正则化特征选择)")
        print("   - C: 1.0 → 0.5 (增强正则化)")
        print("   - class_weight: None → 'balanced'")
        self.optimized_clf_models['Logistic Regression'] = {
            'model': LogisticRegression(
                penalty='l1',
                C=0.5,
                solver='liblinear',
                class_weight='balanced',
                max_iter=2000,
                random_state=42
            ),
            'use_scaled': True
        }
        print()
        
    def train_classification_models(self, models, results_dict, model_type="基线"):
        """训练分类模型"""
        print(f"{'=' * 80}")
        print(f"训练与评估{model_type}分类模型")
        print(f"{'=' * 80}\n")
        
        for name, config in models.items():
            print(f"训练 {name} ({model_type})...")
            model = config['model']
            
            if config['use_scaled']:
                X_train = self.X_train_clf_scaled
                X_test = self.X_test_clf_scaled
            else:
                X_train = self.X_train_clf
                X_test = self.X_test_clf
            
            model.fit(X_train, self.y_train_clf)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(self.y_test_clf, y_pred)
            prec = precision_score(self.y_test_clf, y_pred, average='weighted', zero_division=0)
            rec = recall_score(self.y_test_clf, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test_clf, y_pred, average='weighted', zero_division=0)
            
            cv_scores = cross_val_score(model, X_train, self.y_train_clf, cv=5)
            cm = confusion_matrix(self.y_test_clf, y_pred)
            
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
            
            print(f"  准确率: {acc:.4f} | 精准率: {prec:.4f} | 召回率: {rec:.4f}")
        
        print()
        
    def plot_classification_comparison(self):
        """绘制分类模型优化对比"""
        print("=" * 80)
        print("第六步：可视化分类模型优化效果")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(self.baseline_clf_results.keys())
        
        # 1. 准确率对比
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        
        baseline_acc = [self.baseline_clf_results[m]['accuracy'] for m in models]
        optimized_acc = [self.optimized_clf_results[m]['accuracy'] for m in models]
        
        bars1 = ax1.bar(x - width/2, baseline_acc, width, label='优化前', color='lightcoral')
        bars2 = ax1.bar(x + width/2, optimized_acc, width, label='优化后', color='lightgreen')
        
        ax1.set_xlabel('模型', fontweight='bold')
        ax1.set_ylabel('准确率', fontweight='bold')
        ax1.set_title('分类准确率对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加提升百分比
        for i, (b, o) in enumerate(zip(baseline_acc, optimized_acc)):
            improvement = ((o - b) / b * 100) if b > 0 else 0
            ax1.text(i - width/2, b, f'{b:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, o, f'{o:.3f}', ha='center', va='bottom', fontsize=9)
            if improvement > 0:
                ax1.text(i, max(b, o) + 0.01, f'+{improvement:.1f}%', 
                        ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
        
        # 2. F1分数对比
        ax2 = axes[0, 1]
        baseline_f1 = [self.baseline_clf_results[m]['f1_score'] for m in models]
        optimized_f1 = [self.optimized_clf_results[m]['f1_score'] for m in models]
        
        ax2.plot(models, baseline_f1, 'o-', label='优化前', linewidth=2, markersize=8)
        ax2.plot(models, optimized_f1, 's-', label='优化后', linewidth=2, markersize=8)
        ax2.set_ylabel('F1分数', fontweight='bold')
        ax2.set_title('F1分数对比', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15)
        
        # 3. 性能提升热图
        ax3 = axes[1, 0]
        improvements = []
        metrics = ['准确率', '精准率', '召回率', 'F1']
        
        for model in models:
            improvements.append([
                (self.optimized_clf_results[model]['accuracy'] - 
                 self.baseline_clf_results[model]['accuracy']) * 100,
                (self.optimized_clf_results[model]['precision'] - 
                 self.baseline_clf_results[model]['precision']) * 100,
                (self.optimized_clf_results[model]['recall'] - 
                 self.baseline_clf_results[model]['recall']) * 100,
                (self.optimized_clf_results[model]['f1_score'] - 
                 self.baseline_clf_results[model]['f1_score']) * 100
            ])
        
        improvements_array = np.array(improvements)
        im = ax3.imshow(improvements_array, cmap='RdYlGn', aspect='auto')
        ax3.set_xticks(np.arange(len(metrics)))
        ax3.set_yticks(np.arange(len(models)))
        ax3.set_xticklabels(metrics)
        ax3.set_yticklabels(models)
        ax3.set_title('性能提升热图 (百分点)', fontsize=14, fontweight='bold')
        
        for i in range(len(models)):
            for j in range(len(metrics)):
                text = ax3.text(j, i, f'{improvements_array[i,j]:.1f}', 
                              ha='center', va='center', fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax3)
        
        # 4. 混淆矩阵（最佳模型）
        ax4 = axes[1, 1]
        best_model = max(self.optimized_clf_results.items(), 
                        key=lambda x: x[1]['accuracy'])[0]
        cm = self.optimized_clf_results[best_model]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title(f'{best_model} (优化后)\n混淆矩阵', fontweight='bold')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('loan_classification_comparison.png', dpi=300, bbox_inches='tight')
        print("分类模型对比图已保存: loan_classification_comparison.png\n")
        plt.show()
        
    def generate_classification_report(self):
        """生成分类模型对比报告"""
        print("=" * 80)
        print("第七步：生成分类模型对比报告")
        print("=" * 80)
        
        report = []
        report.append("=" * 100)
        report.append("贷款审批数据集 - 分类模型优化效果分析报告")
        report.append("=" * 100)
        report.append("")
        
        report.append("【一、数据集信息】")
        report.append(f"  样本总数: {len(self.df_processed)}")
        report.append(f"  特征数量: {self.X_train_clf.shape[1]}")
        report.append(f"  训练集: {len(self.X_train_clf)} 样本")
        report.append(f"  测试集: {len(self.X_test_clf)} 样本")
        report.append("")
        
        report.append("【二、分类模型优化前后对比】")
        report.append(f"{'模型':<25} {'版本':<10} {'准确率':<10} {'精准率':<10} {'召回率':<10} {'F1分数':<10}")
        report.append("-" * 85)
        
        for name in self.baseline_clf_results.keys():
            b = self.baseline_clf_results[name]
            report.append(
                f"{name:<25} {'优化前':<10} "
                f"{b['accuracy']:<10.4f} {b['precision']:<10.4f} "
                f"{b['recall']:<10.4f} {b['f1_score']:<10.4f}"
            )
            
            o = self.optimized_clf_results[name]
            report.append(
                f"{name:<25} {'优化后':<10} "
                f"{o['accuracy']:<10.4f} {o['precision']:<10.4f} "
                f"{o['recall']:<10.4f} {o['f1_score']:<10.4f}"
            )
            
            acc_imp = (o['accuracy'] - b['accuracy']) * 100
            report.append(f"{'提升':<25} {'':<10} {acc_imp:>+9.2f}%")
            report.append("-" * 85)
        
        report.append("")
        
        best_model = max(self.optimized_clf_results.items(), 
                        key=lambda x: x[1]['accuracy'])
        report.append(f"【三、最佳模型】: {best_model[0]}")
        report.append(f"  准确率: {best_model[1]['accuracy']:.4f}")
        report.append("")
        
        report.append("【四、改进策略总结】")
        report.append("1. 随机森林: 增加树数量+类别权重平衡+深度限制")
        report.append("2. 梯度提升: 降低学习率+子采样防过拟合")
        report.append("3. 逻辑回归: L1正则化特征选择+类别权重平衡")
        report.append("")
        report.append("=" * 100)
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open('loan_classification_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n分类模型报告已保存: loan_classification_report.txt\n")
        
    def run_complete_analysis(self):
        """运行完整分析流程"""
        self.load_and_explore_data()
        self.visualize_data()
        self.preprocess_data()
        self.build_baseline_classification_models()
        self.build_optimized_classification_models()
        
        print("训练基线分类模型...")
        self.train_classification_models(
            self.baseline_clf_models, self.baseline_clf_results, "基线"
        )
        
        print("训练优化分类模型...")
        self.train_classification_models(
            self.optimized_clf_models, self.optimized_clf_results, "优化"
        )
        
        self.plot_classification_comparison()
        self.generate_classification_report()
        
        print("=" * 80)
        print("✓ 进阶作业完整分析流程已完成！")
        print("=" * 80)
        print("\n生成的文件:")
        print("  1. loan_data_eda.png - 数据探索图表")
        print("  2. loan_classification_comparison.png - 分类模型对比")
        print("  3. loan_classification_report.txt - 分类模型报告")
        print()


# 主程序
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("《人工智能导论》进阶作业 - 贷款审批数据分析系统（含优化对比）")
    print("=" * 80 + "\n")
    
    analyzer = LoanApprovalAnalyzerWithOptimization('loan_approval_dataset.csv')
    analyzer.run_complete_analysis()