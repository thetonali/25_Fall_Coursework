"""
波士顿房价数据回归分析
包含特征关系探索、线性回归和非线性回归模型的对比实验
本代码实现了作业的两个要求：
1. 针对波士顿房价数据，通过实验观测其他特征(单个和多个)和房价之间的关系。
2. 采用非线性回归模型来分析特征和房价之间的关系，并与线性模型进行对比。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

# 忽略不必要的警告，如Scikit-learn的未来版本警告
warnings.filterwarnings('ignore')
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # 支持图表中的中文标题和标签
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像时负号'-'显示为方块的问题

class BostonHousingAnalysis:
    """波士顿房价分析类"""
    
    def __init__(self):
        """初始化分析类，加载并整理波士顿房价数据集"""
        # 从原始数据源加载波士顿房价数据
        # 解决load_boston弃用问题
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        # 重新组织数据
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        
        # 特征名称
        self.feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                              'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        
        # 创建DataFrame
        self.df = pd.DataFrame(data, columns=self.feature_names)
        self.df['PRICE'] = target # 目标变量
        
        # 存储特征名用于后续访问
        self.data = type('obj', (object,), {
            'feature_names': self.feature_names,
            'data': data,
            'target': target
        })()
        
        # 初始化标准化器，用于后续多特征回归的特征缩放
        self.scaler = StandardScaler()
        
        print(f"数据集形状: {self.df.shape}")
        print(f"特征列: {self.feature_names}")
        print("\n特征说明:")
        print("CRIM    - 城镇人均犯罪率")
        print("ZN      - 住宅用地超过25000平方英尺的比例")
        print("INDUS   - 城镇非零售商用土地的比例")
        print("CHAS    - 查理斯河虚拟变量(=1如果邻近河流)")
        print("NOX     - 一氧化氮浓度")
        print("RM      - 住宅平均房间数")
        print("AGE     - 1940年之前建成的自用房屋比例")
        print("DIS     - 到波士顿五个中心区域的加权距离")
        print("RAD     - 辐射性公路的靠近指数")
        print("TAX     - 每10000美元的全值财产税率")
        print("PTRATIO - 城镇师生比例")
        print("B       - 1000(Bk-0.63)^2,其中Bk为黑人比例")
        print("LSTAT   - 人口中地位低下者的比例")
        print("PRICE   - 自住房的中位数报价(单位:千美元)")
        
    def exploratory_analysis(self):
        """
        1. 探索性数据分析
        用于观测特征之间的关系，回答作业要求1的“多个特征”部分。
        包括基本统计、缺失值检查和相关性分析。
        """
        print("1. 探索性数据分析及特征关系观测")
        # 查看数据集前五行以了解该数据集
        print("\n数据集前五行:")
        print(self.df.head())
        
        # 基本统计信息
        print("\n数据集基本统计信息:")
        print(self.df.describe())
        
        # 检查缺失值
        print("\n缺失值检查:")
        print(self.df.isnull().sum())
        
        # 相关性分析-计算所有特征与房价 (PRICE) 的相关系数
        print("\n各特征与房价的相关系数(ρ):")
        correlations = self.df.corr()['PRICE'].sort_values(ascending=False)
        print(correlations)
        
        # 绘制相关性热力图-用于观测特征之间及特征与目标变量之间的关系
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('特征相关性热力图', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('特征相关性热力图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n特征相关性热力图已保存为 '特征相关性热力图.png'")
        
        return correlations
    
    def single_feature_analysis(self, feature_name):
        """
        观测单个特征与房价的关系，并对比线性和非线性拟合效果，
        回答作业要求1的“单个特征”部分，并为要求2奠定基础。
        feature_name: 待分析的特征名称
        return:包含线性/多项式回归性能指标的字典
        """
        print(f"\n---分析特征'{feature_name}'与房价的关系---")
        
        X = self.df[[feature_name]].values
        y = self.df['PRICE'].values
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 1.线性回归 (基准)
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        
        # 2.多项式回归 (2次) - 尝试非线性拟合
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)
        y_pred_poly = poly_model.predict(X_test_poly)
        
        # 评估指标
        lr_r2 = r2_score(y_test, y_pred_lr)
        lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        poly_r2 = r2_score(y_test, y_pred_poly)
        poly_rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))
        
        print(f"线性回归 - R²: {lr_r2:.4f}, RMSE: {lr_rmse:.4f}")
        print(f"多项式回归 - R²: {poly_r2:.4f}, RMSE: {poly_rmse:.4f}")
        
        # 可视化散点图和拟合曲线
        plt.figure(figsize=(14, 5))
        
        # 散点图和线性回归拟合线
        plt.subplot(1, 2, 1)
        plt.scatter(X_test, y_test, alpha=0.5, s=10, label='实际值')
        
        # 排序用于绘制平滑曲线（必须排序）
        sort_idx = X_test.flatten().argsort()
        plt.plot(X_test[sort_idx], y_pred_lr[sort_idx], 'r-', 
                linewidth=2, label=f'线性回归 ($R^2$={lr_r2:.3f})')
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('房价 (PRICE)', fontsize=12)
        plt.title(f'{feature_name} 与房价关系 - 线性回归', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 多项式回归拟合线
        plt.subplot(1, 2, 2)
        plt.scatter(X_test, y_test, alpha=0.5, s=10, label='实际值')
        plt.plot(X_test[sort_idx], y_pred_poly[sort_idx], 'g-', 
                linewidth=2, label=f'2次多项式回归 ($R^2$={poly_r2:.3f})')
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('房价 (PRICE)', fontsize=12)
        plt.title(f'{feature_name} 与房价关系 - 2次多项式回归', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'单特征_{feature_name}分析图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"分析图已保存为 '单特征_{feature_name}分析图.png'")
        
        return {
            'feature': feature_name,
            'linear_r2': lr_r2,
            'linear_rmse': lr_rmse,
            'poly_r2': poly_r2,
            'poly_rmse': poly_rmse
        }
    
    def multiple_features_analysis(self):
        """
        2.多特征回归分析：对比线性和非线性模型的性能。
        回答作业要求2。
        包括：
        - 线性模型：Linear Regression, Ridge, Lasso
        - 非线性/复杂模型：Polynomial Regression (2次), Random Forest, Gradient Boosting, SVR (RBF)
        """
        print("\n" + "="*60)
        print("2.多特征回归分析:线性和非线性模型对比")
        
        # 准备数据
        X = self.df.drop('PRICE', axis=1).values
        y = self.df['PRICE'].values
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集(使用标准化后的数据)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 存储结果
        results = {}
        
        # --- 线性/正则化模型 ---
        # 1.线性回归
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        results['Linear Regression'] = self.evaluate_model(y_test, y_pred_lr)
        
        # 2.Ridge回归(L2正则化,用于减轻多重共线性影响)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        results['Ridge Regression'] = self.evaluate_model(y_test, y_pred_ridge)
        
        # 3.Lasso回归(L1正则化,用于特征选择和稀疏解)
        lasso = Lasso(alpha=0.1) # alpha调优后的值
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        results['Lasso Regression'] = self.evaluate_model(y_test, y_pred_lasso)
        
        # --- 非线性/复杂模型 ---
        # 4.多项式回归(2次) - 线性模型的非线性扩展
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        # 使用线性回归器拟合多项式特征
        poly_lr = LinearRegression()
        poly_lr.fit(X_train_poly, y_train)
        y_pred_poly = poly_lr.predict(X_test_poly)
        results['Polynomial Regression'] = self.evaluate_model(y_test, y_pred_poly)
        
        # 5.随机森林回归(集成学习,天然非线性)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                   random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        results['Random Forest'] = self.evaluate_model(y_test, y_pred_rf)
        
        # 6.梯度提升回归(集成学习,天然非线性)
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                       learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        results['Gradient Boosting'] = self.evaluate_model(y_test, y_pred_gb)
        
        # 7.支持向量回归SVR(非线性)
        svr = SVR(kernel='rbf', C=10, gamma='scale') # C, gamma为调优参数
        svr.fit(X_train, y_train)
        y_pred_svr = svr.predict(X_test)
        results['SVR (RBF)'] = self.evaluate_model(y_test, y_pred_svr)
        
        # 显示结果对比
        self.display_results(results)
        
        # 可视化预测结果对比
        self.visualize_predictions(y_test, {
            'Linear': y_pred_lr,
            'Polynomial': y_pred_poly,
            'Random Forest': y_pred_rf,
            'Gradient Boosting': y_pred_gb,
            'SVR': y_pred_svr
        })
        
        # 特征重要性分析(使用性能较好的非线性模型之一：随机森林)
        self.feature_importance_analysis(rf)
        return results
    
    def evaluate_model(self, y_true, y_pred):
        """计算并返回模型的关键性能指标R², RMSE, MAE, MAPE 的字典"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # 均方根误差(单位与目标变量一致)
        mae = mean_absolute_error(y_true, y_pred) # 平均绝对误差
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def display_results(self, results):
        """将多模型性能结果以表格和图表形式展示。"""
        print("模型性能对比")
        
        results_df = pd.DataFrame(results).T
        # 打印详细表格
        print(results_df.to_string())
        
        # 保存结果到CSV
        results_df.to_csv('模型性能对比数据.csv')
        print("\n结果已保存到 '模型性能对比数据.csv'")
        
        # 可视化对比(R², RMSE, MAE, MAPE)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 字典中的 key 用不含符号的形式
        metric_keys = ['R²', 'RMSE', 'MAE', 'MAPE']
        # 显示给图表用的名称
        metric_labels = ['$R^2$', 'RMSE', 'MAE', 'MAPE']
        
        for idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            # 从 results 中按 key 取值
            values = [results[model][mkey] for model in results.keys()]
            # 选最佳模型
            if mkey == 'R²':
                # R²越大越好
                best_model = max(results.keys(), key=lambda k: results[k]['R²'])
            else:
                # 其他指标越小越好
                best_model = min(results.keys(), key=lambda k: results[k][mkey])
            # 柱子颜色
            colors = ['steelblue'] * len(values)   
            # 突出显示最优模型
            for i, model in enumerate(results.keys()):
                if model == best_model:
                    colors[i] = 'darkorange'
                    
            bars = ax.bar(range(len(results)), values, color=colors, alpha=0.85)
            ax.set_xticks(range(len(results)))
            ax.set_xticklabels(results.keys(), rotation=45, ha='right')
            ax.set_ylabel(mlabel, fontsize=12)
            ax.set_title(f'模型 {mlabel} 对比', fontsize=14, pad=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上显示数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                # 针对R²使用不同的格式
                fmt = f'{height:.4f}' if mkey == 'R²' else f'{height:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       fmt,
                       ha='center', va='bottom', fontsize=9, rotation=0)
        
        plt.tight_layout()
        plt.savefig('模型性能对比图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("对比图已保存为 '模型性能对比图.png'")
    
    def visualize_predictions(self, y_test, predictions_dict):
        """可视化实际值与预测值的散点图"""
        print("\n---预测散点图(实际值与预测值)---")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
            ax = axes[idx]
            
            # 散点图:实际值与预测值
            ax.scatter(y_test, y_pred, alpha=0.5, s=10, color='steelblue')
            
            # 理想预测线(y=x,即预测值=实际值)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='理想预测(Y=X)')
            
            # 计算R²
            r2 = r2_score(y_test, y_pred)
            
            ax.set_xlabel('实际房价(千美元)', fontsize=11)
            ax.set_ylabel('预测房价(千美元)', fontsize=11)
            ax.set_title(f'{model_name}预测结果\n($R^2$ = {r2:.4f})', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(predictions_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('预测结果对比图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("预测对比图已保存为 '预测结果对比图.png'")
    
    def feature_importance_analysis(self, rf_model):
        """基于随机森林模型 (一种非线性模型) 的特征重要性分析"""
        print("="*45)
        print("特征重要性分析(基于随机森林)")
        
        # 获取特征重要性
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 打印特征重要性
        print("特征重要性排序:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {self.data.feature_names[idx]}: {importances[idx]:.4f}")
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances[indices], 
                color='steelblue', alpha=0.7)
        plt.xticks(range(len(importances)), 
                  [self.data.feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('特征', fontsize=12)
        plt.ylabel('重要性', fontsize=12)
        plt.title('特征重要性分析 (随机森林)', fontsize=14, pad=15)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('特征重要性图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n特征重要性图已保存为 '特征重要性图.png'")
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("\n" + "="*80)
        print(" "*20 + "波士顿房价数据回归分析")
        
        # 1.探索性数据分析(对应作业要求1)
        correlations = self.exploratory_analysis()
        
        # 2.单特征分析(选择相关性最强的5个特征)
        print("\n" + "="*50)
        print("单特征与房价关系分析(对比线性和非线性拟合)")
        
        # 选择与房价相关性最强（绝对值最大）的5个特征
        corr_excluding_price = correlations.drop('PRICE')
        top_features = corr_excluding_price.abs().sort_values(ascending=False).head(5).index.tolist()
        
        single_results = []
        for feature in top_features:
            result = self.single_feature_analysis(feature)
            single_results.append(result)
        
        # 显示单特征分析结果
        print("\n单特征分析结果汇总:")
        single_df = pd.DataFrame(single_results)
        # 重新排序以便展示
        single_df = single_df.sort_values(by='poly_r2', ascending=False)
        print(single_df.to_string(index=False, float_format='%.4f'))
        single_df.to_csv('单特征分析数据.csv', index=False)
        
        # 3.多特征分析(对应作业要求2：非线性回归模型对比)
        results = self.multiple_features_analysis()
        
        print("分析完成! 所有结果和图表已保存。")
        
        return results


# 主程序入口
if __name__ == "__main__":
    # 创建分析对象
    analyzer = BostonHousingAnalysis()
    
    # 运行完整分析
    results = analyzer.run_full_analysis()