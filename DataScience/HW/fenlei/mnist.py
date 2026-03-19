"""
手写体数据集分类对比实验
使用SVM和CNN两种分类器对MNIST和Fashion-MNIST数据集进行分类比较
"""
# 0.导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1.数据加载与预处理
def load_and_preprocess_data(dataset_name='mnist'):
    """
    加载并预处理数据集(MNIST或Fashion-MNIST)
    返回-训练集和测试集的数据和标签
    """
    print(f"\n正在加载 {dataset_name.upper()} 数据集——")
    
    if dataset_name == 'mnist':
        # MNIST:0-9手写数字
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        class_names = [str(i) for i in range(10)]
    elif dataset_name == 'fashion_mnist':
        # Fashion-MNIST:10种服装类别
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        raise ValueError("数据集名称必须是'mnist'或'fashion_mnist'")
    
    # 归一化-将像素值从[0,255]缩放到[0,1]区间
    # 有助于加速神经网络收敛，并提高SVM的性能
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test), class_names


def visualize_samples(x_data, y_data, class_names, dataset_name, n_samples=10):
    """
    可视化数据集样本
    参数依次为：图像数据,标签数据,类别名称列表,数据集名称,显示样本数量
    """
    plt.figure(figsize=(15, 3))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(x_data[i], cmap='gray')
        plt.title(f'{class_names[y_data[i]]}')
        plt.axis('off')
    plt.suptitle(f'{dataset_name.upper()} 数据集样本', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2.支持向量机(SVM)分类器
def train_svm(x_train, y_train, x_test, y_test, dataset_name, sample_size=10000):
    """
    训练并评估SVM分类器
    返回-训练好的SVM模型和性能指标
    """
    print(f"\n{'='*50}")
    print(f"SVM分类器 - {dataset_name.upper()}")
    
    # 1.将图像数据展平为一维向量 (28x28->784)
    # SVM是线性模型(即使使用RBF核,其输入也是一维向量)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # 2.采样:SVM计算复杂度高,完整训练60k样本耗时过长,因此进行随机采样
    if x_train_flat.shape[0] > sample_size:
        indices = np.random.choice(x_train_flat.shape[0], sample_size, replace=False)
        x_train_sampled = x_train_flat[indices]
        y_train_sampled = y_train[indices]
        print(f"注意: SVM训练已采样 {sample_size} 个样本以加快速度。")
    else:
        x_train_sampled = x_train_flat
        y_train_sampled = y_train
    
    # 3.标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_sampled) #!如果训练完整数据集这里的参数为x_train_flat
    x_test_scaled = scaler.transform(x_test_flat)
    
    # 4.训练模型:使用RBF核(非线性核函数)
    svm_model = svm.SVC(kernel='rbf', C=5, gamma='scale', verbose=False) # 关闭verbose减少输出
    # 记录训练时间
    start_time = time.time()
    print(f"开始训练SVM (N={len(y_train_sampled)})——")
    svm_model.fit(x_train_scaled, y_train_sampled)
    train_time = time.time() - start_time
    print(f"训练完成! 耗时: {train_time:.2f}秒")
    
    # 5.预测与评估
    start_time = time.time()
    y_pred = svm_model.predict(x_test_scaled)
    test_time = time.time() - start_time
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"预测时间: {test_time:.2f}秒")
    
    # 返回结果
    results = {
        'model': svm_model,
        'accuracy': accuracy,
        'train_time': train_time,
        'test_time': test_time,
        'y_pred': y_pred,
        'scaler': scaler,
        'description': f'SVM (RBF, C=5, Trained on {len(y_train_sampled)} samples)'
    }
    
    return results

# 3.卷积神经网络(CNN)分类器
# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        # 1.第一层卷积层：32个3x3卷积核-提取低级特征
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),  # 批标准化-加速收敛,提高稳定性
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),  # 最大池化-降维,保留最重要特征
        layers.Dropout(0.25),  # Dropout防止过拟合
        
        # 2.第二层卷积层：64个3x3卷积核-提取更复杂的特征
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # 3.第三层卷积层：128个3x3卷积核
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # 4.分类器块
        layers.Flatten(), # 展平:将3D特征图转换为1D向量
        
        # 全连接层
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax') # Softmax用于多分类概率输出
    ])
    
    # 编译模型-使用Adam优化器和稀疏交叉熵损失
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 训练CNN分类器
def train_cnn(x_train, y_train, x_test, y_test, dataset_name, epochs=15):
    print(f"\n{'='*50}")
    
    # 1.添加通道维度(28,28) -> (28,28,1)
    # CNN需要(高,宽,通道数)的输入
    x_train_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # 2.构建CNN模型
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    
    # 3.设置回调函数
    callbacks = [
        # 早停：如果连续5轮验证准确率不再提升，提前停止训练并恢复最佳权重
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, 
                                     restore_best_weights=True),
        # 学习率衰减：当验证损失连续3轮不再下降时，将学习率减半
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                         patience=3, min_lr=1e-7)
    ]
    
    # 4.训练模型
    print(f"\n开始训练CNN ({epochs} 轮)...")
    start_time = time.time()
    history = model.fit(
        x_train_cnn, y_train,
        batch_size=128,
        epochs=epochs,
        validation_split=0.1,  # 使用10%的训练数据作为验证集
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {train_time:.2f}秒")
    
    # 5.评估与预测
    start_time = time.time()
    test_loss, test_accuracy = model.evaluate(x_test_cnn, y_test, verbose=0)
    test_time = time.time() - start_time
    # 预测
    y_pred_probs = model.predict(x_test_cnn, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1) # 将概率转换为类别标签
    
    print(f"\n测试准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"测试损失: {test_loss:.4f}")
    print(f"预测时间: {test_time:.2f}秒")
    
    # 返回结果
    results = {
        'model': model,
        'history': history,
        'accuracy': test_accuracy,
        'loss': test_loss,
        'train_time': train_time,
        'test_time': test_time,
        'y_pred': y_pred,
        'description': 'CNN (Deep ConvNet)'
    }
    return results

# 4.跨数据集测试(迁移性分析)
def cross_dataset_test(trained_model, x_test_cross, y_test_cross, 
                       trained_on_name, test_on_name):
    """
    使用在一个数据集上训练好的CNN模型，在另一个数据集上进行测试，
    分析模型的特征迁移能力。
    """
    print(f"\n{'='*50}")
    print(f"跨数据集测试: {trained_on_name} (训练) -> {test_on_name} (测试)")
    
    # 1.数据形状调整
    x_test_cnn = x_test_cross.reshape(x_test_cross.shape[0], 28, 28, 1)
    
    # 2.预测
    start_time = time.time()
    y_pred_probs = trained_model.predict(x_test_cnn, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    test_time = time.time() - start_time
    
    # 3.评估
    accuracy = accuracy_score(y_test_cross, y_pred)
    
    print(f"\n准确率 (迁移性测试): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"预测时间: {test_time:.2f}秒")
    
    # 返回结果
    results = {
        'accuracy': accuracy,
        'test_time': test_time,
        'y_pred': y_pred,
        'description': f'CNN (Trained on {trained_on_name}, Tested on {test_on_name})'
    }
    return results

# 5.结果可视化与分析
# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# 绘制CNN训练历史曲线
def plot_training_history(history, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率曲线
    axes[0].plot(history.history['accuracy'], label='训练准确率')
    axes[0].plot(history.history['val_accuracy'], label='验证准确率')
    axes[0].set_title(f'CNN模型准确率 - {dataset_name.upper()}', fontsize=14)
    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=12)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    axes[1].plot(history.history['loss'], label='训练损失')
    axes[1].plot(history.history['val_loss'], label='验证损失')
    axes[1].set_title(f'CNN模型损失 - {dataset_name.upper()}', fontsize=14)
    axes[1].set_xlabel('训练轮次 (Epoch)', fontsize=12)
    axes[1].set_ylabel('损失', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_cnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# 对比两种分类器的性能
def compare_classifiers(svm_results, cnn_results, dataset_name):
    # 创建对比表格
    comparison_data = {
        '指标': ['准确率 (%)', '训练时间 (秒)', '测试时间 (秒)'],
        'SVM': [
            f"{svm_results['accuracy']*100:.2f}",
            f"{svm_results['train_time']:.2f}",
            f"{svm_results['test_time']:.2f}"
        ],
        'CNN': [
            f"{cnn_results['accuracy']*100:.2f}",
            f"{cnn_results['train_time']:.2f}",
            f"{cnn_results['test_time']:.2f}"
        ]
    }
    
    print(f"\n{'='*50}")
    print(f"性能对比 - {dataset_name.upper()}")
    print(f"{'='*50}")
    print(f"{'指标':<20} {'SVM':<15} {'CNN':<15}")
    print("-" * 50)
    for i in range(len(comparison_data['指标'])):
        print(f"{comparison_data['指标'][i]:<20} "
              f"{comparison_data['SVM'][i]:<15} "
              f"{comparison_data['CNN'][i]:<15}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率对比
    classifiers = ['SVM', 'CNN']
    accuracies = [svm_results['accuracy']*100, cnn_results['accuracy']*100]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].bar(classifiers, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('准确率 (%)', fontsize=12)
    axes[0].set_title(f'分类器准确率对比 - {dataset_name.upper()}', fontsize=14)
    axes[0].set_ylim([min(accuracies)-5, 100])
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 训练时间对比
    train_times = [svm_results['train_time'], cnn_results['train_time']]
    axes[1].bar(classifiers, train_times, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('训练时间 (秒)', fontsize=12)
    axes[1].set_title(f'训练时间对比 - {dataset_name.upper()}', fontsize=14)
    for i, v in enumerate(train_times):
        axes[1].text(i, v + max(train_times)*0.02, f'{v:.2f}s', 
                    ha='center', va='bottom', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 打印详细分类报告
def print_classification_report(y_true, y_pred, class_names, title):
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 6.主程序
def main():
    print("="*60)
    print("手写体数据集分类对比实验")
    print("SVM vs CNN")
    print("="*60)
    
    # 存储所有结果
    all_results = {}
    
    # 阶段1:加载并处理数据
    # 原始数据集
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test), mnist_class_names = \
        load_and_preprocess_data('mnist')
    # 跨领域数据集
    (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test), fashion_class_names = \
        load_and_preprocess_data('fashion_mnist')
    
    # 阶段2:MNIST数据集上的分类器比较(作业要求1)
    print(f"\n\n{'#'*60}")
    print("# 实验组1:MNIST(手写数字)分类器比较")
    # 2.1训练和评估SVM
    svm_mnist_results = train_svm(mnist_x_train, mnist_y_train, 
                                  mnist_x_test, mnist_y_test, 'mnist', 
                                  sample_size=10000) # 使用10000样本
    all_results['svm_mnist'] = svm_mnist_results
    
    # 2.2训练和评估CNN
    cnn_mnist_results = train_cnn(mnist_x_train, mnist_y_train, 
                                  mnist_x_test, mnist_y_test, 'mnist', epochs=15)
    all_results['cnn_mnist'] = cnn_mnist_results
    
    # 2.3可视化和报告
    plot_training_history(cnn_mnist_results['history'], 'mnist')
    compare_classifiers(svm_mnist_results, cnn_mnist_results, 'mnist')
    
    print_classification_report(mnist_y_test, svm_mnist_results['y_pred'], 
                                mnist_class_names, 'SVM分类报告 - MNIST')
    plot_confusion_matrix(mnist_y_test, svm_mnist_results['y_pred'], mnist_class_names,
                          'SVM混淆矩阵 - MNIST', 'mnist_svm_confusion_matrix.png')
    
    print_classification_report(mnist_y_test, cnn_mnist_results['y_pred'], 
                                mnist_class_names, 'CNN分类报告 - MNIST')
    plot_confusion_matrix(mnist_y_test, cnn_mnist_results['y_pred'], mnist_class_names,
                          'CNN混淆矩阵 - MNIST', 'mnist_cnn_confusion_matrix.png')
    
    # 阶段3:Fashion-MNIST数据集上的分类器训练(作业要求2的独立训练)
    print(f"\n\n{'#'*60}")
    print("# 实验组2:Fashion-MNIST(服装)独立训练")
    # 3.1训练和评估SVM(在Fashion-MNIST上)
    svm_fashion_results = train_svm(fashion_x_train, fashion_y_train, 
                                    fashion_x_test, fashion_y_test, 'fashion_mnist', 
                                    sample_size=10000) # 使用10000样本
    all_results['svm_fashion'] = svm_fashion_results
    
    # 3.2训练和评估CNN(在Fashion-MNIST上)
    cnn_fashion_results = train_cnn(fashion_x_train, fashion_y_train, 
                                    fashion_x_test, fashion_y_test, 'fashion_mnist', epochs=15)
    all_results['cnn_fashion'] = cnn_fashion_results
    
    # 3.3可视化和报告
    plot_training_history(cnn_fashion_results['history'], 'fashion_mnist')
    compare_classifiers(svm_fashion_results, cnn_fashion_results, 'fashion_mnist')
    
    print_classification_report(mnist_y_test, svm_mnist_results['y_pred'], 
                                mnist_class_names, 'SVM分类报告 - Fashion-MNIST')
    plot_confusion_matrix(mnist_y_test, svm_mnist_results['y_pred'], mnist_class_names,
                          'SVM混淆矩阵 - Fashion-MNIST', 'fashion_mnist_svm_confusion_matrix.png')
    
    print_classification_report(fashion_y_test, cnn_fashion_results['y_pred'], 
                                fashion_class_names, 'CNN分类报告 - Fashion-MNIST')
    plot_confusion_matrix(fashion_y_test, cnn_fashion_results['y_pred'], fashion_class_names,
                          'CNN混淆矩阵 - Fashion-MNIST', 'fashion_mnist_cnn_confusion_matrix.png')
                          
    # 阶段4:跨数据集迁移性测试(作业要求2的迁移分析)
    # 使用在MNIST上训练好的CNN模型
    cross_diff_results = cross_dataset_test(
        cnn_mnist_results['model'], 
        fashion_x_test, fashion_y_test, 
        'MNIST', 'Fashion-MNIST (跨领域)'
    )
    all_results['cnn_cross_diff'] = cross_diff_results
    print_classification_report(fashion_y_test, cross_diff_results['y_pred'], 
                                fashion_class_names, '跨数据集 CNN 分类报告 (MNIST -> Fashion-MNIST)')
    plot_confusion_matrix(fashion_y_test, cross_diff_results['y_pred'], fashion_class_names,
                          '跨数据集CNN混淆矩阵 (MNIST -> F-MNIST)', 'cross_diff_confusion_matrix.png')
    
    # 阶段5:最终总结
    print("\n\n" + "="*60)
    print("实验结果总结——")
    
    final_summary = {
        'SVM (MNIST)': f"{all_results['svm_mnist']['accuracy']*100:.2f}% (T: {all_results['svm_mnist']['train_time']:.2f}s)",
        'CNN (MNIST)': f"{all_results['cnn_mnist']['accuracy']*100:.2f}% (T: {all_results['cnn_mnist']['train_time']:.2f}s)",
        'SVM (F-MNIST)': f"{all_results['svm_fashion']['accuracy']*100:.2f}% (T: {all_results['svm_fashion']['train_time']:.2f}s)",
        'CNN (F-MNIST)': f"{all_results['cnn_fashion']['accuracy']*100:.2f}% (T: {all_results['cnn_fashion']['train_time']:.2f}s)",
        'CNN 跨领域迁移 (MNIST->F-MNIST)': f"{all_results['cnn_cross_diff']['accuracy']*100:.2f}%",
    }
    
    for k, v in final_summary.items():
        print(f" - {k:<30}: {v}")
    
    print("\n实验完成! 所有结果已收集，图表已保存到当前目录。")


if __name__ == "__main__":
    main()