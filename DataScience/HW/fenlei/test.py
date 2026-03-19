# ==============================================================================
# 0. 导入必要的库
# ==============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from time import time

# 设置随机种子以保证结果的可复现性
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. 数据集加载与预处理
# ==============================================================================

def load_and_preprocess_data(dataset_name='mnist'):
    """加载和预处理MNIST或Fashion-MNIST数据集"""
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print("--- 加载 MNIST 数据集 ---")
    elif dataset_name == 'fashion_mnist':
        # Fashion-MNIST 数据集作为替代数据集
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        print("--- 加载 Fashion-MNIST 数据集 ---")
    else:
        raise ValueError("数据集名称错误，请使用 'mnist' 或 'fashion_mnist'")

    # 归一化：将像素值从 [0, 255] 缩放到 [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 获取图像尺寸和类别数
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    num_classes = len(np.unique(y_train))

    # 记录原始形状，便于后续处理
    print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")

    return (x_train, y_train), (x_test, y_test), img_rows, img_cols, num_classes

# 加载主数据集 (MNIST)
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test), \
    img_rows, img_cols, num_classes = load_and_preprocess_data('mnist')

# 加载替代数据集 (Fashion-MNIST)
(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test), \
    _, _, _ = load_and_preprocess_data('fashion_mnist')


# ==============================================================================
# 2. 支持向量机 (SVM) 分类器实现
# ==============================================================================

def run_svm_classifier(x_train, y_train, x_test, y_test, max_samples=10000):
    """
    运行基于 Scikit-learn 的 SVM 分类器。
    由于 SVM 训练时间较长，这里使用较小的子集进行演示。
    """
    print("\n" + "="*50)
    print("      2. 运行 支持向量机 (SVM) 分类器      ")
    print("="*50)

    # SVM 对高维数据计算量大，因此需要将图像展平并可能采样
    
    # 1. 展平图像数据 (从 (N, H, W) -> (N, H*W))
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # 2. 采样：为了加快训练速度，仅使用部分样本
    if x_train_flat.shape[0] > max_samples:
        sample_indices = np.random.choice(x_train_flat.shape[0], max_samples, replace=False)
        x_train_sampled = x_train_flat[sample_indices]
        y_train_sampled = y_train[sample_indices]
    else:
        x_train_sampled = x_train_flat
        y_train_sampled = y_train

    print(f"SVM 训练集大小: {x_train_sampled.shape[0]} 样本")
    
    # 3. 特征缩放 (标准化)：对于 SVM 提升性能至关重要
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_sampled)
    x_test_scaled = scaler.transform(x_test_flat)

    # 4. 实例化和训练 SVM 模型
    # 选择 RBF (高斯核) 是处理非线性分类问题的常用选择
    model = SVC(kernel='rbf', gamma='scale', random_state=42, verbose=False)
    
    start_time = time()
    print("开始训练 SVM...")
    model.fit(x_train_scaled, y_train_sampled)
    train_time = time() - start_time
    print(f"SVM 训练完成，耗时: {train_time:.2f} 秒")

    # 5. 预测与评估
    start_time = time()
    y_pred = model.predict(x_test_scaled)
    pred_time = time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    print(f"\nSVM 测试集准确率: {accuracy:.4f}")
    
    results = {
        'name': 'SVM',
        'accuracy': accuracy,
        'train_time': train_time,
        'pred_time': pred_time,
        'report': report
    }
    return results


# ==============================================================================
# 3. 卷积神经网络 (CNN) 分类器实现
# ==============================================================================

def build_cnn_model(input_shape, num_classes):
    """构建一个基础的 CNN 模型"""
    model = Sequential([
        # 卷积层 1: 32个 3x3 滤波器, ReLU 激活
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # 池化层 1: 2x2 最大池化
        MaxPooling2D((2, 2)),
        
        # 卷积层 2: 64个 3x3 滤波器
        Conv2D(64, (3, 3), activation='relu'),
        # 池化层 2
        MaxPooling2D((2, 2)),
        
        # 展平层：将 2D 特征图转换为 1D 向量
        Flatten(),
        
        # 全连接层 1: 128 个神经元
        Dense(128, activation='relu'),
        # Dropout: 防止过拟合
        Dropout(0.5),
        
        # 输出层: num_classes 神经元, Softmax 激活
        Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型：使用 Adam 优化器，交叉熵损失函数
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def run_cnn_classifier(x_train, y_train, x_test, y_test, num_classes, epochs=10):
    """运行基于 Keras 的 CNN 分类器"""
    print("\n" + "="*50)
    print("      3. 运行 卷积神经网络 (CNN) 分类器      ")
    print("="*50)

    # 1. 调整图像形状以适应 CNN (从 (N, H, W) -> (N, H, W, 1))
    input_shape = x_train.shape[1:] + (1,)
    x_train_cnn = x_train.reshape(x_train.shape + (1,))
    x_test_cnn = x_test.reshape(x_test.shape + (1,))
    
    # 2. 独热编码 (One-hot Encoding) 标签
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # 3. 构建和训练模型
    model = build_cnn_model(input_shape, num_classes)
    
    start_time = time()
    print(f"开始训练 CNN ({epochs} 轮)...")
    history = model.fit(x_train_cnn, y_train_cat,
                        epochs=epochs,
                        batch_size=128,
                        validation_data=(x_test_cnn, y_test_cat),
                        verbose=1)
    train_time = time() - start_time
    print(f"CNN 训练完成，耗时: {train_time:.2f} 秒")
    
    # 4. 评估
    start_time = time()
    loss, accuracy = model.evaluate(x_test_cnn, y_test_cat, verbose=0)
    pred_time = time() - start_time

    # 生成分类报告
    y_pred_classes = np.argmax(model.predict(x_test_cnn, verbose=0), axis=1)
    report = classification_report(y_test, y_pred_classes, digits=4, output_dict=True)

    print(f"\nCNN 测试集准确率: {accuracy:.4f}")
    
    results = {
        'name': 'CNN',
        'accuracy': accuracy,
        'train_time': train_time,
        'pred_time': pred_time,
        'report': report,
        'model': model
    }
    return results


# ==============================================================================
# 4. 跨数据集测试 (迁移性分析)
# ==============================================================================

def cross_dataset_test(trained_model, x_test_cross, y_test_cross, 
                       model_name, dataset_name, num_classes):
    """使用在 MNIST 上训练好的 CNN 模型测试 Fashion-MNIST 数据集"""
    print("\n" + "="*50)
    print(f"      4. 跨数据集测试: {model_name} -> {dataset_name}      ")
    print("="*50)
    
    # 1. 调整图像形状 (Fashion-MNIST 图像也需要 (N, H, W, 1) 形状)
    x_test_cnn = x_test_cross.reshape(x_test_cross.shape + (1,))
    
    # 2. 预测
    start_time = time()
    y_pred_probs = trained_model.predict(x_test_cnn, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    pred_time = time() - start_time
    
    # 3. 评估
    # 注意：两个数据集的类别含义完全不同 (数字 vs 服饰)，
    # 这里的准确率将非常低，用于演示模型在不匹配任务上的表现。
    accuracy = accuracy_score(y_test_cross, y_pred_classes)
    report = classification_report(y_test_cross, y_pred_classes, digits=4, 
                                   output_dict=True, zero_division=0)

    print(f"\n{model_name} 在 {dataset_name} 上的准确率: {accuracy:.4f}")

    results = {
        'name': f'{model_name}_Cross_{dataset_name}',
        'accuracy': accuracy,
        'pred_time': pred_time,
        'report': report
    }
    return results

# ==============================================================================
# 5. 主执行部分
# ==============================================================================

if __name__ == "__main__":
    
    all_results = {}
    
    # --- 阶段 1: MNIST 上的分类器比较 ---
    
    # 1.1 运行 SVM
    # 注意：为避免训练时间过长，SVM 仅使用 10000 个 MNIST 样本训练
    svm_results = run_svm_classifier(mnist_x_train, mnist_y_train, 
                                     mnist_x_test, mnist_y_test, 
                                     max_samples=10000)
    all_results['svm_mnist'] = svm_results
    
    # 1.2 运行 CNN
    cnn_results = run_cnn_classifier(mnist_x_train, mnist_y_train, 
                                     mnist_x_test, mnist_y_test, 
                                     num_classes, epochs=5) # 减少 epochs 以缩短运行时间
    all_results['cnn_mnist'] = cnn_results
    
    
    # --- 阶段 2: 跨数据集训练与测试 ---
    
    # 2.1 使用在 MNIST 上训练好的 CNN 模型进行跨数据集测试
    # 测试集是 Fashion-MNIST (10000个样本)
    cross_cnn_results = cross_dataset_test(cnn_results['model'], 
                                           fashion_x_test, fashion_y_test, 
                                           'CNN (Trained on MNIST)', 'Fashion-MNIST', 
                                           num_classes)
    all_results['cnn_cross_fashion'] = cross_cnn_results
    
    
    # --- 阶段 3: 替代数据集上的分类器训练 (Fashion-MNIST) ---
    
    # 3.1 在 Fashion-MNIST 上训练 CNN
    # 这部分满足要求 2 的字面意思：选择其他数据库进行训练
    cnn_fashion_results = run_cnn_classifier(fashion_x_train, fashion_y_train, 
                                             fashion_x_test, fashion_y_test, 
                                             num_classes, epochs=5)
    all_results['cnn_fashion'] = cnn_fashion_results


    # --- 结果汇总 (用于报告) ---
    print("\n" + "="*50)
    print("             实验结果汇总 (用于报告)            ")
    print("="*50)
    
    for key, res in all_results.items():
        print(f"\n模型/数据集: {key}")
        print(f"  准确率: {res['accuracy']:.4f}")
        print(f"  训练时间: {res.get('train_time', 'N/A'):.2f} 秒")
        print(f"  测试/推断时间: {res.get('pred_time', 'N/A'):.4f} 秒")

# 脚本结束