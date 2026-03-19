# -*- coding: utf-8 -*-
#code-7-5.py
#用MLP识别MNIST字符
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(0.1,1.0,5)):
    plt.title(title)#图像标题
    if ylim is not None:#y轴限制不为空时
        plt.ylim(*ylim)
    plt.xlabel("Training examples")#两个标题
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)#获取训练集大小，训练得分集合，测试得分集合
    train_scores_mean=np.mean(train_scores,axis=1)#将训练得分集合按行的到平均值
    train_scores_std=np.std(train_scores,axis=1)#计算训练矩阵的标准方差
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid()#背景设置为网格线
    
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    # plt.fill_between()函数会把模型准确性的平均值的上下方差的空间里用颜色填充。
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='Training score')
    # 然后用plt.plot()函数画出模型准确性的平均值
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='Cross_validation score')
    plt.legend(loc='best')#显示图例
    return plt
    
    
#获取MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
#获取特征和目标值
X = mnist.data
y = mnist.target
#像素灰度值归一化
X = X / 255.

# 分割训练集和测试集
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
#定义MLP模型，1个隐含层，含50个隐结点，训练周期为10
#采用SGD算法，学习率0.1
mlp = MLPClassifier(hidden_layer_sizes=(50), max_iter=10, 
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

print(classification_report(y_test, mlp.predict(X_test)))

#For cross validation    
plot_learning_curve(mlp, 'Learning Curves', X, y, ylim=(0.1, 1.01))#开始绘制曲线

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[1].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
