import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np				# 用 numpy 实现，目的是 pytorch 和 tensorflow 的框架都能使用，label.numpy()
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库: pip install prettytable
    """
    def __init__(self, num_classes: 2, labels: [0,1]):
        self.matrix = np.zeros((num_classes, num_classes))	# 初始化混淆矩阵
        self.num_classes = num_classes
        self.labels = labels

    # 混淆矩阵更新
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    # 计算并打印评价指标
    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]				# 对角线元素求和
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]	# 第一个元素是类别标签
        for i in range(self.num_classes):			# 针对每个类别进行计算
            # 整合其他行列为不属于该类的情况
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.		# 注意分母为 0 的情况
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    # 可视化混淆矩阵
    def plot(self,modelname,title):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)		# 从白色到蓝色
        # 颜色、是否加粗，字体大小，字体采取默认
        fontdict = {'color': 'black',
                    'size': 18}
        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, fontsize=18)	# x 轴标签旋转 45 度方便展示
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels,fontsize=18)
        # 显示colorbar
        plt.colorbar()

        plt.xlabel('True Labels', fontdict=fontdict)
        plt.ylabel('Predicted Labels',fontdict=fontdict)
        plt.title(title,fontdict=fontdict)

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                # 画图的时候横坐标是x，纵坐标是y
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         fontsize=18,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()		# 图形显示更加紧凑
        plt.savefig('result_v/{}'.format(modelname))

