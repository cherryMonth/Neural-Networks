import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 准备数据集
data_path = 'data\Wholesale customers data.csv'

df = pd.read_csv(data_path)
print(df.info())  # 查看数据信息，确保没有错误
dataset = df.values  # 数据加载没有问题
dataset = dataset[:, 2:]  # 本项目只需要后面的6列features即可
col_names = df.columns.tolist()[2:]
print(col_names)


def visual_cluster_effect(cluster, dataset, title, col_id):
    assert isinstance(col_id, list) and len(col_id) == 2, 'col_id must be list type and length must be 2'

    labels = gmm.predict(dataset)  # 每一个样本对应的簇群号码
    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8'
        , 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # 将数据集绘制到图表中
    plt.figure()
    for class_id in set(labels):
        one_class = dataset[class_id == labels]
        print('label: {}, smaple_num: {}'.format(class_id, len(one_class)))
        plt.scatter(one_class[:, 0], one_class[:, 1], marker=markers[class_id % len(markers)],
                    c=colors[class_id % len(colors)], label='class_' + str(class_id))
    plt.legend()

    # 将中心点绘制到图中
    centroids = cluster.means_
    plt.scatter(centroids[:, col_id[0]], centroids[:, col_id[1]], marker='o',
                s=100, linewidths=2, color='k', zorder=5, facecolors='b')
    plt.title(title)
    plt.xlabel('feature_0')
    plt.ylabel('feature_1')
    plt.savefig('gmm.png')
    plt.show()


"""
print(gmmModel.means_)  中心蔟
print(gmmModel.covariances_)  样本协方差矩阵
"""

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)
gmm.fit(dataset)  # 使用评估的带宽构建均值漂移模型，并进行训练
labels = gmm.predict(dataset)

cluster_num = len(np.unique(labels))  # 生成聚类标签
centroids = gmm.means_  # 返回蔟的中心点，这些点是不一定是样本点，该点的坐标对应于各坐标轴
print('\t'.join([col_name[:5] for col_name in col_names]))
for centroid in centroids:
    print('\t'.join(str(int(x)) for x in centroid))
visual_cluster_effect(gmm, dataset, 'MeanShift-X=fresh,y=milk', [0, 1])  # X=fresh， y=milk
# 使用轮廓系数评估模型的优劣
from sklearn.metrics import silhouette_score

si_score = silhouette_score(dataset, labels,
                            metric='euclidean', sample_size=len(dataset))
print('si_score: {:.4f}'.format(si_score))
