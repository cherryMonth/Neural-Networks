import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# 准备数据集
data_path = 'data\Wholesale customers data.csv'

"""
每个特征为某种商品的年花费
此处我们对买新鲜产品和牛奶的用户进行划分
然后对比其他两种聚类方法，分析各种算法的优劣
"""

df = pd.read_csv(data_path)
print(df.info())  # 查看数据信息，确保没有错误
dataset = df.values  # 数据加载没有问题
dataset = dataset[:, 2:]  # 本项目只需要后面的6列features即可
col_names = df.columns.tolist()[2:]
print(col_names)


def visual_cluster_effect(cluster, dataset, title, col_id):
    assert isinstance(col_id, list) and len(col_id) == 2, 'col_id must be list type and length must be 2'

    labels = cluster.labels_  # 每一个样本对应的簇群号码
    #     print(labels.shape) # (440,) 440个样本
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
    plt.title(title)
    plt.xlabel('feature_0')
    plt.ylabel('feature_1')
    plt.show()


from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3)
agg.fit(dataset)  # 使用层次聚类模型，并进行训练
labels = agg.labels_
cluster_num = len(np.unique(labels))  # 生成聚类标签

# 下面打印出簇群种类
print('Number of Clusters: {}'.format(cluster_num))
print('\t'.join([col_name[:5] for col_name in col_names]))
visual_cluster_effect(agg, dataset, 'MeanShift-X=fresh,y=milk', [0, 1])  # X=fresh， y=milk

# 使用轮廓系数评估模型的优劣
from sklearn.metrics import silhouette_score

si_score = silhouette_score(dataset, agg.labels_,
                            metric='euclidean', sample_size=len(dataset))
print('si_score: {:.4f}'.format(si_score))
