import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def visual_cluster_effect(dataset, title, col_id, labels, centroids, num):
    """

    :param dataset: 测试数据集
    :param title:  图像名
    :param col_id:  特征的索引
    :param labels:  样本的标签，对应染色
    :param centroids: 是否含有中心蔟点，若有则在图像上显示
    :param num: 代表当前图像的位置
    :return:
    """

    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D',
               'd', '|']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # 将数据集绘制到图表中
    plt.subplot(num)
    for class_id in set(labels):
        one_class = dataset[class_id == labels]
        print('label: {}, smaple_num: {}'.format(class_id, len(one_class)))
        plt.scatter(one_class[:, 0], one_class[:, 1], marker=markers[class_id % len(markers)],
                    c=colors[class_id % len(colors)], label='class_' + str(class_id))
    plt.legend()

    # 将中心点绘制到图中
    if centroids is not None:
        plt.scatter(centroids[:, col_id[0]], centroids[:, col_id[1]], marker='o', s=100, linewidths=2, color='k',
                    zorder=5, facecolors='b')

    plt.title(title)
    plt.xlabel('feature_0')
    plt.ylabel('feature_1')
    plt.tight_layout()

    # 使用轮廓系数评估模型的优劣
    si_score = silhouette_score(dataset, labels, metric='euclidean', sample_size=len(dataset))
    print('si_score: {:.4f}'.format(si_score))


# 打印测试报告
def echo_centroid(series, labels, col_names):
    print('Number of Clusters: {}'.format(len(np.unique(labels))))  # 显示蔟的数量
    print('\t'.join([col_name[:5] for col_name in col_names]))
    for data in series:
        print('\t'.join(str(int(x)) for x in data))


agg = AgglomerativeClustering(n_clusters=3)
agg.fit(dataset)
labels = agg.labels_
visual_cluster_effect(dataset, 'AGG-X=fresh,y=milk', [0, 1], labels, None, num=311)  # X=fresh， y=milk
echo_centroid([], labels, col_names)

gmm = GaussianMixture(n_components=3)
gmm.fit(dataset)
labels = gmm.predict(dataset)
centroids = gmm.means_  # 返回蔟的中心点，这些点是不一定是样本点，该点的坐标对应于各坐标轴
echo_centroid(centroids, labels, col_names)
visual_cluster_effect(dataset, 'GMM-X=fresh,y=milk', [0, 1], labels, centroids, num=312)

kmeans = KMeans(n_clusters=3)
kmeans.fit(dataset)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
echo_centroid(centroids, labels, col_names)
visual_cluster_effect(dataset, 'K-means-X=fresh,y=milk', [0, 1], labels, centroids, num=313)

# 绘图
plt.savefig('cluster.png')
plt.show()
