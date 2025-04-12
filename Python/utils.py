import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist


# region 加载数据并采样
def load_data(data_path, label_path):
    # 加载数据并按顺序排序，以防采样出现顺序错乱
    features = np.load(data_path)
    labels = np.load(label_path).reshape(-1)
    labels = labels.astype(int)
    unique_label = np.unique(labels)
    features_sort = []
    labels_sort = []
    for i in range(len(unique_label)):
        mask = (labels == i)
        features_sort.append(features[mask])
        labels_sort.append(labels[mask])
    features_sort = np.concatenate(features_sort, axis=0)
    labels_sort = np.concatenate(labels_sort, axis=0)
    return features_sort, labels_sort


# 重载方法
def sample_data(features, labels, kshot, nways=None, sample_type='balance', use_idx=False):
    """
    kshot:每个类的采样数量
    sample_type:采样类型，'balance'下每个类采样kshot，'unbalance'下每个类采样当前类一定比例的样本
    nways:采样的类别数量，若为空，则默认为所有类别
    use_idx:是否返回索引，默认为False
    """
    # 采样比例
    unique_labels = np.unique(labels)
    if nways is None:
        selected_class = unique_labels
        nways = len(unique_labels)
    else:
        selected_class = np.random.choice(unique_labels, nways, replace=False)
        f, l = [], []
        for i in range(nways):
            mask = (labels == selected_class[i])
            f.append(features[mask])
            l.append(labels[mask])
        features = np.concatenate(f, axis=0)
        labels = np.concatenate(l, axis=0)
    ratio = nways * kshot / len(labels)
    idxs = []
    offset = 0
    for i in range(nways):
        l = selected_class[i]
        lens = np.sum(labels == l)
        idx_all = np.arange(lens)
        num_samples = max(int(ratio * lens), 1) if sample_type == 'unbalance' else kshot
        idx = np.random.choice(idx_all, num_samples, replace=False)
        idxs.append(idx + offset)
        offset += lens
    idxs = np.concatenate(idxs, axis=0)
    if use_idx:
        unidxs = np.setdiff1d(np.arange(features.shape[0]), idxs)
        return idxs, unidxs
    sampled_data = features[idxs]
    sampled_label = labels[idxs]
    return sampled_data, sampled_label
# endregion


# region 降维
def decomposition(data: np.array, n_components: int, decom_type: str = 'PCA'):
    """
    data:需要降维的数据
    n_components:降维后的维度
    decom_type:降维类型，可选PCA,ICA,TSNE
    """
    if decom_type is None:
        return data
    if decom_type == 'PCA':
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)
    elif decom_type == 'ICA':
        ica = FastICA(n_components=n_components)
        return ica.fit_transform(data)
    elif decom_type == 'TSNE':
        tsne = TSNE(n_components=min(n_components, 2))
        return tsne.fit_transform(data)


# endregion


# region 计算距离矩阵
def dogdistm(x, sigm1):
    sigm2 = 2 * sigm1
    s = (len(x) + 1) * len(x) // 2
    dis = np.zeros(s)
    if len(x) > 1:
        dis = cdist(x, x, "cosine")
    adjust_dis = (1 / (np.sqrt(2 * np.pi) * sigm1)) * np.exp(-1 / 2 * np.square(dis / sigm1)) - (
                1 / (np.sqrt(2 * np.pi) * sigm2)) * (np.exp(-1 / 2 * np.square(dis / sigm2)))
    return adjust_dis
# endregion


def thetart(x):
    x[x < 0] = 0
    return 1 - np.exp(-x)


def thetea(x):
    return np.sign(x) * (1 - np.exp(-np.abs(x)))


# region 计算连接矩阵
def calculate_link_matrix(predict: np.array):
    num = len(predict)
    matrix = np.zeros((num, num), dtype=int)
    for i in range(1, num):
        for j in range(i):
            if predict[i] == predict[j]:
                matrix[i][j] = 1
    matrix = matrix + matrix.T
    return matrix
# endregion


def sigmadapt(ud_pos, sigm, sigm_max, sigm_min):
    eps = 1e-20
    sigm1U = sigm_max
    sigm1L = sigm_min
    if ud_pos > 1:
        if sigm - sigm1L < eps:
            sigm1L = 0.618 * sigm1L
        sigm1U = sigm
        sigm = sigm1L + 0.618 * (sigm1U - sigm1L)
    elif ud_pos == 0:
        if sigm1U - sigm < eps:
            sigm1U = sigm1U / 0.618
        sigm1L = sigm
        sigm = sigm + 0.618 * (sigm1U - sigm)
    return sigm, sigm1L, sigm1U
