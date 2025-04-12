from utils import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator


class MSNMClassifier(BaseEstimator):
    def __init__(self, sigm_times=30, adaptive_times=100, use_vss=False):
        self.sigm_times = sigm_times
        self.use_vss = use_vss
        self.adaptive_times = adaptive_times
        self.data = None
        self.labels = None

    def fit(self, features: np.array, labels: np.array):
        if self.use_vss:
            self.data, self.labels = self.vsscon(features, labels)
            return
        self.data = features
        self.labels = labels

    def predict(self, data: np.array):
        unique_label = np.unique(self.labels)
        wMatrix = self.__init_matrix__(self.labels)
        num = len(data)
        predict = np.zeros(num,dtype=int)
        for i in range(num):
            sigm1 = 1
            sigm1U = sigm1 * 3
            sigm1L = sigm1 / 3
            x = data[i][None, ...]
            ud = self.__predict__(x, self.data, self.labels, sigm1, wMatrix)
            ud_num = len(ud[ud > 0])
            for _ in range(self.adaptive_times):
                if ud_num != 1:
                    sigm1, sigm1U, sigm1L = sigmadapt(ud_num, sigm1, sigm1U, sigm1L)
                    ud = self.__predict__(x, self.data, self.labels, sigm1, wMatrix)
                    ud_num = len(ud[ud > 0])
                else:
                    break
            predict[i] = unique_label[np.argmax(ud)]
        return predict

    def score(self, data: np.array, labels: np.array):
        predict = self.predict(data)
        return np.mean(predict == labels)

    def vsscon(self, features: np.array, labels: np.array):
        cluster_predicts = self.cluster(features)[0]
        sample_num, n_dim = features.shape
        times = len(cluster_predicts)
        unique_labels = np.unique(labels)
        class_num = len(unique_labels)
        real_link_matrix = calculate_link_matrix(labels)
        link_matrixes = np.zeros((times, sample_num, sample_num))
        for i in range(times):
            link_matrixes[i] = calculate_link_matrix(cluster_predicts[i])
        subtracts = real_link_matrix - link_matrixes
        subtracts_one = (subtracts + 1) // 2
        subtracts_minus_one = (-subtracts + 1) // 2
        t = sample_num ** 2
        kmeans = KMeans(n_clusters=class_num)
        kmeans_predict = kmeans.fit_predict(features)
        kmeans_link_matrix = calculate_link_matrix(kmeans_predict)
        kmeans_matrix_acc = real_link_matrix - kmeans_link_matrix
        kmeans_acc = np.sum(kmeans_matrix_acc == 0) / t
        results = np.zeros(times)
        for i in range(times):
            results[i] = np.sum(subtracts_one[i]) - kmeans_acc * np.sum(subtracts_minus_one[i])
        min_idx = np.argmin(results)
        vss_features = []
        vss_labels = []
        for i in range(class_num):
            current_feature = features[labels == unique_labels[i]]
            predicts, cluster_nums, cluster_results = self.cluster(current_feature)
            cluster_num = cluster_nums[min_idx]
            predict = predicts[min_idx]
            vss_data = np.zeros((cluster_num, n_dim))
            vss_label = np.ones(cluster_num, dtype=int) * unique_labels[i]
            for j in range(cluster_num):
                vss_data[j] = np.mean(current_feature[predict == j], axis=0)
            vss_features.append(vss_data)
            vss_labels.append(vss_label)
        vss_features = np.concatenate(vss_features, axis=0)
        vss_labels = np.concatenate(vss_labels, axis=0)
        return vss_features, vss_labels

    def cluster(self, data: np.array):
        data_num, dim_n = data.shape
        # 可选归一化
        data_norm = data
        sigm_times = self.sigm_times
        sigm_step = (1 - 0.3) / sigm_times
        sigm = np.arange(0.01, 1, sigm_step)
        cluster_times = min(sigm_times, len(sigm))
        # 每次聚类的预测值
        cluster_predicts = [None] * cluster_times
        # 每次聚类的簇数量
        cluster_nums = np.zeros((cluster_times), dtype=int)
        # 每次聚类结果中每个簇对应的样本
        cluster_results = [None] * sigm_times
        for i in range(sigm_times):
            data = data_norm
            sigm1 = sigm[i]
            distance_matrix = dogdistm(data, sigm1)
            sum_matrix = np.sum(distance_matrix, axis=0)
            sort_idx = np.argsort(sum_matrix)[::-1]
            predict = np.zeros((data_num, 1), dtype=int)
            current_x = data[sort_idx[0]]
            current_y = np.array([0])
            curr_cluster_num = 0
            for j in range(1, data_num):
                wMatrix = self.__init_matrix__(current_y)
                idx = sort_idx[j]
                x = data[idx]
                ud = self.__predict__(x, current_x, current_y, sigm1, wMatrix)
                ud_max_idx = np.argmax(ud)
                ud_greater_zero = len(ud[ud > 0])
                if ud_greater_zero == 0:
                    curr_cluster_num = curr_cluster_num + 1
                    predict[idx] = curr_cluster_num
                    current_y = np.append(current_y, curr_cluster_num)
                elif ud_greater_zero == 1:
                    pre = current_y[ud_max_idx]
                    predict[idx] = pre
                    current_y = np.append(current_y, pre)
                else:
                    pre = current_y[ud_max_idx]
                    distance_matrix = dogdistm(normalize(np.vstack((current_x[current_y == pre], x)), norm='l2'), 1)
                    if sum(distance_matrix[-1, :] > 0):
                        current_y = np.append(current_y, pre)
                current_x = np.vstack((current_x, x))
            cluster_predicts[i] = current_y
            cluster_nums[i] = len(np.unique(current_y))
            cluster_num_data = [None] * cluster_nums[i]
            for c in range(cluster_nums[i]):
                cluster_num_data[c] = current_x[current_y == c]
            cluster_results[i] = cluster_num_data
        return cluster_predicts, cluster_nums, cluster_results

    def __init_matrix__(self, labels: np.array):
        unique_label = np.unique(labels)
        lens = len(unique_label)
        matrix = np.zeros((len(labels), lens))
        for i in range(len(labels)):
            l = labels[i]
            for j in range(lens):
                if l == unique_label[j]:
                    matrix[i][j] = 1
                    break
        return matrix

    def __predict__(self, x: np.array, features: np.array, labels: np.array, sigm: float, wMatrix: np.array):
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        sample_num = features.shape[0]
        class_num = len(np.unique(labels))
        combine_f = np.vstack((features, x))
        sample_num += 1
        link_matrix = np.zeros((sample_num, 1))
        link_matrix[-1] = 1
        wMatrix = np.concatenate((wMatrix, np.zeros((1, class_num))))
        distans_matrix = dogdistm(combine_f, sigm)
        v = thetea(distans_matrix @ thetart(link_matrix))
        u = thetart(wMatrix.T @ thetart(v))
        return u

    def cluster_accuracy(self, cluster_predicts, cluster_nums):
        """
        cluster_predicts:MSNMClassifier没次的聚类结果
        cluster_nums:每次聚类结果得到簇数量
        """
        sample_num = len(cluster_predicts[0])
        times = len(cluster_nums)
        offset = np.zeros(times)
        predict = cluster_predicts[0]
        link_matrix = calculate_link_matrix(predict)
        t = sample_num ** 2
        for i in range(1, times):
            current_link_matrix = calculate_link_matrix(cluster_predicts[i])
            subtract = current_link_matrix != link_matrix
            offset[i] = np.sum(subtract) / t
            link_matrix = current_link_matrix
        return offset

    def get_params(self, deep=True):
        return super().get_params(deep)

    def set_params(self, **params):
        return super().set_params(**params)
