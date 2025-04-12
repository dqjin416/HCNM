import os
import pandas as pd
import uuid
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from utils import *
from MSNMClassifier import MSNMClassifier

dataset_name = 'cub'
model_name = 'transformer'
data_path = f"./Datasets/{dataset_name}_{model_name}_features.npy"
label_path = f"./Datasets/{dataset_name}_{model_name}_labels.npy"
save_path_root = f'./CSV'
nways = 5
kshot = 5
decom_types = ['PCA', 'ICA', 'TSNE', None]
decom_type = decom_types[0]
sample_num = 20
# 降维后的维度
n_components = 10
times = 20
# 需要记录时间可以自己加
df_columns = ['KNN', 'SVM', 'MSNM', 'VSS', 'PCA KNN', 'PCA SVM', 'PCA MSNM', 'PCA VSS']
results = np.zeros((times, len(df_columns)))
features, labels = load_data(data_path, label_path)

knn = KNeighborsClassifier(n_neighbors=1)
svm = SVC(kernel='linear', C=5, gamma=1)
msnm = MSNMClassifier()
msnm_vss = MSNMClassifier(use_vss=True)
classifiers = [knn, svm, msnm, msnm_vss]
classifiers_num = len(classifiers)

times_bar = tqdm(range(times), desc='Compare')
for t in times_bar:
    sampled_data, sampled_label = sample_data(features, labels, sample_num, nways=nways)
    sampled_unique_labels = np.unique(sampled_label)
    labeled_idx, unlabeled_idx = sample_data(sampled_data, sampled_label, kshot,use_idx=True)
    labeled_label = sampled_label[labeled_idx]
    unlabeled_label = sampled_label[unlabeled_idx]
    sampled_data_decom = decomposition(sampled_data, n_components, decom_type)
    for c in range(classifiers_num):
        classifier = classifiers[c]
        classifier.fit(sampled_data[labeled_idx], labeled_label)
        results[t, c] = classifier.score(sampled_data[unlabeled_idx], unlabeled_label)
        classifier.fit(sampled_data_decom[labeled_idx], labeled_label)
        results[t, c + classifiers_num] = classifier.score(sampled_data_decom[unlabeled_idx], unlabeled_label)

df_results = pd.DataFrame(results, index=range(1, times + 1), columns=df_columns)
df_results.to_csv(os.path.join(save_path_root, f'{dataset_name}_{model_name}_results_{uuid.uuid4()}.csv'))
