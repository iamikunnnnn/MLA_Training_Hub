from sklearn.datasets import make_classification, make_circles, make_moons, make_regression, make_friedman1
from sklearn.preprocessing import StandardScaler
import numpy as np

def generate_data(dataset_type, n_samples=200, base_n_features=5, selected_derived_features=None):
    """
    生成数据,并可以选择增加衍生特征，衍生特征导致的维度问题在UI界面中用户输入时自动维护
    :param dataset_type:
    :param n_samples:
    :return:
    """

    if dataset_type == "分类":
        X, y = make_classification(n_samples=n_samples, n_features=base_n_features,
                                   n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, random_state=42)
    elif dataset_type == "圆形":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    elif dataset_type == "月形":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif dataset_type == "回归":
        X, y = make_regression(n_samples=1000, n_features=base_n_features, n_informative=2,
                               noise=5.0, bias=1, random_state=42)
    elif dataset_type == "回归2.0":
        X, y = make_friedman1(n_samples=1000, n_features=base_n_features,
                              noise=5.0, random_state=42)
    else:
        X, y = make_moons(n_samples=base_n_features)

    # 记录一下原始的X和y，等会算衍生特征的时候就用这个，不然会混淆
    base_X, base_y = X, y

    if "交叉项" in selected_derived_features:
        # 获取当前特征数量S
        n = base_X.shape[1]
        # 只计算i < j的组合，避免重复
        for i in range(n):
            for j in range(i + 1, n):
                # 计算交叉项（第i列与第j列的乘积）
                cross_term = base_X[:, i] * base_X[:, j]
                # 转换为列向量并拼接
                X = np.hstack([X, cross_term.reshape(-1, 1)])
    if "平方项" in selected_derived_features:
        square_terms = base_X ** 2
        # 将原始特征与平方项横向拼接
        X = np.hstack([X, square_terms])

    if "正弦项" in selected_derived_features:
        sin_terms = np.sin(base_X)
        # 将原始特征与平方项横向拼接
        X = np.hstack([X, sin_terms])

    if "余弦项" in selected_derived_features:
        cos_terms = np.cos(base_X)
        # 将原始特征与平方项横向拼接
        X = np.hstack([X, cos_terms])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # 归一化、不然可能会梯度爆炸
    return X, y