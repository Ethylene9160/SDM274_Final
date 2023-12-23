import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self):
        self.hasTrained = False
        self.mean = np.array([])
        self.transformedMatrix = np.array([])
        self.explained_variance_ratio_ = None
        self.sum_cov = None

    def train(self, data):
        self.mean = np.mean(data, axis=0)
        X = data - self.mean

        cov_mat = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_var
        self.sum_cov = np.cumsum(self.explained_variance_ratio_)

        idx = np.argsort(eigenvalues)[::-1]
        self.transformedMatrix = eigenvectors[:, idx]
        self.hasTrained = True

    def transform(self, data, k):
        if not self.hasTrained:
            print('has not trained!')
            self.train(data)
        if k > self.transformedMatrix.shape[1]:
            raise ValueError("k cannot be larger than the number of eigenvalues/eigenvectors")
        transformk = self.transformedMatrix[:, :k]
        X = data - self.mean
        return np.dot(X, transformk)

    def inverse_transform(self, transformed_data, k):
        if not self.hasTrained:
            raise ValueError("PCA has not been trained. Please call the `train` method first.")
        if k > self.transformedMatrix.shape[1]:
            raise ValueError("k cannot be larger than the number of eigenvalues/eigenvectors")

        # 选择前k个特征向量
        transformk = self.transformedMatrix[:, :k]
        t = transformk.T.dot((transformk.dot(transformk.T))**(-1))
        print('shape of t is:')
        print(t.shape)
        # return transformed_data.dot(t) + self.mean
        # 逆变换到原始空间
        return np.dot(transformed_data, transformk.T) + self.mean

    def getSumCov(self):
        return self.sum_cov

    def draw_variance_plot(self):
        if self.sum_cov is None:
            raise ValueError("PCA has not been trained. Please call the `train` method first.")
        plt.figure(figsize=(8, 5))
        plt.plot(self.sum_cov, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Different Principal Components')
        plt.grid(True)
        plt.show()

    def draw_split_variance_plot(self):
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA has not been trained. Please call the `train` method first.")
        plt.figure(figsize=(8, 5))

        plt.plot(self.explained_variance_ratio_, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance by Different Principal Components')
        plt.grid(True)
        plt.show()
