# Kernel Principal component analysis (KPCA)
from sklearn.decomposition import KernelPCA
transformer = KernelPCA(n_components=7, kernel='linear')
X_transformed = transformer.fit_transform(X)
X_transformed.shape






