# Truncated SVD
# Truncated SVD is different from regular SVDs in that it produces a factorization where the
# number of columns is equal to the specified truncation. For example, given an n x n matrix,
# SVD will produce matrices with n columns, whereas truncated SVD will produce matrices
# with the specified number of columns. This is how the dimensionality is reduced.
import sklearn
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(components)
X_transformed = svd.fit_transform(X)


