# Factor analysis
# help(sklearn.decomposition.FactorAnalysis)
import sklearn
from sklearn.decomposition import FactorAnalysis
transformer = FactorAnalysis(n_components=7, random_state=0)
 X_transformed = transformer.fit_transform(X)

