from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
agg = AgglomerativeClustering(linkage="complete").fit(X)

learned_parameters(agg)

print(agg.children_)
