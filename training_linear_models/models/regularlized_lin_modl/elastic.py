# Elastic Net

# J( \theta ) = MSE( \theta ) +
# r \alpha \sum_{i=1}^n \vert \theta_i \vert +
# \frac{1 - r}{2} \alpha \sum_{i=1}^n \theta_i^2

# Common Imports
import numpy as np

# ML Imports
from sklearn.linear_model import ElasticNet

# Declare variables
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print("Elastic: ", elastic_net.predict([[1.5]]))
