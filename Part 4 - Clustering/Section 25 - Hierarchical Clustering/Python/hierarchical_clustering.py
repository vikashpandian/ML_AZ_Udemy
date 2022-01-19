import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:].values

# dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
# plt.title('Dendogram')
# plt.xlabel('Customers')
# plt.ylabel('Eucledian Distances')
# #plt.show()

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_pred = hc.fit_predict(x)
print(y_pred)

for i in range (5):
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], color = np.random.rand(3,), label = 'Cluster ' + str(i))
plt.title('Hierarchical Clustering Plot')
plt.xlabel('Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()
