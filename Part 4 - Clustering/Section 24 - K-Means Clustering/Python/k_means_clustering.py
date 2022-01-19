import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, color = 'blue')
plt.title('WCSS Values')
plt.xlabel('No. of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_pred = kmeans.fit_predict(x)
print(y_pred)

for i in range (5):
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], color = np.random.rand(3,), label = 'Cluster ' + str(i))
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300 , c = 'yellow', label = 'Centroids')
plt.title('K-Means Plot')
plt.xlabel('Income')
plt.ylabel('Spending score')
plt.legend
plt.show()
