import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)

print(r2_score(y_test, y_pred))

ss_res = sum(map(lambda x: (x[0] - x[1])**2, zip(y_test, y_pred)))
ss_tot = sum((x - np.mean(y_test))**2 for x in y_test)
print(ss_res)
print(ss_tot)
r_sq = 1 - (ss_res / ss_tot)
print(r_sq)
