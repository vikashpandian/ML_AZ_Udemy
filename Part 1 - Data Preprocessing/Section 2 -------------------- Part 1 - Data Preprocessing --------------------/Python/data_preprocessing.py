import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer as imp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

imputer = imp(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:])
x[:, 1:] = imputer.transform(x[:, 1:])
print(x)

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
print(x)

le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# Feature Scaling
sc = StandardScaler()
x_train[:, -2:] = sc.fit_transform(x_train[:, -2:])
x_test[:, -2:] = sc.transform(x_test[:, -2:])
print(x_train)
print(x_test)
