import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

dataset = pd.read_csv('Market_Basket_Optimization.csv', header = None)
transactions = []
for i in range(7501):
    transactions.append([x for x in [*dataset.values[i].astype(str)] if x != 'nan'])

print(transactions)

rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
results = list(rules)
print(results)

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product I', 'Product II', 'Support'])
print(resultsinDataFrame)
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))
