import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimization.csv')

N = dataset.shape[0]
d = dataset.shape[1]
n_round = 1000
ads_selected = []
no_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(n_round):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if(no_of_selections[i] > 0):
            average_reward = sum_of_rewards[i] / no_of_selections[i]
            delta_i = math.sqrt(1.5 * math.log(n + 1) / no_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    no_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward

# print(ads_selected)
print(no_of_selections)
print(sum_of_rewards)
plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('No of times each ad was seleceted')
plt.show()
