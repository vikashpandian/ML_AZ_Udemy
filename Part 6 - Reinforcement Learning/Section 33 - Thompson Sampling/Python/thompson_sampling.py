import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = pd.read_csv('Ads_CTR_Optimization.csv')

N = dataset.shape[0]
d = dataset.shape[1]
n_round = 1000
ads_selected = []
no_of_rewards_1 = [0] * d
no_of_rewards_0 = [0] * d
total_reward = 0

for n in range(n_round):
    ad = 0
    max_random = 0
    for i in range(d):
        random_beta = random.betavariate(no_of_rewards_1[i] + 1, no_of_rewards_0[i] + 1)
        if (random_beta > max_random):
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 0:
        no_of_rewards_0[ad] += 1
    else:
        no_of_rewards_1[ad] += 1
    total_reward += reward

#print(ads_selected)
plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('No of times each ad was seleceted')
plt.show()
