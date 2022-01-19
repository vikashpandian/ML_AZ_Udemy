from nltk import stem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# x = dataset.iloc[:, [0]].values
# y = dataset.iloc[:, 1].values

nltk.download('stopwords')
corpus = []
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
for i in range(1000):
    review = re.sub('[^a-zA-z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() # Stemming the words -> Considering only the root of the word Eg. Loved->Love
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
#print(corpus)

cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray() # To get the words mapped in a certain way like in a dictionary
y = dataset.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Naive Bayes Implementation
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
acs = accuracy_score(y_test, y_pred)
true_preds = cm[0][0] + cm[1][1]
false_preds = cm[0][1] + cm[1][0]
accuracy = true_preds / (true_preds + false_preds) # Accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = cm[1][1] / sum(cm[0]) # Precision = TP / (TP + FP)
recall = cm[1][1] / sum(cm[1]) # TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
print(accuracy)
print(precision)
print(recall)
print(f1_score)

# my_review = 'This is the worst dish I\'ve had in my life'
my_review = 'I hate this dish'
my_review = re.sub('[^a-zA-Z]', ' ', my_review)
my_review = my_review.lower()
my_review = my_review.split()
ps = PorterStemmer()
my_review = [ps.stem(word) for word in my_review if not word in set(all_stopwords)]
my_review = [' '.join(my_review)]
print(my_review)
my_review_pre = cv.transform(my_review).toarray()
print(my_review_pre)
print(classifier.predict(my_review_pre))
print('The review is ' + ('-ve' if (classifier.predict(my_review_pre)[0] == 0) else '+ve'))
