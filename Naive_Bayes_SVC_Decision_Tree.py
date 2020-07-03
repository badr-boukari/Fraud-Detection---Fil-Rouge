import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns
import time

df = pd.read_csv('SMSSpamCollection.txt', sep='\t', header=None, names=['y', 'message'])
df.y = df.y.apply(lambda x: 1 if x == 'spam' else 0)  # dictionary = {ham:0, spam:1}

# first split into train and test
X_train, X_test, y_train, y_test = train_test_split(df.message, df.y, test_size=0.33, random_state=42)

## Vectorisation (TF-IDF)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()


## Naive Bayes classifier
tic_NB = time.time()
gnb = GaussianNB()
y_gnb = gnb.fit(X_train, y_train).predict(X_test)
tac_NB = time.time()

time_NB = tac_NB - tic_NB

print("Naive Bayes Confusion Matrix")
CM_gnb = confusion_matrix(y_test, y_gnb)
print(CM_gnb)
print("Accuracy ", accuracy_score(y_test, y_gnb))
print("F1 Score ", f1_score(y_test, y_gnb))
print("Execution time ", round(time_NB,2),"s")


plot1 = plt.figure(1)
print("Naive Bayes Visualization ...\n")
sns.heatmap(CM_gnb, square=True, annot=True, fmt="d", cmap="RdBu", cbar=True, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title("Naive Bayes Heatmap\n")
plt.xlabel("True Label")
plt.ylabel("Predicted Label")

## SVM Classifier
tic_LSVC = time.time()
svc = SVC(gamma='auto')
svc.fit(X_train, y_train)
y_svc = svc.predict(X_test)
tac_LSVC = time.time()

time_LSVC = tac_LSVC - tic_LSVC

print("SVC Confusion Matrix")
CM_svc = confusion_matrix(y_test, y_svc)
print(CM_svc)
print("Accuracy ", accuracy_score(y_test, y_svc))
print("F1 Score ", f1_score(y_test, y_svc))
print("Execution time ", round(time_LSVC,2),"s")

plot2 = plt.figure(2)
print("SVC Visualization ...\n")
sns.heatmap(CM_svc, square=True, annot=True, fmt="d", cmap="RdBu", cbar=True, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title("SVC Heatmap\n")
plt.xlabel("True Label")
plt.ylabel("Predicted Label")

## Random Forest CLassifier
tic_RF = time.time()
rf = RandomForestClassifier(max_depth=100, random_state=0)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
tac_RF = time.time()

time_RF = tac_RF - tic_RF

print("Random Forest Confusion Matrix")
CM_rf = confusion_matrix(y_test, y_rf)
print(CM_rf)
print("Accuracy ", accuracy_score(y_test, y_rf))
print("F1 Score ", f1_score(y_test, y_rf))
print("Execution time ", round(time_RF,2),"s")

plot3 = plt.figure(3)
print("Random Forest Visualization ...\n")
sns.heatmap(CM_rf, square=True, annot=True, fmt="d", cmap="RdBu", cbar=True, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title("Random Forest Heatmap\n")
plt.xlabel("True Label")
plt.ylabel("Predicted Label")

plt.show()

## Message Predicition :
# msg = ""
# rf.predict(vectorizer.transform(['BangBabes Ur order is on the way']))


