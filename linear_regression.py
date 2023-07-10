import tensorflow
import keras
import sklearn
import matplotlib.pyplot as pyplot
import pickle
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from matplotlib import style
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr

data = pd.read_csv("Maternal Health Risk Data Set.csv")

#print(data.head())

data = data[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]]

le = preprocessing.LabelEncoder()
age = le.fit_transform(list(data["Age"]))
systolicBP = le.fit_transform(list(data["SystolicBP"]))
diastolicBP = le.fit_transform(list(data["DiastolicBP"]))
bS = le.fit_transform(list(data["BS"]))
bodyTemp = le.fit_transform(list(data["BodyTemp"]))
heartRate = le.fit_transform(list(data["HeartRate"]))
riskLevel = le.fit_transform(list(data["RiskLevel"]))

predict = "RiskLevel"

X = list(zip(age, systolicBP, diastolicBP, bS, bodyTemp, heartRate))
y = list(riskLevel)
#X = np.array(data.drop([predict], 1))
#y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
#print(acc)

#Checking correlation among variables
#corr, _ = pearsonr(diastolicBP, riskLevel)
#print('Pearsons correlation: %.3f' % corr)

p = 'BodyTemp'
style.use("ggplot")
pyplot.scatter(data[p],data["RiskLevel"])
pyplot.xlabel(p)
pyplot.ylabel("Risk Level")
pyplot.show()