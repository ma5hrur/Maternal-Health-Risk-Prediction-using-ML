import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn import linear_model, preprocessing, svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

#data = pd.read_csv("Maternal Health Risk Data Set.csv")
data = pd.read_csv("Maternal Health Risk Data Set_updated.csv")

#data = data[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]]
data = data[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel", "Reticulocyte count", "Platelet count"]]

# Normalizing the columns (New columns created)

data["NormAge"] = StandardScaler().fit_transform(np.array(data["Age"]).reshape(-1,1))
data["NormSBP"] = StandardScaler().fit_transform(np.array(data["SystolicBP"]).reshape(-1,1))
data["NormDBP"] = StandardScaler().fit_transform(np.array(data["DiastolicBP"]).reshape(-1,1))
data["NormBS"] = StandardScaler().fit_transform(np.array(data["BS"]).reshape(-1,1))
data["NormBT"] = StandardScaler().fit_transform(np.array(data["BodyTemp"]).reshape(-1,1))
data["NormHR"] = StandardScaler().fit_transform(np.array(data["HeartRate"]).reshape(-1,1))
data["NormRC"] = StandardScaler().fit_transform(np.array(data["Reticulocyte count"]).reshape(-1,1))
data["NormPC"] = StandardScaler().fit_transform(np.array(data["Platelet count"]).reshape(-1,1))

# Deleting old columns
#data = data.drop(["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"], axis = 1)
data = data.drop(["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "Reticulocyte count", "Platelet count"], axis = 1)
data = data.drop(["NormHR"], axis=1)

# Converting categorical variables to numeric
data['RiskLevel'].replace(['high risk', 'mid risk', 'low risk'], [2,1,0], inplace=True)

#print(data.head())

#print(data['RiskLevel'].value_counts())

predict = 'RiskLevel'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# split into 70:30 ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# describes info about train and test set
'''print("X_train dataset: ", X_train.shape)
print("y_train dataset: ", y_train.shape)
print("X_test dataset: ", X_test.shape)
print("y_test dataset: ", y_test.shape)'''

'''print("Before OverSampling, counts of label '2': {}".format(sum(y_train == 2)))
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {}".format(sum(y_train == 0)))'''

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

'''print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {}'.format(y_train_res.shape))'''

'''print("After OverSampling, counts of label '2': {}".format(sum(y_train_res == 2)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))'''

'''for value in range(81, 100):
    clf = RandomForestClassifier(n_estimators=value, random_state=0)
    clf.fit(X_train_res, y_train_res)

    y_predict = clf.predict(X_test)

    # Accuracy
    acc = metrics.accuracy_score(y_test, y_predict)
    print(acc)
    print("Cross Validation Score: ", cross_val_score(clf, X, y, cv=10, scoring ='accuracy').mean())'''

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_res, y_train_res)

y_predict = clf.predict(X_test)

# Accuracy
acc = metrics.accuracy_score(y_test, y_predict)
print("Accuracy: ", acc)
print("Cross Validation Score: ", cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean())
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))






