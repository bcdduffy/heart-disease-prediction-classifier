import pandas as pd
import numpy as np
import sklearn.tree as sk
from sklearn import tree
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import tree
import random
from sklearn.model_selection import train_test_split

heartData = pd.read_csv("heart.csv", header=0)

#Male now = 0 & female = 1
heartData['Sex'] = heartData['Sex'].replace('F', 1)
heartData['Sex'] = heartData['Sex'].replace('M', 0)


#ChestPainType: TA = 0, ATA = 1, NAP = 2, ASY = 3
heartData['ChestPainType'] = heartData['ChestPainType'].replace('TA', 0)
heartData['ChestPainType'] = heartData['ChestPainType'].replace('ATA', 1)
heartData['ChestPainType'] = heartData['ChestPainType'].replace('NAP', 2)
heartData['ChestPainType'] = heartData['ChestPainType'].replace('ASY', 3)

#RestingECG: Normal = 0, ST = 1, LVH = 2
heartData['RestingECG'] = heartData['RestingECG'].replace('Normal', 0)
heartData['RestingECG'] = heartData['RestingECG'].replace('ST', 1)
heartData['RestingECG'] = heartData['RestingECG'].replace('LVH', 2)

#ExcersizeAngina: Y: 0, N: 1
heartData['ExerciseAngina'] = heartData['ExerciseAngina'].replace('Y', 0)
heartData['ExerciseAngina'] = heartData['ExerciseAngina'].replace('N', 1)

#ST_Slope: Up: 0, Flat: 1, Down: 2
heartData['ST_Slope'] = heartData['ST_Slope'].replace('Up', 0)
heartData['ST_Slope'] = heartData['ST_Slope'].replace('Flat', 1)
heartData['ST_Slope'] = heartData['ST_Slope'].replace('Down', 2)

arr1 = []
arr2 = []
arr3 = []
arr4 = []
#Loop through 50 ensembles
ii=0
while ii < 50:
    features = heartData[['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'ST_Slope', 'Oldpeak']]
    result = heartData[['HeartDisease']]
    #80 20 split
    X1, X2, Y1, Y2 = train_test_split(features, result, test_size=0.2)
    #Create NN, SVM, and DT classifiers
    classifier1 = mlp(solver='lbfgs', alpha=10e-5, hidden_layer_sizes=(20,), random_state=0, max_iter=2000, activation='relu', learning_rate='adaptive', learning_rate_init=0.0005)
    classifier2 = SVC(kernel='linear', C=2)
    classifier3 = sk.DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_split=4)

    #Train Them
    Ynew = np.array(Y1)
    classifier1 = classifier1.fit(X1, Ynew)
    classifier2 = classifier2.fit(X1, Ynew)
    classifier3 = classifier3.fit(X1, Ynew)

    #Make Predictions
    pred1 = classifier1.predict(X2)
    pred2 = classifier2.predict(X2)
    pred3 = classifier3.predict(X2)
    pred4 = np.zeros(len(pred1))
    #Majority rules if at least 2 classifiers say one 1 or 0 choose that one
    i = 0
    while i < len(pred1):
        if (pred1[i] + pred2[i] + pred3[i]) > 1:
            pred4[i] = 1
        else:
            pred4[i] = 0
        i+=1

    #see accuracy of all of them
    accuracy1 = metrics.accuracy_score(Y2, pred1)
    accuracy2 = metrics.accuracy_score(Y2, pred2)
    accuracy3 = metrics.accuracy_score(Y2, pred3)
    accuracy4 = metrics.accuracy_score(Y2, pred4)
    arr1.append(accuracy1)
    arr2.append(accuracy2)
    arr3.append(accuracy3)
    arr4.append(accuracy4)
    ii+=1

print("Accuracy NN: " + str(sum(arr1)/ len(arr1)))
print("Accuracy SVM: " + str(sum(arr2)/ len(arr2)))
print("Accuracy DT: " + str(sum(arr3)/ len(arr3)))
print("Accuracy Ensemble: " + str(sum(arr4)/ len(arr4)))

