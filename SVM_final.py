import pandas as pd
import numpy as np
import sklearn.svm as sk
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#   This script is the final SVM model training and testing script
#   The scrit will create a SVM classifier using the best kernel
#   and C Parameter found for an SVm and calculates its accuracy.

#Reading the data set
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

arr = []

#Producing features and result matrices
features = heartData[['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]
result = heartData[['HeartDisease']]

#Splitting the data and shuffling using the train_test_split function
X1, X2, Y1, Y2 = train_test_split(features, result, test_size = 0.2)
    
#Creating the SVM model using the best found parameters
classifier = SVC(kernel='linear', C=2)

classifier = classifier.fit(X1, np.ravel(Y1))  ###

#Running predicitons using the trained model to later find the accuracy
pred = classifier.predict(X2)

#From the oredicitons calculatingthet oal accuracy of the model
accuracy = metrics.accuracy_score(Y2, pred)
arr.append(accuracy)

#Printing the final accuracy along with the kernel used and C parameter value chosen
print("SVM with ", 'linear', " kernel with C parameter value ", 2, " accuracy calculated as: ", accuracy)



