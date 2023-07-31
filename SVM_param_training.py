import pandas as pd
import numpy as np
import sklearn.svm as sk
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#   The purpose of this file is for hyperparameter tuning. Below the loop will for a
#   selected kernel iterate through C parameter values from the array below producing
#   the accuracy of each model with each combination. This could be used to find
#   the most accurate kernels with their possible C parameters.

#Reading the data set from the kaggle file
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

#Array of all possible kernels to choose, the user can choose which kernel by choosing the kerneal index
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kern = 3

#Array of C Parameter values to iterate through when finding the best value to choose given the kernel
c_vals = [0.1, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
i = 0
arr = []
#Iterates 15 times through the possible C vlaues in the array to test each ones accuracy given the kernel
while i < 15:

	#Producing features and result matrices
	features = heartData[['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]
	result = heartData[['HeartDisease']]

	#Splits and shuffles data using the train_test_split function
	X1, X2, Y1, Y2 = train_test_split(features, result, test_size = 0.2)
    
	#Creates the classifier using the kernel selected and the iteration of the C value choasen
	classifier = SVC(kernel=kernels[kern], C=c_vals[i])

	classifier = classifier.fit(X1, np.ravel(Y1))  ###

	#Running predicitons using the trained model to later find the accuracy
	pred = classifier.predict(X2)

	#From the oredicitons calculatingthet oal accuracy of the model
	accuracy = metrics.accuracy_score(Y2, pred)
	arr.append(accuracy)

	#Printing the final accuracy along with the kernel used and C parameter of the current iteration
	print("SVM with ", kernels[kern], " kernel with C parameter value ", c_vals[i], " accuracy calculated as: ", accuracy)

	i += 1

plt.show()