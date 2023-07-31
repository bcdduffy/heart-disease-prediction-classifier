import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt


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

best = 0
i = 0
arr = []
#Run 100 times to get a good average
while i < 100:
	#Get data into arrays
	features = heartData[['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'ST_Slope','Oldpeak']]
	result = heartData[['HeartDisease']]
	#Split 80 20
	X1, X2, Y1, Y2 = train_test_split(features, result, test_size=0.2)
	#Create the classifier
	classifier = mlp(solver='lbfgs', alpha=10e-5, hidden_layer_sizes=(20,), random_state=0, max_iter=2000, activation='relu')
	#Train Classifier
	Ynew = np.array(Y1)
	classifier = classifier.fit(X1, Ynew)
	#Test Classifier
	pred = classifier.predict(X2)
	#Get accuracy
	accuracy = metrics.accuracy_score(Y2, pred)
	arr.append(accuracy)
	#Look for best accuracy
	if accuracy > best:
		best = accuracy
		# print("New Best: " + str(accuracy) + " i = " + str(i))
	i += 1
print("Average accuracy: " + str(sum(arr)/ len(arr)))
print("Best accuracy: " + str(best))


