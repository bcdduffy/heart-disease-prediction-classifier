import pandas as pd
import sklearn.tree as sk
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Load in .csv file
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

#Get data into columns.
features = heartData[['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]
result = heartData[['HeartDisease']]

#Split with and 80/20 train/test split
X1, X2, Y1, Y2 =  train_test_split(features, result, test_size=.2)

#create classifer with hyperparameter
classifier = sk.DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_split=4)

#Fit the model
classifier = classifier.fit(X1, Y1)

#Predictions for all of test set
pred = classifier.predict(X2)

#Compute accuracy
accuracy = metrics.accuracy_score(Y2, pred)

print(accuracy)

#Plot decision tree.
tree.plot_tree(classifier)
plt.show()

