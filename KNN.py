import pandas as pd
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
# data = pd.read_csv("/content/data.csv")
data = pd.read_csv("indian_liver_patient (1).csv")
Alamine = data['Alamine_Aminotransferase']
Bilirubin = data['Direct_Bilirubin']
Alkaline = data['Alkaline_Phosphotase']
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
Alamine_encoded=le.fit_transform(Alamine)
print (Alamine_encoded)
# Converting string labels into numbers
Bilirubin_encoded=le.fit_transform(Bilirubin)

label=le.fit_transform(Alkaline)
print ("Direct_Bilirubin:",Bilirubin_encoded)
print ("Alkaline_Phosphotase:",label)
#Combinig Alamine and Bilirubin into single listof tuples
features=zip(Alamine_encoded,Bilirubin_encoded)
features=list(features)
print (features)
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(features,label)
#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print ("Predicted Value:", predicted)