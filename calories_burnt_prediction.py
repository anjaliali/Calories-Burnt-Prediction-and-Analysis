# -*- coding: utf-8 -*-
"""Calories_burnt_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HxLa--PxOHSAbBBFiIPH41jvTNuxtgd0

*Importing the libraries*
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics

"""*Data Collection & Pre-Processing*"""

# loading the calories data from csv file to a DataFrame
calories=pd.read_csv('/content/calories.csv')
# print the top 5 rows of the dataframe
calories.head()

# print the bottom 5 rows of the dataframe
calories.tail()

# loading the exercise data from csv file to a DataFrame
exercise=pd.read_csv('/content/exercise.csv')
# print the top 5 rows of the dataframe
exercise.head()

# print the bottom 5 rows of the dataframe
exercise.tail()

"""*Combining the two Dataframes*"""

combined_data=pd.concat([exercise, calories['Calories']], axis=1)
combined_data.head()

combined_data.tail()

# checking the number of rows and columns
combined_data.shape

# getting some information about the data
combined_data.info()

# checking for the missing values
combined_data.isnull().sum()

"""*Data Analysis*"""

# get some statistical measures about the data
combined_data.describe()

# drop User_ID column because this is not required from main dataframe itself
combined_data.drop(columns = ["User_ID"],axis=1,inplace =True)
combined_data.head()

"""*Data Visualization*"""

sns.set()

sns.lineplot(x='Duration', y='Calories', data=combined_data)

sns.lineplot(x='Body_Temp', y='Calories', data=combined_data)

sns.lineplot(x='Heart_Rate', y='Calories', data=combined_data)

# plotting the gender column in count plot
sns.countplot(combined_data['Gender'])
plt.show()

# finding the distribution of "Age" column
sns.distplot(combined_data['Age'])

# finding the distribution of "Height" column
sns.distplot(combined_data['Height'])

# finding the distribution of "Weight" column
sns.distplot(combined_data['Weight'])

# finding the distribution of "Duration" column
sns.distplot(combined_data['Duration'])

# finding the distribution of "Heart_Rate" column
sns.distplot(combined_data['Heart_Rate'])

# finding the distribution of "Body_Temp" column
sns.distplot(combined_data['Body_Temp'])

# converting categorical data of gender column into numerical data
combined_data.replace({"Gender":{'male':0,'female':1}},inplace=True)
combined_data.head()

combined_data.head()

"""*Correlation in the data set*
1. Positive Correlation
2. Negative Correlation
"""

correlation=combined_data.corr()
# constructing a heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')

"""*Separating Features and Target*


"""

X=combined_data.drop(columns=['Calories'],axis=1)
Y=combined_data['Calories']

print(X)

print(Y)

"""*Splitting the data into training data and test data*"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

print(X.shape,X_train.shape,X_test.shape)

"""*XGBRegressor Model Training*"""

# loading the model
xgb=XGBRegressor()
# training the model with x train
xgb.fit(X_train,Y_train)

"""*Model Evaluation*"""

# prediction on test data
test_data_prediction=xgb.predict(X_test)
print(test_data_prediction)
# mean absolute error
mae=metrics.mean_absolute_error(Y_test,test_data_prediction)
print("Mean Absolute Error: ",mae)
# mean square error
mse=metrics.mean_squared_error(Y_test,test_data_prediction)
print("Mean Squared Error: ",mse)
# r square
r2=metrics.r2_score(Y_test,test_data_prediction)
print("R Square: ",r2)

def predict_calories_burnt():
    age = int(input("Enter age: "))
    gender = int(input("Enter gender (0 for male, 1 for female): "))
    height = float(input("Enter height in centimeters: "))
    weight = float(input("Enter weight in kilograms: "))
    duration = float(input("Enter duration in minutes: "))
    heart_rate = int(input("Enter heart rate: "))
    body_temp = float(input("Enter body temperature in degrees Celsius: "))

    # Create user_data DataFrame with the correct feature order
    user_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                             columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

    prediction = xgb.predict(user_data)
    print(f"Prediction: {prediction[0]}") # Assuming prediction is an array, access the first element

# Call the user input function to test
predict_calories_burnt()

"""*Decision Tree Regressor Model Training*"""

# loading the model
dt_model = DecisionTreeRegressor(random_state=2)
# training the model with x train
dt_model.fit(X_train,Y_train)

"""*Model Evaluation*"""

# prediction on test data
test_data_prediction=dt_model.predict(X_test)
print(test_data_prediction)
# mean absolute error
mae=metrics.mean_absolute_error(Y_test,test_data_prediction)
print("Mean Absolute Error: ",mae)
# mean square error
mse=metrics.mean_squared_error(Y_test,test_data_prediction)
print("Mean Squared Error: ",mse)
# r square
r2=metrics.r2_score(Y_test,test_data_prediction)
print("R Square: ",r2)

def predict_calories_burnt():
    age = int(input("Enter age: "))
    gender = int(input("Enter gender (0 for male, 1 for female): "))
    height = float(input("Enter height in centimeters: "))
    weight = float(input("Enter weight in kilograms: "))
    duration = float(input("Enter duration in minutes: "))
    heart_rate = int(input("Enter heart rate: "))
    body_temp = float(input("Enter body temperature in degrees Celsius: "))

    # Create user_data DataFrame with the correct feature order
    user_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                             columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

    prediction = dt_model.predict(user_data)
    print(f"Prediction: {prediction[0]}") # Assuming prediction is an array, access the first element

# Call the user input function to test
predict_calories_burnt()

"""*Random Forest Regressor Model Training*"""

# loading the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=2)  # Adjust n_estimators as needed
# training the model with x train
rf_model.fit(X_train,Y_train)

"""*Model Evaluation*"""

# prediction on test data
test_data_prediction=rf_model.predict(X_test)
print(test_data_prediction)
# mean absolute error
mae=metrics.mean_absolute_error(Y_test,test_data_prediction)
print("Mean Absolute Error: ",mae)
# mean square error
mse=metrics.mean_squared_error(Y_test,test_data_prediction)
print("Mean Squared Error: ",mse)
# r square
r2=metrics.r2_score(Y_test,test_data_prediction)
print("R Square: ",r2)

def predict_calories_burnt():
    age = int(input("Enter age: "))
    gender = int(input("Enter gender (0 for male, 1 for female): "))
    height = float(input("Enter height in centimeters: "))
    weight = float(input("Enter weight in kilograms: "))
    duration = float(input("Enter duration in minutes: "))
    heart_rate = int(input("Enter heart rate: "))
    body_temp = float(input("Enter body temperature in degrees Celsius: "))

    # Create user_data DataFrame with the correct feature order
    user_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                             columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

    prediction = rf_model.predict(user_data)
    print(f"Prediction: {prediction[0]}") # Assuming prediction is an array, access the first element

# Call the user input function to test
predict_calories_burnt()

"""*Support Vector Regressor Model Training*"""

# loading the model
svr_model = SVR(kernel='rbf')
# training the model with x train
svr_model.fit(X_train,Y_train)

"""*Model Evaluation*"""

# prediction on test data
test_data_prediction=svr_model.predict(X_test)
print(test_data_prediction)
# mean absolute error
mae=metrics.mean_absolute_error(Y_test,test_data_prediction)
print("Mean Absolute Error: ",mae)
# mean square error
mse=metrics.mean_squared_error(Y_test,test_data_prediction)
print("Mean Squared Error: ",mse)
# r square
r2=metrics.r2_score(Y_test,test_data_prediction)
print("R Square: ",r2)

def predict_calories_burnt():
    age = int(input("Enter age: "))
    gender = int(input("Enter gender (0 for male, 1 for female): "))
    height = float(input("Enter height in centimeters: "))
    weight = float(input("Enter weight in kilograms: "))
    duration = float(input("Enter duration in minutes: "))
    heart_rate = int(input("Enter heart rate: "))
    body_temp = float(input("Enter body temperature in degrees Celsius: "))

    # Create user_data DataFrame with the correct feature order
    user_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                             columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

    prediction = svr_model.predict(user_data)
    print(f"Prediction: {prediction[0]}") # Assuming prediction is an array, access the first element

# Call the user input function to test
predict_calories_burnt()
import pickle
pickle.dump(xgb,open('xgb.pkl','wb'))
