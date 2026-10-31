import pandas as pd
import numpy as np
import dtale as d
import sweetviz as s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

print("\n" + "-" * 50)
print("       Data set")
print("-" * 50)
df=pd.read_csv("House_price1.csv")

print("\n" + "-" * 50)
print("       Understanding Dataset")
print("-" * 50)
print("\nShape :", df.shape)
print("\nColumns :", df.columns.tolist()) 

print("\n" + "-" * 50)
print("       Quantitative Data")
print("-" * 50)
print("\nquantitative data:int64 and float64")
quantitative=df.select_dtypes(include=['int64','float64'])
print("quantitative data:",quantitative)
print("quantitative column labels:\n",quantitative.columns)

print("\n" + "-" * 50)
print("       Qualitative Data")
print("-" * 50)
print("\n qualitative data:object and category")
qualitative=df.select_dtypes(include=['object','category'])
print("Qualitative data:",qualitative)
print("qualitative column labels:\n",qualitative.columns)

print("\n" + "-" * 50)
print("       EDA")
print("-" * 50)
print(df.head(5))
print(df.tail(5))
print(df.isnull().sum())
print(df.describe())
print(df.nunique())

print("\n" + "-" * 50)
print("       Correlation and covariation")
print("-" * 50)
print(df.cov(numeric_only=True))
print(df.corr(numeric_only=True))

print("\n" + "-" * 50)
print("       Value Counts")
print("-" * 50)
print(df['Area'].value_counts())
print(df['BHK'].value_counts())
print(df['Size_sqft'].value_counts())
print(df['Bathrooms'].value_counts())
print(df['Parking'].value_counts())
print(df['Age_Years'].value_counts())
print(df['Floor'].value_counts())
print(df['Total_Floors'].value_counts())
print(df['Furnished'].value_counts())
print(df['Price_INR'].value_counts())

print("\n" + "-" * 50)
print("       Data preprocessing")
print("-" * 50)
x=df.drop(columns=['Price_INR'])
y=df['Price_INR']


print("\n" + "-" * 50)
print("       categorical and numerical columns")
print("-" * 50)
categorical_columns=['Area','Furnished']
numeric_columns=[
    'BHK',
    'Size_sqft',
    'Bathrooms',
    'Parking',
    'Age_Years',
    'Floor',
    'Total_Floors']

preprocessor=ColumnTransformer(transformers=[('cat',OneHotEncoder(handle_unknown='ignore'),categorical_columns),('num','passthrough',numeric_columns)])

print("\n" + "-" * 50)
print("       Regression")
print("-" * 50)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=Pipeline(steps=[('preprocessor',preprocessor),('regressor',LinearRegression())])
    
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("\n" + "-" * 50)
print("      R2 and RSS calculation")
print("-" * 50)
r2=r2_score(y_test,y_pred)
print("r2 value:",r2)

rss=((y_test-y_pred)**2).sum()
print("RSS:",rss)

print("\n" + "-" * 50)
print("       MAE and MSE calculation")
print("-" * 50)

mae=mean_absolute_error(y_test,y_pred)
print("mean mae:",mae)
mse=mean_squared_error(y_test,y_pred)
print("mean mse:",mse)

print("\n" + "-" * 50)
print("       RMSE calculation")
print("-" * 50)

rmse=np.sqrt(mse)
print("RMSE:",rmse)

area = input("Enter Area: ")
bhk = int(input("Enter BHK: "))
size = float(input("Enter Size in sqft: "))
bathrooms = int(input("Enter Bathrooms: "))
parking = int(input("Enter Parking: "))
age = int(input("Enter Age in Years: "))
floor = int(input("Enter Floor: "))
total_floors = int(input("Enter Total Floors: "))
furnished = input("Enter Furnished Type: ")

house = pd.DataFrame([[area, bhk, size, bathrooms, parking, age, floor, total_floors, furnished]],columns=x.columns)
    
prediction = model.predict(house)

print("\n" + "-" * 50)
print("             PREDICTION RESULT")
print("-" * 50)

print("Area              :", area)
print("BHK               :", bhk)
print("Size              :", size, "sqft")
print("Bathrooms         :", bathrooms)
print("Parking           :", parking)
print("Age               :", age, "years")
print("Floor             :", floor)
print("Total Floors      :", total_floors)
print("Furnished Type    :", furnished)

print("\nPredicted House Price : ₹", format(prediction[0], ",.2f"))

print("\n" + "=" * 50)
print("          PREDICTION COMPLETED")
print("=" * 50)    