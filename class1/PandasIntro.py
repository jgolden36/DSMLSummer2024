#Pandas
import pandas as pd
import numpy as np
import matplotlib as plt

# Creating Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

#Creating Dataframe
data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Age': [28, 24, 35, 32],
        'City': ['New York', 'Paris', 'Berlin', 'London']}
df = pd.DataFrame(data)
print(df)

#Viewing Data
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

#Data Types
print(df.dtypes)
df['Age'] = df['Age'].astype(float)
print(df.dtypes)

#Data Selection
print(df['Name'])
print(df[['Name', 'City']])

#Indexing
print(df.loc[1])
print(df.iloc[1])
print(df[df['Age'] > 30])
df.set_index('Name', inplace=True)
print(df)
df.reset_index(inplace=True)
print(df)

#Data Cleaning
print(df.isnull())
df['Age'].fillna(df['Age'].mean(), inplace=True)
print(df)
df.dropna(inplace=True)
print(df)

#Removing Duplicates
df.drop_duplicates(inplace=True)
print(df)

#Renaming Columns and Index
df.rename(columns={'Age': 'Years'}, inplace=True)
print(df)

#Applying Functions
print(df.apply(np.sum))

#Lambda Functions
print(df['Years'].map(lambda x: x * 2))

#Groupby and Aggregation
grouped = df.groupby('City')
print(grouped.mean())

#Pivot Tables
pivot_table = df.pivot_table(values='Years', index='City', aggfunc=np.mean)
print(pivot_table)

#Concatenation, appending, and joining
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2']})
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                    'B': ['B3', 'B4', 'B5']})
result = pd.concat([df1, df2])
print(result)

left = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                     'A': ['A0', 'A1', 'A2']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K3'],
                      'B': ['B0', 'B1', 'B3']})
merged = pd.merge(left, right, on='key', how='inner')
print(merged)

# Datetime Operations

df['Date'] = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
print(df)

print(df['Date'].dt.year)
print(df['Date'].dt.month)
print(df['Date'].dt.day)

#Input and Output
df = pd.read_csv('ExampleDataset.csv')
print(df)

df = pd.read_excel('ExampleExcel.xlsx', sheet_name='Sheet1')
print(df)

df.to_csv('output.csv', index=False)

df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

#Visualization

df.plot(kind='line', x='Name', y='Years')
plt.show()

#Basic Integration with Sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = {'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

# Splitting the data
X = df[['feature1', 'feature2']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)
print(X_test_scaled)

#Example with Real Dataset
