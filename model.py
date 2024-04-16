import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from an Excel file
df = pd.read_excel('/Users/amir/Desktop/own-interpreter/interpret/lib/python3.11/site-packages/mongodb data for clustering.xlsx')

'''Data Processing '''

# Handling missing values
# You can choose to fill missing values with a specific value, mean, median or mode, or drop them
#df.fillna(df.mean(), inplace=True)  # Replace NaN with the mean of each column

# Optionally, drop rows with any NaN values if needed
df.dropna(inplace=True)

# Ensuring correct data types
df['createdAt'] = pd.to_datetime(df['createdAt'])
df['updatedAt'] = pd.to_datetime(df['updatedAt'])

# Convert categorical data to category type
df['Day'] = df['Day'].astype('category')
df['Month'] = df['Month'].astype('category')

# Example of removing outliers for a hypothetical numerical column
# Assuming 'noOfLikes' could have outliers
q_low = df['noOfLikes'].quantile(0.01)
q_hi  = df['noOfLikes'].quantile(0.99)

df = df[(df['noOfLikes'] > q_low) & (df['noOfLikes'] < q_hi)]


'''Feature Engineering '''

# Extracting more detailed time-based features
df['created_hour'] = df['createdAt'].dt.hour
df['created_weekday'] = df['createdAt'].dt.dayofweek  # Monday=0, Sunday=6

# Time between created and updated dates might be a useful feature
df['time_delta_days'] = (df['updatedAt'] - df['createdAt']).dt.days

'''Exploratory Data Analysis'''

# Histograms for numerical features
num_features = ['noOfLikes', 'created_hour', 'time_delta_days']
df[num_features].hist(bins=15, figsize=(15, 6), layout=(2, 2))
plt.show()

# Boxplots to check for outliers in numerical features
plt.figure(figsize=(12, 6))
df.boxplot(column=['noOfLikes'])
plt.show()

# Bar plots for categorical data
cat_features = ['Day', 'Month', 'type']
for feature in cat_features:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=feature, data=df)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
    plt.show()
