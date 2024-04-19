import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime


'''Data Loading'''
df = pd.read_excel("/Users/amir/Desktop/own-interpreter/interpret/lib/python3.11/site-packages/mongodb data for clustering.xlsx")
print(df.head())
print("The columns are :", df.columns)

'''Data Cleaning'''
#Impute missing values for numerical columns with the median
num_features = df.select_dtypes(include= ['int64', 'float64']).columns.tolist()
# Replace missing categorical values with the mode
cat_features = df.select_dtypes(include=['object']).columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

''' Feature Engineering '''
# Time-based features
df['createdAt'] = pd.to_datetime(df['createdAt'])
df['day_of_week'] = df['createdAt'].dt.dayofweek
df['hour_of_day'] = df['createdAt'].dt.hour
df['month'] = df['createdAt'].dt.month

# Advanced features might include time since last session, session duration, etc.
# Example: Calculate time since last login for each user
df.sort_values(by=['username', 'createdAt'], inplace=True)
df['time_since_last_login'] = df.groupby('username')['createdAt'].diff().apply(lambda x: x.total_seconds()/3600.0)

''' Label Definition '''
# Calculate the number of sessions per user using 'username' as the identifier
df['session_count'] = df.groupby('username')['username'].transform('count')

# Define the 'retained' status based on session_count and recent activity
# For example, users with more than 3 sessions and who logged in within the last 24 hours are considered retained
df['retained'] = (df['session_count'] > 3) & (df['time_since_last_login'] <= 24)

''' Preprocess the Dataset '''
# Apply the preprocessing pipeline to the DataFrame
df_processed = preprocessor.fit_transform(df)

# Get feature names from one-hot encoded categorical features
# Combining original numerical features with new one-hot encoded feature names
feature_names = num_features + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features))

# Convert the processed matrix back to a DataFrame
df_processed = pd.DataFrame(df_processed, columns=feature_names)
print(df_processed.head())

''' Data Preparation for Model Training '''
# Assuming you would want to separate features and target label
X = df_processed.drop('retained', axis=1)  # Drop the target variable to create a features only DataFrame
y = df['retained']  # Target variable

# If 'retained' became part of the processed DataFrame, use the following instead:
# X = df_processed.drop(columns='retained')
# y = df_processed['retained']

print("Features and Labels prepared for model training:")
print(X.head())
print(y.head())


