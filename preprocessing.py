"""
Stroke Prediction - Data Preprocessing
Student: [Your Name]
Date: [Date]

Following the TA's approach from class examples
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import os

# Load the data
print("Loading stroke data...")

# Check if file exists in data folder
if os.path.exists('data/healthcare-dataset-stroke-data.csv'):
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    print(f"✓ Data loaded! Shape: {df.shape}")
else:
    print("❌ ERROR: Can't find healthcare-dataset-stroke-data.csv in data folder!")
    exit()

print("\nFirst few rows:")
print(df.head())

# Check the data
print("\nData info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Fix BMI column - it has 'N/A' as string
print("\nFixing BMI column...")
df['bmi'] = df['bmi'].replace('N/A', np.nan)
df['bmi'] = pd.to_numeric(df['bmi'])

# Remove ID column (not needed for prediction)
print("\nRemoving ID column...")
df = df.drop('id', axis=1)

# IMPORTANT: Shuffle the data to mix stroke cases
print("\nShuffling data to randomize order...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Done! Data is now randomized")

# Separate the target (what we want to predict)
print("\nSeparating target variable (stroke)...")
y = df['stroke']
X = df.drop('stroke', axis=1)
print(f"Target distribution: \n{y.value_counts()}")

# ============================================================================
# NUMERICAL FEATURES - Following TA's approach
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Handle Numerical Features")
print("="*70)

# Select numerical features (exclude 'object' type) - same as TA
Numerical_Features = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Numerical features: {Numerical_Features}")

# Create dataframe with numerical features only
dataframe_N = X[Numerical_Features]
print(f"\nShape: {dataframe_N.shape}")
print(f"Missing values:\n{dataframe_N.isnull().sum()}")

# Impute missing values with mean - same as TA
print("\nImputing missing values with mean...")
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(dataframe_N)
dataframe_N = imp_mean.transform(dataframe_N)
print("Done!")

# Scale numerical features - same as TA
print("\nScaling numerical features with MinMaxScaler...")
scaler = MinMaxScaler()
dataframe_N = scaler.fit_transform(dataframe_N)
print("Done! All values now between 0 and 1")

# Convert back to DataFrame to keep column names
dataframe_N = pd.DataFrame(dataframe_N, columns=Numerical_Features)
print("\nNumerical features after preprocessing:")
print(dataframe_N.head())

# ============================================================================
# CATEGORICAL FEATURES - Following TA's approach
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Handle Categorical Features")
print("="*70)

# Select categorical features (include 'object' type) - same as TA
Categ_Feature = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical features: {Categ_Feature}")

# Create dataframe with categorical features only
dataframe_C = X[Categ_Feature]
print(f"\nShape: {dataframe_C.shape}")
print(f"Missing values:\n{dataframe_C.isnull().sum()}")

# Impute missing values with most_frequent - same as TA
print("\nImputing missing values with most_frequent...")
imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mean.fit(dataframe_C)
dataframe_C = imp_mean.transform(dataframe_C)
print("Done!")

# Convert back to DataFrame
dataframe_C = pd.DataFrame(dataframe_C, columns=Categ_Feature)
print("\nCategorical features after imputation:")
print(dataframe_C.head())

# Need to convert categorical to numbers for the model
print("\nConverting categorical features to numbers...")
from sklearn.preprocessing import LabelEncoder

for col in Categ_Feature:
    le = LabelEncoder()
    dataframe_C[col] = le.fit_transform(dataframe_C[col])
    print(f"  {col}: converted to numbers")

print("\nCategorical features after encoding:")
print(dataframe_C.head())

# ============================================================================
# COMBINE EVERYTHING
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Combine All Features")
print("="*70)

# Put numerical and categorical features back together
X_processed = pd.concat([dataframe_N, dataframe_C], axis=1)
print(f"Final shape: {X_processed.shape}")
print(f"Features: {X_processed.columns.tolist()}")

print("\nFinal processed data:")
print(X_processed.head())

# ============================================================================
# SAVE THE PROCESSED DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Save Processed Data")
print("="*70)

# Make sure data folder exists
if not os.path.exists('data'):
    os.makedirs('data')
    print("Created 'data' folder")

# Save to CSV files ONLY in data folder
X_processed.to_csv('data/X_processed.csv', index=False)
y.to_csv('data/y_processed.csv', index=False)

print("\n✓ Saved files in data folder:")
print("  - data/X_processed.csv (features)")
print("  - data/y_processed.csv (target)")
print("\nNote: Files are saved ONLY in data folder, not in root!")

print("\n" + "="*70)
print("DONE! ✓")
print("="*70)
print(f"\nProcessed {X_processed.shape[0]} samples")
print(f"With {X_processed.shape[1]} features")
print("\nReady for model training!")