# Titanic Survival Prediction - CodSoft Internship Project

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Dataset
df = pd.read_csv("titanic.csv")  # Use your actual file name if different
print("First 5 rows of dataset:")
print(df.head())

# Step 3: Exploratory Data Analysis (EDA)
print("\nBasic Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Step 4: Data Preprocessing
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)  # Drop unused columns
df.dropna(inplace=True)  # Remove rows with missing values

# Convert categorical variables to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Step 5: Feature Selection
X = df.drop('Survived', axis=1)
y = df['Survived']

# Step 6: Split into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Visualization (Optional)
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()
