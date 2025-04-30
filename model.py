import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Data cleaning
# df = pd.read_csv('/Users/cengiz/Desktop/Titanic-Project/titanic.csv')
# df['Age'].fillna(df['Age'].mean(), inplace=True)
# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# df.drop(columns=['Cabin'], inplace=True)
# df.to_csv("titanic_clean.csv", index=False)

df = pd.read_csv('/Users/cengiz/Desktop/Titanic-Project/titanic_clean.csv')

# Convert string columns to numerical
df['Sex'] = df['Sex'].map({'male': 0, "female": 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Define features and target for model training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy and confusion matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model to a .pkl file
joblib.dump(model, 'titanic_model.pkl')
