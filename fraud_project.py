# Import libraries and machine learning tools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the credit card transaction dataset
df = pd.read_csv("creditcard.csv.zip")

# Explore the dataset
print(df.head())
print(df.shape)
print(df.columns)
df.info()
print(df.isnull().sum())
print(df["Class"].value_counts())

# Separate the input features from the fraud target
x = df.drop("Class", axis=1)
y = df["Class"]

# Split the data into 80% training data and 20% test data
x_train, x_test, y_train, y_test = train_test_split(
x,
y,
test_size=0.2,
random_state=42,
stratify=y)

# Scale Time and amount using values learned from the training data
scaler = StandardScaler()
x_train[["Time", "Amount"]] = scaler.fit_transform(x_train[["Time", "Amount"]])
x_test[["Time", "Amount"]] = scaler.fit_transform(x_test[["Time", "Amount"]])

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Make predictions using Logistic Regression
predictions = model.predict(x_test)

# Evaluate Logistic Regression performance
print("LOGISTIC REGRESSION RESULTS")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Train Random Forest model
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(x_train, y_train)

# Make predictions using Random Forest
rf_predictions = random_forest.predict(x_test)

# Evaluate Random Forest performance
print("RANDOM FOREST RESULTS")
print(classification_report(y_test, rf_predictions))
print(confusion_matrix(y_test, rf_predictions))
