import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Step 1: Extract the dataset from the ZIP archive
with zipfile.ZipFile(r"C:\Users\harsh\OneDrive\Desktop\New folder\archive (2).zip", "r") as zip_ref:
    zip_ref.extractall("extracted_data")  # Destination folder for extracted files

# Step 2: Load the dataset
data = pd.read_csv("extracted_data/fraudTest.csv")

# Step 3: Data Exploration
print(data.head())
print(data.info())
print(data.columns)  # Check column names

# Check for the correct column containing class labels
column = 'is_fraud'
print(data[column].value_counts())

# Step 4: Data Preprocessing
# Drop non-numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Split the dataset into features (X) and target (y)
X = numeric_data.drop(column, axis=1)
y = numeric_data[column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection and Training
# Logistic Regression
lo_model = LogisticRegression()
lo_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Step 6: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(classification_report(y_test, y_predicted))

print("Logistic Regression Metrics:")
evaluate_model(lo_model, X_test, y_test)

print("Random Forest Metrics:")
evaluate_model(rf_model, X_test, y_test)




