from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names) #to convert the data into a DataFrame
df['target'] = data.target
df.head()
df.describe()  #to get the summary statistics of the DataFrame
df.sample(5)  #to get a random sample of 5 rows from the DataFrame

# Save the DataFrame to a CSV file
df.to_csv('cancer_data.csv', index=False)


X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features to ensure they have a mean of 0 and a standard deviation of 1 and have similar ranges

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the processed data to CSV files
pd.DataFrame(X_train, columns=X.columns).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv('X_test.csv', index=False)
pd.DataFrame(y_train, columns=['target']).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test, columns=['target']).to_csv('y_test.csv', index=False)

# Train different models 

lr = LogisticRegression()  # Logistic Regression
lr.fit(X_train, y_train)


svc = SVC(kernel='linear') # Support Vector Classifier
svc.fit(X_train, y_train)


rf = RandomForestClassifier(n_estimators=100) # Random Forest Classifier
rf.fit(X_train, y_train)


# Evaluate the models
models = {'Logistic Regression': lr, 'SVC': svc, 'Random Forest': rf}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))
    print("="*50)

# Save the models

joblib.dump(rf, 'random_forest_model.pkl')
