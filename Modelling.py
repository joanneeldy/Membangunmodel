import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('mushrooms_preprocessing/mushrooms_preprocessed.csv')
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# AKTIFKAN AUTOLOG
mlflow.autolog()

with mlflow.start_run(run_name="Basic RandomForest"):
    # Model dengan parameter tetap (tanpa tuning)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("Basic model training complete with autolog.")
