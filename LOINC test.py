from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd

# Load the data
loinc_df = pd.read_csv("Loinc.csv", on_bad_lines='skip')

# Prepare the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(loinc_df["COMPONENT"])
y = loinc_df["LOINC_NUM"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")