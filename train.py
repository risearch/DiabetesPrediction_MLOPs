# train.py
#Import benoenigter Bibliotheken

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Datenset von Quelle laden (Kaggle/hosted)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("Spalten:", df.columns.tolist()) 

#Daten Bereinigung
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].isin([0]).sum()

# Daten vorbereiten
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Datensatz aufsplitten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Speichern
joblib.dump(model, "diabetes_model.pkl")
print("Modell gespeichert als diabetes_model.pkl")
