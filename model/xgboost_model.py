import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

dataset = pd.read_csv('data/Dataset.csv')

target_columns = dataset.iloc[:, -10:]
print("\nMissing values in target columns before cleaning:\n", target_columns.isnull().sum())

cleaned_dataset = dataset.dropna(subset=target_columns.columns)
print("\nMissing values in target columns after cleaning:\n", cleaned_dataset.iloc[:, -10:].isnull().sum())

X = cleaned_dataset.iloc[:, :-10]
y = cleaned_dataset.iloc[:, -10:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {}
y_pred = pd.DataFrame()

for disease in y.columns:
    print(f"Training model for {disease}")
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train[disease])
    models[disease] = xgb_model    

    y_pred[disease] = xgb_model.predict(X_test)

overall_accuracy = 0
for disease in y.columns:
    print(f"\nClassification Report for {disease}:")
    print(classification_report(y_test[disease], y_pred[disease], zero_division=0))
    
    disease_accuracy = accuracy_score(y_test[disease], y_pred[disease])
    print(f"Accuracy for {disease}: {disease_accuracy}")
    
    overall_accuracy += disease_accuracy

overall_accuracy /= len(y.columns)
overall_accuracy = overall_accuracy * 100
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
