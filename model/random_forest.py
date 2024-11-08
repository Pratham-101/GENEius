import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

dataset = pd.read_csv('data/Dataset.csv')

X = dataset.iloc[:, :-10]
y = dataset.iloc[:, -10:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

individual_accuracies = {}
individual_f1_scores = {}

for target_col in y.columns:
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train[target_col])
    
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test[target_col], y_pred)
    f1 = f1_score(y_test[target_col], y_pred, average='weighted')
    
    individual_accuracies[target_col] = acc
    individual_f1_scores[target_col] = f1

overall_accuracy = sum(individual_accuracies.values()) / len(individual_accuracies)
overall_accuracy = overall_accuracy*100
print("Individual Accuracies for Each Disease:")
for disease, acc in individual_accuracies.items():
    print(f"{disease}: Accuracy = {acc:.4f}, F1 Score = {individual_f1_scores[disease]:.4f}")
    
print(f"\nOverall Accuracy of the Model: {overall_accuracy:.4f}")
