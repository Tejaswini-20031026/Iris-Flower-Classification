# ==========================================
# IRIS FLOWER CLASSIFICATION USING ML
# ==========================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Iris.csv")

# Remove unnecessary Id column
df = df.drop("Id", axis=1)

print("Dataset after removing Id:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ==========================================
# 2. DATA UNDERSTANDING
# ==========================================

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpecies Distribution:")
print(df["Species"].value_counts())


# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

# Species distribution
plt.figure(figsize=(7, 5))

sns.countplot(x="Species", data=df)

plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")

plt.show()


# Pairplot
sns.pairplot(df, hue="Species")

plt.show()


# Correlation heatmap
plt.figure(figsize=(8, 6))

correlation = df.drop("Species", axis=1).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.show()


# Feature distributions
df.drop("Species", axis=1).hist(
    figsize=(10, 8),
    bins=15
)

plt.suptitle("Distribution of Iris Features")

plt.show()


# ==========================================
# 4. FEATURE AND TARGET SELECTION
# ==========================================

X = df.drop("Species", axis=1)
y = df["Species"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 6. RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nRandom Forest model training completed successfully!")


# Make predictions
y_pred = model.predict(X_test)

print("\nPredicted Species:")
print(y_pred)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:", accuracy * 100, "%")


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    ],
    yticklabels=[
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    ]
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.show()


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy


print("\nModel Comparison:")

for name, accuracy in results.items():

    print(f"{name}: {accuracy * 100:.2f}%")


# ==========================================
# 9. MODEL ACCURACY COMPARISON GRAPH
# ==========================================

model_names = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(8, 5))

plt.bar(
    model_names,
    [accuracy * 100 for accuracy in accuracies]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 110)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# ==========================================
# 10. TRAIN BEST MODEL - KNN
# ==========================================

best_model = KNeighborsClassifier()

best_model.fit(X_train, y_train)

print("\nBest Model: K-Nearest Neighbors")


# ==========================================
# 11. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    best_model,
    "iris_knn_model.pkl"
)

print("Model saved successfully!")


# ==========================================
# 12. NEW FLOWER PREDICTION
# ==========================================

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = best_model.predict(new_flower)

print("\nNew Flower Prediction:")
print("Predicted Species:", prediction[0])


# ==========================================
# PROJECT COMPLETED
# ==========================================

print("\n==========================================")
print("Iris Flower Classification Project Completed!")
print("==========================================")