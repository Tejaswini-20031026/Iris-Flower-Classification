import pandas as pd
import joblib

# Load the saved model
model = joblib.load("iris_knn_model.pkl")

# Take flower measurements from the user
sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

# Create input DataFrame
new_flower = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

# Make prediction
prediction = model.predict(new_flower)

print("\nPredicted Iris Species:", prediction[0])