# Iris Flower Classification Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to predict the species of an Iris flower based on its sepal and petal measurements.

The three Iris species considered are:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Objective

To build a machine learning model that can accurately classify Iris flowers using four measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Machine Learning Models

The following models were trained and compared:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.67% |
| K-Nearest Neighbors | 100.00% |
| Decision Tree | 93.33% |
| Random Forest | 90.00% |

## Best Model

K-Nearest Neighbors achieved the highest accuracy of 100% on the test dataset used in this experiment.

## Dataset

The Iris dataset contains 150 samples with four numerical features and one target variable.

## Project Workflow

Dataset Loading  
↓  
Data Cleaning  
↓  
Exploratory Data Analysis  
↓  
Feature Selection  
↓  
Train-Test Split  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Model Comparison  
↓  
Best Model Selection  
↓  
New Flower Prediction

## Prediction

The trained KNN model was saved using Joblib as:

`iris_knn_model.pkl`

The model can be loaded later and used to predict the species of a new Iris flower.

## Conclusion

The project successfully demonstrates a complete machine learning classification workflow. Among the four tested models, K-Nearest Neighbors achieved the highest test accuracy of 100% for the selected train-test split.