Titanic Survival Prediction Project
This project implements a machine learning pipeline using scikit-learn to predict passenger survival based on the Titanic dataset. It covers the complete workflow from data cleaning and feature engineering to hyperparameter tuning and model evaluation.

Features
Custom Data Preprocessing: Handles missing values by filling the Fare column with its median and calculating Age based on passenger class specific medians.

Feature Engineering: Creates new features to improve model performance, including FamilySize, isAlone, FareBin, and AgeBin.

Hyperparameter Tuning: Utilizes GridSearchCV to optimize the KNeighborsClassifier, testing various configurations for n_neighbors, metric, and weights.

Performance Visualization: Generates a confusion matrix heatmap using Seaborn and Matplotlib to visualize the accuracy of predictions.

Data Scaling: Uses MinMaxScaler to normalize features before training the model.

Project Structure
main.py: The primary script containing the data processing logic, model training, and evaluation.

tested.csv: The dataset containing passenger information such as Pclass, Sex, Age, Fare, and Survived status.

Requirements
To run this project, you will need the following Python libraries:

pandas

numpy

scikit-learn

matplotlib

seaborn

How to Run
Ensure tested.csv and main.py are in the same directory.

Execute the script:

Bash

python main.py
The script will output the model's accuracy to the console and display a Confusion Matrix plot.

Model Evaluation
The project evaluates the optimized model using:

Accuracy Score: Provides a percentage of correct survival predictions.

Confusion Matrix: Details true positives, true negatives, and misclassifications for "Survived" vs. "Not Survived".