import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Cross Validation Examples
iris = load_iris()
X = iris.data
y = iris.target

#TrainTestSplitExample
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize K-Fold CV
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize model
model = LogisticRegression(max_iter=1000)

# Perform K-Fold CV
cv_scores = cross_val_score(model, X, y, cv=kf)

# Print results
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")


# Initialize LOO CV
loo = LeaveOneOut()

# Perform LOO CV
cv_scores_loo = cross_val_score(model, X, y, cv=loo)

# Print results
print(f"Leave-One-Out CV Scores: {cv_scores_loo}")
print(f"Average Accuracy: {np.mean(cv_scores_loo):.2f} (+/- {np.std(cv_scores_loo):.2f})")


# Initialize Stratified K-Fold CV
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Perform Stratified K-Fold CV
cv_scores_stratified = cross_val_score(model, X, y, cv=skf)

# Print results
print(f"Stratified K-Fold CV Scores: {cv_scores_stratified}")
print(f"Average Accuracy: {np.mean(cv_scores_stratified):.2f} (+/- {np.std(cv_scores_stratified):.2f})")

#Example with real data