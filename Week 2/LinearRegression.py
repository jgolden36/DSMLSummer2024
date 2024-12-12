import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

# Linear regression using analytical techniques
# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = np.dot(X, np.array([1, 2])) + 3
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Example feature matrix with shape (4, 2)
y = np.array([3, 7, 11, 15])  # Example target vector with shape (4,)
X_b = np.c_[np.ones((X.shape[0], 1)), X]
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(beta)
# Linear Regression with maximum likelihood estimation

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)
    for i in range(num_iters):
        # Hypothesis function h(x)
        h = np.dot(X, theta)
        # Calculate the error
        error = h - y
        # Calculate the gradient
        gradient = np.dot(X.T, error) / m
        # Update theta using gradient descent
        theta -= alpha * gradient
        # Calculate cost function
        cost = np.sum((error ** 2)) / (2 * m)
        cost_history[i] = cost
    return theta, cost_history

def linear_regression(X, y, alpha, num_iters):
    # Add intercept term to X
    X = np.column_stack((np.ones(len(X)), X))
    # Initialize theta with zeros
    theta = np.zeros(X.shape[1])
    # Perform gradient descent
    theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)
    return theta, cost_history

# Example usage:
# Assuming X and y are your feature matrix and target vector respectively
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Example feature matrix with shape (4, 2)
y = np.array([3, 7, 11, 15])  # Example target vector with shape (4,)
alpha = 0.01  # Learning rate
num_iters = 1000  # Number of iterations

theta, cost_history = linear_regression(X, y, alpha, num_iters)
plt.plot(range(num_iters), cost_history, 'b-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()

y_pred = X_b.dot(theta)

plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_pred, color='red', label='Linear regression fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()

print("Beta:", beta)

#Regularization: LASSO

def lasso_regression(X, y, lambda_, num_iters=1000, tol=1e-4):
    m, n = X.shape
    beta = np.zeros(n + 1)
    X_b = np.c_[np.ones((m, 1)), X]
    for iteration in range(num_iters):
        beta_old = beta.copy()
        for j in range(n + 1):
            if j == 0:
                beta[j] = np.sum(y - X_b @ beta + beta[j] * X_b[:, j]) / m
            else:
                rho = np.sum(X_b[:, j] * (y - (X_b @ beta - beta[j] * X_b[:, j])))
                if rho < -lambda_ / 2:
                    beta[j] = (rho + lambda_ / 2) / np.sum(X_b[:, j] ** 2)
                elif rho > lambda_ / 2:
                    beta[j] = (rho - lambda_ / 2) / np.sum(X_b[:, j] ** 2)
                else:
                    beta[j] = 0
        if np.sum(np.abs(beta - beta_old)) < tol:
            break
    return beta

lambda_ = 0.1
beta_lasso = lasso_regression(X, y, lambda_)
print("Lasso Coefficients:", beta_lasso)

#Regularization: Ridge

def ridge_regression(X, y, lambda_):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    I = np.eye(X_b.shape[1])
    I[0, 0] = 0  # Don't regularize the intercept term
    beta = np.linalg.inv(X_b.T @ X_b + lambda_ * I) @ X_b.T @ y
    return beta

lambda_ = 1.0
beta_ridge = ridge_regression(X, y, lambda_)
print("Ridge Coefficients:", beta_ridge)

def Ridge_Regression_Optimization(X, y, lambda_, num_iters=1000, tol=1e-4):
    m, n = X.shape
    beta = np.zeros(n + 1)
    X_b = np.c_[np.ones((m, 1)), X]
    for iteration in range(num_iters):
        beta_old = beta.copy()
        for j in range(n + 1):
            if j == 0:
                beta[j] = np.sum(y - X_b @ beta + beta[j] * X_b[:, j]) / m
            else:
                rho = np.sum(X_b[:, j] * (y - (X_b @ beta - beta[j] * X_b[:, j])))
                if rho < -lambda_ / 2:
                    beta[j] = (rho + lambda_ / 2) / (np.sum(X_b[:, j] ** 2))
                elif rho > lambda_ / 2:
                    beta[j] = (rho - lambda_ / 2) / (np.sum(X_b[:, j] ** 2))
                else:
                    beta[j] = 0
        if np.sum(np.abs(beta - beta_old)) < tol:
            break
    return beta

lambda_ = 1.0
beta_ridge = Ridge_Regression_Optimization(X, y, lambda_)
print("Ridge Coefficients:", beta_ridge)

#Regularization: Elastic Net

def elastic_net_regression(X, y, lambda1, lambda2, num_iters=1000, tol=1e-4):
    m, n = X.shape
    beta = np.zeros(n + 1)
    X_b = np.c_[np.ones((m, 1)), X]
    for iteration in range(num_iters):
        beta_old = beta.copy()
        for j in range(n + 1):
            if j == 0:
                beta[j] = np.sum(y - X_b @ beta + beta[j] * X_b[:, j]) / m
            else:
                rho = np.sum(X_b[:, j] * (y - (X_b @ beta - beta[j] * X_b[:, j])))
                if rho < -lambda1 / 2:
                    beta[j] = (rho + lambda1 / 2) / (np.sum(X_b[:, j] ** 2) + lambda2)
                elif rho > lambda1 / 2:
                    beta[j] = (rho - lambda1 / 2) / (np.sum(X_b[:, j] ** 2) + lambda2)
                else:
                    beta[j] = 0
        if np.sum(np.abs(beta - beta_old)) < tol:
            break
    return beta

lambda1 = 0.1
lambda2 = 0.1
beta_elastic_net = elastic_net_regression(X, y, lambda1, lambda2)
print("Elastic Net Coefficients:", beta_elastic_net)

#Linear Regression with sklearn

# Create a linear regression model
lin_reg = LinearRegression()

# Train the model
lin_reg.fit(X, y)

# Predict the target variable
y_pred = lin_reg.predict(X)
print("Linear Regression Coefficients:", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)

#LASSO with Sklearn

# Create a lasso regression model
lasso_reg = Lasso(alpha=0.1)

# Train the model
lasso_reg.fit(X, y)

# Predict the target variable
y_pred_lasso = lasso_reg.predict(X)
print("Lasso Regression Coefficients:", lasso_reg.coef_)
print("Intercept:", lasso_reg.intercept_)

#Ridge with Sklearn

# Create a ridge regression model
ridge_reg = Ridge(alpha=1.0)

# Train the model
ridge_reg.fit(X, y)

# Predict the target variable
y_pred_ridge = ridge_reg.predict(X)
print("Ridge Regression Coefficients:", ridge_reg.coef_)
print("Intercept:", ridge_reg.intercept_)

#Elastic Net with sklearn

# Create an elastic net regression model
elastic_net_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train the model
elastic_net_reg.fit(X, y)

# Predict the target variable
y_pred_elastic_net = elastic_net_reg.predict(X)
print("Elastic Net Coefficients:", elastic_net_reg.coef_)
print("Intercept:", elastic_net_reg.intercept_)

#Example with real world data