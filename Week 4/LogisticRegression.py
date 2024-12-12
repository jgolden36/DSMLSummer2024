import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
#Initialized data
X = np.array([[0.50, 1], [0.75, 1], [1.00, 1], [1.25, 0], [1.50, 0], [1.75, 0], [2.00, 0], [2.25, 1], [2.50, 0], [2.75, 1]])
y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 1])
#Logistic Regression with maximum likelihood

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, weights):
    z = np.dot(X, weights)
    ll = np.sum(y * z - np.log(1 + np.exp(z)))
    return ll

def logistic_regression(X, y, num_steps, learning_rate, add_intercept=True):
    if add_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    weights = np.zeros(X.shape[1])
    for step in range(num_steps):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        # Update weights
        gradient = np.dot(X.T, y - predictions)
        weights += learning_rate * gradient
        # Print log-likelihood every 1000 steps
        if step % 1000 == 0:
            ll = log_likelihood(X, y, weights)
            print(f'Step {step}: log likelihood = {ll}')
    return weights

weights = logistic_regression(X, y, num_steps=10000, learning_rate=0.01)
print(f'Final weights: {weights}')

#Logistic Regression with sklearn
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)
#Logistic Regression with Pytorch

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

def train_model(model, criterion, optimizer, X_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        # Convert numpy arrays to torch tensors
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(y_train).float().view(-1, 1)
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

# Define model, loss function, and optimizer
input_dim = X.shape[1]
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10000
trained_model = train_model(model, criterion, optimizer, X, y, num_epochs)

# Print final weights
for name, param in trained_model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

#Another example with Iris dataset
X, y = load_iris(return_X_y=True)
weights = logistic_regression(X, y, num_steps=10000, learning_rate=0.01)
clf = LogisticRegression(random_state=0).fit(X, y)
trained_model = train_model(model, criterion, optimizer, X, y, num_epochs)