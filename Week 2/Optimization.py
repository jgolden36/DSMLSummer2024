import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import torch
import torch.optim as optim
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

# Example: Gradient Descent for a simple quadratic function
def f(x):
    return x**2 + 4*x + 4

def gradient(x):
    return 2*x + 4

x = 10
learning_rate = 0.1
for i in range(10000):
    grad = gradient(x)
    x = x - learning_rate * grad
    print(f"Iteration {i}: x = {x}, f(x) = {f(x)}")

# Example: Newton's Method for a simple quadratic function
def secondDerivative(x):
    return 2

for i in range(100):
    Newtons = gradient(x)/secondDerivative(x)
    x = x - learning_rate * Newtons
    print(f"Iteration {i}: x = {x}, f(x) = {f(x)}")

def objective(x):
    return x**2 + 4*x + 4

x0 = 10
result = minimize(objective, x0, method='BFGS')
print(result)

# Example: Hyperparameter tuning for logistic regression
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Example: Using Adam optimizer in PyTorch
model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

#Example: Using SciPy minimize with Constraints

def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0] + x[1] - 1

x0 = [0.5, 0.5]
con = {'type': 'eq', 'fun': constraint}
result = minimize(objective, x0, method='SLSQP', constraints=con)
print(result)

#Example: Global optimization
def objective(x):
    return x[0]**2 + x[1]**2

bounds = [(-5, 5), (-5, 5)]
result = differential_evolution(objective, bounds)
print(result)

#Example: Rock paer scissors game thoery with linear programming
class RPSAgent(object):
    def __init__(self):
        pass

    def solve(self, R):
        RockReward=R[1][2]
        PaperReward=R[0][1]
        SReward=R[2][0]
        print(SReward)
        print(RockReward)
        print(PaperReward)
        RockProb = cp.Variable()
        SProb=cp.Variable()
        PaperProb = cp.Variable()
        utility = cp.Variable()
        constraints = [utility<=PaperProb*SReward-SProb*PaperReward,utility<=SProb*RockReward-RockProb*SReward,utility<=RockProb*PaperReward-PaperProb*RockReward,PaperProb+RockProb+SProb==1,PaperProb>=0,RockProb>=0,SProb>=0]
        objective = cp.Maximize(utility)
        problem = cp.Problem(objective, constraints)
        solution = problem.solve()
        strategyMatrix=[RockProb.value,SProb.value,PaperProb.value]
        print(strategyMatrix)
        return strategyMatrix

import unittest

class TestRPS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = RPSAgent()

    def test_case_1(self):
        R = [
            [0,1,-1],[-1,0,1],[1,-1,0]
        ]

        np.testing.assert_almost_equal(
            self.agent.solve(R),
            np.array([0.333, 0.333, 0.333]),
            decimal=3
        )
        
    def test_case_2(self):
        R = [[0,  2, -1],
            [-2,  0,  1],
            [1, -1,  0]]
    
        np.testing.assert_almost_equal(
            self.agent.solve(R),
            np.array([0.250, 0.250, 0.500]),
            decimal=3
        )

unittest.main(argv=[''], verbosity=2, exit=False)