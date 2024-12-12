import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Dot Product
np.dot(3, 4)
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
np.dot(a, b)

#Creating Arrays
np.arange(3,7)
np.arange(3,7,2)
np.ones(5)
np.zeros(5)
np.ones((5,5))
np.zeros((5,5))
np.eye(5)
np.ones((5,5,2))
np.zeros((5,5,2))
np.zeros((5,5))+5
np.linspace(2.0, 3.0, num=5)
np.linspace(2.0, 3.0, num=5, endpoint=False)
#Get Dimmensions of Matrix
a = np.array([[1, 0],[0, 1]])
b = np.array([[4, 1],[2, 2]])
c=np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
a.shape
b.shape
c.shape

#Basic indexing and slicing
b[0,0]
b[:,1]
b[1:2,0]

#Boolean Mask Example

#Transpose Matrix
a.T
np.transpose(a)
v=np.array([1, 2, 3, 4])
np.transpose(v)

#Simple elementwise operations
a+b
a-b
a+c
a+v

#Matrix Multiplication
np.matmul(a, b)
np.matmul(a, c)
np.matmul(a, c)
d=np.array([ [[1, 2,9], [3, 4,8]], [[1, 2,7], [2, 1,6]], [[1, 3,5], [3, 1,4]] ])
np.matmul(a, d)

#While * provides the elementwise product, @ provides the matrix product
a*b
a@b

#Type Conversion
x = np.array([1, 2, 2.5])
x
x.astype(int)

#Rank
b = np.array([[4, 1],[2, 2]])
np.linalg.matrix_rank(b)
np.linalg.matrix_rank(np.ones((4,)))
np.linalg.matrix_rank(np.zeros((4,)))

#Trace
np.trace(np.eye(3))

#Matrix Inversion
a = np.array([[1., 2.], [3., 4.]])
ainv = np.linalg.inv(a)
ainv
ainv = np.linalg.inv(np.matrix(a))
ainv
a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
np.linalg.inv(a)

#Determinant
a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
np.linalg.det(a)

#Solving a linear system
CoeficientMatrix = np.array([[1, 2], [3, 5]])
OutputMatrix = np.array([1, 2])
SolutionMatrix = np.linalg.solve(a, b)
SolutionMatrix

#Eigendecompisition
eigenvalues, eigenvectors =np.linalg.eig(b)
eigenvalues, eigenvectors
eigenvalues, eigenvectors =np.linalg.eig(a)
eigenvalues, eigenvectors

#Raise matrix to power
i = np.array([[0, 1], [-1, 0]])
np.linalg.matrix_power(i, 3)
np.linalg.matrix_power(i, 0)

#Cholesky Decompsition
A = np.array([[1,-2j],[2j,5]])
A
L = np.linalg.cholesky(A)
L
np.dot(L, L.T.conj())

#Creating a random matrix using standard normal random variable
RandomMatrix = np.random.randn(9, 6)

#Uniform Random Matrix
np.random.rand(3,2)

#Normal Random variable
mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 1000)
abs(mu - np.mean(s))
abs(sigma - np.std(s, ddof=1))
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.show()

#Set Seed
np.random.seed(seed=20)


#Random Sampling
np.random.choice(5, 3)
np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
np.random.choice(5, 3, replace=False)

#Shuffle
arr = np.arange(9).reshape((3, 3))
np.random.shuffle(arr)
arr

#Basic Mathematical Functions
np.log(RandomMatrix)
np.exp(RandomMatrix)
np.sqrt(RandomMatrix)
np.sin(RandomMatrix)
np.cos(RandomMatrix)

#Basic Statistical Operations
RandomMatrix.sum(axis=1)
RandomMatrix.sum(axis=0)
RandomMatrix.mean()
RandomMatrix.mean(0)
RandomMatrix.mean(1)
RandomMatrix.median()
RandomMatrix.std()
RandomMatrix.var()
RandomMatrix.min()
RandomMatrix.max()

#QR Decomposition
Q, R = np.linalg.qr(RandomMatrix)
Q,R
np.allclose(RandomMatrix, np.dot(Q, R))

#Norm
a = np.arange(9) - 4
a
b = a.reshape((3, 3))
b
np.linalg.norm(a)
np.linalg.norm(a,2)
np.linalg.norm(a,-2)
np.linalg.norm(a,1)
np.linalg.norm(a,np.inf)

#Singular Value Decomposition
RandomMatrix=np.random.rand(40,38)
U, S, Vh = np.linalg.svd(RandomMatrix, full_matrices=True)
U, S, Vh

#Stacking vectors
x = np.array([0, 1, 2, 3])
A = np.vstack([x, np.ones(len(x))]).T
A
B=np.hstack([x, np.ones(len(x))])
B
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)

#Splitting Vectors
x = np.arange(9.0)
np.split(x, 3)

#Linear Regression with Numpy
x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.vstack([x, np.ones(len(x))]).T
Reg=np.linalg.inv(A.T@A)@A.T@y
Reg
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
m,c
_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, Reg[0]*x + Reg[1], 'r', label='Fitted line')
_ = plt.legend()
plt.show()
_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, m*x + c, 'r', label='Fitted line')
_ = plt.legend()
plt.show()

