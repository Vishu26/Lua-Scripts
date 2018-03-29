import time
t1 = time.time()

import numpy as np



X = 50 * (2 * np.random.randn(100, 1) - 1)

ones = np.ones((100, 1))

X_b = np.c_[ones, X]

y = 3*X - 2 + 2 * (2 * np.random.rand(100, 1) - 1)

epochs = 50000
epsilon = 0.000001
m = 100
batch_size = 32

theta = np.random.randn(2, 1)

for i in range(epochs):

	index = np.random.permutation(100)

	xi = X_b[index]
	yi = y[index]

	er = xi.dot(theta) - yi

	gradient = xi.T.dot(er)

	theta = theta - gradient*epsilon

error = np.sum(np.square(X_b.dot(theta) - y))

print('Mean Squared Error :', error)
print('Weight', theta)
print('Time Taken', time.time() - t1)