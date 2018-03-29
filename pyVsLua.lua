t1 = os.clock()

torch.manualSeed(42)

X = 50 * (2 * torch.randn(100, 1) - 1)

ones = torch.Tensor(100, 1):fill(1)

X_b = torch.cat(ones, X, 2)

y = 3*X - 2 + 2 * (2 * torch.rand(100, 1) - 1)

epochs = 50000
epsilon = 0.000001
m = 100
batch_size = 32

theta = torch.randn(2, 1)

for i=1, epochs do

	index = torch.randperm(m)
	index = index:narrow(1, 1, batch_size)

	xi = X_b:index(1, index:long())
	yi = y:index(1, index:long())

	er = torch.mm(xi, theta) - yi


	gradient = 2 * torch.mm(xi:t(), er)
	theta = theta - gradient*epsilon
end

err = torch.sum(torch.pow(torch.mm(X_b, theta) - y, 2))

print('Mean Squared Error :', err)
print('Weight', theta)
print('Time Taken', os.clock() - t1)