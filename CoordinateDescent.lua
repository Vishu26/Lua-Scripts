t1 = os.clock()

torch.manualSeed(42)

X =  torch.randn(100, 1) - 1
X2 = torch.randn(100, 1) - 1
X3 = torch.randn(100, 1) - 1


X_b = torch.Tensor(100, 4)
X_b[{{1, 100}, 1}]:fill(1)
X_b[{{1, 100}, 2}] = X
X_b[{{1, 100}, 3}] = X2
X_b[{{1, 100}, 4}] = X3

X_b1 = X_b:clone()
X_b[{{1, 100}, 3}]:fill(0)
X_b[{{1, 100}, 4}]:fill(0)

X_b2 = X_b:clone()
X_b[{{1, 100}, 2}]:fill(0)
X_b[{{1, 100}, 4}]:fill(0)

X_b3 = X_b:clone()
X_b[{{1, 100}, 2}]:fill(0)
X_b[{{1, 100}, 3}]:fill(0)


y = 3*X - 2*X2 + 4*X3 - 7 + 2 * (2 * torch.rand(100, 1) - 1)


epochs = 100
epsilon = 0.001
m = 100
batch_size = 32

theta = torch.randn(4, 1)

for i=1, epochs do
	for k=1, m do

		index = torch.randperm(m)
		index = index:narrow(1, 1, batch_size)

		xi = X_b1:index(1, index:long())
		yi = y:index(1, index:long())

		er = torch.mm(xi, theta) - yi


		gradient = 2 * torch.mm(xi:t(), er)
		theta = theta - gradient*epsilon
	end
end

for i=1, epochs do
	for k=1, m do


		index = torch.randperm(m)
		index = index:narrow(1, 1, batch_size)

		xi = X_b2:index(1, index:long())
		yi = y:index(1, index:long())

		er = torch.mm(xi, theta) - yi


		gradient = 2 * torch.mm(xi:t(), er)
		theta = theta - gradient*epsilon
	end
end

for i=1, epochs do
	for k=1, m do

		index = torch.randperm(m)
		index = index:narrow(1, 1, batch_size)

		xi = X_b3:index(1, index:long())
		yi = y:index(1, index:long())

		er = torch.mm(xi, theta) - yi


		gradient = 2 * torch.mm(xi:t(), er)
		theta = theta - gradient*epsilon
	end
end

err = torch.sum(torch.pow(torch.mm(X_b, theta) - y, 2))

print('Mean Squared Error :', err)
print('Weight', theta)
print('Time Taken', os.clock() - t1)