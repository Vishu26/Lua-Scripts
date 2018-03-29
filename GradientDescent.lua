torch.manualSeed(42)

X = 50 * (2 * torch.randn(100, 1) - 1)

ones = torch.Tensor(100, 1):fill(1)

X_b = torch.cat(ones, X, 2)

y = 3*X - 2 + 2 * (2 * torch.rand(100, 1) - 1)

epochs = 5000
epsilon = 0.00001
m = 100

theta = torch.randn(2, 1)

for i=1, epochs do

	index = torch.random(1, m)

	xi = X_b[index]
	yi = y[index]

	z = torch.Tensor(2, 1)
	z2 = torch.Tensor(1, 2)

	for i=1, 2 do
		z[i][1] = xi[i]
	end

	for i=1, 2 do
		z2[1][i] = xi[i]
	end

	er = torch.mm(z2, theta) - yi


	gradient = 2 * torch.mm(z, er)
	theta = theta - gradient*epsilon
end

err = torch.sum(torch.pow(torch.mm(X_b, theta) - y, 2))

print('Mean Squared Error :', err)
print('Weight', theta)