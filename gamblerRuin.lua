x = {{1, 0, 0, 0, 0, 0},
     {0.6, 0, 0.4, 0, 0, 0},
     {0, 0.6, 0, 0.4, 0, 0},
     {0, 0, 0.6, 0, 0.4, 0},
     {0, 0, 0, 0.6, 0, 0.4},
     {0, 0, 0, 0, 0, 1}}

P = torch.Tensor(x)

Praise = P:clone()

for i=1, 10000 do
	Praise = torch.mm(Praise, Praise)
end

-- Case 1 : Start with $2 --

v = {{0, 0, 1 ,0, 0, 0}}

V = torch.Tensor(v)

Prob = torch.mm(V, Praise)

print('Class Probability', Prob)


-- Case 1 : Start with $4 --

v = {{0, 0, 0 ,0, 1, 0}}

V = torch.Tensor(v)

Prob = torch.mm(V, Praise)

print('Class Probability', Prob)