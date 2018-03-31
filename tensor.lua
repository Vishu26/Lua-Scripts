--- Tensor ---

--- Create a Tensor of dimension 6x7 ---
x = torch.Tensor(6, 7)

--- Create an arbitrary Tensor ---
 a = torch.LongStorage(6)
 a[1], a[2], a[3], a[4], a[5], a[6] = 1, 2, 3, 4, 5, 6;
 x = torch.Tensor(a)

 --- Get dimension of Tensor ---
 x:nDimension()

 --- Get Each dimension of Tensor ---
 x:size()

--- Tensor is contained into a Storage ---
x:storage()

--- the first position used in the Storage is given by storageOffset() ---
x:storageOffset()

---  jump needed to go from one element to another element in the i-th dimension is given by stride(i) ---
x:stride(i) 


 --- Initialize a Tensor ---
x = torch.Tensor(4,5)
s = x:storage()
for i=1,s:size() do  -- fill up the Storage
  s[i] = i
end

x = torch.Tensor(4,5)
i = 0

x:apply(function()
  i = i + 1
  return i
end)

--- Tensor with zeros ---
x = torch.Tensor(5):zero()

--- narrow is used for slicing ---
x:narrow(1, 2, 3):fill(1)

--- Copy Tensor ---
y = torch.Tensor(x:size()):copy(x)
y = x:clone()

x = torch.Tensor(2,5):fill(3.14)

-- Narrow --
x = torch.Tensor(7, 8):fill(1)
y = x:narrow(2, 3, 4)

-- Slicing --
x = torch.Tensor(7, 8):fill(1)
y = x:sub(2, 3)

-- Select --
x = torch.Tensor(7, 8):fill(1)
y = x:select(1, 5)



