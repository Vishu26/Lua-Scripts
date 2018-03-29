require 'dp'
require 'nn'
require 'optim'

ds = dp.Mnist()

X_train = ds:get('train', 'input', 'bchw')
y_train = ds:get('train', 'target', 'b')

X_test = ds:get('test', 'input', 'bchw')
y_test = ds:get('test', 'target', 'b')

model = nn.Sequential()
model:add(nn.Convert('bchw', 'bf'))
model:add(nn.Linear(1*28*28, 20))
model:add(nn.Tanh())
model:add(nn.Linear(20, 10))
model:add(nn.LogSoftMax())



loss = nn.ClassNLLCriterion()


function trainEpoch(model, loss, input, target, batch_size)

    local idxs = torch.randperm(input:size(1))

    for i=1, input:size(1), batch_size do

        if i + batch_size > input:size(1) then
            idx = idxs:narrow(1, i, input:size(1) - i)
        else
            idx = idxs:narrow(1, i, batch_size)
        end

        local batchInputs = input:index(1, idx:long())
        local batchLabels = target:index(1, idx:long())
        local params, grad = model:getParameters()
        local optimState = {learningRate = 0.1, momentum = 0.9}

        function feval(params)

            grad: zero()
            local outputs = model:forward(batchInputs)
            local loss = loss:forward(outputs, batchLabels)
            local dloss_doutputs = loss:backward(outputs, batchLabels)
            model:backward(batchInputs, dloss_doutputs)
            return loss, grad
        end

        optim.sgd(feval, params, optimState)

    end
    idx = nil
    collectgarbage()
end

trainEpoch(model, loss, X_train, y_train, 32)





