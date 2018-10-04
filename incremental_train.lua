require 'compute_loss'
require 'structural_modifications'
function incremental_train(dataset,model_in,opt_in,epoch_in,itrs_policy,mapping)
   -- epoch tracker
   local epoch = epoch_in
   local opt = opt_in
   local parameters,gradParameters = model_in:getParameters()
   local linear_nodes, container_nodes = model_in:findModules('nn.Linear')
   local NL = #linear_nodes
   epoch = epoch or 1
   

   -- local vars
   local time = sys.clock()

   local in_all = torch.Tensor(dataset:size(),1,geometry[1],geometry[2])
   local t_all = torch.Tensor(dataset:size())
   for i = 1,dataset:size() do
     -- load new sample
     local sample = dataset[i]
     local input = sample[1]:clone()
     local _,target = sample[2]:clone():max(1)
     target = target:squeeze()
     in_all[i] = input
     t_all[i] = target
   end
  
--   parameters:fill(1)
   -- do one epoch
--   print('<trainer> on training set:')
--   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   local losses = torch.Tensor(math.floor(epoch*dataset:size()/opt.batchSize))
   local grads = torch.Tensor(math.floor(epoch*dataset:size()/opt.batchSize))
   local grads_1 = torch.Tensor(math.floor(epoch*dataset:size()/opt.batchSize))
   local grads_2 = torch.Tensor(math.floor(epoch*dataset:size()/opt.batchSize))
   local grads_l = torch.Tensor(math.floor(epoch*dataset:size()/opt.batchSize))
   local itrs = torch.Tensor(math.floor(epoch*dataset:size()/opt.batchSize))
   local loss_index = 1
   local large_structure = false
   for tt = 1, epoch do 
     if tt>1 then
      print('-----------epoch:' .. tt .. ',loss=' .. losses[loss_index-1] .. ',grad=' .. grads[loss_index-1])
     end
     local modification = false
     local index = 1
     for k=1,itrs_policy:size(1) do 
       if itrs_policy[k][1] == tt then 
        modification = true
        index = k
        break
       end
     end
     if modification == true then 
        if itrs_policy[index][4] == -1 then 
--         remove nodes
           print('===== removing nodes\n') 
           structure_policy = torch.Tensor(NL)
           structure_policy:fill(itrs_policy[index][3])
           structure_policy[-1] = 0
           model_in = shrink_net(model_in,structure_policy,mapping)
           large_structure = false
           print(model_in)
        else 
--         add nodes
           print('===== adding nodes\n')
           large_structure = true
           structure_policy = torch.Tensor(NL)
           structure_policy:fill(itrs_policy[index][3])
           structure_policy[-1] = 0
           print(structure_policy)
           model_in = expand_net(model_in,structure_policy,mapping,itrs_policy[index][2])
           print(model_in)
        end
        parameters,gradParameters = model_in:getParameters()
     end
     for t = 1,dataset:size(),opt.batchSize do
        -- create mini batch
        local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
        local targets = torch.Tensor(opt.batchSize)
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
           -- load new sample
           local sample = dataset[i]
           local input = sample[1]:clone()
           local _,target = sample[2]:clone():max(1)
           target = target:squeeze()
           inputs[k] = input
           targets[k] = target
           k = k + 1
        end
  
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
           -- just in case:
           collectgarbage()
  
           -- get new parameters
           if x ~= parameters then
              parameters:copy(x)
           end
  
           -- reset gradients
           gradParameters:zero()
  
           -- evaluate function for complete mini batch
           local outputs = model_in:forward(inputs)
           local f = criterion:forward(outputs, targets)
  
           -- estimate df/dW
           local df_do = criterion:backward(outputs, targets)
           model_in:backward(inputs, df_do)
  
           -- penalties (L1 and L2):
           if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
              -- locals:
              local norm,sign= torch.norm,torch.sign
  
              -- Loss:
              f = f + opt.coefL1 * norm(parameters,1)
              f = f + opt.coefL2 * norm(parameters,2)^2/2
  
              -- Gradients:
              gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
           end
  
           -- update confusion
           for i = 1,opt.batchSize do
              confusion:add(outputs[i], targets[i])
           end
  
           -- return f and df/dX
           return f,gradParameters
        end
  
        -- optimize on current mini-batch
        if opt.optimization == 'LBFGS' then
  
           -- Perform LBFGS step:
           lbfgsState = lbfgsState or {
              maxIter = opt.maxIter,
              lineSearch = optim.lswolfe
           }
           optim.lbfgs(feval, parameters, lbfgsState)
         
           -- disp report:
           print('LBFGS step')
           print(' - progress in batch: ' .. t .. '/' .. dataset:size())
           print(' - nb of iterations: ' .. lbfgsState.nIter)
           print(' - nb of function evalutions: ' .. lbfgsState.funcEval)
  
        elseif opt.optimization == 'SGD' then
          
           -- Perform SGD step:
           sgdState = {
              learningRate = opt.learningRate,
              momentum = opt.momentum,
              learningRateDecay = 5e-7
           }
           optim.sgd(feval, parameters, sgdState)
           
        elseif opt.optimization == 'ADAGRAD' then
            adagradState = {
              learningRate = opt.learningRate,
              learningRateDecay = 5e-7
           }
           optim.adagrad(feval, parameters, adagradState)
        elseif opt.optimization == 'ADAM' then
          adamState = {
            learningRate = opt.learningRate,
            learningRateDecay = 5e-7
         }
         optim.adam(feval, parameters, adamState)   
        
        else
           error('unknown optimization method')
        end
        local loss,full_grad = f_all(in_all,t_all,model_in,opt)
         local W,WG = model_in:parameters()
  --           print(GW)
--         print('\n loss = ' .. loss .. '\n')
--         print(' norm of gradient :' .. torch.norm(full_grad) .. '\n')
--         print('parametes:[0]=' .. parameters[1] .. ",[1]=" .. parameters[2])
         -- disp progress
         xlua.progress(t, dataset:size())
         losses[loss_index] = loss
         grads[loss_index] = torch.norm(full_grad)
         grads_1[loss_index] = torch.norm(WG[1]:reshape(W[1]:size(1)*W[1]:size(2)))
         grads_2[loss_index] = torch.norm(WG[3]:reshape(W[3]:size(1)*W[3]:size(2)))
         grads_l[loss_index] = torch.norm(WG[17]:reshape(W[17]:size(1)*W[17]:size(2)))
         itrs[loss_index] = tt + t/dataset:size()
         loss_index = loss_index + 1
     end
   end
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
--   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  
  
   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
--   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
   return losses,grads,itrs,grads_1,grads_2,grads_l,model_in
end