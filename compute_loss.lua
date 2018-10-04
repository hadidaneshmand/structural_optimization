 f_all = function(in_all,t_all,model,opt)
       -- just in case:
       collectgarbage()
       local model_in = model:clone()
       local parameters,gradParameters = model_in:getParameters()

       -- reset gradients
       gradParameters:zero()

       -- evaluate function for complete mini batch
       local outputs = model_in:forward(in_all)
--       print('---------')
--       print(outputs[1])
--       print(outputs[outputs:size(1)] )
--       print('-----')
       local f = criterion:forward(outputs, t_all)

       -- estimate df/dW
       local df_do = criterion:backward(outputs, t_all)
       model_in:backward(in_all, df_do)

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

       -- return f and df/dX
       return f,gradParameters
 end 