---- add nodes to the layers based on add_policy which is array of size of layers where element i is the number of additional nodes to the layer i. 

function add_node(model,add_policy)
   linear_nodes, container_nodes = model:findModules('nn.Linear')
   local model_out = nn.Sequential()
   model_out:add(nn.Reshape(1024)) -- just for mnist dataset
   outputSize_last = linear_nodes[1].weight:size(2)
   for i=1,#linear_nodes do 
     outputSize = linear_nodes[i].weight:size(1) 
     add_nodes_i = add_policy[i]
     noutputSize = outputSize + add_nodes_i
     ninputSize = outputSize_last
     model_out:add(nn.Linear(ninputSize, noutputSize))
     if i<#linear_nodes then
      model_out:add(nn.Tanh())
     end
     outputSize_last = noutputSize
   end
    model_out:add(nn.LogSoftMax()) -- just for multiclass classification
    return model_out
end

---- add nodes to different layers of network and initializing with different mapping of Amari's paper ... 
function expand_net(smodel,add_policy,mapping,var)
  lmodel = add_node(smodel,add_policy)
  lnodes,_ = lmodel:findModules('nn.Linear')
  snodes,_ = smodel:findModules('nn.Linear')
  for i=1,#lnodes do 
    lnodes[i].weight:zero()
    lnodes[i].bias:zero()
  end
  for i=1,#lnodes do 
    nrow = snodes[i].weight:size(1) 
    ncol = snodes[i].weight:size(2)
    lnodes[i].weight:sub(1,nrow,1,ncol):copy(snodes[i].weight)
    lnodes[i].bias:sub(1,nrow):copy(snodes[i].bias)
  end
  if mapping == 'beta' then 
    print("==beta mapping...")
    for i=1,add_policy:size(1)-1 do
      nsi = add_policy[i]
      input_bias_size = snodes[i].weight:size(1) 
      next_output_size = snodes[i+1].weight:size(1)
      for j=1,nsi do
        input_bias =torch.rand(1)[1]*var
        index = j+input_bias_size
        lnodes[i].bias[index]= input_bias 
       
        for k=1,next_output_size do 
          output_weight = torch.rand(1)[1]*var
          lnodes[i+1].weight[k][index] = output_weight
          lnodes[i+1].bias[k] = lnodes[i+1].bias[k] - math.tanh(input_bias)*output_weight
        end     
      end
    end
  end
  if mapping == 'alpha' then 
    print("-------alpha mapping...")
    outputSize_last = linear_nodes[1].weight:size(2)
    for i=1,add_policy:size(1)-1 do
      nsi = add_policy[i]
      nrow = snodes[i].weight:size(1)
      ncol = lnodes[i].weight:size(2)
      if nsi > 0 then 
       lnodes[i].weight:sub(nrow+1,nsi+nrow):copy(torch.rand(nsi,ncol):mul(var)) 
      end
    end
  end
    
  return lmodel
end
--- shrink down the network structure based on the given policy 
function remove_nodes(model, remove_policy)
   linear_nodes, container_nodes = model:findModules('nn.Linear')
   local model_out = nn.Sequential()
   model_out:add(nn.Reshape(1024)) -- just for mnist dataset
   outputSize_last = linear_nodes[1].weight:size(2)
   for i=1,#linear_nodes do 
     outputSize = linear_nodes[i].weight:size(1) 
     add_nodes_i = remove_policy[i]
     noutputSize = outputSize - add_nodes_i
     ninputSize = outputSize_last
     nmodule = nn.Linear(ninputSize, noutputSize)
     nmodule.weight:copy(linear_nodes[i].weight:sub(1,noutputSize,1,ninputSize))
     nmodule.bias:copy(linear_nodes[i].bias:sub(1,noutputSize))
     model_out:add(nmodule)
     if i<#linear_nodes then
      model_out:add(nn.Tanh())
     end
     outputSize_last = noutputSize
   end
    model_out:add(nn.LogSoftMax()) -- just for multiclass classification
    return model_out
end
function add_layer(model,init_policy,nlayer,geometry)
   
   linear_nodes, container_nodes = model:findModules('nn.Linear')
   local model_out = nn.Sequential()
   model_out:add(nn.Reshape(geometry[1]*geometry[2]*geometry[3])) -- just for mnist dataset
   outputSize_last = linear_nodes[1].weight:size(2)
   for i=1,#linear_nodes-1 do 
     outputSize = linear_nodes[i].weight:size(1) 
     noutputSize = outputSize 
     ninputSize = outputSize_last
     nmodule = nn.Linear(ninputSize, noutputSize)
     nmodule.weight:copy(linear_nodes[i].weight)
     nmodule.bias:copy(linear_nodes[i].bias)
     model_out:add(nmodule)
     if i<#linear_nodes then
      model_out:add(nn.Tanh())
     end
     outputSize_last = noutputSize
   end
   for i =1,nlayer do 
    model_out:add(nn.Linear(outputSize_last,outputSize_last))
    model_out:add(nn.Tanh())
   end
    model_out:add(nn.Linear(outputSize_last,#classes))
    model_out:add(nn.Tanh())
    if init_policy == "incermental" then 
      for i =1,nlayer do 
--      model_out:get(#linear_nodes+i-1).weight:copy(linear_nodes[#linear_nodes-1].weight);
--      model_out:get(#linear_nodes+i-1).bias:copy(linear_nodes[#linear_nodes-1].bias);
        model_out:get(#linear_nodes+i-1).weight:fill(0)
        model_out:get(#linear_nodes+i-1).bias:fill(0)
      end 
--      model_out:get(#linear_nodes+1).weight:copy(linear_nodes[#linear_nodes].weight);
--      model_out:get(#linear_nodes+1).bias:copy(linear_nodes[#linear_nodes].bias);
        model_out:get(#linear_nodes+1).weight:fill(0)
        model_out:get(#linear_nodes+1).bias:fill(0)
    end
    model_out:add(nn.LogSoftMax()) -- just for multiclass classification
    return model_out
end

function copy_layer(source_model,destination_model,nlayer)
   linear_nodes, container_nodes = destination_model:findModules('nn.Linear')
   linear_nodes_s, container_nodes_s = source_model:findModules('nn.Linear')
   for i=1,nlayer do 
     nmodule = nn.Linear(ninputSize, noutputSize)
     linear_nodes[i].weight:copy(linear_nodes_s[i].weight)
     linear_nodes[i].bias:copy(linear_nodes_s[i].bias)
   end
end
function shrink_net(model,remove_policy,mapping)
  smodel = remove_nodes(model,remove_policy)
  if mapping == 'beta' then
    snodes,_ = smodel:findModules('nn.Linear')
    lnodes,_ = model:findModules('nn.Linear')
    for i=1,#lnodes-1 do 
      nr = remove_policy[i]
      ls = lnodes[i].weight:size(1)
      ns = snodes[i+1].weight:size(1)
      if nr > 0 then 
        for j=0,nr-1 do 
          index = ls - j
          bias_r = lnodes[i].bias[index]
          for k=1,ns do 
            vw = lnodes[i+1].weight[k][index]
            snodes[i+1].bias[k] = snodes[i+1].bias[k] + vw*math.tanh(bias_r)
          end
        end
      end
    end
  end
  return smodel
end 

