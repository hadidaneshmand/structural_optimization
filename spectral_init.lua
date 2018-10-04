----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------
itrs = 5
require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'gnuplot'
require 'create_model'
require 'train'
require 'compute_loss'
require 'structural_modifications'
require 'incremental_train'
require 'noisy_train'
----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
]]
--opt.coefL1 = 0.001
opt.optimization = 'LBFGS'
opt.model = 'mlp'
opt.batchSize = 8000
opt.learningRate = 0.0005
--opt.momentum = 1
rep = 1

 losses_2 = nil
 grads_2 = nil 
 itrs_2 = nil
  losses_1 = nil
 grads_1 = nil 
 itrs_1 = nil


-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

classes = {'1','2','3','4','5','6','7','8','9','10'}
  -- geometry: width and height of input images
geometry = {32,32}
 

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 8000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-----------------
--
--
--
  in_all = torch.Tensor(trainData:size(),1,geometry[1],geometry[2])
  t_all = torch.Tensor(trainData:size())
 for i = 1,trainData:size() do
   -- load new sample
   local sample = trainData[i]
   local input = sample[1]:clone()
   local _,target = sample[2]:clone():max(1)
   target = target:squeeze()
   in_all[i] = input
   t_all[i] = target
 end
  dataset = trainData
----------------------------------------------------------------------
-- and train!
--

-- model =  torch.load('outputs/net')
for kk=1,rep do 
torch.manualSeed(19300008865773+kk*1382398)
print("pass k = " .. kk .. "------------")
    
    NL = 3
    HN = 100
     
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
      model = create_model(opt,HN,HN,NL)
      copy_model = model:clone()
      criterion = nn.ClassNLLCriterion()
      outputs = model:forward(in_all)
      linear_nodes, container_nodes = model:findModules('nn.Linear')
      main_input = model:get(1).output 
      cov_mat = main_input:t()*main_input/main_input:size(1)
      U,s,V = torch.svd(cov_mat)
      U = U:t()
      target_node = 2 
      print(torch.norm(model:get(target_node).weight))
      w =  model:get(target_node).weight 
      abs_s = s:abs()
      inds = abs_s:lt(0.00001)
--      for i=1,w:size(1) do
--          r = torch.random(1,inds:size(1))
--    --    for j=1,abs_s:size(1) do  
--    --      if abs_s[j]< 0.0001 then
--          if abs_s[r] > 0 and abs_s[r]<0.01 then
--            w[i] = w[i] + U[r]
--          end 
--    --    end 
--      end
      model:get(target_node).weight = w*cov_mat
      if losses_2 == nil then 
        losses_1,grads_1,itrs_1 = train(trainData,copy_model:clone(),opt,itrs)
        losses_2,grads_2,itrs_2 = train(trainData,copy_model:clone(),opt,itrs)
      else 
        losses_1t, grads_1t, itrs_1t = train(trainData,model:clone(),opt,itrs)
        losses_2t, grads_2t, itrs_2t = train(trainData,copy_model:clone(),opt,itrs)
        
        losses_1 = losses_1 + losses_1t
        grads_1 = grads_1 + grads_1t 
        losses_2 = losses_2+ losses_2t
        grads_2 = grads_2+ grads_2t 
      end  
 end 
 losses_1 = losses_1/rep
 losses_2 = losses_2/rep 

 gnuplot.figure(1)
 gnuplot.plot({'loss homot',itrs_1,((losses_1)),'~'},{'loss random',itrs_2,((losses_2)),'~'})
 gnuplot.plot({'grad1 homot',itrs_1,((f_grads_1)),'~'},{'grad1 random',itrs_2,((f_grads_2)),'~'})
 gnuplot.plot({'gradl homot',itrs_1,((l_grads_1)),'~'},{'gradl random',itrs_2,((l_grads_2)),'~'})
-- gnuplot.figure(2)
-- gnuplot.plot({'grad homot',itrs_1,((grads_1)),'~'},{'grad random',itrs_2,((grads_2)),'~'})
--  linear_nodes, container_nodes = model:findModules('nn.Linear')
--  the_node = linear_nodes[4]
--  the_out = the_node.output
--  print(linear_nodes)
--  U, s, V = torch.svd(the_out)
--  U = U:t()
--  labels_1 = t_all:eq(1)
--  labels_1 = labels_1:float()
--  M = torch.Tensor(HN)
--  M:zero()
--  dots = torch.addmv(M,U,labels_1)
--  inner_node = linear_nodes[0]
--  U2,s2,V2 = torch.svd(inner_node.output)
--  U2 = U2:t()
--  dots2 = torch.addmv(M,U2,labels_1)
--  last_node = linear_nodes[8]
--  U3,s3,V3 = torch.svd(last_node.output)
--  U3 = U3:t()
--  dots3 = torch.addmv(M,U3,labels_1)
--  torch.save('outputs/net_spectral',model)
--  print(dots)
--gnuplot.figure('')
--gnuplot.plot({'layer 4',s:log(),dots:abs():log(),'+'},{'layer 1',s2:log(),dots2:abs():log(),'+'},{'layer 8',s3:log(),dots3:abs():log(),'+'})

