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
itrs = 50
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
opt.model = 'mlp'
opt.batchSize = 50000
opt.learningRate = 0.1
opt.optimization = 'LBFGS'
--opt.momentum = 1
-- fix seed
torch.manualSeed(193000088765774583)

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
   nbTrainingPatches = 5000
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
  
----------------------------------------------------------------------
-- and train!
--

-- model =  torch.load('outputs/net')
seed = 193000088765774583
torch.manualSeed(seed)
NL = 10
HN = 100
model = create_model(opt,HN,HN,NL)
-- model = torch.load('outputs/net_spectral_10')
criterion = nn.ClassNLLCriterion()
 dataset = trainData
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
  outputs = model:forward(in_all)
  linear_nodes, container_nodes = model:findModules('nn.Linear')
  the_node = linear_nodes[9]
  the_out = the_node.output
  U, s_b, V = torch.svd(the_out)
  U = U:t()
--  labels_1 = t_all:eq(1)
--  labels_1 = labels_1:float()
  labels_1  = t_all/t_all:size(1)
  M = torch.Tensor(HN)
  M:zero()
  dots_b = torch.addmv(M,U,labels_1)
train(trainData,model,opt,80)

 criterion = nn.ClassNLLCriterion()
 dataset = trainData
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
  outputs = model:forward(in_all)
  linear_nodes, container_nodes = model:findModules('nn.Linear')
  the_node = linear_nodes[9]
  print(the_node)
  the_out = the_node.output
  print(linear_nodes)
  U, s, V = torch.svd(the_out)
  
  U = U:t()
--  labels_1 = t_all:eq(1)
--  labels_1 = labels_1:float()
 
  labels_1  = t_all/t_all:size(1)
  M = torch.Tensor(HN)
  M:zero()
   print(labels_1)
  dots = torch.addmv(M,U,labels_1)
  
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
gnuplot.figure('')
s_b = s_b:pow(2)
s = s:pow(2)
dots_b = dots_b:pow(2)
dots = dots:pow(2)
--s_b = s_b/s_b:sum()
--s = s/s:sum()
--dots_b = dots_b/dots_b:sum()
--dots = dots/dots:sum()
gnuplot.plot({'initial features',s_b:log(),dots_b:log(),'+'},{'trained features',s:log(),dots:log(),'+'})
gnuplot.movelegend('left','top')
