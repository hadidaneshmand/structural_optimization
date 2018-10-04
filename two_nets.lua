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
itrs = 100
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
opt.batchSize = 10000
opt.learningRate = 0.1
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
   nbTrainingPatches = 10000
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
NL = 4
HN = 80

model2 = torch.load('outputs/model2')
model3 = torch.load('outputs/model3')
--model3 = create_model(opt,HN,HN,NL+2)
--
--train(trainData,model3,opt,14)
--
--torch.save('outputs/model3',model3)
--
--torch.save('outputs/model2',model2)
--
----model2 = torch.load('outputs/model2')
----model3 = torch.load('outputs/model3')
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
  outputs2 = model2:forward(in_all)
  outputs3 = model3:forward(in_all)
  ind = 4 
  x2 = model2:get(ind).output
  cov2 = x2:t()*x2/x2:size(1)
  x3 = model3:get(ind).output
  cov3 = x3:t()*x3/x3:size(1)
  print(torch.norm(cov2-cov3))

