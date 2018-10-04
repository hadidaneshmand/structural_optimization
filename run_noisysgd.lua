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
opt.batchSize = 100
opt.learningRate = 0.005
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
   nbTrainingPatches = 2000
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
 
 criterion = nn.ClassNLLCriterion()
 noisy_itrs = torch.Tensor(3)
 noisy_itrs[1] = 10
 noisy_itrs[2] = 50
 noisy_itrs[3] = 100
 nois_var = 1 
 itrs = 2000
 seed = 1930000887657745831
 torch.manualSeed(seed)
   NL = 6
   HN = 4
   HN2 = 6
 model = create_model(opt,HN,HN,NL+3)
--  model =  torch.load('outputs/net')
-- noisy_model = model:clone()
-- losses_n,grads_n,itrs_n = noisy_train(trainData,noisy_model,opt,itrs,noisy_itrs,nois_var)
 losses,grads,itrs = train(trainData,model,opt,itrs)
 gnuplot.figure(1)
 gnuplot.plot({'sgd',itrs,losses,'~'})
 gnuplot.figure(2)
 gnuplot.plot({'sgd',itrs,grads,'~'})
-- nmodel = expand_net(model,policy,'beta',2)
-- nmodel = shrink_net(nmodel,policy,'beta')
-- loss,norm_grad = f_all(in_all,t_all,model,opt)
-- print('initial loss:' .. loss .. ", norm grad:" .. torch.norm(norm_grad))
-- loss,norm_grad = f_all(in_all,t_all,nmodel,opt)
-- print('after mapping loss:' .. loss .. ", norm grad:" .. torch.norm(norm_grad))

