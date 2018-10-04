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
opt.batchSize = 100
opt.learningRate = 0.005
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

-- model =  torch.load('outputs/net')
seed = 193000088765774583
 torch.manualSeed(seed)
   NL = 14
   HN = 20
 model = create_model(opt,HN,HN,NL)
-- model = torch.load('outputs/net12')
-- opt.optimization = 'ADAGRAD'
 itrs_policy = torch.Tensor(10,4)
 itrs_policy:fill(0)
 for i=1,2 do 
  itrs_policy[i][1] = i*15 - 5
  itrs_policy[i][2] = 2 
  itrs_policy[i][3] = 4
  itrs_policy[i][4] = 1
 end
 for i=3,10 do 
  itrs_policy[i][1] = 50+(i-2)*8
  itrs_policy[i][3] = 1
  itrs_policy[i][4] = -1
 end
 
 itrs = 300
 noise_policy = torch.Tensor(2)
 noise_policy[1] = 10

-- noise_policy = torch.Tensor(1)
-- noise_policy[1] = 10
 criterion = nn.ClassNLLCriterion()
-- nmodel = expand_net(model,policy,'beta',2)
-- nmodel = shrink_net(nmodel,policy,'beta')
-- loss,norm_grad = f_all(in_all,t_all,model,opt)
-- print('initial loss:' .. loss .. ", norm grad:" .. torch.norm(norm_grad))
-- loss,norm_grad = f_all(in_all,t_all,nmodel,opt)
-- print('after mapping loss:' .. loss .. ", norm grad:" .. torch.norm(norm_grad))
 
 losses_2,grads_2,itrs_2,grads_f2,grads_s2,grads_l2,_ = incremental_train(trainData,model:clone(),opt,itrs,itrs_policy,'beta')
 losses_1,grads_1,itrs_1,grads_f1,grads_s1,grads_l1 = train(trainData,model:clone(),opt,itrs)
----
 losses_3,grads_3,itrs_3,grads_f3,grads_s3,grads_l3 = noisy_train(trainData,model:clone(),opt,itrs, noise_policy, 5.0)
 losses_4,grads_4,itrs_4,grads_f3,grads_s3,grads_l3 = noisy_train(trainData,model:clone(),opt,itrs, noise_policy, 10.0)
  

 gnuplot.figure(1)
 gnuplot.plot({'sgd',itrs_1,((losses_1)),'~'},{'incremental',itrs_2,((losses_2)),'~'},{'noisy 5',itrs_3,((losses_3)),'~'},{'noisy 10',itrs_4,((losses_4)),'~'})
-- gnuplot.plot({'sgd',itrs_1,((losses_1)),'~'})
-- torch.save('outputs/net12',model)