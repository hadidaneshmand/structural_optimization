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
require 'train_layerwise'
require 'compute_loss'
require 'structural_modifications'
require 'incremental_train'
require 'noisy_train'
torch.setheaptracking(true)
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
opt.batchSize = 50000
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
subclasses = torch.Tensor(2)
subclasses[1] = 1
subclasses[2] = 2

  -- geometry: width and height of input images
geometry = {1,32,32}
 

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 50000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData.data,trainData.labels = trainData:binary(subclasses,geometry)
print(trainData)
classes = {classes[subclasses[1]],classes[subclasses[2]]}
--trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--
-----------------
--
--
--
  in_all = trainData.data
  t_all = trainData.labels
 
----------------------------------------------------------------------
-- and train!
-------------------------------------------
--#criterion = nn.ClassNLLCriterion()
model =  torch.load('outputs/bnet_100_4_mnist')
criterion =   nn.SoftMarginCriterion(1)
f,gd = f_all(in_all,t_all,model,opt)
outputs = model:forward(in_all)
------------------------------------
---- saving statistics 
--------------------------------------
--file = torch.DiskFile('outputs/optstat_bnet_100_4_mnist.txt', 'w')
--file:writeString(4 .. '\n')
--for i=1,4 do 
--  outs = model:get(2*i-1).output
--  ws = model:get(2*i).weight
--  file:writeString(ws:size(1) .. '\n')
--  for j=1,ws:size(1) do 
--    dots = outs*ws[j]
--    for k=1,2 do
--      if k == 1 then 
--        inds = t_all:eq(1)
--      else 
--        inds = t_all:eq(-1)
--      end 
--  --  gnuplot.hist(dots[inds])
--      for m=1,inds:size(1) do 
--        if inds[m] == 1 then
--          file:writeString(dots[m] .. ',')
--        end
--      end
--      file:writeString('\n')
--    end
--  end
--end
--file:close()
--
-- the input of the function is m \times n matrix and it tries to estimate the first row by 
function linear_dep(x)

  a = torch.reshape(x[1],1,x[1]:size(1))
  subx = x:sub(2,-1)
  m = subx*subx:t()
--  if x:size(2) ~= 1024 then
--    t,p = torch.symeig(m)
--    print(t)
--  end
  coefs = a*subx:t()*torch.inverse(m)
  local err = torch.norm(a-coefs*subx)
--  print('error = ' .. err)
--  print(coefs)
  if x:size(2) ~= 1024 then
    print(coefs)
    print('error = ' .. err)
  end
  return err
end
--x = torch.randn(3,3)
--x[3] = x[1] + x[2]
--linear_dep(x)
for i=1,4 do 
  outs = model:get(2*i-1).output
  d = outs:size(2)
  cov = torch.Tensor(2,d,d)
  cov = cov:fill(0)
  mu = torch.Tensor(2,d)
  mu = mu:fill(0)
  nnn = outs:size(1)
  for ii=1,nnn do
--    print(mu) 
--    print(torch.mul(outs[ii],t_all[ii]/(1.0*nnn)))
   
    if t_all[ii] == 1 then 
      ind = 1
    else 
      ind =2
    end
    mu[ind] = mu[ind] + torch.mul(outs[ii],1.0/(1.0*nnn))
    cov[ind] = cov[ind]:addr(1,1.0/(1.0*nnn),outs[ii],outs[ii])
--    cov[ind] = cov[ind] + torch.mul(torch.diag(outs[ii]),1.0/(1.0*nnn))
  end
  err = 0 
  ws = model:get(2*i).weight
  print(ws[1]:size())
  for j=1,ws:size(1) do 
    wi = ws[j]
    x = torch.Tensor(4,d)
    x[1] = mu[1] 
    x[2] = cov[1]*wi
    x[3] = (cov[2])*wi
    x[4] = mu[2]
    oerr = linear_dep(x)
    err = err + oerr/(1.0*ws:size(1))
  end
  print('layer[' .. i .. ']:' .. err)
end

