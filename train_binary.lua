itrs = 8
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
--opt.coefL2 = 0.001
opt.optimization = 'LBFGS'
opt.model = 'mlp'

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

opt.batchSize = trainData.data:size(1)
----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
--confusion = optim.ConfusionMatrix(classes)


-----------------
--
--
--

----------------------------------------------------------------------
-- and train!
-------------------------------------------
HN = 3 
NL = 1
model = create_model(opt,HN,HN,NL,1,1)
criterion =   nn.SoftMarginCriterion(1)
--outputs = model:forward(trainData.data)
--print('-----')
--print(outputs[1])
--print('++++++')
--print(outputs[outputs:size(1)])
print(model)
train(trainData,model,opt,itrs)
torch.save('outputs/bnet_100_2_mnist',model)