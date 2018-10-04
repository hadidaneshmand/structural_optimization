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
require 'gaussian-dataset'
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
function math.sign(x)
   if x<0 then
     return -1
   elseif x>0 then
     return 1
   else
     return 0
   end
end
--opt.coefL2 = 0.001
opt.optimization = 'LBFGS'
torch.manualSeed(10)

opt.learningRate = 0.01
n = 1000 
d = 2 
mu = 1
m = 10
opt.batchSize = n
geometry = torch.Tensor({d,1,1})
dataset = gaussian.loadDataset(n,d)
model = nn.Sequential()
model:add(nn.Reshape(d))
model:add(nn.Linear(d, m))
--model:add(nn.Tanh())
--model:add(nn.Linear(m,m))
model:add(nn.Tanh())
model:add(nn.Linear(m,1,false))
criterion = nn.MSECriterion()
--criterion =   nn.SoftMarginCriterion(1)
losses0,grads0,itrs0 = train(dataset,model,opt,100)
--layer = model:get(3) 
--outs = layer.output
--print(torch.mean(outs,1))
--print(torch.mean(torch.mean(outs,1)))
--print(outs[2])

w = model:get(2).weight
for i=1,4 do 
  y = math.sign(i % 2 -0.5 )
--  means[i] = y*means[i]
end
--print(torch.dot(w[1],w[2]))
print(w*means:t())
--means = torch.Tensor(2,d)
--means:fill(0)
--for i=1,dataset.size(1) do 
--  if dataset.labels[i] == 1 then 
--   ind = 1
--  else 
--   ind = 2 
--  end
--  means[ind] = means[ind] + dataset.data[i]/(2*n)
--end
pos = dataset.labels:eq(1)
neg = dataset.labels:eq(-1)
indices = torch.linspace(1,pos:size(1),pos:size(1)):long()
--print(dataset.data:sub({indices[dataset.labels:eq(1)]:,1}):size())
--print(dataset.data:sub({neg,2}):size())
--print(dataset.data:size())
--gnuplot.plot(dataset.data:t()[1],dataset.data:t()[2],'+')
t = model:get(4).weight
torch.save('data.csv',dataset.data,'ascii')
torch.save('weights.csv',model:get(2).weight,'ascii')
torch.save('b.csv',model:get(2).bias,'ascii')
torch.save('labels.csv',dataset.labels,'ascii')
torch.save('weights2.csv',t,'ascii')
--print(pos)
--mu1 = torch.mean(dataset.data[pos])
--print(mu1)
--w_opt = torch.Tensor({1,0,1,0,1})
--error = 0 
--for i=1,n do 
-- dotb = (torch.dot(w_opt,dataset.data[i]))
---- error = error + math.pow(out_b-dataset.labels[i],2)
-- if (dotb*dataset.labels[i])<0 then 
--   error = error + 1 
-- end
--end
--print(error/n)

--dataset.data = dataset.data + 1
--losses,grads,itrs = train(dataset,model,opt,1000)
--gnuplot.plot({'0',itrs0,losses0},{'1',itrs,losses})