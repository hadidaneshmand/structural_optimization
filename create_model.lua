require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'gnuplot'

function create_model(opt,NH,FH,NL,nbands,size_class)
  ----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
  
  local model = nn.Sequential()
  if opt.network == '' then
     -- define model to train
  
     if opt.model == 'convnet' then
        ------------------------------------------------------------
        -- convolutional network 
        ------------------------------------------------------------
        -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
        model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
        model:add(nn.Tanh())
        model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
        -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
        for i=1,NL do 
          model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
          model:add(nn.Tanh())
          model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        end
        -- stage 3 : standard 2-layer MLP:
        model:add(nn.Reshape(64*3*3))
        model:add(nn.Linear(64*3*3, HN))
        model:add(nn.Tanh())
        model:add(nn.Linear(HN, size_class))
        ------------------------------------------------------------

     elseif opt.model == 'mlp' then
        ------------------------------------------------------------
        -- regular 2-layer MLP
        ------------------------------------------------------------
        model:add(nn.Reshape(nbands*32*32))
        model:add(nn.Linear(nbands*32*32, FH))
--        model:add(nn.Tanh())
--        model:add(nn.Linear(FH, NH))
        model:add(nn.Tanh())
        for i=1,NL-3 do 
          model:add(nn.Linear(NH, NH))
          model:add(nn.Tanh())
        end
        model:add(nn.Linear(NH,size_class))
        ------------------------------------------------------------
  
     elseif opt.model == 'linear' then
        ------------------------------------------------------------
        -- simple linear model: logistic regression
        ------------------------------------------------------------
        model:add(nn.Reshape(1024))
        model:add(nn.Linear(1024,size_class))
        ------------------------------------------------------------
  
     else
        print('Unknown model type')
        cmd:text()
        error()
     end
  else
     print('<trainer> reloading previously trained network')
     model = torch.load(opt.network)
  end
    -- verbose
  print('<mnist> using model:')
--  print(model)
  
  ----------------------------------------------------------------------
  -- loss function: negative log-likelihood
  --
  if size_class == 1 then 
--    model:add(nn.Sign())
  else 
    model:add(nn.LogSoftMax())
  
  end 
  
--  criterion = nn.nn.BCECriterion()
  return model
end