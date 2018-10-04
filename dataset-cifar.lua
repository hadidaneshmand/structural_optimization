require 'torch'
require 'paths'
require 'image'
cifar = {}
cifar.classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

cifar.path_remote = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
cifar.path_dataset = 'cifar-10-batches-t7'
cifar.path_trainset = {}
cifar.path_trainset[1] = paths.concat(cifar.path_dataset, 'data_batch_1.t7')
cifar.path_trainset[2] = paths.concat(cifar.path_dataset, 'data_batch_2.t7')
cifar.path_trainset[3] = paths.concat(cifar.path_dataset, 'data_batch_3.t7')
cifar.path_trainset[4] = paths.concat(cifar.path_dataset, 'data_batch_4.t7')
cifar.path_testset = paths.concat(cifar.path_dataset, 'test_batch.t7')



function cifar.loadTestSet(maxLoad, geometry)
--   return cifar.loadDataset(cifar.path_testset, maxLoad, geometry)
end

function cifar.loadTrainSet(maxload)
  
   local data = torch.Tensor(50000, 3072)
   local labels = torch.Tensor(50000)
   for i = 0,4 do
     subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
     if i>maxload then 
        break
     end
     data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
     labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
   end
   
   data = data[{ {1,maxload} }]
  
   labels = labels[{{1,maxload}}]
   local nExample =data:size(1)
   data = data:reshape(nExample,3,32,32)
   print('<cifar> done')

   local dataset = {}
   dataset.data = data
   dataset.labels = labels
   
--   function dataset:normalize(mean_, std_)
--      
--      local mean = mean_ or data:view(data:size(1), -1):mean(1)
--      local std = std_ or data:view(data:size(1), -1):std(1, true)
--      print("\n----------------\n")
--      print(data:dim())
----      for i=1,data:size(1) do
----         data[i]:add(-mean[1][i])
----         if std[1][i] > 0 then
----            tensor:select(2, i):mul(1/std[1][i])
----         end
----      end
----      return mean, std
--   end

   function dataset:normalizeGlobal(mean_, std_)
      print("\n----------------\n")
--      print(data:dim())
      normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
      for i = 1,nExample do
         -- rgb -> yuv
         local rgb = data[i]
--         print(rgb)
         local yuv = image.rgb2yuv(rgb)
         -- normalize y locally:
         yuv[1] = normalization(yuv[{{1}}])
         data[i] = yuv
      end
      -- normalize u globally:
      mean_u = data[{ {},2,{},{} }]:mean()
      std_u = data[{ {},2,{},{} }]:std()
      data[{ {},2,{},{} }]:add(-mean_u)
      data[{ {},2,{},{} }]:div(-std_u)
      -- normalize v globally:
      mean_v = data[{ {},3,{},{} }]:mean()
      std_v = data[{ {},3,{},{} }]:std()
      data[{ {},3,{},{} }]:add(-mean_v)
      data[{ {},3,{},{} }]:div(-std_v)
   end

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
           local input = self.data[index]
           local class = self.labels[index]
           local label = labelvector:zero()
           label[class+1] = 1
           local example = {input, label}
                                       return example
   end})

   return dataset
end
