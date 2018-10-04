require 'torch'
require 'paths'
require 'distributions'
gaussian = {}
gaussian.mu = 0
gaussian.d = 10
function gaussian.loadDataset(n,d)
   means = torch.Tensor(4,d)
   means = means:fill(1)
   means[2][1] = -1
   means[4][2] = -1 
   means[3] = -1*means[1]
--   means[2][4] = 0 
--   means[2][3] = 0 
   sigma = 0.1*torch.eye(d)
   local data = torch.Tensor(4*n,d)
   local labels = torch.Tensor(4*n)
   for i=1,4*n do 
      
      y = math.sign(i%2-0.5)
      ind = i % 4 +1
      print(ind)
      data[i] = distributions.mvn.rnd(means[ind], sigma)
      data[i] = torch.reshape(data[i],d,1,1)
      labels[i] = y
   end
   dataset = {}
   dataset.data = data
   dataset.labels = labels
   function dataset:size()
      return dataset.data:size(1)
   end
   return dataset
end

