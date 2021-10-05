import torch
from torch import Tensor
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, reduce_batch_first = False, epsilon=1e-6):
        super(DiceLoss,self).__init__()
        
        self.rbf = reduce_batch_first
        self.e = epsilon        
                 
    def dice_coeff(self, input: Tensor, target: Tensor):
      assert input.size() == target.size()
      if input.dim() == 2 and self.rbf:
          raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

      if input.dim() == 2 or self.rbf:
          intersection = torch.dot(input.reshape(-1), target.reshape(-1))
          sets_sum = torch.sum(input) + torch.sum(target)
          if sets_sum.item() == 0:
              sets_sum = 2 * inter

          return (2 * intersection + self.e) / (sets_sum + self.e)
      else:
          dice = 0
          for i in range(input.shape[0]):
              dice += dice_coeff(input[i, ...], target[i, ...])
          return dice / input.shape[0]


    def dice_loss(self, input: Tensor, target: Tensor):
      assert input.size() == target.size()
      dice = 0
      for channel in range(input.shape[1]):
          dice += self.dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
      dice = dice/input.shape[1]    
      return 1 - dice
