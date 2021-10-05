import torch
from torch import Tensor
import torch.nn as nn

class DiceLoss():

    def __init__(self):

        super(DiceLoss,self).__init__()
 
    def dice_coefficient(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
        
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        if input.dim() == 2 and reduce_batch_first:
            raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

        if input.dim() == 2 or reduce_batch_first:
            inter = torch.dot(input.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(input) + torch.sum(target)
        
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0
            for i in range(input.shape[0]):
                dice += self.dice_coefficient(input[i, ...], target[i, ...])
            return dice / input.shape[0]


    def multiclass(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):

        # Average of Dice coefficient for all classes
        assert input.size() == target.size()
        dice = 0
        for channel in range(input.shape[1]):
            dice += self.dice_coefficient(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

        return dice / input.shape[1]


    def dice_loss(self, input: Tensor, target: Tensor, multiclass: bool = False):

        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        fn = self.multiclass if multiclass else self.dice_coefficient
        return 1 - fn(input, target, reduce_batch_first=True) , fn(input, target, reduce_batch_first=True) 
