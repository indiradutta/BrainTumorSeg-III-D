## Model Architecture
![linknet3d_architecture](https://user-images.githubusercontent.com/66861243/135907960-04689cc1-dbab-4128-8f7f-e4912a730198.png)



## Model Initialization
The LinkNet model may be initialized and the state dict may be viewed by running the following code snippet:

```python
from linknet import LinkNet

net = LinkNet()
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
